import asyncio
import base64
import logging
import random
import time
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple

# OpenAI specific imports
from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError, APIConnectionError, BadRequestError

# Google Gemini specific imports
from google import genai as google_genai
from google.genai import types as google_types
from google.api_core import exceptions as google_api_exceptions

from .constants import PROVIDERS_SUPPORTING_USERNAMES, GEMINI_SAFETY_SETTINGS_DICT, AllKeysFailedError
from .rate_limiter import get_db_manager, get_available_keys
from .utils import _truncate_base64_in_payload, default_serializer

async def generate_response_stream(
    provider: str,
    model_name: str,
    history_for_llm: List[Dict[str, Any]],
    system_prompt_text: Optional[str],
    provider_config: Dict[str, Any],
    extra_params: Dict[str, Any]
) -> AsyncGenerator[Tuple[Optional[str], Optional[str], Optional[Any], Optional[str]], None]:
    """
    Generates a response stream from the specified LLM provider.

    Handles API key selection, retries, payload preparation, streaming, and error handling.

    Args:
        provider: The name of the LLM provider (e.g., "openai", "google").
        model_name: The specific model name (e.g., "gpt-4.1", "gemini-2.0-flash").
        history_for_llm: The conversation history formatted for the LLM API.
        system_prompt_text: The system prompt string, if any.
        provider_config: Configuration dictionary for the provider (base_url, api_keys).
        extra_params: Dictionary of extra API parameters for the model.

    Yields:
        Tuple containing:
        - text_chunk (Optional[str]): A chunk of text from the response stream.
        - finish_reason (Optional[str]): The reason the generation finished (e.g., "stop", "length", "safety").
        - grounding_metadata (Optional[Any]): Grounding metadata from the response (currently Gemini specific).
        - error_message (Optional[str]): An error message if a non-retryable error occurred during streaming.

    Raises:
        AllKeysFailedError: If all available API keys for the provider fail.
        ValueError: If configuration is invalid (e.g., missing keys for required provider).
    """
    all_api_keys = provider_config.get("api_keys", [])
    base_url = provider_config.get("base_url")
    is_gemini = provider == "google"

    # Check if keys are required for this provider
    keys_required = provider not in ["ollama", "lmstudio", "vllm", "oobabooga", "jan"] # Add other keyless providers

    if keys_required and not all_api_keys:
        raise ValueError(f"No API keys configured for the selected provider '{provider}' which requires keys.")

    available_llm_keys = await get_available_keys(provider, all_api_keys)
    llm_db_manager = get_db_manager(provider)

    if keys_required and not available_llm_keys:
        raise AllKeysFailedError(provider, ["No available (non-rate-limited) API keys."])

    random.shuffle(available_llm_keys)
    keys_to_loop = available_llm_keys if keys_required else ["dummy_key"] # Loop once for keyless
    llm_errors = []
    last_error_type = None

    # --- History Correction Logic ---
    # This logic prepares `history_to_use_for_api_call` based on whether the current call is for Gemini or OpenAI-compatible.
    # `history_for_llm` is the original history built by `build_message_history`.
    history_to_use_for_api_call = []
    if is_gemini:
        logging.debug("Preparing history for Gemini API format from input history_for_llm.")
        for msg_index, original_msg_data in enumerate(history_for_llm):
            # original_msg_data could be from build_message_history (Gemini format or OpenAI format)
            gemini_parts_for_this_msg = []
            
            # Determine the source of parts/content for this message
            source_parts_or_content = None
            if "parts" in original_msg_data and isinstance(original_msg_data["parts"], list):
                # Already has a "parts" key, likely Gemini-native or pre-converted
                source_parts_or_content = original_msg_data["parts"]
            elif "content" in original_msg_data:
                # Has a "content" key, likely OpenAI-native
                openai_content = original_msg_data["content"]
                if isinstance(openai_content, str):
                    source_parts_or_content = [{"type": "text", "text": openai_content}] # Convert OpenAI string content to part structure
                elif isinstance(openai_content, list):
                    source_parts_or_content = openai_content # Already a list of OpenAI parts
                else:
                    logging.warning(f"Message {msg_index} (role '{original_msg_data['role']}') has 'content' of unexpected type {type(openai_content)}. Skipping.")
                    continue
            else:
                logging.warning(f"Message {msg_index} (role '{original_msg_data['role']}') has neither 'parts' nor 'content'. Skipping.")
                continue

            # Convert source_parts_or_content to Gemini Parts
            for part_idx, part_item in enumerate(source_parts_or_content):
                if isinstance(part_item, google_types.Part):
                    gemini_parts_for_this_msg.append(part_item)
                elif isinstance(part_item, dict): # OpenAI part dict
                    part_type = part_item.get("type")
                    if part_type == "text":
                        gemini_parts_for_this_msg.append(google_types.Part.from_text(text=part_item.get("text", "")))
                    elif part_type == "image_url":
                        image_url_dict = part_item.get("image_url", {})
                        data_url = image_url_dict.get("url")
                        if isinstance(data_url, str) and data_url.startswith("data:image") and ";base64," in data_url:
                            try:
                                header, encoded_data = data_url.split(";base64,", 1)
                                mime_type_str = header.split(":")[1] if ":" in header else "image/png"
                                img_bytes = base64.b64decode(encoded_data)
                                gemini_parts_for_this_msg.append(google_types.Part.from_bytes(data=img_bytes, mime_type=mime_type_str))
                            except Exception as e:
                                logging.warning(f"Error converting OpenAI image_url part to Gemini Part for msg {msg_index}, part {part_idx}: {e}. Skipping part.")
                        else:
                            logging.warning(f"Invalid data URL in OpenAI image_url part for msg {msg_index}, part {part_idx}. Skipping part.")
                    else:
                        logging.warning(f"Unsupported OpenAI part type '{part_type}' for msg {msg_index}, part {part_idx}. Skipping part.")
                elif isinstance(part_item, str): # Should ideally be caught by OpenAI content wrapping
                    gemini_parts_for_this_msg.append(google_types.Part.from_text(text=part_item))
                else:
                    logging.warning(f"Unsupported part item type {type(part_item)} for msg {msg_index}, part {part_idx}. Skipping part.")
            
            if gemini_parts_for_this_msg:
                history_to_use_for_api_call.append({"role": original_msg_data["role"], "parts": gemini_parts_for_this_msg})
            else:
                logging.warning(f"Message {msg_index} (role '{original_msg_data['role']}') resulted in no valid Gemini parts after conversion. Skipping.")
    else: # Not Gemini, assume history_for_llm is for OpenAI-compatible
        # For OpenAI, the history should already be in the format: {"role": ..., "content": "text" or [{"type":"text",...}]}
        # No significant transformation needed here, but ensure it's a deep copy if modifications were planned.
        history_to_use_for_api_call = [msg.copy() for msg in history_for_llm]


    # --- Start the API call loop ---
    for key_index, current_api_key in enumerate(keys_to_loop):
        key_display = f"...{current_api_key[-4:]}" if current_api_key != "dummy_key" else "N/A (keyless)"
        logging.info(f"Attempting LLM request with provider '{provider}' using key {key_display} ({key_index+1}/{len(keys_to_loop)})")

        llm_client = None
        api_config = None
        api_content_kwargs = {}
        payload_to_print = {}
        is_blocked_by_safety = False
        is_stopped_by_recitation = False
        content_received = False
        chunk_processed_successfully = False
        stream_finish_reason = None
        stream_grounding_metadata = None

        try: # Inner try for the specific API call attempt
            # --- Initialize Client and Prepare Payload ---
            if is_gemini:
                if current_api_key == "dummy_key":
                    raise ValueError("Gemini requires an API key.")
                llm_client = google_genai.Client(api_key=current_api_key)

                # --- THIS IS THE CRITICAL PART ---
                # Use the history_to_use_for_api_call which has been prepared
                gemini_contents = []
                for msg_data in history_to_use_for_api_call: 
                    role = msg_data["role"]
                    # 'parts' here should already be a list of google_types.Part objects due to the correction logic above
                    parts_list = msg_data.get("parts", []) 
                    if not parts_list: # Should not happen if correction logic is sound and msg was not skipped
                        logging.warning(f"Message with role '{role}' has no parts after correction. Skipping for Gemini contents.")
                        continue
                    try:
                        gemini_contents.append(google_types.Content(role=role, parts=parts_list))
                    except Exception as content_creation_error:
                         logging.error(f"FATAL: Failed to create google_types.Content! Role: {role}, Parts: {parts_list}", exc_info=True)
                         yield None, None, None, f"Internal error creating Gemini content structure: {content_creation_error}"
                         return 

                if not gemini_contents: # If all messages were skipped or had no parts
                    logging.error("Gemini contents list is empty after processing history. Cannot make API call.")
                    yield None, None, None, "Internal error: No valid content to send to Gemini after history processing."
                    return

                api_content_kwargs["contents"] = gemini_contents

                gemini_extra_params = extra_params.copy()
                if "max_tokens" in gemini_extra_params:
                    gemini_extra_params["max_output_tokens"] = gemini_extra_params.pop("max_tokens")
                
                # --- ADDED: Handle thinking_budget for Gemini ---
                gemini_thinking_budget_val = gemini_extra_params.pop("thinking_budget", None)
                # --- END ADDED ---

                gemini_safety_settings_list = [
                    google_types.SafetySetting(category=category, threshold=threshold)
                    for category, threshold in GEMINI_SAFETY_SETTINGS_DICT.items()
                ]

                api_config = google_types.GenerateContentConfig(
                    **gemini_extra_params,
                    safety_settings=gemini_safety_settings_list,
                    tools=[google_types.Tool(google_search=google_types.GoogleSearch())] # Enable grounding
                )
                # --- ADDED: Apply thinking_budget if valid ---
                if gemini_thinking_budget_val is not None:
                    budget_to_apply: Optional[int] = None
                    if isinstance(gemini_thinking_budget_val, int):
                        budget_to_apply = gemini_thinking_budget_val
                    elif isinstance(gemini_thinking_budget_val, str):
                        try:
                            budget_to_apply = int(gemini_thinking_budget_val)
                        except ValueError:
                            logging.warning(f"Non-integer string for thinking_budget ('{gemini_thinking_budget_val}') for Gemini. Ignoring.")
                    else:
                        logging.warning(f"Unsupported type for thinking_budget ('{type(gemini_thinking_budget_val).__name__}') for Gemini. Ignoring.")

                    if budget_to_apply is not None:
                        if 0 <= budget_to_apply <= 24576: # Max value for thinking_budget (as per Gemini docs)
                            if not hasattr(api_config, 'thinking_config') or api_config.thinking_config is None:
                                api_config.thinking_config = google_types.ThinkingConfig()
                            api_config.thinking_config.thinking_budget = budget_to_apply
                            logging.debug(f"Applied thinking_budget: {budget_to_apply} to Gemini ThinkingConfig")
                        else:
                            logging.warning(f"Invalid thinking_budget value ({budget_to_apply}) for Gemini. Must be 0-24576. Ignoring.")
                # --- END ADDED ---

                if system_prompt_text:
                     api_config.system_instruction = google_types.Part.from_text(text=system_prompt_text)

                payload_to_print = {
                    "model": model_name,
                    "contents": [c.model_dump(mode='json', exclude_none=True) for c in api_content_kwargs["contents"]],
                    "generationConfig": api_config.model_dump(mode='json', exclude_none=True) if api_config else {},
                }
                payload_to_print["generationConfig"] = {k: v for k, v in payload_to_print["generationConfig"].items() if v}

            else: # OpenAI compatible
                api_key_to_use = current_api_key if current_api_key != "dummy_key" else None
                # Ensure base_url is provided for OpenAI compatible APIs
                if not base_url:
                    raise ValueError(f"base_url is required for OpenAI-compatible provider '{provider}' but not found in config.")
                llm_client = AsyncOpenAI(base_url=base_url, api_key=api_key_to_use)

                # Ensure OpenAI correction logic is added here if needed in the future
                # For OpenAI, history_to_use_for_api_call should be a list of {"role": ..., "content": ...}
                openai_messages = history_to_use_for_api_call[:] 
                if system_prompt_text:
                    # Check if system prompt already exists to avoid duplicates if history_to_use_for_api_call was pre-processed
                    if not openai_messages or openai_messages[0].get("role") != "system":
                        openai_messages.insert(0, {"role": "system", "content": system_prompt_text})
                    elif openai_messages[0].get("role") == "system": # Update existing system prompt if necessary
                        openai_messages[0]["content"] = system_prompt_text


                api_content_kwargs["messages"] = openai_messages
                api_config = extra_params.copy()
                api_config["stream"] = True # Always stream

                payload_to_print = {
                    "model": model_name,
                    "messages": api_content_kwargs["messages"],
                    **api_config
                }

            # --- Print Payload ---
            try:
                print(f"\n--- LLM Request Payload (Provider: {provider}, Model: {model_name}) ---")
                # THIS LINE CALLS THE TRUNCATION FUNCTION
                payload_for_printing = _truncate_base64_in_payload(payload_to_print)
                # THIS LINE PRINTS THE TRUNCATED VERSION
                print(json.dumps(payload_for_printing, indent=2, default=default_serializer))
                print("--- End LLM Request Payload ---\n")
            except Exception as print_err:
                logging.error(f"Error printing LLM payload: {print_err}")
                # Fallback to printing raw if serialization/truncation fails
                print(f"Raw Payload Data (may contain unserializable objects):\n{payload_to_print}\n")

            # --- Make API Call and Process Stream ---
            stream_response = None
            if is_gemini:
                if not llm_client: raise ValueError("Gemini client not initialized.")
                stream_response = await llm_client.aio.models.generate_content_stream(
                    model=model_name,
                    contents=api_content_kwargs["contents"],
                    config=api_config
                )
            else:
                if not llm_client: raise ValueError("OpenAI client not initialized.")
                stream_response = await llm_client.chat.completions.create(
                    model=model_name,
                    messages=api_content_kwargs["messages"],
                    **api_config
                )

            # --- Stream Processing Loop ---
            async for chunk in stream_response:
                new_content_chunk = ""
                chunk_finish_reason = None
                chunk_grounding_metadata = None
                chunk_processed_successfully = False # Reset for each chunk

                try: # Inner try for stream processing errors
                    if is_gemini:
                        if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                            logging.warning(f"Gemini Prompt Blocked (reason: {chunk.prompt_feedback.block_reason}) with key {key_display}. Aborting.")
                            llm_errors.append(f"Key {key_display}: Prompt Blocked ({chunk.prompt_feedback.block_reason})")
                            is_blocked_by_safety = True
                            last_error_type = "safety"
                            break # Exit inner stream loop

                        if hasattr(chunk, 'text') and chunk.text:
                            new_content_chunk = chunk.text
                            content_received = True
                            logging.debug(f"Received Gemini chunk text (len: {len(new_content_chunk)})")

                        if hasattr(chunk, 'candidates') and chunk.candidates:
                             candidate = chunk.candidates[0]
                             if hasattr(candidate, 'finish_reason') and candidate.finish_reason and candidate.finish_reason != google_types.FinishReason.FINISH_REASON_UNSPECIFIED:
                                  reason_map = {
                                       google_types.FinishReason.STOP: "stop",
                                       google_types.FinishReason.MAX_TOKENS: "length",
                                       google_types.FinishReason.SAFETY: "safety",
                                       google_types.FinishReason.RECITATION: "recitation",
                                       google_types.FinishReason.OTHER: "other",
                                  }
                                  chunk_finish_reason = reason_map.get(candidate.finish_reason, str(candidate.finish_reason))
                                  logging.info(f"Gemini finish reason received: {chunk_finish_reason} ({candidate.finish_reason})")

                                  if chunk_finish_reason:
                                      finish_reason_lower = chunk_finish_reason.lower()
                                      if finish_reason_lower == "safety":
                                          logging.warning(f"Gemini Response Blocked (finish_reason=SAFETY) with key {key_display}. Check prompt/safety settings.")
                                          llm_errors.append(f"Key {key_display}: Response Blocked (Safety)")
                                          is_blocked_by_safety = True
                                          last_error_type = "safety"
                                      elif finish_reason_lower == "recitation":
                                          logging.warning(f"Gemini Response stopped due to Recitation (finish_reason=RECITATION) with key {key_display}.")
                                          llm_errors.append(f"Key {key_display}: Response Stopped (Recitation)")
                                          is_stopped_by_recitation = True
                                          last_error_type = "recitation"
                                      elif finish_reason_lower == "other":
                                          logging.warning(f"Gemini Response Blocked (finish_reason=OTHER) with key {key_display}.")
                                          llm_errors.append(f"Key {key_display}: Response Blocked (Other)")
                                          is_blocked_by_safety = True
                                          last_error_type = "other"

                             if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                                  chunk_grounding_metadata = candidate.grounding_metadata
                             if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                                 for rating in candidate.safety_ratings:
                                     if rating.probability in (google_types.HarmProbability.MEDIUM, google_types.HarmProbability.HIGH):
                                         logging.warning(f"Gemini content potentially blocked by safety rating: Category {rating.category}, Probability {rating.probability}. Key: {key_display}")
                                         if not content_received:
                                             llm_errors.append(f"Key {key_display}: Response Blocked (Safety Rating: {rating.category}={rating.probability})")
                                             is_blocked_by_safety = True
                                             last_error_type = "safety"

                    else: # OpenAI
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            chunk_finish_reason = chunk.choices[0].finish_reason
                            if delta and delta.content:
                                new_content_chunk = delta.content
                                content_received = True

                    # Update overall finish reason and grounding metadata
                    if chunk_finish_reason:
                        stream_finish_reason = chunk_finish_reason
                    if chunk_grounding_metadata:
                        stream_grounding_metadata = chunk_grounding_metadata

                    # Yield content if not empty and not blocked/stopped
                    if new_content_chunk and not is_blocked_by_safety and not is_stopped_by_recitation:
                        yield new_content_chunk, None, None, None # Yield text chunk

                    chunk_processed_successfully = True

                    # Break inner stream loop if finished or blocked/stopped
                    if stream_finish_reason:
                        logging.info(f"Stream finished with reason: {stream_finish_reason}. Exiting inner loop.")
                        break # Exit inner stream loop

                except google_api_exceptions.GoogleAPIError as stream_err:
                    logging.warning(f"Google API error during streaming with key {key_display}: {type(stream_err).__name__} - {stream_err}. Trying next key.")
                    llm_errors.append(f"Key {key_display}: Stream Google API Error - {type(stream_err).__name__}: {stream_err}")
                    last_error_type = "google_api"
                    if isinstance(stream_err, google_api_exceptions.ResourceExhausted):
                        if current_api_key != "dummy_key": llm_db_manager.add_key(current_api_key)
                        last_error_type = "rate_limit"
                    break # Exit inner loop on stream error
                except APIConnectionError as stream_err:
                    logging.warning(f"Connection error during streaming with key {key_display}: {stream_err}. Trying next key.")
                    llm_errors.append(f"Key {key_display}: Stream Connection Error - {stream_err}")
                    last_error_type = "connection"
                    break # Exit inner loop on stream error
                except APIError as stream_err:
                    logging.warning(f"API error during streaming with key {key_display}: {stream_err}. Trying next key.")
                    llm_errors.append(f"Key {key_display}: Stream API Error - {stream_err}")
                    last_error_type = "api"
                    if isinstance(stream_err, RateLimitError):
                        if current_api_key != "dummy_key": llm_db_manager.add_key(current_api_key)
                        last_error_type = "rate_limit"
                    break # Exit inner loop on stream error
                except Exception as stream_err:
                    logging.exception(f"Unexpected error during streaming with key {key_display}")
                    llm_errors.append(f"Key {key_display}: Unexpected Stream Error - {type(stream_err).__name__}")
                    last_error_type = "unexpected"
                    break # Exit inner loop on stream error
            # --- End Stream Processing Loop ---

            # --- After Stream Processing Loop ---
            if not content_received and not is_blocked_by_safety and not is_stopped_by_recitation:
                is_successful_empty_finish = stream_finish_reason and (stream_finish_reason.lower() == "stop" or (is_gemini and stream_finish_reason == str(google_types.FinishReason.FINISH_REASON_UNSPECIFIED)))
                if is_successful_empty_finish:
                    logging.warning(f"LLM stream finished successfully for key {key_display} but NO content was received. Finish reason: {stream_finish_reason}. Aborting retries.")
                    llm_errors.append(f"Key {key_display}: No content received (Successful Finish)")
                    last_error_type = "no_content_success_finish"
                    # Yield final state indicating failure
                    yield None, stream_finish_reason, stream_grounding_metadata, f"No content received (Successful Finish: {stream_finish_reason})"
                    return # Stop generation
                elif stream_finish_reason:
                    logging.warning(f"LLM stream finished with reason '{stream_finish_reason}' for key {key_display} but NO content was received. Aborting retries.")
                    llm_errors.append(f"Key {key_display}: No content received (Finish Reason: {stream_finish_reason})")
                    last_error_type = "no_content_other_finish"
                    yield None, stream_finish_reason, stream_grounding_metadata, f"No content received (Finish Reason: {stream_finish_reason})"
                    return # Stop generation
                else:
                    logging.warning(f"LLM stream ended prematurely for key {key_display} and NO content was received. Last error type: {last_error_type}. Aborting retries.")
                    if not llm_errors or not llm_errors[-1].startswith(f"Key {key_display}"):
                        llm_errors.append(f"Key {key_display}: Stream ended prematurely with no content")
                    last_error_type = "premature_empty_stream"
                    yield None, stream_finish_reason, stream_grounding_metadata, "Stream ended prematurely with no content"
                    return # Stop generation

            elif is_blocked_by_safety:
                logging.warning(f"LLM response blocked due to safety/other with key {key_display}. Aborting retries.")
                yield None, "safety", stream_grounding_metadata, "Response blocked by safety filters."
                return # Stop generation
            elif is_stopped_by_recitation:
                logging.warning(f"LLM response stopped due to recitation with key {key_display}. Aborting retries.")
                yield None, "recitation", stream_grounding_metadata, "Response stopped due to recitation."
                return # Stop generation

            elif stream_finish_reason and content_received:
                is_successful_finish = stream_finish_reason.lower() in ("stop", "end_turn") or (is_gemini and stream_finish_reason == str(google_types.FinishReason.FINISH_REASON_UNSPECIFIED))
                if is_successful_finish:
                    logging.info(f"LLM request successful with key {key_display}")
                    # Yield final state after successful completion
                    yield None, stream_finish_reason, stream_grounding_metadata, None
                    return # Successful completion
                else:
                    logging.warning(f"LLM stream finished with non-stop reason '{stream_finish_reason}' for key {key_display}. Trying next key.")
                    llm_errors.append(f"Key {key_display}: Finished Reason '{stream_finish_reason}'")
                    last_error_type = stream_finish_reason
                    continue # Try next key

            elif not chunk_processed_successfully:
                 logging.warning(f"LLM stream broke during processing for key {key_display}. Last error type: {last_error_type}. Trying next key.")
                 continue # Try next key

            else: # Fallback case: Content received, not blocked/stopped, but no finish reason
                 # Check for retry condition
                 if not is_gemini and content_received and stream_finish_reason is None and not is_blocked_by_safety and not is_stopped_by_recitation:
                     logging.warning(f"Non-Gemini LLM stream for key {key_display} ended without finish reason but with content. Signaling for Gemini retry.")
                     yield None, None, stream_grounding_metadata, "RETRY_WITH_GEMINI_NO_FINISH_REASON"
                     return # Stop this attempt

                 logging.warning(f"LLM stream ended without a finish reason for key {key_display}, but content was received. Treating as successful completion. Content received: {content_received}, Finish reason: {stream_finish_reason}, Blocked: {is_blocked_by_safety}, Stopped: {is_stopped_by_recitation}.")
                 # Treat as successful completion since content was received
                 logging.info(f"LLM request considered successful (stream ended without finish reason) with key {key_display}")
                 # Yield final state indicating success, but note the missing finish reason
                 yield None, stream_finish_reason, stream_grounding_metadata, None # Yield None for error message
                 return # Successful completion

        # --- Handle Initial API Call Errors ---
        except (RateLimitError, google_api_exceptions.ResourceExhausted) as e:
            logging.warning(f"Initial Request Rate limit hit for provider '{provider}' with key {key_display}. Error: {e}. Trying next key.")
            if current_api_key != "dummy_key": llm_db_manager.add_key(current_api_key)
            llm_errors.append(f"Key {key_display}: Initial Rate Limited")
            last_error_type = "rate_limit"
            continue
        except (AuthenticationError, google_api_exceptions.PermissionDenied) as e:
            logging.error(f"Initial Request Authentication failed for provider '{provider}' with key {key_display}. Error: {e}. Aborting retries.")
            llm_errors.append(f"Key {key_display}: Initial Authentication Failed")
            raise AllKeysFailedError(provider, llm_errors) from e
        except (APIConnectionError, google_api_exceptions.ServiceUnavailable, google_api_exceptions.DeadlineExceeded) as e:
            logging.warning(f"Initial Request Connection/Service error for provider '{provider}' with key {key_display}. Error: {e}. Trying next key.")
            llm_errors.append(f"Key {key_display}: Initial Connection/Service Error - {type(e).__name__}")
            last_error_type = "connection"
            continue
        except (BadRequestError, google_api_exceptions.InvalidArgument) as e:
             logging.error(f"Initial Request Bad request error for provider '{provider}' with key {key_display}. Error: {e}. Aborting retries.")
             if isinstance(e, google_api_exceptions.InvalidArgument):
                 error_details = getattr(e, 'details', None)
                 if error_details: logging.error(f"Google API InvalidArgument Details: {error_details}")
             llm_errors.append(f"Key {key_display}: Initial Bad Request - {e}")
             raise AllKeysFailedError(provider, llm_errors) from e
        except APIError as e: # Catch other OpenAI API errors
            logging.exception(f"Initial Request OpenAI API Error for key {key_display}")
            llm_errors.append(f"Key {key_display}: Initial API Error - {type(e).__name__}: {e}")
            last_error_type = "api"
            continue
        except google_api_exceptions.GoogleAPICallError as e: # Catch other Google API errors
            logging.exception(f"Initial Request Google API Call Error for key {key_display}")
            error_details = getattr(e, 'details', None)
            if error_details: logging.error(f"Google API Error Details: {error_details}")
            llm_errors.append(f"Key {key_display}: Initial Google API Error - {type(e).__name__}: {e}")
            last_error_type = "google_api"
            continue
        except Exception as e: # Catch other unexpected errors
            logging.exception(f"Unexpected error during initial LLM call with key {key_display}")
            llm_errors.append(f"Key {key_display}: Unexpected Initial Error - {type(e).__name__}")
            last_error_type = "unexpected"
            continue

    # --- Post-Retry Loop ---
    # If the loop finishes without returning/raising, all keys failed
    logging.error(f"All LLM API keys failed for provider '{provider}'. Errors: {llm_errors}")
    raise AllKeysFailedError(provider, llm_errors)
