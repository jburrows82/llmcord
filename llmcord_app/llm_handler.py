import base64
import logging
import random
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
import io
from PIL import Image

# OpenAI specific imports
from openai import (
    AsyncOpenAI,
    APIError,
    RateLimitError,
    AuthenticationError,
    APIConnectionError,
    BadRequestError,
    UnprocessableEntityError,
)

# Google Gemini specific imports
from google import genai as google_genai
from google.genai import types as google_types
from google.api_core import exceptions as google_api_exceptions

from .constants import (
    AllKeysFailedError,
    PROVIDERS_SUPPORTING_USERNAMES,
    GEMINI_SAFETY_SETTINGS_CONFIG_KEY,  # New
    PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY, # New
)
from .rate_limiter import get_db_manager, get_available_keys
from .utils import _truncate_base64_in_payload, default_serializer


async def generate_response_stream(
    provider: str,
    model_name: str,
    history_for_llm: List[Dict[str, Any]],
    system_prompt_text: Optional[str],
    provider_config: Dict[str, Any],
    extra_params: Dict[str, Any],
    # Add app_config to access Gemini safety settings
    app_config: Dict[str, Any],
) -> AsyncGenerator[
    Tuple[Optional[str], Optional[str], Optional[Any], Optional[str]], None
]:
    # Track image compression information
    compression_occurred = False
    final_quality = 100  # Start with original quality
    final_resize = 1.0  # Start with original size
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
        app_config: The main application configuration dictionary.

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
    keys_required = provider not in [
        "ollama",
        "lmstudio",
        "vllm",
        "oobabooga",
        "jan",
    ]  # Add other keyless providers

    if keys_required and not all_api_keys:
        raise ValueError(
            f"No API keys configured for the selected provider '{provider}' which requires keys."
        )

    available_llm_keys = await get_available_keys(provider, all_api_keys, app_config)
    llm_db_manager = await get_db_manager(provider)

    if keys_required and not available_llm_keys:
        raise AllKeysFailedError(
            provider, ["No available (non-rate-limited) API keys."]
        )

    random.shuffle(available_llm_keys)
    keys_to_loop = (
        available_llm_keys if keys_required else ["dummy_key"]
    )  # Loop once for keyless
    llm_errors = []
    last_error_type = None

    # --- History Correction Logic ---
    # This logic prepares `history_for_api_call_current_key_attempt` based on whether the current call is for Gemini or OpenAI-compatible.
    # `history_for_llm` is the original history built by `build_message_history`.

    # The `history_for_api_call_current_key_attempt` will be used and potentially modified by compression
    # within the compression retry loop for a single API key.
    # It's initialized from `history_for_llm` at the start of each API key attempt.

    # --- Start the API call loop (outer loop for API keys) ---
    for key_index, current_api_key in enumerate(keys_to_loop):
        key_display = (
            f"...{current_api_key[-4:]}"
            if current_api_key != "dummy_key"
            else "N/A (keyless)"
        )
        logging.info(
            f"Attempting LLM request with provider '{provider}' using key {key_display} ({key_index + 1}/{len(keys_to_loop)})"
        )

        # Initialize history for the current API key attempt from the original, unmodified history_for_llm
        # This ensures that each API key starts with fresh, uncompressed history.
        # Deepcopy here if history_for_llm items are complex mutable objects that might be altered by format_history_for_provider
        # For now, assuming format_history_for_provider returns a new list or handles copying internally.

        # Format history specifically for the current provider (OpenAI or Gemini)
        # This step is crucial and should produce `current_history_for_api_call`
        current_history_for_api_call = []
        if is_gemini:
            logging.debug(
                "Preparing history for Gemini API format from input history_for_llm for current key."
            )
            for msg_index, original_msg_data in enumerate(history_for_llm):
                gemini_parts_for_this_msg = []
                source_parts_or_content = None
                if "parts" in original_msg_data and isinstance(
                    original_msg_data["parts"], list
                ):
                    source_parts_or_content = original_msg_data["parts"]
                elif "content" in original_msg_data:
                    openai_content = original_msg_data["content"]
                    if isinstance(openai_content, str):
                        source_parts_or_content = [
                            {"type": "text", "text": openai_content}
                        ]
                    elif isinstance(openai_content, list):
                        source_parts_or_content = openai_content
                    else:
                        logging.warning(
                            f"Msg {msg_index} (role '{original_msg_data['role']}') has 'content' of unexpected type {type(openai_content)}. Skipping."
                        )
                        continue
                else:
                    logging.warning(
                        f"Msg {msg_index} (role '{original_msg_data['role']}') has neither 'parts' nor 'content'. Skipping."
                    )
                    continue

                for part_idx, part_item in enumerate(source_parts_or_content):
                    if isinstance(part_item, google_types.Part):
                        gemini_parts_for_this_msg.append(part_item)
                    elif isinstance(part_item, dict):  # OpenAI part dict
                        part_type = part_item.get("type")
                        if part_type == "text":
                            gemini_parts_for_this_msg.append(
                                google_types.Part.from_text(
                                    text=part_item.get("text", "")
                                )
                            )
                        elif part_type == "image_url":
                            image_url_dict = part_item.get("image_url", {})
                            data_url = image_url_dict.get("url")
                            if (
                                isinstance(data_url, str)
                                and data_url.startswith("data:image")
                                and ";base64," in data_url
                            ):
                                try:
                                    header, encoded_data = data_url.split(";base64,", 1)
                                    mime_type_str = (
                                        header.split(":")[1]
                                        if ":" in header
                                        else "image/png"
                                    )
                                    img_bytes = base64.b64decode(encoded_data)
                                    gemini_parts_for_this_msg.append(
                                        google_types.Part.from_bytes(
                                            data=img_bytes, mime_type=mime_type_str
                                        )
                                    )
                                except Exception as e:
                                    logging.warning(
                                        f"Error converting OpenAI image_url part to Gemini Part for msg {msg_index}, part {part_idx}: {e}. Skipping part."
                                    )
                            else:
                                logging.warning(
                                    f"Invalid data URL in OpenAI image_url part for msg {msg_index}, part {part_idx}. Skipping part."
                                )
                        else:
                            logging.warning(
                                f"Unsupported OpenAI part type '{part_type}' for msg {msg_index}, part {part_idx}. Skipping part."
                            )
                    elif isinstance(part_item, str):
                        gemini_parts_for_this_msg.append(
                            google_types.Part.from_text(text=part_item)
                        )
                    else:
                        logging.warning(
                            f"Unsupported part item type {type(part_item)} for msg {msg_index}, part {part_idx}. Skipping part."
                        )

                if gemini_parts_for_this_msg:
                    # Map OpenAI roles to Gemini roles (Gemini only accepts "user" and "model")
                    gemini_role = "user"
                    if (
                        original_msg_data["role"] == "assistant"
                        or original_msg_data["role"] == "model"
                    ):
                        gemini_role = "model"
                    elif original_msg_data["role"] == "system":
                        gemini_role = "user"  # Use "user" for system, as Gemini doesn't have a system role

                    current_history_for_api_call.append(
                        {
                            "role": gemini_role,
                            "parts": gemini_parts_for_this_msg,
                        }
                    )
                else:
                    logging.warning(
                        f"Msg {msg_index} (role '{original_msg_data['role']}') resulted in no valid Gemini parts. Skipping."
                    )
        else:  # OpenAI-compatible
            current_history_for_api_call = [msg.copy() for msg in history_for_llm]

        # --- Compression Retry Loop (inner loop for a single API key) ---
        compression_attempt = 0
        max_compression_attempts = 5  # Example: 5 attempts
        # Initial compression parameters (can be adjusted)
        current_compression_quality = 90  # Start with higher quality
        current_resize_factor = 1.0  # No resize initially

        # This history will be modified by compression attempts for the current API key
        history_for_current_compression_cycle = [
            msg.copy() for msg in current_history_for_api_call
        ]

        while compression_attempt < max_compression_attempts:
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

            try:  # Innermost try for the specific API call attempt (with potentially compressed data)
                # --- Initialize Client and Prepare Payload (using history_for_current_compression_cycle) ---
                if is_gemini:
                    if current_api_key == "dummy_key":
                        raise ValueError("Gemini requires an API key.")
                    llm_client = google_genai.Client(api_key=current_api_key)
                    gemini_contents = []
                    for msg_data in (
                        history_for_current_compression_cycle
                    ):  # Use the potentially compressed history
                        role = msg_data["role"]
                        parts_list = msg_data.get("parts", [])
                        if not parts_list:
                            continue
                        try:
                            gemini_contents.append(
                                google_types.Content(role=role, parts=parts_list)
                            )
                        except Exception as content_creation_error:
                            logging.error(
                                f"FATAL: Failed to create google_types.Content! Role: {role}, Parts: {parts_list}",
                                exc_info=True,
                            )
                            yield (
                                None,
                                None,
                                None,
                                f"Internal error creating Gemini content structure: {content_creation_error}",
                            )
                            return
                    if not gemini_contents:
                        logging.error(
                            "Gemini contents list is empty. Cannot make API call."
                        )
                        yield (
                            None,
                            None,
                            None,
                            "Internal error: No valid content to send to Gemini.",
                        )
                        return
                    api_content_kwargs["contents"] = gemini_contents
                    gemini_extra_params = extra_params.copy()
                    if "max_tokens" in gemini_extra_params:
                        gemini_extra_params["max_output_tokens"] = (
                            gemini_extra_params.pop("max_tokens")
                        )
                    gemini_thinking_budget_val = gemini_extra_params.pop(
                        "thinking_budget", None
                    )

                    # Load safety settings from app_config
                    gemini_safety_settings_from_config = app_config.get(
                        GEMINI_SAFETY_SETTINGS_CONFIG_KEY, {}
                    )
                    gemini_safety_settings_list = []
                    for (
                        category_str,
                        threshold_str,
                    ) in gemini_safety_settings_from_config.items():
                        try:
                            category_enum = getattr(
                                google_types.HarmCategory, category_str.upper(), None
                            )
                            threshold_enum = getattr(
                                google_types.HarmBlockThreshold,
                                threshold_str.upper(),
                                None,
                            )
                            if category_enum and threshold_enum:
                                gemini_safety_settings_list.append(
                                    google_types.SafetySetting(
                                        category=category_enum, threshold=threshold_enum
                                    )
                                )
                            else:
                                logging.warning(
                                    f"Invalid Gemini safety category ('{category_str}') or threshold ('{threshold_str}') in config. Skipping."
                                )
                        except Exception as e_safety:
                            logging.warning(
                                f"Error processing Gemini safety setting {category_str}={threshold_str}: {e_safety}"
                            )

                    # Use default if no valid settings were parsed from config
                    if not gemini_safety_settings_list:
                        logging.warning(
                            "No valid Gemini safety settings found in config, using BLOCK_NONE for all."
                        )
                        gemini_safety_settings_list = [
                            google_types.SafetySetting(
                                category=c,
                                threshold=google_types.HarmBlockThreshold.BLOCK_NONE,
                            )
                            for c in [
                                google_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                google_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                google_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                google_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            ]
                        ]

                    # Process thinking_budget if provided
                    thinking_config = None
                    if gemini_thinking_budget_val is not None:
                        budget_to_apply: Optional[int] = None
                        if isinstance(gemini_thinking_budget_val, int):
                            budget_to_apply = gemini_thinking_budget_val
                        elif isinstance(gemini_thinking_budget_val, str):
                            try:
                                budget_to_apply = int(gemini_thinking_budget_val)
                            except ValueError:
                                logging.warning(
                                    f"Non-integer string for thinking_budget ('{gemini_thinking_budget_val}') for Gemini. Ignoring."
                                )
                        else:
                            logging.warning(
                                f"Unsupported type for thinking_budget ('{type(gemini_thinking_budget_val).__name__}') for Gemini. Ignoring."
                            )
                        if (
                            budget_to_apply is not None
                            and 0 <= budget_to_apply <= 24576
                        ):
                            thinking_config = google_types.ThinkingConfig(
                                thinking_budget=budget_to_apply
                            )
                            logging.debug(
                                f"Applied thinking_budget: {budget_to_apply} to Gemini ThinkingConfig"
                            )
                        elif budget_to_apply is not None:
                            logging.warning(
                                f"Invalid thinking_budget value ({budget_to_apply}) for Gemini. Must be 0-24576. Ignoring."
                            )

                    # Create GenerateContentConfig with thinking_config if available
                    api_config = google_types.GenerateContentConfig(
                        **gemini_extra_params,
                        safety_settings=gemini_safety_settings_list,
                        tools=[
                            google_types.Tool(google_search=google_types.GoogleSearch())
                        ],
                        thinking_config=thinking_config,
                    )
                    if system_prompt_text:
                        api_config.system_instruction = google_types.Part.from_text(
                            text=system_prompt_text
                        )
                    payload_to_print = {
                        "model": model_name,
                        "contents": [
                            c.model_dump(mode="json", exclude_none=True)
                            for c in api_content_kwargs["contents"]
                        ],
                        "generationConfig": api_config.model_dump(
                            mode="json", exclude_none=True
                        )
                        if api_config
                        else {},
                    }
                    payload_to_print["generationConfig"] = {
                        k: v
                        for k, v in payload_to_print["generationConfig"].items()
                        if v
                    }

                else:  # OpenAI compatible
                    api_key_to_use = (
                        current_api_key if current_api_key != "dummy_key" else None
                    )
                    if not base_url:
                        raise ValueError(
                            f"base_url is required for OpenAI-compatible provider '{provider}' but not found in config."
                        )
                    llm_client = AsyncOpenAI(base_url=base_url, api_key=api_key_to_use)
                    openai_messages = [
                        msg.copy() for msg in history_for_current_compression_cycle
                    ]  # Use the potentially compressed history
                    if system_prompt_text:
                        if (
                            not openai_messages
                            or openai_messages[0].get("role") != "system"
                        ):
                            openai_messages.insert(
                                0, {"role": "system", "content": system_prompt_text}
                            )
                        elif openai_messages[0].get("role") == "system":
                            openai_messages[0]["content"] = system_prompt_text
                    if provider not in PROVIDERS_SUPPORTING_USERNAMES:
                        for msg_data in openai_messages:
                            if msg_data.get("role") == "user" and "name" in msg_data:
                                del msg_data["name"]
                                logging.debug(
                                    f"Removed 'name' field for user message for provider '{provider}'"
                                )
                    api_content_kwargs["messages"] = openai_messages
                    api_config = extra_params.copy()
                    api_config["stream"] = True
                    payload_to_print = {
                        "model": model_name,
                        "messages": api_content_kwargs["messages"],
                        **api_config,
                    }

                # --- Print Payload ---
                try:
                    print(
                        f"\n--- LLM Request Payload (Provider: {provider}, Model: {model_name}, Key: {key_display}, Attempt: {compression_attempt + 1}) ---"
                    )
                    payload_for_printing = _truncate_base64_in_payload(payload_to_print)
                    print(
                        json.dumps(
                            payload_for_printing, indent=2, default=default_serializer
                        )
                    )
                    print("--- End LLM Request Payload ---\n")
                except Exception as print_err:
                    logging.error(f"Error printing LLM payload: {print_err}")
                    print(
                        f"Raw Payload Data (may contain unserializable objects):\n{payload_to_print}\n"
                    )

                # --- Make API Call and Process Stream ---
                stream_response = None
                if is_gemini:
                    if not llm_client:
                        raise ValueError("Gemini client not initialized.")
                    stream_response = (
                        await llm_client.aio.models.generate_content_stream(
                            model=model_name,
                            contents=api_content_kwargs["contents"],
                            config=api_config,
                        )
                    )
                else:
                    if not llm_client:
                        raise ValueError("OpenAI client not initialized.")
                    stream_response = await llm_client.chat.completions.create(
                        model=model_name,
                        messages=api_content_kwargs["messages"],
                        **api_config,
                    )

                # --- Stream Processing Loop ---
                async for chunk in stream_response:
                    new_content_chunk = ""
                    chunk_finish_reason = None
                    chunk_grounding_metadata = None
                    chunk_processed_successfully = False

                    try:  # Inner try for stream processing errors
                        if is_gemini:
                            if (
                                hasattr(chunk, "prompt_feedback")
                                and chunk.prompt_feedback
                                and chunk.prompt_feedback.block_reason
                            ):
                                logging.warning(
                                    f"Gemini Prompt Blocked (reason: {chunk.prompt_feedback.block_reason}) with key {key_display}. Aborting."
                                )
                                llm_errors.append(
                                    f"Key {key_display}: Prompt Blocked ({chunk.prompt_feedback.block_reason})"
                                )
                                is_blocked_by_safety = True
                                last_error_type = "safety"
                                break
                            if hasattr(chunk, "text") and chunk.text:
                                new_content_chunk = chunk.text
                                content_received = True
                            if hasattr(chunk, "candidates") and chunk.candidates:
                                candidate = chunk.candidates[0]
                                if (
                                    hasattr(candidate, "finish_reason")
                                    and candidate.finish_reason
                                    and candidate.finish_reason
                                    != google_types.FinishReason.FINISH_REASON_UNSPECIFIED
                                ):
                                    reason_map = {
                                        google_types.FinishReason.STOP: "stop",
                                        google_types.FinishReason.MAX_TOKENS: "length",
                                        google_types.FinishReason.SAFETY: "safety",
                                        google_types.FinishReason.RECITATION: "recitation",
                                        google_types.FinishReason.OTHER: "other",
                                    }
                                    chunk_finish_reason = reason_map.get(
                                        candidate.finish_reason,
                                        str(candidate.finish_reason),
                                    )
                                    if chunk_finish_reason:
                                        finish_reason_lower = (
                                            chunk_finish_reason.lower()
                                        )
                                        if finish_reason_lower == "safety":
                                            is_blocked_by_safety = True
                                            last_error_type = "safety"
                                            llm_errors.append(
                                                f"Key {key_display}: Response Blocked (Safety)"
                                            )
                                        elif finish_reason_lower == "recitation":
                                            is_stopped_by_recitation = True
                                            last_error_type = "recitation"
                                            llm_errors.append(
                                                f"Key {key_display}: Response Stopped (Recitation)"
                                            )
                                        elif finish_reason_lower == "other":
                                            is_blocked_by_safety = True
                                            last_error_type = "other"
                                            llm_errors.append(
                                                f"Key {key_display}: Response Blocked (Other)"
                                            )
                                if (
                                    hasattr(candidate, "grounding_metadata")
                                    and candidate.grounding_metadata
                                ):
                                    chunk_grounding_metadata = (
                                        candidate.grounding_metadata
                                    )
                                if (
                                    hasattr(candidate, "safety_ratings")
                                    and candidate.safety_ratings
                                ):
                                    for rating in candidate.safety_ratings:
                                        if rating.probability in (
                                            google_types.HarmProbability.MEDIUM,
                                            google_types.HarmProbability.HIGH,
                                        ):
                                            if not content_received:
                                                is_blocked_by_safety = True
                                                last_error_type = "safety"
                                                llm_errors.append(
                                                    f"Key {key_display}: Response Blocked (Safety Rating: {rating.category}={rating.probability})"
                                                )
                        else:  # OpenAI
                            if chunk.choices:
                                delta = chunk.choices[0].delta
                                chunk_finish_reason = chunk.choices[0].finish_reason
                                if delta and delta.content:
                                    new_content_chunk = delta.content
                                    content_received = True

                        if chunk_finish_reason:
                            stream_finish_reason = chunk_finish_reason
                        if chunk_grounding_metadata:
                            stream_grounding_metadata = chunk_grounding_metadata
                        if (
                            new_content_chunk
                            and not is_blocked_by_safety
                            and not is_stopped_by_recitation
                        ):
                            yield new_content_chunk, None, None, None
                        chunk_processed_successfully = True
                        if stream_finish_reason:
                            break

                    except google_api_exceptions.GoogleAPIError as stream_err:
                        llm_errors.append(
                            f"Key {key_display}: Stream Google API Error - {type(stream_err).__name__}: {stream_err}"
                        )
                        last_error_type = "google_api"
                        if isinstance(
                            stream_err, google_api_exceptions.ResourceExhausted
                        ):
                            if current_api_key != "dummy_key":
                                await llm_db_manager.add_key(current_api_key)
                            last_error_type = "rate_limit"
                        break
                    except APIConnectionError as stream_err:
                        llm_errors.append(
                            f"Key {key_display}: Stream Connection Error - {stream_err}"
                        )
                        last_error_type = "connection"
                        break
                    except APIError as stream_err:
                        llm_errors.append(
                            f"Key {key_display}: Stream API Error - {stream_err}"
                        )
                        last_error_type = "api"
                        if isinstance(stream_err, RateLimitError):
                            if current_api_key != "dummy_key":
                                await llm_db_manager.add_key(current_api_key)
                            last_error_type = "rate_limit"
                        # Check for 413 within stream error for OpenAI
                        if (
                            not is_gemini
                            and hasattr(stream_err, "status_code")
                            and stream_err.status_code == 413
                        ):
                            logging.warning(
                                f"OpenAI API Error 413 (Request Entity Too Large) during stream with key {key_display}. Attempting compression."
                            )
                            # This break will exit the stream processing loop, and the outer compression loop will handle it.
                            # No need to set last_error_type to 'api_413' here, the initial API call error handler will do it.
                            break
                        break
                    except Exception as stream_err:
                        llm_errors.append(
                            f"Key {key_display}: Unexpected Stream Error - {type(stream_err).__name__}"
                        )
                        last_error_type = "unexpected"
                        break
                # --- End Stream Processing Loop for this attempt ---

                # --- After Stream Processing Loop for this attempt ---
                if (
                    is_blocked_by_safety or is_stopped_by_recitation
                ):  # Safety/Recitation block is final for this key.
                    logging.warning(
                        f"LLM response blocked/stopped for key {key_display}. Type: {'safety' if is_blocked_by_safety else 'recitation'}. Aborting retries for this key."
                    )
                    yield (
                        None,
                        ("safety" if is_blocked_by_safety else "recitation"),
                        stream_grounding_metadata,
                        f"Response {'blocked by safety' if is_blocked_by_safety else 'stopped by recitation'}.",
                    )
                    return  # Stop generation entirely if safety/recitation block from any key.

                if (
                    stream_finish_reason and content_received
                ):  # Successful stream completion
                    is_successful_finish = stream_finish_reason.lower() in (
                        "stop",
                        "end_turn",
                    ) or (
                        is_gemini
                        and stream_finish_reason
                        == str(google_types.FinishReason.FINISH_REASON_UNSPECIFIED)
                    )
                    if is_successful_finish:
                        logging.info(
                            f"LLM request successful with key {key_display} on compression attempt {compression_attempt + 1}"
                        )

                        # Add compression warning if compression occurred
                        if compression_occurred:
                            quality_pct = final_quality
                            resize_pct = int(final_resize * 100)
                            user_warning = f"⚠️ The image is at {quality_pct}% of the original quality and has been resized to {resize_pct}% so the request works."
                            yield None, None, None, f"COMPRESSION_INFO:{user_warning}"

                        yield (
                            None,
                            stream_finish_reason,
                            stream_grounding_metadata,
                            None,
                        )
                        return  # Successful completion
                    else:  # Non-stop finish reason, but content received. This is unusual.
                        logging.warning(
                            f"LLM stream finished with non-stop reason '{stream_finish_reason}' for key {key_display} but content was received. This is unexpected. Continuing to next API key if any."
                        )
                        llm_errors.append(
                            f"Key {key_display}: Finished Reason '{stream_finish_reason}' with content"
                        )
                        last_error_type = stream_finish_reason
                        break

                if (
                    not content_received and stream_finish_reason
                ):  # No content, but a finish reason (e.g. max_tokens without output)
                    logging.warning(
                        f"LLM stream for key {key_display} finished with reason '{stream_finish_reason}' but NO content was received. Aborting retries for this key."
                    )
                    llm_errors.append(
                        f"Key {key_display}: No content (Finish: {stream_finish_reason})"
                    )
                    last_error_type = f"no_content_{stream_finish_reason}"
                    # If this was a 413 attempt that led to no content, we might not want to yield error yet,
                    # but let the compression loop try again if not max attempts.
                    # However, if it's not a 413, then it's a genuine "no content" scenario.
                    # For now, if it's not a 413 that led here, we yield error.
                    # The 413 handling is primarily in the except APIError block.
                    if (
                        last_error_type != "api_413"
                    ):  # If not currently in a 413 retry that resulted in no content
                        yield (
                            None,
                            stream_finish_reason,
                            stream_grounding_metadata,
                            f"No content received (Finish Reason: {stream_finish_reason})",
                        )
                        return  # Stop generation
                    # If it WAS a 413 that led to no content, the compression loop should continue or break based on attempts.
                    # The `continue` for compression is handled in the except APIError block.

                if (
                    not chunk_processed_successfully and not stream_finish_reason
                ):  # Stream broke mid-way without a finish reason
                    logging.warning(
                        f"LLM stream broke for key {key_display}. Last error type: {last_error_type}. Trying next compression or API key."
                    )
                    # This break will exit the stream processing loop.
                    # If it was a 413 that caused the break, the APIError handler below will catch it.
                    # Otherwise, the compression loop will continue or break.
                    break

                # If we reach here, it means the stream ended, possibly without a finish reason, but content might have been received.
                # This can happen if the connection drops or the server closes the stream unexpectedly after sending some data.
                if content_received and not stream_finish_reason:
                    if not is_gemini:  # OpenAI-like stream ended without finish_reason but sent content
                        logging.warning(
                            f"Non-Gemini LLM stream for key {key_display} ended without finish reason but with content. Signaling for Gemini retry if applicable."
                        )
                        yield (
                            None,
                            None,
                            stream_grounding_metadata,
                            "RETRY_WITH_GEMINI_NO_FINISH_REASON",
                        )
                        return  # Stop this attempt, let outer logic handle retry signal
                    else:  # Gemini stream ended with content but no explicit finish reason (should be rare)
                        logging.warning(
                            f"Gemini stream for key {key_display} ended with content but no explicit finish reason. Treating as complete."
                        )
                        yield (
                            None,
                            "stop",
                            stream_grounding_metadata,
                            None,
                        )  # Assume stop
                        return

                # If no content, no finish reason, and stream didn't break unexpectedly (e.g. empty response)
                if (
                    not content_received
                    and not stream_finish_reason
                    and chunk_processed_successfully
                ):
                    logging.warning(
                        f"LLM stream for key {key_display} yielded no content and no finish reason. Aborting for this key."
                    )
                    llm_errors.append(f"Key {key_display}: Empty response")
                    last_error_type = "empty_response"
                    break

                # If we fall through here, it implies an issue not caught above, break compression loop.
                break

            # --- Handle Initial API Call Errors / Compression Trigger ---
            except APIError as e:  # OpenAI specific
                if not is_gemini and hasattr(e, "status_code") and e.status_code == 413:
                    logging.warning(
                        f"OpenAI API Error 413 (Request Entity Too Large) for key {key_display}, attempt {compression_attempt + 1}."
                    )
                    last_error_type = "api_413"  # Mark as 413
                    compression_attempt += 1
                    if compression_attempt >= max_compression_attempts:
                        logging.error(
                            f"Max compression attempts reached for key {key_display} after 413 error. Trying next API key."
                        )
                        llm_errors.append(
                            f"Key {key_display}: Max compression retries for 413 failed."
                        )
                        break  # Break compression loop, try next API key

                    logging.info(
                        f"Attempting image compression (Attempt {compression_attempt}/{max_compression_attempts}) for key {key_display}. Quality: {current_compression_quality}, Resize: {current_resize_factor:.2f}"
                    )
                    history_was_modified_by_compression = False
                    temp_compressed_history = [
                        msg.copy() for msg in current_history_for_api_call
                    ]  # Work on a copy for this compression cycle

                    for msg_data in temp_compressed_history:
                        if msg_data.get("role") == "user" and isinstance(
                            msg_data.get("content"), list
                        ):
                            new_content_parts = []
                            for part in msg_data["content"]:
                                if part.get("type") == "image_url" and isinstance(
                                    part.get("image_url"), dict
                                ):
                                    image_data_url = part["image_url"].get("url", "")
                                    if (
                                        image_data_url.startswith("data:image")
                                        and ";base64," in image_data_url
                                    ):
                                        try:
                                            header, encoded_image_data = (
                                                image_data_url.split(";base64,", 1)
                                            )
                                            mime_type = (
                                                header.split(":", 1)[1]
                                                if ":" in header
                                                else "image/png"
                                            )  # Default to png
                                            image_bytes = base64.b64decode(
                                                encoded_image_data
                                            )

                                            img = Image.open(io.BytesIO(image_bytes))
                                            try:
                                                original_format = img.format or (
                                                    mime_type.split("/")[-1].upper()
                                                    if "/" in mime_type
                                                    else "PNG"
                                                )

                                                # Apply resizing
                                                if current_resize_factor < 1.0:
                                                    new_width = int(
                                                        img.width
                                                        * current_resize_factor
                                                    )
                                                    new_height = int(
                                                        img.height
                                                        * current_resize_factor
                                                    )
                                                    if (
                                                        new_width > 0 and new_height > 0
                                                    ):  # Ensure dimensions are positive
                                                        logging.debug(
                                                            f"Resizing image from {img.size} to ({new_width}, {new_height})"
                                                        )
                                                        img = img.resize(
                                                            (new_width, new_height),
                                                            Image.Resampling.LANCZOS,
                                                        )
                                                        history_was_modified_by_compression = True

                                                output_buffer = io.BytesIO()
                                                save_params = {}
                                                target_format = original_format

                                                if original_format in ["JPEG", "WEBP"]:
                                                    save_params["quality"] = (
                                                        current_compression_quality
                                                    )
                                                    target_format = "JPEG"  # Prefer JPEG for quality adjustments
                                                    if (
                                                        img.mode == "RGBA"
                                                        or img.mode == "LA"
                                                        or (
                                                            img.mode == "P"
                                                            and "transparency"
                                                            in img.info
                                                        )
                                                    ):
                                                        logging.debug(
                                                            f"Image has alpha, converting to RGB for JPEG. Original mode: {img.mode}"
                                                        )
                                                        img = img.convert("RGB")
                                                    history_was_modified_by_compression = True
                                                elif original_format == "PNG":
                                                    save_params["optimize"] = True
                                                    # Could also consider reducing colors for PNG if size is still an issue: img = img.quantize(colors=128)
                                                    # For now, rely on resize and potential future conversion to JPEG if PNGs are too large.

                                                img.save(
                                                    output_buffer,
                                                    format=target_format,
                                                    **save_params,
                                                )
                                                compressed_image_bytes = (
                                                    output_buffer.getvalue()
                                                )
                                            finally:
                                                img.close()

                                            new_mime_type = (
                                                f"image/{target_format.lower()}"
                                            )
                                            new_encoded_data = base64.b64encode(
                                                compressed_image_bytes
                                            ).decode("utf-8")

                                            new_part = part.copy()
                                            new_part["image_url"]["url"] = (
                                                f"data:{new_mime_type};base64,{new_encoded_data}"
                                            )
                                            new_content_parts.append(new_part)
                                            logging.debug(
                                                f"Compressed image. Original size: {len(image_bytes)}, New size: {len(compressed_image_bytes)}, Format: {target_format}, Quality: {current_compression_quality if target_format == 'JPEG' else 'N/A'}, Resize: {current_resize_factor:.2f}"
                                            )
                                        except Exception as compress_exc:
                                            logging.error(
                                                f"Error during image compression: {compress_exc}",
                                                exc_info=True,
                                            )
                                            new_content_parts.append(
                                                part
                                            )  # Add original part back if compression failed
                                    else:
                                        new_content_parts.append(
                                            part
                                        )  # Not a data URL image
                                else:
                                    new_content_parts.append(
                                        part
                                    )  # Not an image_url part
                            msg_data["content"] = new_content_parts

                    if history_was_modified_by_compression:
                        # Track compression information
                        compression_occurred = True
                        final_quality = min(final_quality, current_compression_quality)
                        final_resize = min(final_resize, current_resize_factor)

                        history_for_current_compression_cycle = temp_compressed_history  # Use compressed history for next attempt with this key
                        # Adjust parameters for the *next* compression attempt with this key
                        current_compression_quality = max(
                            10, current_compression_quality - 20
                        )  # Decrease quality more aggressively
                        current_resize_factor = max(
                            0.2, current_resize_factor - 0.15
                        )  # Decrease size more aggressively
                        logging.info(
                            f"Retrying API call for key {key_display} with compressed images (Attempt {compression_attempt}/{max_compression_attempts})."
                        )
                        continue
                    else:
                        logging.warning(
                            f"413 error for key {key_display}, but no images were compressed/modified. Trying next API key."
                        )
                        llm_errors.append(
                            f"Key {key_display}: 413 but no images to compress or compression ineffective."
                        )
                        break
                else:  # Other APIError for OpenAI
                    logging.warning(
                        f"Initial Request OpenAI API Error for key {key_display}: {type(e).__name__} - {e}. Trying next key."
                    )
                    llm_errors.append(
                        f"Key {key_display}: Initial API Error - {type(e).__name__}: {e}"
                    )
                    last_error_type = "api"
                    if isinstance(e, RateLimitError):
                        if current_api_key != "dummy_key":
                            await llm_db_manager.add_key(current_api_key)
                        last_error_type = "rate_limit"
                    break

            except (RateLimitError, google_api_exceptions.ResourceExhausted) as e:
                logging.warning(
                    f"Initial Request Rate limit hit for provider '{provider}' with key {key_display}. Error: {e}. Trying next key."
                )
                if current_api_key != "dummy_key":
                    await llm_db_manager.add_key(current_api_key)
                llm_errors.append(f"Key {key_display}: Initial Rate Limited")
                last_error_type = "rate_limit"
                break
            except (AuthenticationError, google_api_exceptions.PermissionDenied) as e:
                logging.error(
                    f"Initial Request Authentication failed for provider '{provider}' with key {key_display}. Error: {e}. Aborting retries for all keys."
                )
                llm_errors.append(f"Key {key_display}: Initial Authentication Failed")
                raise AllKeysFailedError(provider, llm_errors) from e
            except (
                APIConnectionError,
                google_api_exceptions.ServiceUnavailable,
                google_api_exceptions.DeadlineExceeded,
            ) as e:
                logging.warning(
                    f"Initial Request Connection/Service error for provider '{provider}' with key {key_display}. Error: {e}. Trying next key."
                )
                llm_errors.append(
                    f"Key {key_display}: Initial Connection/Service Error - {type(e).__name__}"
                )
                last_error_type = "connection"
                break

            except (BadRequestError, google_api_exceptions.InvalidArgument) as e:
                # Check if Gemini error is due to "request entity too large"
                is_gemini_too_large = False
                if is_gemini and isinstance(e, google_api_exceptions.InvalidArgument):
                    if "request entity too large" in str(e).lower() or (
                        hasattr(e, "message")
                        and "request entity too large" in e.message.lower()
                    ):
                        is_gemini_too_large = True

                if is_gemini_too_large:
                    logging.warning(
                        f"Gemini API Error (Request Entity Too Large) for key {key_display}, attempt {compression_attempt + 1}."
                    )
                    last_error_type = "api_too_large_gemini"
                    compression_attempt += 1
                    if compression_attempt >= max_compression_attempts:
                        logging.error(
                            f"Max compression attempts reached for Gemini key {key_display} after 'too large' error. Trying next API key."
                        )
                        llm_errors.append(
                            f"Key {key_display}: Max compression retries for Gemini 'too large' failed."
                        )
                        break

                    logging.info(
                        f"Attempting image compression for Gemini (Attempt {compression_attempt}/{max_compression_attempts}) for key {key_display}. Quality: {current_compression_quality}, Resize: {current_resize_factor:.2f}"
                    )
                    history_was_modified_by_compression = False
                    temp_compressed_history = [
                        msg.copy() for msg in current_history_for_api_call
                    ]

                    for msg_data in temp_compressed_history:
                        if isinstance(msg_data.get("parts"), list):
                            new_gemini_parts = []
                            for part in msg_data["parts"]:
                                if (
                                    isinstance(part, google_types.Part)
                                    and hasattr(part, "inline_data")
                                    and part.inline_data
                                    and part.inline_data.mime_type.startswith("image/")
                                ):
                                    try:
                                        image_bytes = part.inline_data.data
                                        img = Image.open(io.BytesIO(image_bytes))
                                        try:
                                            original_format = img.format or (
                                                part.inline_data.mime_type.split("/")[
                                                    -1
                                                ].upper()
                                                if "/" in part.inline_data.mime_type
                                                else "PNG"
                                            )

                                            if current_resize_factor < 1.0:
                                                new_width = int(
                                                    img.width * current_resize_factor
                                                )
                                                new_height = int(
                                                    img.height * current_resize_factor
                                                )
                                                if new_width > 0 and new_height > 0:
                                                    img = img.resize(
                                                        (new_width, new_height),
                                                        Image.Resampling.LANCZOS,
                                                    )
                                                    history_was_modified_by_compression = True

                                            output_buffer = io.BytesIO()
                                            save_params = {}
                                            target_format = original_format

                                            if original_format in ["JPEG", "WEBP"]:
                                                save_params["quality"] = (
                                                    current_compression_quality
                                                )
                                                target_format = "JPEG"
                                                if (
                                                    img.mode == "RGBA"
                                                    or img.mode == "LA"
                                                    or (
                                                        img.mode == "P"
                                                        and "transparency" in img.info
                                                    )
                                                ):
                                                    img = img.convert("RGB")
                                                history_was_modified_by_compression = (
                                                    True
                                                )
                                            elif original_format == "PNG":
                                                save_params["optimize"] = True

                                            img.save(
                                                output_buffer,
                                                format=target_format,
                                                **save_params,
                                            )
                                            compressed_image_bytes = (
                                                output_buffer.getvalue()
                                            )
                                        finally:
                                            img.close()

                                        new_mime_type = f"image/{target_format.lower()}"
                                        new_gemini_parts.append(
                                            google_types.Part.from_bytes(
                                                data=compressed_image_bytes,
                                                mime_type=new_mime_type,
                                            )
                                        )
                                        logging.debug(
                                            f"Gemini Compressed image. Original size: {len(image_bytes)}, New size: {len(compressed_image_bytes)}, Format: {target_format}, Quality: {current_compression_quality if target_format == 'JPEG' else 'N/A'}, Resize: {current_resize_factor:.2f}"
                                        )
                                    except Exception as compress_exc:
                                        logging.error(
                                            f"Error during Gemini image compression: {compress_exc}",
                                            exc_info=True,
                                        )
                                        new_gemini_parts.append(part)
                                else:
                                    new_gemini_parts.append(part)
                            msg_data["parts"] = new_gemini_parts

                    if history_was_modified_by_compression:
                        # Track compression information
                        compression_occurred = True
                        final_quality = min(final_quality, current_compression_quality)
                        final_resize = min(final_resize, current_resize_factor)

                        history_for_current_compression_cycle = temp_compressed_history
                        current_compression_quality = max(
                            10, current_compression_quality - 20
                        )
                        current_resize_factor = max(0.2, current_resize_factor - 0.15)
                        logging.info(
                            f"Retrying Gemini API call for key {key_display} with compressed images (Attempt {compression_attempt}/{max_compression_attempts})."
                        )
                        continue
                    else:
                        logging.warning(
                            f"Gemini 'too large' error for key {key_display}, but no images were compressed/modified. Trying next API key."
                        )
                        llm_errors.append(
                            f"Key {key_display}: Gemini 'too large' but no images to compress or compression ineffective."
                        )
                        break
                else:  # Other BadRequestError or InvalidArgument
                    logging.warning(  # Changed from error to warning, as we are retrying
                        f"Initial Request Bad request error for provider '{provider}' with key {key_display}. Error: {e}. Trying next key."  # Updated message
                    )
                    if isinstance(e, google_api_exceptions.InvalidArgument) and hasattr(
                        e, "details"
                    ):
                        logging.warning(  # Consistent logging level
                            f"Google API InvalidArgument Details: {e.details}"
                        )
                    llm_errors.append(f"Key {key_display}: Initial Bad Request - {e}")
                    last_error_type = (
                        "bad_request"  # Set last_error_type for consistency
                    )
                    break  # Changed from raise to break to try next key

            except UnprocessableEntityError as e:  # OpenAI specific 422
                logging.warning(
                    f"Initial Request Unprocessable Entity error for provider '{provider}' with key {key_display}. Error: {e}. Signaling for fallback."
                )
                llm_errors.append(
                    f"Key {key_display}: Initial Unprocessable Entity - {e}"
                )
                last_error_type = "unprocessable_entity"
                yield None, None, None, "RETRY_WITH_FALLBACK_MODEL_UNPROCESSABLE_ENTITY"
                return
            except (
                google_api_exceptions.GoogleAPICallError
            ) as e:  # Other Google API errors
                logging.warning(
                    f"Initial Request Google API Call Error for key {key_display}: {type(e).__name__} - {e}. Trying next key."
                )
                if hasattr(e, "details"):
                    logging.error(f"Google API Error Details: {e.details}")
                llm_errors.append(
                    f"Key {key_display}: Initial Google API Error - {type(e).__name__}: {e}"
                )
                last_error_type = "google_api"
                break
            except Exception as e:  # Catch other unexpected errors
                logging.exception(
                    f"Unexpected error during initial LLM call with key {key_display}"
                )
                llm_errors.append(
                    f"Key {key_display}: Unexpected Initial Error - {type(e).__name__}"
                )
                last_error_type = "unexpected"
                break
        # --- End of inner compression retry while-loop ---

        # If the compression loop was broken (e.g. by a non-413 error, or max compression attempts for 413),
        # this `continue` will move to the next API key in the outer `for` loop.
        if compression_attempt >= max_compression_attempts and last_error_type in [
            "api_413",
            "api_too_large_gemini",
        ]:
            logging.error(
                f"Max compression attempts reached for key {key_display}. All compression strategies failed for this key."
            )
        # If we broke from the while loop due to a non-413 error, or if the stream completed successfully,
        # the logic above the `except` blocks (like `return` or `continue` for the outer loop) would have handled it.
        # This `continue` ensures we try the next key if the `while` loop exhausted or broke due to an error that means "try next key".
        continue

    # --- Post-Retry Loop (after all API keys tried) ---
    logging.error(
        f"All LLM API keys failed for provider '{provider}'. Errors: {json.dumps(llm_errors)}"
    )
    raise AllKeysFailedError(provider, llm_errors)


async def enhance_prompt_with_llm(
    prompt_to_enhance: str,
    prompt_design_strategies_doc: str,
    prompt_guide_2_doc: str,
    prompt_guide_3_doc: str,
    provider: str,
    model_name: str,
    provider_config: Dict[str, Any],
    extra_params: Dict[str, Any],
    app_config: Dict[str, Any],
) -> AsyncGenerator[
    Tuple[Optional[str], Optional[str], Optional[Any], Optional[str]], None
]:
    """
    Enhances a given prompt using a specified LLM and provided documentation.

    Args:
        prompt_to_enhance: The user's original prompt.
        prompt_design_strategies_doc: Content of the 'prompt design strategies' document.
        prompt_guide_2_doc: Content of the 'prompt guide 2' document.
        prompt_guide_3_doc: Content of the 'prompt guide 3' document.
        provider: The LLM provider.
        model_name: The LLM model name.
        provider_config: Configuration for the provider.
        extra_params: Extra API parameters for the model.
        app_config: Main application configuration.

    Yields:
        Same as generate_response_stream.
    """
    enhancement_user_prompt_content = f"""Improve the following user prompt so it follows the prompt design strategies and prompt guides provided below.
Output only the improved prompt, without any preamble or explanation.

<my prompt>
{prompt_to_enhance}
</my prompt>

<prompt design strategies>
{prompt_design_strategies_doc}
</prompt design strategies>

<prompt guide 2>
{prompt_guide_2_doc}
</prompt guide 2>

<prompt guide 3>
{prompt_guide_3_doc}
</prompt guide 3>

Improved Prompt:"""

    # The main instruction and content are now part of the user message.
    history_for_enhancement_llm = [
        {"role": "user", "content": enhancement_user_prompt_content}
    ]

    logging.info(
        f"Attempting to enhance prompt using {provider}/{model_name} with instructions as user prompt."
    )

    # Use the existing generate_response_stream function
    async for chunk_tuple in generate_response_stream(
        provider=provider,
        model_name=model_name,
        history_for_llm=history_for_enhancement_llm,
        system_prompt_text=app_config.get(PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY),
        provider_config=provider_config,
        extra_params=extra_params,
        app_config=app_config,
    ):
        yield chunk_tuple
