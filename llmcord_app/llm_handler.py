import base64
import logging
import random
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple

from datetime import date  # Specifically for enhance_prompt_with_llm

# OpenAI specific imports
from openai import (
    APIError,
    RateLimitError,
    AuthenticationError,
    APIConnectionError,
    BadRequestError,
    UnprocessableEntityError,
)

# Google Gemini specific imports
from google.genai import (
    types as google_types,
)  # Re-add for type hints and internal logic
from google.api_core import (
    exceptions as google_api_exceptions,
)  # Re-add for exception handling

from .constants import (
    AllKeysFailedError,
    PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY,  # New
)
from .rate_limiter import get_db_manager, get_available_keys
from .image_utils import compress_images_in_history
from .llm_providers.gemini_provider import (
    generate_gemini_stream,
    generate_gemini_image_stream,
)  # Added import
from .llm_providers.openai_provider import generate_openai_stream  # Added import


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
    Tuple[
        Optional[str],
        Optional[str],
        Optional[Any],
        Optional[str],
        Optional[bytes],
        Optional[str],
    ],
    None,
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
        - image_data (Optional[bytes]): Image data if available (for image generation models).
        - image_mime_type (Optional[str]): MIME type of the image if available.

    Raises:
        AllKeysFailedError: If all available API keys for the provider fail.
        ValueError: If configuration is invalid (e.g., missing keys for required provider).
    """
    all_api_keys = provider_config.get("api_keys", [])
    base_url = provider_config.get("base_url")
    is_gemini = provider == "google"
    is_image_generation_model = (
        is_gemini and model_name == "gemini-2.0-flash-preview-image-generation"
    )

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
            is_blocked_by_safety = False
            is_stopped_by_recitation = False
            content_received = False
            chunk_processed_successfully = False
            stream_finish_reason = None
            stream_grounding_metadata = None

            try:  # Innermost try for the specific API call attempt (with potentially compressed data)
                # --- Delegate to Provider-Specific Stream Generation ---
                stream_generator_func = None
                if is_gemini:
                    if is_image_generation_model:
                        # Use special image generation function (no system prompts, no grounding)
                        stream_generator_func = generate_gemini_image_stream(
                            api_key=current_api_key,
                            model_name=model_name,
                            history_for_api_call=history_for_current_compression_cycle,
                            extra_params=extra_params,
                            app_config=app_config,
                        )
                    else:
                        # Use regular Gemini function
                        stream_generator_func = generate_gemini_stream(
                            api_key=current_api_key,
                            model_name=model_name,
                            history_for_api_call=history_for_current_compression_cycle,
                            system_instruction_text=system_prompt_text,
                            extra_params=extra_params,
                            app_config=app_config,
                        )
                else:  # OpenAI compatible
                    stream_generator_func = generate_openai_stream(
                        api_key=(
                            current_api_key if current_api_key != "dummy_key" else None
                        ),
                        base_url=base_url,
                        model_name=model_name,
                        history_for_api_call=history_for_current_compression_cycle,
                        system_prompt_text=system_prompt_text,
                        extra_params=extra_params,
                        current_provider_name=provider,  # Pass current provider name
                    )

                # --- Stream Processing Loop (now consumes from provider-specific generator) ---
                if is_image_generation_model:
                    # Handle image generation model with expanded tuple
                    async for (
                        text_chunk,
                        chunk_finish_reason,
                        chunk_grounding_metadata,
                        error_msg_chunk,
                        image_data,
                        image_mime_type,
                    ) in stream_generator_func:
                        chunk_processed_successfully = (
                            False  # Reset for each chunk from provider
                        )

                        if error_msg_chunk:
                            # Provider function signals an error
                            llm_errors.append(
                                f"Key {key_display}: Provider Stream Error - {error_msg_chunk}"
                            )
                            last_error_type = "provider_stream_error"

                            # Decide if this error from provider is a rate limit
                            if (
                                "rate limit" in error_msg_chunk.lower()
                                or "resourceexhausted" in error_msg_chunk.lower()
                            ):
                                if current_api_key != "dummy_key":
                                    await llm_db_manager.add_key(current_api_key)
                                last_error_type = "rate_limit"

                            break  # Break from stream consumption, try next key

                        # If no error from provider chunk, process as before
                        if chunk_finish_reason:
                            stream_finish_reason = chunk_finish_reason
                        if chunk_grounding_metadata:
                            stream_grounding_metadata = chunk_grounding_metadata

                        if text_chunk or image_data:
                            content_received = True
                            # Yield content if not blocked
                            if (
                                not is_blocked_by_safety
                                and not is_stopped_by_recitation
                            ):
                                yield (
                                    text_chunk,
                                    None,
                                    None,
                                    None,
                                    image_data,
                                    image_mime_type,
                                )

                        chunk_processed_successfully = True

                        if stream_finish_reason:
                            # Check if Gemini specific safety/recitation finish reasons were passed up
                            if stream_finish_reason.lower() == "safety":
                                is_blocked_by_safety = True
                                last_error_type = "safety"
                                llm_errors.append(
                                    f"Key {key_display}: Response Blocked (Safety via provider)"
                                )
                            elif stream_finish_reason.lower() == "recitation":
                                is_stopped_by_recitation = True
                                last_error_type = "recitation"
                                llm_errors.append(
                                    f"Key {key_display}: Response Stopped (Recitation via provider)"
                                )
                            break  # Break from stream consumption
                else:
                    # Handle regular models - now also expect 6-tuple but ignore image data
                    async for (
                        text_chunk,
                        chunk_finish_reason,
                        chunk_grounding_metadata,
                        error_msg_chunk,
                        image_data,  # Will be None for regular models
                        image_mime_type,  # Will be None for regular models
                    ) in stream_generator_func:
                        chunk_processed_successfully = (
                            False  # Reset for each chunk from provider
                        )

                        if error_msg_chunk:
                            # Provider function signals an error
                            # Specific error strings can be used for special handling (e.g., compression)
                            if (
                                error_msg_chunk
                                == "OPENAI_API_ERROR_413_PAYLOAD_TOO_LARGE"
                            ):
                                if (
                                    not is_gemini
                                ):  # This error is specific to OpenAI path
                                    logging.warning(
                                        f"OpenAI API Error 413 (Request Entity Too Large) for key {key_display}, attempt {compression_attempt + 1}."
                                    )
                                    last_error_type = "api_413"
                                    # Break from this inner stream consumption to trigger compression logic below
                                    break
                            elif error_msg_chunk == "OPENAI_UNPROCESSABLE_ENTITY_422":
                                if not is_gemini:
                                    logging.warning(
                                        f"OpenAI Unprocessable Entity (422) for key {key_display}. Signaling for fallback."
                                    )
                                    last_error_type = "unprocessable_entity"
                                    # Yield a specific signal that the main handler in response_sender.py can catch
                                    yield (
                                        None,
                                        None,
                                        None,
                                        "RETRY_WITH_FALLBACK_MODEL_UNPROCESSABLE_ENTITY",
                                        None,
                                        None,
                                    )
                                    return  # Stop this entire generation attempt

                            # Handle other errors from provider stream
                            llm_errors.append(
                                f"Key {key_display}: Provider Stream Error - {error_msg_chunk}"
                            )
                            last_error_type = "provider_stream_error"  # Generic type for errors from provider stream

                            # Decide if this error from provider is a rate limit
                            if (
                                "rate limit" in error_msg_chunk.lower()
                                or "resourceexhausted" in error_msg_chunk.lower()
                            ):
                                if current_api_key != "dummy_key":
                                    await llm_db_manager.add_key(current_api_key)
                                last_error_type = "rate_limit"  # Be more specific if it's a rate limit

                            break  # Break from stream consumption, try next compression or key

                        # If no error from provider chunk, process as before
                        if chunk_finish_reason:
                            stream_finish_reason = chunk_finish_reason
                        if chunk_grounding_metadata:  # Mainly for Gemini
                            stream_grounding_metadata = chunk_grounding_metadata

                        if text_chunk:
                            content_received = True
                            # Yield content if not blocked
                            if (
                                not is_blocked_by_safety
                                and not is_stopped_by_recitation
                            ):  # These flags are set by Gemini provider
                                yield text_chunk, None, None, None, None, None

                        chunk_processed_successfully = (
                            True  # If we got here without error_msg_chunk
                        )

                        if stream_finish_reason:
                            # Check if Gemini specific safety/recitation finish reasons were passed up
                            if stream_finish_reason.lower() == "safety":
                                is_blocked_by_safety = True
                                last_error_type = "safety"
                                llm_errors.append(
                                    f"Key {key_display}: Response Blocked (Safety via provider)"
                                )
                            elif stream_finish_reason.lower() == "recitation":
                                is_stopped_by_recitation = True
                                last_error_type = "recitation"
                                llm_errors.append(
                                    f"Key {key_display}: Response Stopped (Recitation via provider)"
                                )
                            break  # Break from stream consumption

                # --- After Stream Processing Loop for this attempt ---

                # and inside the `try` that catches APIError, but before generic error handling.
                # The logic for compression retry should be triggered if `last_error_type` is `api_413`.

                if (
                    last_error_type == "api_413"
                ):  # Check if the stream broke due to 413 signal
                    compression_attempt += 1
                    if compression_attempt >= max_compression_attempts:
                        logging.error(
                            f"Max compression attempts for 413 error on key {key_display}. Trying next API key."
                        )
                        llm_errors.append(
                            f"Key {key_display}: Max compression for 413 failed."
                        )
                        break  # Break compression loop, try next API key

                    logging.info(
                        f"Attempting image compression for OpenAI (Attempt {compression_attempt}/{max_compression_attempts}) for key {key_display}. Quality: {current_compression_quality}, Resize: {current_resize_factor:.2f}"
                    )
                    (
                        history_for_current_compression_cycle,
                        history_was_modified_by_compression,
                    ) = await compress_images_in_history(
                        history=current_history_for_api_call,  # Start with the history for this API key
                        is_gemini_provider=False,  # OpenAI
                        compression_quality=current_compression_quality,
                        resize_factor=current_resize_factor,
                    )
                    if history_was_modified_by_compression:
                        compression_occurred = True
                        final_quality = min(final_quality, current_compression_quality)
                        final_resize = min(final_resize, current_resize_factor)
                        current_compression_quality = max(
                            10, current_compression_quality - 20
                        )
                        current_resize_factor = max(0.2, current_resize_factor - 0.15)
                        logging.info(
                            f"Retrying API call for key {key_display} with compressed images."
                        )
                        continue  # Continue to next compression attempt
                    else:
                        logging.warning(
                            f"413 error for key {key_display}, but no images were compressed/modified. Trying next API key."
                        )
                        llm_errors.append(
                            f"Key {key_display}: 413 but no images to compress or compression ineffective."
                        )
                        break  # Break compression loop

                # The rest of the logic after stream processing (safety blocks, successful finish, etc.)
                # remains largely the same, using `stream_finish_reason`, `content_received`, etc.
                # which are now populated by consuming the provider-specific stream.
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
                        None,
                        None,
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
                            yield (
                                None,
                                None,
                                None,
                                None,
                                None,
                                f"COMPRESSION_INFO:{user_warning}",
                            )

                        yield (
                            None,
                            stream_finish_reason,
                            stream_grounding_metadata,
                            None,
                            None,
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
                            None,
                            None,
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
                            None,
                            None,
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
                            None,
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

                    # Use the new image utility function
                    (
                        temp_compressed_history,
                        history_was_modified_by_compression,
                    ) = await compress_images_in_history(
                        history=current_history_for_api_call,  # Pass the current version of history for this key
                        is_gemini_provider=False,  # OpenAI
                        compression_quality=current_compression_quality,
                        resize_factor=current_resize_factor,
                    )

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

                    # Use the new image utility function
                    (
                        temp_compressed_history,
                        history_was_modified_by_compression,
                    ) = await compress_images_in_history(
                        history=current_history_for_api_call,  # Pass current version of history for this key
                        is_gemini_provider=True,  # Gemini
                        compression_quality=current_compression_quality,
                        resize_factor=current_resize_factor,
                    )

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
                yield None, None, None, None, None, None
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
        system_prompt_text=(
            f"{app_config.get(PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY, '')}\n"
            f"Current date: {date.today().strftime('%Y-%m-%d')}"  # Use imported date
        ),
        provider_config=provider_config,
        extra_params=extra_params,
        app_config=app_config,
    ):
        # Extract only the first 4 elements from the 6-tuple returned by generate_response_stream
        text_chunk, finish_reason, grounding_metadata, error_message = chunk_tuple[:4]
        yield text_chunk, finish_reason, grounding_metadata, error_message
