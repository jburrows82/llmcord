import base64
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple

# OpenAI specific imports
from openai import (
    APIError,
    RateLimitError,
    AuthenticationError,
    APIConnectionError,
    BadRequestError,
)

# Google Gemini specific imports
from google.genai import types as google_types
from google.api_core import exceptions as google_api_exceptions

from ..core.image_utils import compress_images_in_history
from .providers.gemini_provider import (
    generate_gemini_stream,
    generate_gemini_image_stream,
)
from .providers.imagen_provider import generate_imagen_image_stream
from .providers.openai_provider import generate_openai_stream


def format_history_for_gemini(
    history_for_llm: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert history format from OpenAI to Gemini format."""
    current_history_for_api_call = []

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
                source_parts_or_content = [{"type": "text", "text": openai_content}]
            elif isinstance(openai_content, list):
                source_parts_or_content = openai_content
            else:
                logging.warning(
                    f"Msg {msg_index} has 'content' of unexpected type {type(openai_content)}. Skipping."
                )
                continue
        else:
            logging.warning(
                f"Msg {msg_index} has neither 'parts' nor 'content'. Skipping."
            )
            continue

        for part_idx, part_item in enumerate(source_parts_or_content):
            if isinstance(part_item, google_types.Part):
                gemini_parts_for_this_msg.append(part_item)
            elif isinstance(part_item, dict):  # OpenAI part dict
                part_type = part_item.get("type")
                if part_type == "text":
                    gemini_parts_for_this_msg.append(
                        google_types.Part.from_text(text=part_item.get("text", ""))
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
                                header.split(":")[1] if ":" in header else "image/png"
                            )
                            img_bytes = base64.b64decode(encoded_data)
                            gemini_parts_for_this_msg.append(
                                google_types.Part.from_bytes(
                                    data=img_bytes, mime_type=mime_type_str
                                )
                            )
                        except Exception as e:
                            logging.warning(
                                f"Error converting OpenAI image_url part: {e}. Skipping part."
                            )
                    else:
                        logging.warning(
                            "Invalid data URL in OpenAI image_url part. Skipping part."
                        )
                else:
                    logging.warning(
                        f"Unsupported OpenAI part type '{part_type}'. Skipping part."
                    )
            elif isinstance(part_item, str):
                gemini_parts_for_this_msg.append(
                    google_types.Part.from_text(text=part_item)
                )
            else:
                logging.warning(
                    f"Unsupported part item type {type(part_item)}. Skipping part."
                )

        if gemini_parts_for_this_msg:
            # Map OpenAI roles to Gemini roles
            gemini_role = "user"
            if original_msg_data["role"] in ["assistant", "model"]:
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
                f"Msg {msg_index} resulted in no valid Gemini parts. Skipping."
            )

    return current_history_for_api_call


def format_history_for_openai(
    history_for_llm: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return a copy of history for OpenAI-compatible providers."""
    return [msg.copy() for msg in history_for_llm]


def get_stream_generator(
    provider: str,
    model_name: str,
    is_gemini: bool,
    current_api_key: str,
    history_for_current_compression_cycle: List[Dict[str, Any]],
    system_prompt_text: Optional[str],
    extra_params: Dict[str, Any],
    app_config: Dict[str, Any],
    base_url: Optional[str] = None,
):
    """Get the appropriate stream generator based on provider and model."""
    if is_gemini:
        if model_name == "gemini-2.0-flash-preview-image-generation":
            return generate_gemini_image_stream(
                api_key=current_api_key,
                model_name=model_name,
                history_for_api_call=history_for_current_compression_cycle,
                extra_params=extra_params,
                app_config=app_config,
            )
        elif model_name.startswith("imagen-"):
            return generate_imagen_image_stream(
                api_key=current_api_key,
                model_name=model_name,
                history_for_api_call=history_for_current_compression_cycle,
                extra_params=extra_params,
                app_config=app_config,
            )
        else:
            return generate_gemini_stream(
                api_key=current_api_key,
                model_name=model_name,
                history_for_api_call=history_for_current_compression_cycle,
                system_instruction_text=system_prompt_text,
                extra_params=extra_params,
                app_config=app_config,
            )
    else:  # OpenAI compatible
        return generate_openai_stream(
            api_key=(current_api_key if current_api_key != "dummy_key" else None),
            base_url=base_url,
            model_name=model_name,
            history_for_api_call=history_for_current_compression_cycle,
            system_prompt_text=system_prompt_text,
            extra_params=extra_params,
            current_provider_name=provider,
        )


async def process_stream_chunks(
    stream_generator_func,
    is_image_generation_model: bool,
    is_gemini: bool,
    key_display: str,
    llm_db_manager,
    current_api_key: str,
    llm_errors: List[str],
) -> AsyncGenerator[Tuple, None]:
    """Process chunks from the stream generator and return status flags."""
    is_blocked_by_safety = False
    is_stopped_by_recitation = False
    stream_finish_reason = None

    if is_image_generation_model:
        async for (
            text_chunk,
            chunk_finish_reason,
            chunk_grounding_metadata,
            error_msg_chunk,
            image_data,
            image_mime_type,
        ) in stream_generator_func:
            if error_msg_chunk:
                llm_errors.append(
                    f"Key {key_display}: Provider Stream Error - {error_msg_chunk}"
                )

                if (
                    "rate limit" in error_msg_chunk.lower()
                    or "resourceexhausted" in error_msg_chunk.lower()
                ):
                    if current_api_key != "dummy_key":
                        await llm_db_manager.add_key(current_api_key)
                elif error_msg_chunk.lower() in [
                    "jsondecodeerror",
                    "stream timeout error",
                ]:
                    pass
                break

            if chunk_finish_reason:
                stream_finish_reason = chunk_finish_reason
            if chunk_grounding_metadata:
                pass  # Grounding metadata available but not currently used

            if text_chunk or image_data:
                if not is_blocked_by_safety and not is_stopped_by_recitation:
                    yield (text_chunk, None, None, None, image_data, image_mime_type)

            if stream_finish_reason:
                if stream_finish_reason.lower() == "safety":
                    is_blocked_by_safety = True
                    llm_errors.append(
                        f"Key {key_display}: Response Blocked (Safety via provider)"
                    )
                elif stream_finish_reason.lower() == "recitation":
                    is_stopped_by_recitation = True
                    llm_errors.append(
                        f"Key {key_display}: Response Stopped (Recitation via provider)"
                    )
                break
    else:
        async for (
            text_chunk,
            chunk_finish_reason,
            chunk_grounding_metadata,
            error_msg_chunk,
            image_data,
            image_mime_type,
        ) in stream_generator_func:
            if error_msg_chunk:
                if error_msg_chunk == "OPENAI_API_ERROR_413_PAYLOAD_TOO_LARGE":
                    if not is_gemini:
                        logging.warning(f"OpenAI API Error 413 for key {key_display}")
                        break
                elif error_msg_chunk == "OPENAI_UNPROCESSABLE_ENTITY_422":
                    if not is_gemini:
                        logging.warning(
                            f"OpenAI Unprocessable Entity (422) for key {key_display}"
                        )
                        yield (
                            None,
                            None,
                            None,
                            "RETRY_WITH_FALLBACK_MODEL_UNPROCESSABLE_ENTITY",
                            None,
                            None,
                        )
                        return

                llm_errors.append(
                    f"Key {key_display}: Provider Stream Error - {error_msg_chunk}"
                )

                if (
                    "rate limit" in error_msg_chunk.lower()
                    or "resourceexhausted" in error_msg_chunk.lower()
                ):
                    if current_api_key != "dummy_key":
                        await llm_db_manager.add_key(current_api_key)
                elif error_msg_chunk.lower() in [
                    "jsondecodeerror",
                    "stream timeout error",
                ]:
                    pass
                break

            if chunk_finish_reason:
                stream_finish_reason = chunk_finish_reason
            if chunk_grounding_metadata:
                pass  # Grounding metadata available but not currently used

            if text_chunk:
                if not is_blocked_by_safety and not is_stopped_by_recitation:
                    yield text_chunk, None, None, None, None, None

            if stream_finish_reason:
                if stream_finish_reason.lower() == "safety":
                    is_blocked_by_safety = True
                    llm_errors.append(
                        f"Key {key_display}: Response Blocked (Safety via provider)"
                    )
                elif stream_finish_reason.lower() == "recitation":
                    is_stopped_by_recitation = True
                    llm_errors.append(
                        f"Key {key_display}: Response Stopped (Recitation via provider)"
                    )
                break

    # Function ends naturally - async generator complete


async def handle_compression_retry(
    current_api_key: str,
    key_display: str,
    provider: str,
    model_name: str,
    current_history_for_api_call: List[Dict[str, Any]],
    system_prompt_text: Optional[str],
    extra_params: Dict[str, Any],
    app_config: Dict[str, Any],
    base_url: Optional[str],
    is_gemini: bool,
    llm_db_manager,
    llm_errors: List[str],
    compression_occurred: bool,
    final_quality: int,
    final_resize: float,
) -> AsyncGenerator[Optional[Tuple], None]:
    """Handle compression retry logic for a single API key."""
    compression_attempt = 0
    max_compression_attempts = 5
    current_compression_quality = 90
    current_resize_factor = 1.0

    history_for_current_compression_cycle = [
        msg.copy() for msg in current_history_for_api_call
    ]

    is_image_generation_model = is_gemini and (
        model_name == "gemini-2.0-flash-preview-image-generation"
        or model_name.startswith("imagen-")
    )

    while compression_attempt < max_compression_attempts:
        is_blocked_by_safety = False
        is_stopped_by_recitation = False
        content_received = False
        stream_finish_reason = None
        stream_grounding_metadata = None

        try:
            # Get stream generator
            if is_gemini:
                if model_name == "gemini-2.0-flash-preview-image-generation":
                    stream_generator_func = generate_gemini_image_stream(
                        api_key=current_api_key,
                        model_name=model_name,
                        history_for_api_call=history_for_current_compression_cycle,
                        extra_params=extra_params,
                        app_config=app_config,
                    )
                elif model_name.startswith("imagen-"):
                    stream_generator_func = generate_imagen_image_stream(
                        api_key=current_api_key,
                        model_name=model_name,
                        history_for_api_call=history_for_current_compression_cycle,
                        extra_params=extra_params,
                        app_config=app_config,
                    )
                else:
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
                    current_provider_name=provider,
                )

            # Process stream - yield all chunks, then handle completion
            stream_error_occurred = False

            async for (
                text_chunk,
                chunk_finish_reason,
                chunk_grounding_metadata,
                error_msg_chunk,
                image_data,
                image_mime_type,
            ) in stream_generator_func:
                if error_msg_chunk:
                    # Handle specific error conditions
                    if (
                        error_msg_chunk == "OPENAI_API_ERROR_413_PAYLOAD_TOO_LARGE"
                        and not is_gemini
                    ):
                        stream_error_occurred = True
                        break  # Will trigger compression retry
                    elif (
                        error_msg_chunk == "OPENAI_UNPROCESSABLE_ENTITY_422"
                        and not is_gemini
                    ):
                        yield (
                            None,
                            None,
                            None,
                            "RETRY_WITH_FALLBACK_MODEL_UNPROCESSABLE_ENTITY",
                            None,
                            None,
                        )
                        return

                    llm_errors.append(
                        f"Key {key_display}: Stream Error - {error_msg_chunk}"
                    )
                    if (
                        "rate limit" in error_msg_chunk.lower()
                        or "resourceexhausted" in error_msg_chunk.lower()
                    ):
                        if current_api_key != "dummy_key":
                            await llm_db_manager.add_key(current_api_key)
                    stream_error_occurred = True
                    break

                # Update metadata
                if chunk_finish_reason:
                    stream_finish_reason = chunk_finish_reason
                if chunk_grounding_metadata:
                    stream_grounding_metadata = chunk_grounding_metadata

                # Yield content chunks
                if text_chunk or image_data:
                    content_received = True
                    if not is_blocked_by_safety and not is_stopped_by_recitation:
                        if is_image_generation_model:
                            yield (
                                text_chunk,
                                None,
                                None,
                                None,
                                image_data,
                                image_mime_type,
                            )
                        else:
                            yield (text_chunk, None, None, None, None, None)

                # Check for safety/recitation blocks
                if stream_finish_reason:
                    if stream_finish_reason.lower() == "safety":
                        is_blocked_by_safety = True
                        yield (
                            None,
                            "safety",
                            stream_grounding_metadata,
                            "Response blocked by safety.",
                            None,
                            None,
                        )
                        return
                    elif stream_finish_reason.lower() == "recitation":
                        is_stopped_by_recitation = True
                        yield (
                            None,
                            "recitation",
                            stream_grounding_metadata,
                            "Response stopped by recitation.",
                            None,
                            None,
                        )
                        return

            # If there was a stream error, handle compression retry
            if stream_error_occurred:
                continue  # Go to compression retry logic

            # After the stream processing loop, handle completion
            if stream_finish_reason and content_received:
                is_successful_finish = stream_finish_reason.lower() in (
                    "stop",
                    "end_turn",
                ) or (
                    is_gemini
                    and stream_finish_reason
                    == str(google_types.FinishReason.FINISH_REASON_UNSPECIFIED)
                )
                is_acceptable_with_content = stream_finish_reason.lower() in (
                    "content_filter",
                    "length",
                    "max_tokens",
                )

                if is_successful_finish or is_acceptable_with_content:
                    if compression_occurred:
                        quality_pct = final_quality
                        resize_pct = int(final_resize * 100)
                        user_warning = f"⚠️ Image at {quality_pct}% quality, resized to {resize_pct}%"
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
                    return

            # Handle stream ending with content but no finish reason
            if content_received and not stream_finish_reason:
                if not is_gemini:
                    yield (
                        None,
                        None,
                        stream_grounding_metadata,
                        "RETRY_WITH_GEMINI_NO_FINISH_REASON",
                        None,
                        None,
                    )
                    return
                else:
                    yield (None, "stop", stream_grounding_metadata, None, None, None)
                    return

            # If we get here without content or finish reason, try compression or next key
            if not content_received and not stream_finish_reason:
                logging.warning(
                    f"No content or finish reason for key {key_display}, trying next compression attempt or key"
                )
                break

        except APIError as e:
            if not is_gemini and hasattr(e, "status_code") and e.status_code == 413:
                compression_attempt += 1
                if compression_attempt >= max_compression_attempts:
                    llm_errors.append(
                        f"Key {key_display}: Max compression retries failed."
                    )
                    break

                (
                    history_for_current_compression_cycle,
                    history_was_modified,
                ) = await compress_images_in_history(
                    history=current_history_for_api_call,
                    is_gemini_provider=False,
                    compression_quality=current_compression_quality,
                    resize_factor=current_resize_factor,
                )

                if history_was_modified:
                    compression_occurred = True
                    final_quality = min(final_quality, current_compression_quality)
                    final_resize = min(final_resize, current_resize_factor)
                    current_compression_quality = max(
                        10, current_compression_quality - 20
                    )
                    current_resize_factor = max(0.2, current_resize_factor - 0.15)
                    continue
                else:
                    llm_errors.append(f"Key {key_display}: No images to compress.")
                    break
            else:
                llm_errors.append(f"Key {key_display}: API Error - {type(e).__name__}")
                if isinstance(e, RateLimitError):
                    if current_api_key != "dummy_key":
                        await llm_db_manager.add_key(current_api_key)
                break

        except (BadRequestError, google_api_exceptions.InvalidArgument) as e:
            is_gemini_too_large = False
            if is_gemini and isinstance(e, google_api_exceptions.InvalidArgument):
                if "request entity too large" in str(e).lower():
                    is_gemini_too_large = True

            if is_gemini_too_large:
                compression_attempt += 1
                if compression_attempt >= max_compression_attempts:
                    break

                (
                    history_for_current_compression_cycle,
                    history_was_modified,
                ) = await compress_images_in_history(
                    history=current_history_for_api_call,
                    is_gemini_provider=True,
                    compression_quality=current_compression_quality,
                    resize_factor=current_resize_factor,
                )

                if history_was_modified:
                    compression_occurred = True
                    final_quality = min(final_quality, current_compression_quality)
                    final_resize = min(final_resize, current_resize_factor)
                    current_compression_quality = max(
                        10, current_compression_quality - 20
                    )
                    current_resize_factor = max(0.2, current_resize_factor - 0.15)
                    continue
                else:
                    break
            else:
                llm_errors.append(f"Key {key_display}: Bad Request - {e}")
                break

        except Exception as e:
            llm_errors.append(
                f"Key {key_display}: Unexpected Error - {type(e).__name__}"
            )
            break

    yield None  # Signal that this key failed


def handle_api_errors(
    e: Exception,
    provider: str,
    key_display: str,
    is_gemini: bool,
    llm_errors: List[str],
    current_api_key: str,
    llm_db_manager,
) -> str:
    """Handle various API errors and return error type."""
    if isinstance(e, (RateLimitError, google_api_exceptions.ResourceExhausted)):
        logging.warning(
            f"Rate limit hit for provider '{provider}' with key {key_display}"
        )
        if current_api_key != "dummy_key":
            llm_db_manager.add_key(current_api_key)
        llm_errors.append(f"Key {key_display}: Rate Limited")
        return "rate_limit"
    elif isinstance(e, (AuthenticationError, google_api_exceptions.PermissionDenied)):
        logging.error(
            f"Authentication failed for provider '{provider}' with key {key_display}"
        )
        llm_errors.append(f"Key {key_display}: Authentication Failed")
        return "auth_failed"
    elif isinstance(e, (APIConnectionError, google_api_exceptions.ServiceUnavailable)):
        logging.warning(
            f"Connection error for provider '{provider}' with key {key_display}"
        )
        llm_errors.append(f"Key {key_display}: Connection Error")
        return "connection"
    else:
        logging.exception(f"Unexpected error with key {key_display}")
        llm_errors.append(f"Key {key_display}: Unexpected Error")
        return "unexpected"
