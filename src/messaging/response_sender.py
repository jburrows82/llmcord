import asyncio
import logging
import io
from datetime import datetime as dt
from typing import List, Dict, Optional, Any, Set

import discord
from google.genai import types as google_types  # For finish_reason comparison

from ..core import models  # Use relative import
from ..core.constants import (
    EMBED_COLOR_COMPLETE,
    EMBED_COLOR_INCOMPLETE,
    EMBED_COLOR_ERROR,
    STREAMING_INDICATOR,
    EDIT_DELAY_SECONDS_CONFIG_KEY,  # New
    MAX_EMBED_DESCRIPTION_LENGTH,
    MAX_PLAIN_TEXT_LENGTH,
    IMGUR_HEADER,
    IMGUR_URL_PATTERN,
    IMGUR_URL_PREFIX,
    AllKeysFailedError,
    GEMINI_USE_THINKING_BUDGET_CONFIG_KEY,
    GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY,
    FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY,  # New
    FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY,
)
from ..ui.ui import ResponseActionView
from ..llm.handler import generate_response_stream
from ..services.prompt_utils import prepare_system_prompt  # <-- ADDED
from ..bot.commands import (
    get_user_gemini_thinking_budget_preference,
)

# Forward declaration for LLMCordClient to resolve circular import for type hinting if needed
# However, it's better to pass necessary attributes directly if possible.


async def send_initial_processing_message(
    new_msg: discord.Message, use_plain_responses: bool
) -> Optional[discord.Message]:
    """Sends the initial 'Processing request...' message."""
    processing_msg: Optional[discord.Message] = None
    try:
        if use_plain_responses:
            processing_msg = await new_msg.reply(
                "⏳ Processing request...", mention_author=False, suppress_embeds=True
            )
        else:
            processing_embed = discord.Embed(
                description="⏳ Processing request...", color=EMBED_COLOR_INCOMPLETE
            )
            processing_msg = await new_msg.reply(
                embed=processing_embed, mention_author=False
            )
    except discord.HTTPException as e:
        logging.warning(f"Failed to send initial 'Processing request...' message: {e}")
    except Exception as e:
        logging.error(
            f"Unexpected error sending initial 'Processing request...' message: {e}",
            exc_info=True,
        )
    return processing_msg


async def handle_llm_response_stream(
    client,  # Actually an LLMCordClient instance, pass necessary attributes/methods
    new_msg: discord.Message,
    processing_msg: Optional[discord.Message],
    provider: str,
    model_name: str,
    history_for_llm: List[Dict[str, Any]],
    system_prompt_text: Optional[str],
    provider_config: Dict[str, Any],
    extra_api_params: Dict[str, Any],
    app_config: Dict[str, Any],
    initial_user_warnings: Set[str],
    use_plain_responses_config: bool,
    split_limit_config: int,
    custom_search_queries_generated: bool,  # New parameter
    successful_api_results_count: int,  # New parameter
):
    """Handles the streaming, editing, and sending of LLM responses."""
    response_msgs: List[discord.Message] = []
    final_text_to_return = ""
    llm_call_successful_final = False

    edit_task = None  # Keep edit_task for embed streaming

    should_retry_with_gemini_signal = False
    should_retry_due_to_unprocessable_entity = False  # New flag for 422 error
    should_retry_due_to_all_keys_failed = False  # New flag for all keys failed

    original_provider_param = provider
    original_model_name_param = model_name
    original_provider_config_param = provider_config.copy()
    original_extra_api_params_param = extra_api_params.copy()
    original_system_prompt_text = system_prompt_text  # Store original system prompt

    current_provider = original_provider_param
    current_model_name = original_model_name_param
    current_provider_config = original_provider_config_param
    current_extra_params = original_extra_api_params_param
    current_system_prompt_text = original_system_prompt_text  # Initialize with original

    for attempt_num in range(2):  # 0 for original, 1 for retry
        if attempt_num == 1:  # This block is for the retry attempt
            if not (
                should_retry_with_gemini_signal
                or should_retry_due_to_unprocessable_entity
                or should_retry_due_to_all_keys_failed  # Check new flag
            ):
                break  # No signal to retry, so exit the loop

            if should_retry_with_gemini_signal:
                logging.info(
                    f"Original model '{original_model_name_param}' stream ended without finish reason. Deleting incomplete messages and retrying with fallback model..."
                )
            elif should_retry_due_to_unprocessable_entity:
                logging.info(
                    f"Original model '{original_model_name_param}' failed with Unprocessable Entity. Deleting incomplete messages and retrying with fallback model..."
                )
            elif should_retry_due_to_all_keys_failed:
                logging.info(
                    f"Original model '{original_model_name_param}' failed due to all API keys exhausted. Deleting incomplete messages and retrying with fallback model..."
                )

            # --- Deletion Logic for Incomplete Messages ---
            deleted_processing_msg_original_id = None
            if processing_msg:  # Store original processing_msg's ID
                deleted_processing_msg_original_id = processing_msg.id

            if response_msgs:
                logging.info(
                    f"Deleting {len(response_msgs)} incomplete response message(s) before retrying."
                )
                # Iterate over a copy for safe modification and access to original objects
                temp_response_msgs_copy = list(response_msgs)
                response_msgs.clear()  # Clear the main list that will be used for the retry attempt

                for msg_to_delete in reversed(temp_response_msgs_copy):
                    try:
                        await msg_to_delete.delete()
                        await asyncio.sleep(
                            0.2
                        )  # Small delay to avoid hitting Discord rate limits on delete
                    except discord.NotFound:
                        logging.warning(
                            f"Message {msg_to_delete.id} was already deleted (NotFound)."
                        )
                    except discord.HTTPException as e:
                        logging.error(
                            f"Failed to delete message {msg_to_delete.id}: {e}"
                        )
                    finally:
                        # Clean up from msg_nodes cache
                        if msg_to_delete.id in client.msg_nodes:
                            client.msg_nodes.pop(msg_to_delete.id, None)
                            # Lock handling removed here, as it's managed by the context manager
                            # where the lock was acquired. If a node is deleted, its lock should
                            # ideally be released if held, or the task using it cancelled.
                            # For simplicity, we assume locks are released before node deletion,
                            # or that the context manager handles this implicitly upon task completion/cancellation.
                        # If the original processing_msg was one of the deleted messages, nullify the variable
                        if (
                            deleted_processing_msg_original_id
                            and deleted_processing_msg_original_id == msg_to_delete.id
                        ):
                            processing_msg = None
            # --- End Deletion Logic ---

            fallback_model_str = client.config.get(
                FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY,
                "google/gemini-2.5-flash-preview-05-20",
            )
            if should_retry_with_gemini_signal:
                warning_message = f"⚠️ Original model ({original_model_name_param}) stream incomplete. Retrying with `{fallback_model_str}`..."
            elif should_retry_due_to_unprocessable_entity:
                warning_message = f"⚠️ Original model ({original_model_name_param}) failed (422 Error). Retrying with `{fallback_model_str}`..."
            elif should_retry_due_to_all_keys_failed:
                warning_message = f"⚠️ Original model ({original_model_name_param}) failed (All API keys exhausted). Retrying with `{fallback_model_str}`..."
            else:  # Should not happen if loop condition is correct
                warning_message = f"⚠️ Retrying with `{fallback_model_str}` due to an unspecified issue with {original_model_name_param}."
            initial_user_warnings.add(
                warning_message
            )  # This warning will be shown in the retry message

            try:
                current_provider, current_model_name = fallback_model_str.split("/", 1)
            except ValueError:
                logging.error(
                    f"Invalid format for '{FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY}': '{fallback_model_str}'. Cannot retry."
                )
                error_text = f"⚠️ Internal configuration error: Invalid fallback model format ('{fallback_model_str}' for key '{FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY}'). Cannot retry."
                await _handle_llm_exception(
                    new_msg,
                    processing_msg,
                    response_msgs,
                    error_text,
                    use_plain_responses_config,
                    client.config.get("use_plain_responses", False),
                )
                return llm_call_successful_final, final_text_to_return, response_msgs

            # Ensure client.config is available and correctly structured
            if not hasattr(client, "config") or not isinstance(client.config, dict):
                logging.error(
                    "Client configuration is missing or invalid. Cannot proceed with Gemini retry."
                )
                error_text = "⚠️ Internal configuration error. Cannot retry with Gemini."
                await _handle_llm_exception(
                    new_msg,
                    processing_msg,
                    response_msgs,
                    error_text,
                    use_plain_responses_config,
                    client.config.get("use_plain_responses", False),
                )
                return llm_call_successful_final, final_text_to_return, response_msgs

            current_provider_config = client.config.get("providers", {}).get(
                current_provider, {}
            )
            if not current_provider_config or not current_provider_config.get(
                "api_keys"
            ):
                logging.error(
                    f"Cannot retry with Gemini: Provider '{current_provider}' not configured with API keys."
                )
                error_text = f"⚠️ Cannot retry with Gemini: '{current_provider}' not configured. Original attempt also failed to complete."
                await _handle_llm_exception(
                    new_msg,
                    processing_msg,
                    response_msgs,
                    error_text,
                    use_plain_responses_config,
                    client.config.get("use_plain_responses", False),
                )
                return llm_call_successful_final, final_text_to_return, response_msgs

            current_extra_params = client.config.get("extra_api_parameters", {}).copy()
            # Determine if the fallback model is Gemini to adjust parameters
            is_fallback_gemini = current_provider == "google"

            if "max_tokens" in current_extra_params and is_fallback_gemini:
                current_extra_params["max_output_tokens"] = current_extra_params.pop(
                    "max_tokens"
                )
            # Remove max_output_tokens if fallback is not Gemini and it was added for Gemini
            elif (
                "max_output_tokens" in current_extra_params
                and not is_fallback_gemini
                and "max_tokens" not in current_extra_params
            ):
                current_extra_params["max_tokens"] = current_extra_params.pop(
                    "max_output_tokens"
                )

            if is_fallback_gemini:
                global_use_thinking_budget = client.config.get(
                    GEMINI_USE_THINKING_BUDGET_CONFIG_KEY, False
                )
                user_wants_thinking_budget = get_user_gemini_thinking_budget_preference(
                    new_msg.author.id, global_use_thinking_budget
                )
                if user_wants_thinking_budget:
                    current_extra_params["thinking_budget"] = client.config.get(
                        GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY, 0
                    )
            else:  # Remove Gemini specific params if fallback is not Gemini
                current_extra_params.pop("thinking_budget", None)
                current_extra_params.pop(
                    "max_output_tokens", None
                )  # Ensure this is removed if not already handled by max_tokens logic

            # Get and prepare the system prompt for the fallback model
            fallback_system_prompt_text_from_config = client.config.get(
                FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY
            )
            current_system_prompt_text = prepare_system_prompt(
                is_fallback_gemini,
                current_provider,
                fallback_system_prompt_text_from_config,
            )
            logging.info(
                f"Using fallback system prompt for retry: '{current_system_prompt_text}'"
            )

            final_text_for_this_attempt = ""
            grounding_metadata_for_this_attempt = None

            if processing_msg:
                try:
                    retry_status_text = "⏳ Retrying with Gemini..."
                    if use_plain_responses_config:
                        await processing_msg.edit(
                            content=retry_status_text, embed=None, view=None
                        )
                    else:
                        retry_embed = discord.Embed(
                            description=retry_status_text, color=EMBED_COLOR_INCOMPLETE
                        )
                        await processing_msg.edit(embed=retry_embed, view=None)

                    if (
                        response_msgs
                        and processing_msg.id == response_msgs[0].id
                        and len(response_msgs) == 1
                    ):
                        pass
                    else:
                        response_msgs = [processing_msg] if processing_msg else []
                except discord.HTTPException as e:
                    logging.warning(
                        f"Failed to edit processing_msg for Gemini retry: {e}"
                    )
                    response_msgs = []
            else:
                response_msgs = []

            # Reset all retry flags after initiating the retry
            should_retry_with_gemini_signal = False
            should_retry_due_to_unprocessable_entity = False
            should_retry_due_to_all_keys_failed = False

        base_embed = discord.Embed()
        # Footer text construction moved to where it's set on the final segment
        for warning_text in sorted(
            list(initial_user_warnings)
        ):  # Use list() for sorting a set
            base_embed.add_field(name=warning_text, value="", inline=False)

        current_attempt_llm_successful = False
        final_text_for_this_attempt = ""
        grounding_metadata_for_this_attempt = None

        # Special handling for image generation model
        is_image_generation_model = (
            current_provider == "google"
            and (
                current_model_name == "gemini-2.0-flash-preview-image-generation"
                or current_model_name.startswith("imagen-")
            )
        )
        accumulated_image_data = None
        accumulated_image_mime_type = None

        try:
            async with new_msg.channel.typing():
                stream_generator = generate_response_stream(
                    provider=current_provider,
                    model_name=current_model_name,
                    history_for_llm=history_for_llm,
                    system_prompt_text=current_system_prompt_text,  # Use current_system_prompt_text
                    provider_config=current_provider_config,
                    extra_params=current_extra_params,
                    app_config=client.config,  # Pass app_config
                )

                async for (
                    text_chunk,
                    finish_reason,
                    chunk_grounding_metadata,
                    error_message,
                    image_data,
                    image_mime_type,
                ) in stream_generator:
                    if error_message == "RETRY_WITH_GEMINI_NO_FINISH_REASON":
                        if (
                            original_provider_param != "google" and attempt_num == 0
                        ):  # Only on first attempt for non-Gemini
                            should_retry_with_gemini_signal = True
                            logging.info(
                                "Signal received: RETRY_WITH_GEMINI_NO_FINISH_REASON."
                            )
                            break  # Break from stream_generator loop to trigger retry logic
                        else:  # If it's already Gemini or a retry attempt, treat as a normal error
                            error_message = "Stream ended without finish reason."
                    elif (
                        error_message
                        == "RETRY_WITH_FALLBACK_MODEL_UNPROCESSABLE_ENTITY"
                    ):
                        if attempt_num == 0:  # Only on first attempt
                            should_retry_due_to_unprocessable_entity = True
                            logging.info(
                                "Signal received: RETRY_WITH_FALLBACK_MODEL_UNPROCESSABLE_ENTITY."
                            )
                            break  # Break from stream_generator loop to trigger retry logic
                        else:  # If it's already a retry attempt, treat as a normal error
                            error_message = (
                                "Unprocessable Entity error on retry attempt."
                            )

                    # Check for compression warning message
                    if error_message and error_message.startswith("COMPRESSION_INFO:"):
                        try:
                            # Extract compression information from the error_message
                            compression_msg = error_message.replace(
                                "COMPRESSION_INFO:", ""
                            ).strip()
                            logging.info(
                                f"Image compression detected: {compression_msg}"
                            )

                            # Create a new embed with the same properties
                            new_embed = discord.Embed(color=base_embed.color)

                            # Copy footer if exists
                            if base_embed.footer:
                                new_embed.set_footer(text=base_embed.footer.text)

                            # Copy fields except existing compression warnings
                            for field in base_embed.fields:
                                if not field.name.startswith(
                                    "⚠️ The image is at"
                                ) and not field.name.startswith("⚠️ the image is at"):
                                    new_embed.add_field(
                                        name=field.name,
                                        value=field.value,
                                        inline=field.inline,
                                    )

                            # Add the new compression warning
                            warning_field_name = compression_msg
                            new_embed.add_field(
                                name=warning_field_name, value="", inline=False
                            )

                            # Replace the base_embed with our new one
                            base_embed = new_embed

                            # Continue processing, this isn't a fatal error
                            error_message = None
                        except Exception as e:
                            logging.error(f"Error processing compression warning: {e}")
                            # Continue with normal processing if parsing fails

                    if error_message:  # Handles cases where error_message was set above or passed directly
                        logging.error(
                            f"LLM stream failed for {current_provider}/{current_model_name} (Attempt {attempt_num + 1}): {error_message}"
                        )
                        await _handle_stream_error(
                            new_msg,
                            processing_msg,
                            response_msgs,
                            error_message,
                            use_plain_responses_config,
                        )
                        return (
                            llm_call_successful_final,
                            final_text_for_this_attempt,
                            response_msgs,
                        )

                    if chunk_grounding_metadata:
                        grounding_metadata_for_this_attempt = chunk_grounding_metadata
                    if text_chunk:
                        if isinstance(text_chunk, list):
                            final_text_for_this_attempt += "".join(text_chunk)
                        else:
                            final_text_for_this_attempt += text_chunk

                    # For image generation model, accumulate image data but don't send yet
                    if is_image_generation_model and image_data and image_mime_type:
                        accumulated_image_data = image_data
                        accumulated_image_mime_type = image_mime_type
                        logging.info(
                            f"Accumulated image data: {len(image_data)} bytes, MIME type: {image_mime_type}"
                        )

                    # Skip embed processing entirely for image generation model
                    if is_image_generation_model:
                        if finish_reason:
                            current_attempt_llm_successful = (
                                finish_reason.lower()
                                in (
                                    "stop",
                                    "end_turn",
                                )
                                or (
                                    current_provider == "google"
                                    and finish_reason
                                    == str(
                                        google_types.FinishReason.FINISH_REASON_UNSPECIFIED
                                    )
                                )
                            ) or finish_reason.lower() in (
                                "content_filter",
                                "length",
                                "max_tokens",
                            )
                            break
                        continue

                    # Regular embed processing for non-image generation models
                    if not use_plain_responses_config:
                        is_final_chunk_for_attempt = finish_reason is not None
                        if (
                            not final_text_for_this_attempt
                            and not is_final_chunk_for_attempt
                        ):
                            continue

                        current_msg_idx = (
                            (len(final_text_for_this_attempt) - 1) // split_limit_config
                            if final_text_for_this_attempt
                            else 0
                        )
                        start_next_msg = current_msg_idx >= len(response_msgs)

                        ready_to_edit = (edit_task is None or edit_task.done()) and (
                            dt.now().timestamp() - client.last_task_time
                            >= client.config.get(EDIT_DELAY_SECONDS_CONFIG_KEY, 1.0)
                        )

                        if (
                            start_next_msg
                            or ready_to_edit
                            or is_final_chunk_for_attempt
                        ):
                            if edit_task is not None and not edit_task.done():
                                try:
                                    await edit_task
                                except asyncio.CancelledError:
                                    logging.warning("Previous edit task cancelled.")
                                except Exception as e:
                                    logging.error(
                                        f"Error waiting for previous edit task: {e}"
                                    )
                                edit_task = None

                            if start_next_msg and response_msgs:
                                prev_msg_idx = current_msg_idx - 1
                                if (
                                    0 <= prev_msg_idx < len(response_msgs)
                                    and response_msgs[prev_msg_idx].embeds
                                ):
                                    prev_text = final_text_for_this_attempt[
                                        prev_msg_idx
                                        * split_limit_config : current_msg_idx
                                        * split_limit_config
                                    ]
                                    prev_text = prev_text[:MAX_EMBED_DESCRIPTION_LENGTH]
                                    temp_prev_embed = discord.Embed.from_dict(
                                        response_msgs[prev_msg_idx].embeds[0].to_dict()
                                    )
                                    stripped_prev_text = (
                                        prev_text.strip() if prev_text else ""
                                    )
                                    temp_prev_embed.description = (
                                        stripped_prev_text or "..."
                                    )
                                    temp_prev_embed.color = EMBED_COLOR_COMPLETE
                                    try:
                                        await response_msgs[prev_msg_idx].edit(
                                            embed=temp_prev_embed, view=None
                                        )
                                    except discord.HTTPException as e:
                                        logging.error(
                                            f"Failed to finalize prev msg {prev_msg_idx}: {e}"
                                        )

                            current_display_text = final_text_for_this_attempt[
                                current_msg_idx * split_limit_config : (
                                    current_msg_idx + 1
                                )
                                * split_limit_config
                            ]
                            current_display_text = current_display_text[
                                :MAX_EMBED_DESCRIPTION_LENGTH
                            ]

                            view_to_attach = None
                            is_successful_finish_attempt = finish_reason and (
                                (
                                    finish_reason.lower() in ("stop", "end_turn")
                                    or (
                                        current_provider == "google"
                                        and finish_reason
                                        == str(
                                            google_types.FinishReason.FINISH_REASON_UNSPECIFIED
                                        )
                                    )
                                )
                                or finish_reason.lower()
                                in (
                                    "content_filter",
                                    "length",
                                    "max_tokens",
                                )
                            )
                            is_blocked_attempt = (
                                finish_reason
                                and finish_reason.lower()
                                in ("safety", "recitation", "other")
                            )

                            current_segment_embed = (
                                discord.Embed()
                            )  # Start with a fresh embed for the segment
                            # Footer from base_embed is not copied here anymore, it's constructed later if needed.
                            for (
                                field
                            ) in base_embed.fields:  # Copy warnings from base_embed
                                current_segment_embed.add_field(
                                    name=field.name,
                                    value=field.value,
                                    inline=field.inline,
                                )

                            # Ensure display text for description is never just whitespace
                            stripped_current_display_text = (
                                current_display_text.strip()
                                if current_display_text
                                else ""
                            )
                            effective_display_text_for_segment = (
                                stripped_current_display_text or "..."
                            )

                            current_segment_embed.description = (
                                effective_display_text_for_segment
                                if is_final_chunk_for_attempt
                                else (
                                    effective_display_text_for_segment
                                    + STREAMING_INDICATOR
                                )
                            )

                            if is_final_chunk_for_attempt and is_blocked_attempt:
                                # Use the already stripped and defaulted text if available
                                base_blocked_text = stripped_current_display_text  # This was from current_display_text

                                block_message_addon = ""
                                if finish_reason.lower() == "safety":
                                    block_message_addon = (
                                        "⚠️ Response blocked by safety filters."
                                    )
                                elif finish_reason.lower() == "recitation":
                                    block_message_addon = (
                                        "⚠️ Response stopped due to recitation."
                                    )
                                else:
                                    block_message_addon = (
                                        f"⚠️ Response blocked (Reason: {finish_reason})."
                                    )

                                if base_blocked_text:
                                    current_segment_embed.description = (
                                        f"{base_blocked_text}\n\n{block_message_addon}"
                                    )
                                else:
                                    current_segment_embed.description = (
                                        block_message_addon
                                    )

                                current_segment_embed.description = (
                                    current_segment_embed.description[
                                        :MAX_EMBED_DESCRIPTION_LENGTH
                                    ]
                                )
                                current_segment_embed.color = EMBED_COLOR_ERROR
                            else:
                                current_segment_embed.color = (
                                    EMBED_COLOR_COMPLETE
                                    if is_final_chunk_for_attempt
                                    and is_successful_finish_attempt
                                    else EMBED_COLOR_INCOMPLETE
                                )
                                if (
                                    is_final_chunk_for_attempt
                                    and is_successful_finish_attempt
                                ):
                                    # Add footer only to the final, successful message segment
                                    footer_text_final = f"Model: {current_model_name}"
                                    if attempt_num == 1:  # This is for retry logic
                                        footer_text_final += f" (Retried from {original_model_name_param})"

                                    internet_info_parts = []

                                    # --- Begin Gemini-specific grounding logic for footer ---
                                    is_gemini = (
                                        current_provider == "google"
                                        and "gemini" in current_model_name.lower()
                                    )
                                    if is_gemini:
                                        # Determine if grounding was actually used
                                        has_gemini_grounding = (
                                            grounding_metadata_for_this_attempt
                                            and (
                                                getattr(
                                                    grounding_metadata_for_this_attempt,
                                                    "web_search_queries",
                                                    None,
                                                )
                                                or getattr(
                                                    grounding_metadata_for_this_attempt,
                                                    "grounding_chunks",
                                                    None,
                                                )
                                                or getattr(
                                                    grounding_metadata_for_this_attempt,
                                                    "search_entry_point",
                                                    None,
                                                )
                                            )
                                        )
                                        if has_gemini_grounding:
                                            internet_info_parts.append("Internet used")
                                            # Optionally, count search results if available
                                            if (
                                                hasattr(
                                                    grounding_metadata_for_this_attempt,
                                                    "grounding_chunks",
                                                )
                                                and getattr(
                                                    grounding_metadata_for_this_attempt,
                                                    "grounding_chunks",
                                                    None,
                                                )
                                                is not None
                                            ):
                                                num_results = len(
                                                    getattr(
                                                        grounding_metadata_for_this_attempt,
                                                        "grounding_chunks",
                                                        [],
                                                    )
                                                )
                                                internet_info_parts.append(
                                                    f"{num_results} search result{'s' if num_results != 1 else ''} processed"
                                                )
                                        else:
                                            internet_info_parts.append(
                                                "Internet not used"
                                            )
                                    else:
                                        # Non-Gemini: keep existing logic
                                        if custom_search_queries_generated:
                                            internet_info_parts.append("Internet used")
                                            internet_info_parts.append(
                                                f"{successful_api_results_count} search result{'s' if successful_api_results_count != 1 else ''} processed"
                                            )
                                        else:
                                            internet_info_parts.append(
                                                "Internet not used"
                                            )
                                    # --- End Gemini-specific grounding logic for footer ---

                                    if internet_info_parts:
                                        footer_text_final += " | " + ", ".join(
                                            internet_info_parts
                                        )

                                    current_segment_embed.set_footer(
                                        text=footer_text_final
                                    )

                                    has_sources = (
                                        grounding_metadata_for_this_attempt
                                        and (
                                            getattr(
                                                grounding_metadata_for_this_attempt,
                                                "web_search_queries",
                                                None,
                                            )
                                            or getattr(
                                                grounding_metadata_for_this_attempt,
                                                "grounding_chunks",
                                                None,
                                            )
                                            or getattr(
                                                grounding_metadata_for_this_attempt,
                                                "search_entry_point",
                                                None,
                                            )
                                        )
                                    )
                                    has_text_content = bool(final_text_for_this_attempt)
                                    if has_sources or has_text_content:
                                        # Determine whether the response used the internet.
                                        internet_used_flag = (
                                            has_sources
                                            if current_provider == "google"  # Gemini grounding implies web usage
                                            else custom_search_queries_generated
                                        )

                                        view_to_attach = ResponseActionView(
                                            grounding_metadata=grounding_metadata_for_this_attempt,
                                            full_response_text=final_text_for_this_attempt,
                                            model_name=current_model_name,
                                            app_config=client.config,
                                            original_user_message=new_msg,
                                            internet_used=internet_used_flag,
                                        )

                                if "view_to_attach" in locals():
                                    if (
                                        not view_to_attach
                                        or len(view_to_attach.children) == 0
                                    ):
                                        view_to_attach = None

                            target_msg_for_node_update = None
                            if start_next_msg:
                                if (
                                    not response_msgs and processing_msg
                                ):  # First message, edit processing_msg
                                    await processing_msg.edit(
                                        content=None,
                                        embed=current_segment_embed,
                                        view=view_to_attach,
                                    )
                                    response_msg = processing_msg
                                    target_msg_for_node_update = response_msg
                                    if view_to_attach:
                                        view_to_attach.message = response_msg
                                else:  # Subsequent messages, or no processing_msg
                                    reply_target = (
                                        new_msg
                                        if not response_msgs
                                        else response_msgs[-1]
                                    )
                                    response_msg = await reply_target.reply(
                                        embed=current_segment_embed,
                                        view=view_to_attach,
                                        mention_author=False,
                                    )
                                    target_msg_for_node_update = response_msg
                                    if view_to_attach:  # Set message for the view
                                        view_to_attach.message = response_msg
                                response_msgs.append(response_msg)
                                if (
                                    target_msg_for_node_update
                                    and target_msg_for_node_update.id
                                    not in client.msg_nodes
                                ):
                                    client.msg_nodes[target_msg_for_node_update.id] = (
                                        models.MsgNode(parent_msg=new_msg)
                                    )
                                    # Lock acquisition will be handled by a context manager if needed
                                    # around operations that require exclusive access to the node.
                                    # For now, direct acquire/release is removed.
                                if view_to_attach:
                                    view_to_attach.message = response_msg
                            elif response_msgs and current_msg_idx < len(
                                response_msgs
                            ):  # Editing existing message in stream
                                target_msg = response_msgs[current_msg_idx]
                                if target_msg:
                                    edit_task = asyncio.create_task(
                                        target_msg.edit(
                                            embed=current_segment_embed,
                                            view=view_to_attach,
                                        )
                                    )
                                    if view_to_attach:
                                        view_to_attach.message = target_msg
                                    target_msg_for_node_update = target_msg
                            elif (
                                not response_msgs and is_final_chunk_for_attempt
                            ):  # Short final response, no prior stream msgs
                                if processing_msg:
                                    await processing_msg.edit(
                                        content=None,
                                        embed=current_segment_embed,
                                        view=view_to_attach,
                                    )
                                    response_msg = processing_msg
                                    target_msg_for_node_update = response_msg
                                    if view_to_attach:  # Set message for the view
                                        view_to_attach.message = response_msg
                                else:
                                    response_msg = await new_msg.reply(
                                        embed=current_segment_embed,
                                        view=view_to_attach,
                                        mention_author=False,
                                    )
                                    target_msg_for_node_update = response_msg
                                    if view_to_attach:  # Set message for the view
                                        view_to_attach.message = response_msg
                                response_msgs.append(response_msg)
                                if (
                                    target_msg_for_node_update
                                    and target_msg_for_node_update.id
                                    not in client.msg_nodes
                                ):
                                    client.msg_nodes[target_msg_for_node_update.id] = (
                                        models.MsgNode(parent_msg=new_msg)
                                    )
                                    # Lock acquisition will be handled by a context manager if needed
                                    # around operations that require exclusive access to the node.
                                    # For now, direct acquire/release is removed.
                                if view_to_attach:
                                    view_to_attach.message = response_msg

                            if (
                                is_final_chunk_for_attempt
                                and target_msg_for_node_update
                                and target_msg_for_node_update.id in client.msg_nodes
                            ):
                                client.msg_nodes[
                                    target_msg_for_node_update.id
                                ].full_response_text = final_text_for_this_attempt

                            client.last_task_time = dt.now().timestamp()

                    if finish_reason:
                        current_attempt_llm_successful = (
                            finish_reason.lower()
                            in (
                                "stop",
                                "end_turn",
                            )
                            or (
                                current_provider == "google"
                                and finish_reason
                                == str(
                                    google_types.FinishReason.FINISH_REASON_UNSPECIFIED
                                )
                            )
                        ) or finish_reason.lower() in (
                            "content_filter",
                            "length",
                            "max_tokens",
                        )
                        break

                if (
                    should_retry_with_gemini_signal
                    or should_retry_due_to_unprocessable_entity
                ):
                    final_text_for_this_attempt = ""  # Discard text from this attempt
                    # The outer loop (for attempt_num in range(2)) will continue to the retry logic
                    continue

                # Handle image generation model - send accumulated data at the end
                if is_image_generation_model and current_attempt_llm_successful:
                    try:
                        if accumulated_image_data and accumulated_image_mime_type:
                            # Create a file from the accumulated image data
                            image_file = io.BytesIO(accumulated_image_data)

                            # Determine file extension from mime type
                            file_extension = "png"  # default
                            if (
                                "jpeg" in accumulated_image_mime_type
                                or "jpg" in accumulated_image_mime_type
                            ):
                                file_extension = "jpg"
                            elif "webp" in accumulated_image_mime_type:
                                file_extension = "webp"
                            elif "gif" in accumulated_image_mime_type:
                                file_extension = "gif"

                            filename = f"generated_image.{file_extension}"
                            discord_file = discord.File(
                                fp=image_file, filename=filename
                            )

                            # Send the image as a reply with any text content
                            content_to_send = (
                                final_text_for_this_attempt.strip()
                                if final_text_for_this_attempt.strip()
                                else None
                            )

                            if processing_msg and not response_msgs:
                                # Edit the processing message to show completion and send image
                                await processing_msg.edit(
                                    content="✅ Image generated successfully!",
                                    embed=None,
                                    view=None,
                                )
                                response_msg = await processing_msg.reply(
                                    content=content_to_send,
                                    file=discord_file,
                                    mention_author=False,
                                )
                            else:
                                # Send as a new reply
                                reply_target = (
                                    response_msgs[-1] if response_msgs else new_msg
                                )
                                response_msg = await reply_target.reply(
                                    content=content_to_send,
                                    file=discord_file,
                                    mention_author=False,
                                )

                            response_msgs.append(response_msg)

                            # Update msg_nodes cache
                            if response_msg.id not in client.msg_nodes:
                                client.msg_nodes[response_msg.id] = models.MsgNode(
                                    parent_msg=new_msg
                                )
                                client.msg_nodes[
                                    response_msg.id
                                ].full_response_text = final_text_for_this_attempt

                    except Exception as e:
                        logging.error(f"Error sending generated image: {e}")
                        # Fall back to text-only response
                        if processing_msg:
                            await processing_msg.edit(
                                content=f"Generated image but failed to send: {str(e)}",
                                embed=None,
                                view=None,
                            )

                llm_call_successful_final = current_attempt_llm_successful
                final_text_to_return = final_text_for_this_attempt

                if use_plain_responses_config and llm_call_successful_final:
                    final_messages_content = [
                        final_text_to_return[i : i + split_limit_config]
                        for i in range(0, len(final_text_to_return), split_limit_config)
                    ]
                    if not final_messages_content:
                        final_messages_content.append("...")  # type: ignore

                    temp_response_msgs_plain = []
                    start_index_plain = 0
                    if processing_msg and (
                        not response_msgs
                        or (response_msgs and processing_msg.id == response_msgs[0].id)
                    ):
                        await processing_msg.edit(
                            content=final_messages_content[0] or "...",
                            embed=None,
                            view=None,
                        )
                        temp_response_msgs_plain.append(processing_msg)
                        if processing_msg.id in client.msg_nodes:
                            client.msg_nodes[
                                processing_msg.id
                            ].full_response_text = final_text_to_return
                        else:  # Should not happen if processing_msg was handled correctly
                            client.msg_nodes[processing_msg.id] = models.MsgNode(
                                parent_msg=new_msg,
                                full_response_text=final_text_to_return,
                            )
                        # await client.msg_nodes[processing_msg.id].lock.acquire() # Temporarily commented out for refactoring
                    start_index_plain = 1

                    reply_target_plain = (
                        temp_response_msgs_plain[-1]
                        if temp_response_msgs_plain
                        else new_msg
                    )

                    for i in range(start_index_plain, len(final_messages_content)):
                        content_chunk_plain = final_messages_content[i]
                        response_msg_plain = await reply_target_plain.reply(
                            content=content_chunk_plain or "...",
                            suppress_embeds=True,
                            view=None,
                            mention_author=False,
                        )
                        temp_response_msgs_plain.append(response_msg_plain)
                        # Ensure new node for new reply
                        client.msg_nodes[response_msg_plain.id] = models.MsgNode(
                            parent_msg=new_msg
                        )
                        node_plain = client.msg_nodes[response_msg_plain.id]
                        # Lock acquisition will be handled by a context manager if needed
                        if i == len(final_messages_content) - 1:
                            node_plain.full_response_text = final_text_to_return
                        reply_target_plain = response_msg_plain

                    response_msgs = temp_response_msgs_plain
                break  # type: ignore
        except AllKeysFailedError as e:
            logging.error(
                f"LLM generation failed for message {new_msg.id} (Attempt {attempt_num + 1}, Model {current_model_name}): {e}"
            )
            if attempt_num == 0:  # Original attempt failed due to all keys
                logging.info(
                    f"AllKeysFailedError on original attempt for {current_provider}/{current_model_name}. Setting flag to retry with fallback."
                )
                should_retry_due_to_all_keys_failed = True
                # Don't break yet, let the loop continue to the retry logic
                continue
            else:  # AllKeysFailedError on the retry attempt, this is a final failure
                error_text = f"⚠️ All API keys for fallback provider `{e.service_name}` also failed."
                last_err_str = str(e.errors[-1]) if e.errors else "Unknown reason."
                error_text += f"\nLast error: `{last_err_str[:100]}{'...' if len(last_err_str) > 100 else ''}`"
                await _handle_llm_exception(
                    new_msg,
                    processing_msg,
                    response_msgs,
                    error_text,
                    use_plain_responses_config,
                    client.config.get("use_plain_responses", False),
                )
                llm_call_successful_final = False
                break

        except Exception as outer_e:
            logging.exception(
                f"Unhandled error during message processing for {new_msg.id} (Attempt {attempt_num + 1}, Model {current_model_name})."
            )
            error_text = f"⚠️ An unexpected error occurred: {type(outer_e).__name__}"
            await _handle_llm_exception(
                new_msg,
                processing_msg,
                response_msgs,
                error_text,
                use_plain_responses_config,
                client.config.get("use_plain_responses", False),
            )
            llm_call_successful_final = False  # type: ignore
            break  # type: ignore

    return llm_call_successful_final, final_text_to_return, response_msgs


async def _handle_stream_error(
    new_msg: discord.Message,
    processing_msg: Optional[discord.Message],
    response_msgs: List[discord.Message],
    error_message: str,
    use_plain_responses_config: bool,
):
    """Handles errors that occur mid-stream."""
    if not use_plain_responses_config and response_msgs and response_msgs[-1].embeds:
        try:
            embed = response_msgs[-1].embeds[0]
            embed.description = (
                (embed.description or "").replace(STREAMING_INDICATOR, "").strip()
            )
            embed.description += f"\n\n⚠️ Error: {error_message}"
            embed.description = embed.description[:MAX_EMBED_DESCRIPTION_LENGTH]
            embed.color = EMBED_COLOR_ERROR
            await response_msgs[-1].edit(embed=embed, view=None)
        except Exception as edit_err:
            logging.error(f"Failed to edit message to show stream error: {edit_err}")
            # Fallback reply
            target = (
                processing_msg
                if processing_msg and processing_msg.id == response_msgs[-1].id
                else new_msg
            )
            await target.reply(
                f"⚠️ Error during response generation: {error_message}",
                mention_author=False,
            )
    else:
        error_text_plain = f"⚠️ Error during response generation: {error_message}"
        if processing_msg:
            await processing_msg.edit(content=error_text_plain, embed=None, view=None)
        else:
            await new_msg.reply(error_text_plain, mention_author=False)


async def _handle_llm_exception(
    new_msg: discord.Message,
    processing_msg: Optional[discord.Message],
    response_msgs: List[discord.Message],
    error_text: str,
    use_plain_responses_stream: bool,  # Whether the stream was plain
    use_plain_initial_status: bool,  # Whether the initial status message was plain
):
    """Handles exceptions from the LLM call (e.g., AllKeysFailedError)."""
    if processing_msg:
        if use_plain_initial_status:
            await processing_msg.edit(content=error_text, embed=None, view=None)
        else:  # Initial status was an embed
            if (
                not use_plain_responses_stream
                and response_msgs
                and response_msgs[-1].embeds
            ):  # Stream was also embed
                target_edit_msg = response_msgs[-1]
                embed_to_edit = discord.Embed.from_dict(
                    target_edit_msg.embeds[0].to_dict()
                )
                current_desc = embed_to_edit.description or ""
                embed_to_edit.description = current_desc.replace(
                    STREAMING_INDICATOR, ""
                ).strip()
                embed_to_edit.description += f"\n\n{error_text}"
                embed_to_edit.description = embed_to_edit.description[
                    :MAX_EMBED_DESCRIPTION_LENGTH
                ]
                embed_to_edit.color = EMBED_COLOR_ERROR
                await target_edit_msg.edit(embed=embed_to_edit, view=None)
            else:  # Stream was plain or no stream messages, edit initial embed status
                await processing_msg.edit(
                    content=None,
                    embed=discord.Embed(
                        description=error_text, color=EMBED_COLOR_ERROR
                    ),
                    view=None,
                )
    elif (
        not use_plain_responses_stream and response_msgs and response_msgs[-1].embeds
    ):  # No processing_msg, but stream had embeds
        target_edit_msg = response_msgs[-1]
        embed_to_edit = discord.Embed.from_dict(target_edit_msg.embeds[0].to_dict())
        current_desc = embed_to_edit.description or ""
        embed_to_edit.description = current_desc.replace(
            STREAMING_INDICATOR, ""
        ).strip()
        embed_to_edit.description += f"\n\n{error_text}"
        embed_to_edit.description = embed_to_edit.description[
            :MAX_EMBED_DESCRIPTION_LENGTH
        ]
        embed_to_edit.color = EMBED_COLOR_ERROR
        await target_edit_msg.edit(embed=embed_to_edit, view=None)
    else:  # Fallback: No processing_msg and no stream to edit, send new reply
        await new_msg.reply(error_text, mention_author=False)


async def resend_imgur_urls(
    new_msg: discord.Message,  # Original user message
    response_msgs: List[discord.Message],  # List of bot's response messages
    final_text: str,
):
    """Checks for Imgur URLs in the final_text and resends them if found."""
    if not final_text:
        return

    lines = final_text.strip().split("\n")
    imgur_urls_to_resend = []
    found_header = False
    for line in lines:
        stripped_line = line.strip()
        if stripped_line == IMGUR_HEADER:
            found_header = True
            continue
        if found_header and stripped_line.startswith(IMGUR_URL_PREFIX):
            if IMGUR_URL_PATTERN.match(stripped_line):
                imgur_urls_to_resend.append(stripped_line)
            else:
                break
        elif found_header and stripped_line:
            break

    if imgur_urls_to_resend:
        logging.info(
            f"Detected Imgur URLs in response to message {new_msg.id}. Resending."
        )
        formatted_urls = imgur_urls_to_resend
        max_chars = MAX_PLAIN_TEXT_LENGTH
        messages_to_send_content = []
        current_message_content = ""

        for url_str in formatted_urls:
            needed_len = len(url_str) + (2 if current_message_content else 0)
            if len(current_message_content) + needed_len > max_chars:
                if current_message_content:
                    messages_to_send_content.append(current_message_content)
                current_message_content = (
                    url_str[:max_chars] if len(url_str) > max_chars else url_str
                )
                if len(url_str) > max_chars:
                    logging.warning(f"Single Imgur URL too long: {url_str}")
            else:
                current_message_content += (
                    "\n\n" if current_message_content else ""
                ) + url_str

        if current_message_content:
            messages_to_send_content.append(current_message_content)

        reply_target = response_msgs[-1] if response_msgs else new_msg
        last_sent_msg = reply_target
        for i, msg_content in enumerate(messages_to_send_content):
            try:
                target_to_reply_to = last_sent_msg
                if isinstance(target_to_reply_to, discord.Message):
                    sent_msg = await target_to_reply_to.reply(
                        content=msg_content, mention_author=False
                    )
                    last_sent_msg = sent_msg
                else:
                    logging.warning(
                        f"Invalid reply target type: {type(target_to_reply_to)}. Replying to original."
                    )
                    sent_msg = await new_msg.reply(
                        content=msg_content, mention_author=False
                    )
                    last_sent_msg = sent_msg
                await asyncio.sleep(0.1)
            except discord.HTTPException as send_err:
                logging.error(f"Failed to resend Imgur URL chunk {i + 1}: {send_err}")
                try:
                    await new_msg.reply(
                        f"(Error sending previous chunk)\n{msg_content}",
                        mention_author=False,
                    )
                except discord.HTTPException as fallback_err:
                    logging.error(f"Fallback Imgur resend failed: {fallback_err}")
            except Exception as e:
                logging.error(
                    f"Unexpected error resending Imgur URL chunk {i + 1}: {e}"
                )
