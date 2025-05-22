import logging
from typing import Optional, List, Dict, Any
import asyncio
import base64
import httpx
import re
import tiktoken  # Added tiktoken

import discord

from .constants import AT_AI_PATTERN, STREAMING_INDICATOR
from . import models
# Assuming google.genai.types will be passed as google_types_module
# Assuming extract_text_from_pdf_bytes will be passed as a function


# --- ADDED: Tiktoken helper ---
def _get_tokenizer_for_model(model_name: str):
    """Gets the tiktoken encoder. Always uses 'o200k_base'."""
    # User request: Always use o200k_base
    logging.debug(
        f"Requested tokenizer for model '{model_name}', returning 'o200k_base'."
    )
    return tiktoken.get_encoding("o200k_base")


def _truncate_text_by_tokens(text: str, tokenizer, max_tokens: int) -> tuple[str, int]:
    """Truncates text to a maximum number of tokens and returns the truncated text and token count."""
    if not text:
        return "", 0
    tokens = tokenizer.encode(text)
    actual_token_count = len(tokens)
    if actual_token_count > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        try:
            truncated_text = tokenizer.decode(truncated_tokens)
        except UnicodeDecodeError:
            try:
                truncated_text = tokenizer.decode(truncated_tokens[:-1])
            except Exception as e:
                logging.error(
                    f"Failed to decode truncated tokens even after removing one: {e}. Returning empty string."
                )
                return "", actual_token_count  # Return original token count
        if (
            truncated_text and max_tokens > 0
        ):  # Ensure ellipsis isn't added to an empty or zero-token result
            truncated_text += "..."
        return truncated_text, actual_token_count  # Return original token count
    return text, actual_token_count


async def find_parent_message(
    message: discord.Message,
    current_bot_user: discord.User,  # Pass the bot's user object
    is_dm: bool,
) -> Optional[discord.Message]:
    """Determines the logical parent message for conversation history."""
    try:
        # Check if the current message explicitly triggers the bot
        mentions_bot_in_current = current_bot_user.mentioned_in(message)
        contains_at_ai_in_current = AT_AI_PATTERN.search(message.content) is not None
        is_explicit_trigger = mentions_bot_in_current or contains_at_ai_in_current

        # 1. Explicit Reply always takes precedence
        if message.reference and message.reference.message_id:
            try:
                ref_msg = message.reference.cached_message
                if not ref_msg:
                    ref_msg = await message.channel.fetch_message(
                        message.reference.message_id
                    )
                if ref_msg and ref_msg.type in (
                    discord.MessageType.default,
                    discord.MessageType.reply,
                ):
                    return ref_msg
                else:
                    logging.debug(
                        f"Referenced message {message.reference.message_id} is not usable type {getattr(ref_msg, 'type', 'N/A')}"
                    )
            except (discord.NotFound, discord.HTTPException) as e:
                logging.warning(
                    f"Could not fetch referenced message {message.reference.message_id}: {e}"
                )

        # 2. Thread Start: If it's the first user message in a thread (not a reply)
        if (
            isinstance(message.channel, discord.Thread)
            and message.channel.parent
            and not message.reference
        ):
            try:
                starter_msg = message.channel.starter_message
                if not starter_msg:
                    starter_msg = await message.channel.parent.fetch_message(
                        message.channel.id
                    )
                if starter_msg and starter_msg.type in (
                    discord.MessageType.default,
                    discord.MessageType.reply,
                ):
                    return starter_msg
                else:
                    logging.debug(
                        f"Thread starter message {message.channel.id} is not usable type {getattr(starter_msg, 'type', 'N/A')}"
                    )
            except (discord.NotFound, discord.HTTPException, AttributeError) as e:
                logging.warning(
                    f"Could not fetch thread starter message for thread {message.channel.id}: {e}"
                )

        # 3. Automatic Chaining (Only if NOT explicitly triggered)
        if not is_explicit_trigger:
            prev_msg_in_channel = None
            try:
                async for m in message.channel.history(before=message, limit=1):
                    prev_msg_in_channel = m
                    break
            except (discord.Forbidden, discord.HTTPException) as e:
                logging.warning(
                    f"Could not fetch history in channel {message.channel.id}: {e}"
                )

            if prev_msg_in_channel and prev_msg_in_channel.type in (
                discord.MessageType.default,
                discord.MessageType.reply,
            ):
                if (is_dm and prev_msg_in_channel.author == current_bot_user) or (
                    not is_dm and prev_msg_in_channel.author == message.author
                ):
                    return prev_msg_in_channel

        # 4. No logical parent found
        return None

    except Exception:
        logging.exception(f"Error determining parent message for {message.id}")
        return None


async def build_message_history(
    new_msg: discord.Message,
    initial_cleaned_content: str,
    combined_context: str,
    max_messages: int,
    max_tokens_for_text: int,  # This is the TOTAL history token limit from config's "max_text"
    max_files_per_message: int,
    accept_files: bool,
    use_google_lens: bool,
    is_target_provider_gemini: bool,
    target_provider_name: str,
    target_model_name: str,
    user_warnings: set,
    current_message_url_fetch_results: Optional[List["models.UrlFetchResult"]],
    msg_nodes_cache: dict,
    bot_user_obj: discord.User,
    httpx_async_client: "httpx.AsyncClient",
    models_module: Any,
    google_types_module: Any,
    extract_text_from_pdf_bytes_func: callable,
    at_ai_pattern_re: Any,
    providers_supporting_usernames_const: tuple,
    system_prompt_text_for_budgeting: Optional[str] = None,
) -> List[Dict[str, Any]]:
    raw_history_entries_reversed = []  # Oldest last initially, will be reversed

    # 1. Populate MsgNodes and collect raw message data (reversed chronological)
    #    Loop backwards from new_msg to gather messages up to max_messages.
    #    MsgNode.text will store full, untruncated text.
    #    MsgNode.external_content is set only for the new_msg's node.

    _curr_msg_for_loop = new_msg
    is_dm_current_msg_channel = isinstance(new_msg.channel, discord.DMChannel)

    processed_msg_count = 0
    while _curr_msg_for_loop is not None and processed_msg_count < max_messages:
        current_msg_id = _curr_msg_for_loop.id
        if current_msg_id not in msg_nodes_cache:
            logging.debug(
                f"Node for message {current_msg_id} not in cache. Fetching message."
            )
            try:
                if current_msg_id != new_msg.id:  # Don't re-fetch the initial new_msg
                    fetched_msg = await new_msg.channel.fetch_message(current_msg_id)
                    if not fetched_msg:
                        logging.warning(
                            f"Failed to fetch message {current_msg_id} for history building."
                        )
                        user_warnings.add(
                            f"⚠️ Couldn't fetch full history (message {current_msg_id} missing)."
                        )
                        break
                    _curr_msg_for_loop = (
                        fetched_msg  # Update _curr_msg_for_loop with fetched message
                    )
            except (discord.NotFound, discord.HTTPException) as fetch_err:
                logging.warning(
                    f"Failed to fetch message {current_msg_id} for history building: {fetch_err}"
                )
                user_warnings.add(
                    f"⚠️ Couldn't fetch full history (message {current_msg_id} missing)."
                )
                break  # Stop if a message in the chain can't be fetched
            msg_nodes_cache[current_msg_id] = models_module.MsgNode()

        curr_node = msg_nodes_cache[current_msg_id]

        async with curr_node.lock:
            is_current_message_node = current_msg_id == new_msg.id
            current_role = (
                "model" if _curr_msg_for_loop.author == bot_user_obj else "user"
            )

            # Populate node only if it's new or the current message being processed
            should_populate_node = (curr_node.text is None) or is_current_message_node

            if should_populate_node:
                curr_node.has_bad_attachments = False
                # Reset api_file_parts only for the current message node to avoid reprocessing old ones if they were already processed
                if is_current_message_node:
                    curr_node.api_file_parts = []
                    # Set external_content only for the current message node
                    curr_node.external_content = (
                        combined_context if combined_context else None
                    )
                    logging.debug(
                        f"Set external_content for node {current_msg_id} to {'present' if combined_context else 'None'}."
                    )

                content_to_store = ""
                if current_role == "model":
                    if (
                        curr_node.full_response_text
                    ):  # Prefer full response if available
                        content_to_store = curr_node.full_response_text
                    else:  # Fallback to embed or raw content
                        if (
                            _curr_msg_for_loop.embeds
                            and _curr_msg_for_loop.embeds[0].description
                        ):
                            content_to_store = (
                                _curr_msg_for_loop.embeds[0]
                                .description.replace(STREAMING_INDICATOR, "")
                                .strip()
                            )
                        else:
                            content_to_store = _curr_msg_for_loop.content
                else:  # User message
                    content_to_store = (
                        initial_cleaned_content
                        if is_current_message_node
                        else _curr_msg_for_loop.content
                    )
                    is_dm_iter_msg_channel = isinstance(
                        _curr_msg_for_loop.channel, discord.DMChannel
                    )
                    if not is_dm_iter_msg_channel and bot_user_obj.mentioned_in(
                        _curr_msg_for_loop
                    ):
                        content_to_store = content_to_store.replace(
                            bot_user_obj.mention, ""
                        ).strip()
                    if not is_current_message_node:  # Don't re-sub @ai for the current message as it's already cleaned
                        content_to_store = at_ai_pattern_re.sub(" ", content_to_store)
                    content_to_store = re.sub(r"\s{2,}", " ", content_to_store).strip()

                current_attachments = _curr_msg_for_loop.attachments
                MAX_ATTACHMENTS_TO_DOWNLOAD_IN_HISTORY = 5  # Limit for history messages
                attachments_to_fetch = []
                unfetched_unsupported_types = False

                for att_idx, att in enumerate(current_attachments):
                    if (
                        len(attachments_to_fetch)
                        >= MAX_ATTACHMENTS_TO_DOWNLOAD_IN_HISTORY
                    ):
                        curr_node.has_bad_attachments = True
                        break
                    if att.content_type:
                        is_relevant_for_download = False
                        if att.content_type.startswith("text/"):
                            is_relevant_for_download = True
                        elif att.content_type.startswith("image/"):
                            if accept_files or (
                                is_current_message_node and use_google_lens
                            ):
                                is_relevant_for_download = True
                        elif att.content_type == "application/pdf":
                            if (
                                is_target_provider_gemini and accept_files
                            ) or not is_target_provider_gemini:
                                is_relevant_for_download = True
                        if is_relevant_for_download:
                            attachments_to_fetch.append(att)
                        else:
                            unfetched_unsupported_types = True
                if unfetched_unsupported_types:
                    curr_node.has_bad_attachments = True

                attachment_responses = await asyncio.gather(
                    *[
                        httpx_async_client.get(att.url, timeout=15.0)
                        for att in attachments_to_fetch
                    ],
                    return_exceptions=True,
                )

                text_parts = [content_to_store] if content_to_store else []
                if current_role == "user":  # Add embed text for user messages
                    text_parts.extend(
                        filter(
                            None, (embed.title for embed in _curr_msg_for_loop.embeds)
                        )
                    )
                    text_parts.extend(
                        filter(
                            None,
                            (embed.description for embed in _curr_msg_for_loop.embeds),
                        )
                    )

                for att, resp in zip(attachments_to_fetch, attachment_responses):
                    if (
                        isinstance(resp, httpx.Response)
                        and resp.status_code == 200
                        and att.content_type.startswith("text/")
                    ):
                        try:
                            text_parts.append(resp.text)
                        except Exception as e:
                            logging.warning(
                                f"Failed to decode text attachment {att.filename}: {e}"
                            )
                            curr_node.has_bad_attachments = True
                    elif isinstance(resp, Exception) and att.content_type.startswith(
                        "text/"
                    ):
                        logging.warning(
                            f"Failed to fetch text attachment {att.filename}: {resp}"
                        )
                        curr_node.has_bad_attachments = True

                curr_node.text = "\n".join(
                    filter(None, text_parts)
                )  # Store full text, no per-node truncation here

                if (
                    current_role == "user" and not is_target_provider_gemini
                ):  # Append PDF text for non-Gemini user messages
                    pdf_texts_to_append = []
                    for att, resp in zip(attachments_to_fetch, attachment_responses):
                        if att.content_type == "application/pdf":
                            if (
                                isinstance(resp, httpx.Response)
                                and resp.status_code == 200
                            ):
                                try:
                                    extracted_pdf_text = (
                                        await extract_text_from_pdf_bytes_func(
                                            resp.content
                                        )
                                    )
                                    if extracted_pdf_text:
                                        pdf_texts_to_append.append(
                                            f"\n\n--- Content from PDF: {att.filename} ---\n{extracted_pdf_text}\n--- End of PDF: {att.filename} ---"
                                        )
                                    else:
                                        curr_node.has_bad_attachments = True
                                except Exception as pdf_e:
                                    logging.error(
                                        f"Error extracting PDF {att.filename}: {pdf_e}"
                                    )
                                    curr_node.has_bad_attachments = True
                            elif isinstance(resp, Exception):
                                curr_node.has_bad_attachments = True
                    if pdf_texts_to_append:
                        curr_node.text = (curr_node.text or "") + "".join(
                            pdf_texts_to_append
                        )

                # Populate api_file_parts (only if not already populated or if it's the current message)
                # This logic assumes api_file_parts are only truly needed for the current message or if re-evaluating history.
                # For simplicity in this refactor, we'll repopulate if should_populate_node is true.
                # A more optimized approach might check if curr_node.api_file_parts is already set from a previous build.

                temp_api_file_parts = []  # Build fresh for this scope
                files_processed_for_api_count = 0
                is_lens_trigger_message = is_current_message_node and use_google_lens
                should_process_files_for_api = (
                    current_role == "user" or is_lens_trigger_message
                ) and (accept_files or is_lens_trigger_message)

                if (
                    should_process_files_for_api and not is_lens_trigger_message
                ):  # Lens images are handled by external_content
                    for att, resp in zip(attachments_to_fetch, attachment_responses):
                        is_api_relevant_type = False
                        mime_type_for_api = att.content_type
                        file_bytes_for_api = None
                        if att.content_type.startswith("image/"):
                            is_api_relevant_type = True
                            if (
                                isinstance(resp, httpx.Response)
                                and resp.status_code == 200
                            ):
                                file_bytes_for_api = resp.content
                            else:
                                curr_node.has_bad_attachments = True
                                continue
                        elif (
                            att.content_type == "application/pdf"
                            and is_target_provider_gemini
                            and accept_files
                        ):
                            is_api_relevant_type = True
                            mime_type_for_api = "application/pdf"
                            if (
                                isinstance(resp, httpx.Response)
                                and resp.status_code == 200
                            ):
                                file_bytes_for_api = resp.content
                            else:
                                curr_node.has_bad_attachments = True
                                continue
                        if not is_api_relevant_type or file_bytes_for_api is None:
                            continue
                        if files_processed_for_api_count >= max_files_per_message:
                            curr_node.has_bad_attachments = True
                            break
                        try:
                            if is_target_provider_gemini:
                                temp_api_file_parts.append(
                                    google_types_module.Part.from_bytes(
                                        data=file_bytes_for_api,
                                        mime_type=mime_type_for_api,
                                    )
                                )
                            else:
                                # Offload base64 encoding
                                base64_encoded_image = await asyncio.to_thread(
                                    lambda b: base64.b64encode(b).decode("utf-8"),
                                    file_bytes_for_api,
                                )
                                temp_api_file_parts.append(
                                    dict(
                                        type="image_url",
                                        image_url=dict(
                                            url=f"data:{mime_type_for_api};base64,{base64_encoded_image}"
                                        ),
                                    )
                                )
                            files_processed_for_api_count += 1
                        except Exception as e:
                            curr_node.has_bad_attachments = True
                            logging.error(
                                f"Error preparing attachment {att.filename} for API: {e}"
                            )

                    if (
                        is_current_message_node and current_message_url_fetch_results
                    ):  # Add fetched image URLs for current message
                        for fetched_url_res in current_message_url_fetch_results:
                            if (
                                fetched_url_res.type == "image_url_content"
                                and isinstance(fetched_url_res.content, bytes)
                                and not fetched_url_res.error
                            ):
                                if (
                                    files_processed_for_api_count
                                    >= max_files_per_message
                                ):
                                    curr_node.has_bad_attachments = True
                                    user_warnings.add("⚠️ Max files reached.")
                                    break
                                img_bytes = fetched_url_res.content
                                url_lower = fetched_url_res.url.lower()
                                mime_type = "image/png"
                                if url_lower.endswith((".jpg", ".jpeg")):
                                    mime_type = "image/jpeg"
                                elif url_lower.endswith(".gif"):
                                    mime_type = "image/gif"
                                elif url_lower.endswith(".webp"):
                                    mime_type = "image/webp"
                                elif url_lower.endswith(".bmp"):
                                    mime_type = "image/bmp"
                                try:
                                    if is_target_provider_gemini:
                                        temp_api_file_parts.append(
                                            google_types_module.Part.from_bytes(
                                                data=img_bytes, mime_type=mime_type
                                            )
                                        )
                                    else:
                                        # Offload base64 encoding
                                        base64_encoded_image = await asyncio.to_thread(
                                            lambda b: base64.b64encode(b).decode(
                                                "utf-8"
                                            ),
                                            img_bytes,
                                        )
                                        temp_api_file_parts.append(
                                            dict(
                                                type="image_url",
                                                image_url=dict(
                                                    url=f"data:{mime_type};base64,{base64_encoded_image}"
                                                ),
                                            )
                                        )
                                    files_processed_for_api_count += 1
                                except Exception as e:
                                    curr_node.has_bad_attachments = True
                                    user_warnings.add(
                                        f"⚠️ Error processing image URL: {fetched_url_res.url[:50]}..."
                                    )
                                    logging.error(
                                        f"Error preparing image URL {fetched_url_res.url} for API: {e}"
                                    )
                curr_node.api_file_parts = (
                    temp_api_file_parts  # Assign the newly built parts
                )

                curr_node.role = current_role
                curr_node.user_id = (
                    _curr_msg_for_loop.author.id if curr_node.role == "user" else None
                )

                if curr_node.parent_msg is None and not curr_node.fetch_parent_failed:
                    parent = await find_parent_message(
                        _curr_msg_for_loop, bot_user_obj, is_dm_current_msg_channel
                    )
                    if (
                        parent is None
                        and _curr_msg_for_loop.reference
                        and _curr_msg_for_loop.reference.message_id
                    ):
                        curr_node.fetch_parent_failed = True
                    curr_node.parent_msg = parent

            # Add to raw_history_entries_reversed (oldest will be at the end)
            # Only include external_content for the current message being processed
            history_entry = {
                "id": current_msg_id,
                "role": curr_node.role,
                "text": curr_node.text,
                "files": curr_node.api_file_parts,
                "user_id": curr_node.user_id,
            }
            
            # Only add external_content for the current message node
            # This ensures it's used for the current request but not saved in history
            if is_current_message_node:
                history_entry["external_content"] = curr_node.external_content
            else:
                history_entry["external_content"] = None
                
            raw_history_entries_reversed.append(history_entry)
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Some attachments might not have been processed.")
            if curr_node.fetch_parent_failed:
                user_warnings.add("⚠️ Couldn't fetch full history")

            _curr_msg_for_loop = curr_node.parent_msg
        processed_msg_count += 1

    if processed_msg_count >= max_messages:
        user_warnings.add(f"⚠️ Only using last {max_messages} messages")

    chronological_entries_data = raw_history_entries_reversed[
        ::-1
    ]  # Reverse to make it chronological

    # 2. Tokenize and Prepare for Truncation
    tokenizer = _get_tokenizer_for_model(target_model_name)

    # Adjust max_tokens_for_text budget for the system prompt
    effective_max_tokens_for_messages = max_tokens_for_text
    if system_prompt_text_for_budgeting:
        system_prompt_tokens = len(tokenizer.encode(system_prompt_text_for_budgeting))
        effective_max_tokens_for_messages = max(
            0, max_tokens_for_text - system_prompt_tokens
        )
        logging.debug(
            f"Original max_tokens_for_text: {max_tokens_for_text}. System prompt tokens: {system_prompt_tokens}. Effective budget for messages: {effective_max_tokens_for_messages}"
        )

    tokenized_entries = []
    for entry_data in chronological_entries_data:
        # This text_for_token_count is ONLY for calculating this specific entry's token contribution.
        # It includes its own text and its own external_content.
        # Image tokens (if any, for OpenAI) would be added here if we were to count them against this entry's budget.
        # However, the prompt error was about total prompt size, so we focus on text for now.
        current_entry_text_content = (
            (
                entry_data["external_content"]
                + "\n\nUser's query:\n"
                + (entry_data["text"] or "")
            )
            if entry_data["external_content"]
            else (entry_data["text"] or "")
        )

        # Calculate token count for text and external content first
        text_only_token_count = len(tokenizer.encode(current_entry_text_content))
        image_token_cost_for_entry = 0

        # Add fixed token cost for images if files exist
        if entry_data["files"]:
            num_images_in_entry = 0
            for file_part_struct in entry_data["files"]:
                if (
                    isinstance(file_part_struct, dict)
                    and file_part_struct.get("type") == "image_url"
                ):
                    # Check if it's a data URL, implying an image processed for an API
                    img_url_dict = file_part_struct.get("image_url", {})
                    data_url_val = img_url_dict.get("url", "")
                    if (
                        data_url_val.startswith(
                            "data:image"
                        )  # Covers OpenAI and similar base64 images
                        and ";base64," in data_url_val
                    ) or (  # Also consider Gemini parts if they are image bytes
                        is_target_provider_gemini
                        and isinstance(file_part_struct, google_types_module.Part)
                        and hasattr(file_part_struct, "inline_data")
                        and file_part_struct.inline_data
                        and file_part_struct.inline_data.mime_type.startswith("image/")
                    ):
                        num_images_in_entry += 1
            image_token_cost_for_entry = num_images_in_entry * 765

        token_count = text_only_token_count + image_token_cost_for_entry
        tokenized_entries.append({**entry_data, "token_count": token_count})

    # 3. Separate Latest Query
    latest_query_data = None
    history_data_to_truncate = []
    if tokenized_entries and tokenized_entries[-1]["id"] == new_msg.id:
        latest_query_data = tokenized_entries.pop()
    history_data_to_truncate = tokenized_entries  # Remaining entries are prior history

    # 4. Perform Truncation using effective_max_tokens_for_messages
    final_api_message_parts = []

    if not history_data_to_truncate:  # No prior history
        if latest_query_data:
            # latest_query_data["token_count"] includes its text, external_content, and any image data tokens
            query_actual_tokens = latest_query_data["token_count"]

            if query_actual_tokens > effective_max_tokens_for_messages:
                # Truncate the 'text' field of latest_query_data
                budget_for_latest_query_node_text_field = (
                    effective_max_tokens_for_messages
                )

                # Subtract tokens for external_content and image data from the budget for the text field
                # External content will only exist for the current message
                if latest_query_data.get("external_content"):
                    budget_for_latest_query_node_text_field -= len(
                        tokenizer.encode(
                            latest_query_data["external_content"]
                            + "\n\nUser's query:\n"
                        )
                    )

                if latest_query_data["files"]:
                    num_images_in_latest_query = 0
                    for file_part_struct in latest_query_data["files"]:
                        if (
                            isinstance(file_part_struct, dict)
                            and file_part_struct.get("type") == "image_url"
                        ):
                            # Check if it's a data URL, implying an image processed for an API
                            img_url_dict = file_part_struct.get("image_url", {})
                            data_url_val = img_url_dict.get("url", "")
                            if (
                                data_url_val.startswith(
                                    "data:image"
                                )  # Covers OpenAI and similar base64 images
                                and ";base64," in data_url_val
                            ) or (  # Also consider Gemini parts if they are image bytes
                                is_target_provider_gemini
                                and isinstance(
                                    file_part_struct, google_types_module.Part
                                )
                                and hasattr(file_part_struct, "inline_data")
                                and file_part_struct.inline_data
                                and file_part_struct.inline_data.mime_type.startswith(
                                    "image/"
                                )
                            ):
                                num_images_in_latest_query += 1
                    if num_images_in_latest_query > 0:
                        budget_for_latest_query_node_text_field -= (
                            num_images_in_latest_query * 765
                        )

                budget_for_latest_query_node_text_field = max(
                    0, budget_for_latest_query_node_text_field
                )

                truncated_user_text, _ = _truncate_text_by_tokens(
                    latest_query_data["text"] or "",
                    tokenizer,
                    budget_for_latest_query_node_text_field,
                )
                latest_query_data["text"] = truncated_user_text
                user_warnings.add(
                    f"⚠️ Query truncated to fit token limit ({effective_max_tokens_for_messages:,})"
                )
            final_api_message_parts.append(latest_query_data)
    else:  # Prior history exists
        current_history_token_sum = sum(
            e["token_count"] for e in history_data_to_truncate
        )  # This sum includes text, external_content, and image data for each history entry.

        latest_query_actual_tokens = (
            latest_query_data["token_count"] if latest_query_data else 0
        )
        # latest_query_actual_tokens also includes its text, external_content, and image data.

        if (
            current_history_token_sum + latest_query_actual_tokens
            > effective_max_tokens_for_messages
        ):
            user_warnings.add(
                f"⚠️ History truncated to fit token limit ({effective_max_tokens_for_messages:,})"
            )
            retained_history_data = list(history_data_to_truncate)

            while retained_history_data:
                sum_retained_tokens = sum(
                    e["token_count"] for e in retained_history_data
                )
                if (
                    sum_retained_tokens + latest_query_actual_tokens
                    <= effective_max_tokens_for_messages
                ):
                    break
                if (
                    len(retained_history_data) >= 2
                    and retained_history_data[0]["role"] == "user"
                    and retained_history_data[1]["role"] in ("assistant", "model")
                ):
                    retained_history_data.pop(0)
                    if retained_history_data:
                        retained_history_data.pop(0)
                elif retained_history_data:
                    retained_history_data.pop(0)
                else:
                    break

            history_data_to_truncate = retained_history_data
            current_history_token_sum = sum(
                e["token_count"] for e in history_data_to_truncate
            )

            if latest_query_data and (
                current_history_token_sum + latest_query_actual_tokens
                > effective_max_tokens_for_messages
            ):
                budget_for_latest_query_node_text_field = (
                    effective_max_tokens_for_messages - current_history_token_sum
                )

                # External content will only exist for the current message
                if latest_query_data.get("external_content"):
                    budget_for_latest_query_node_text_field -= len(
                        tokenizer.encode(
                            latest_query_data["external_content"]
                            + "\n\nUser's query:\n"
                        )
                    )

                if latest_query_data["files"]:
                    num_images_in_latest_query = 0
                    for file_part_struct in latest_query_data["files"]:
                        if (
                            isinstance(file_part_struct, dict)
                            and file_part_struct.get("type") == "image_url"
                        ):
                            # Check if it's a data URL, implying an image processed for an API
                            img_url_dict = file_part_struct.get("image_url", {})
                            data_url_val = img_url_dict.get("url", "")
                            if (
                                data_url_val.startswith(
                                    "data:image"
                                )  # Covers OpenAI and similar base64 images
                                and ";base64," in data_url_val
                            ) or (  # Also consider Gemini parts if they are image bytes
                                is_target_provider_gemini
                                and isinstance(
                                    file_part_struct, google_types_module.Part
                                )
                                and hasattr(file_part_struct, "inline_data")
                                and file_part_struct.inline_data
                                and file_part_struct.inline_data.mime_type.startswith(
                                    "image/"
                                )
                            ):
                                num_images_in_latest_query += 1
                    if num_images_in_latest_query > 0:
                        budget_for_latest_query_node_text_field -= (
                            num_images_in_latest_query * 765
                        )

                budget_for_latest_query_node_text_field = max(
                    0, budget_for_latest_query_node_text_field
                )

                truncated_user_text, _ = _truncate_text_by_tokens(
                    latest_query_data["text"] or "",
                    tokenizer,
                    budget_for_latest_query_node_text_field,
                )
                latest_query_data["text"] = truncated_user_text

        final_api_message_parts.extend(history_data_to_truncate)
        if latest_query_data:
            final_api_message_parts.append(latest_query_data)

    # 5. Format for API
    api_formatted_history = []
    for entry in final_api_message_parts:
        # Construct the text content that goes into the API call for this message
        # Only include external_content if it's the current message
        is_current_message = entry.get("id") == new_msg.id
        text_content_for_api = entry["text"] or ""
        
        # Only use external_content for the new message, even if somehow it got saved in history
        if is_current_message and entry.get("external_content"):
            text_content_for_api = entry["external_content"] + "\n\nUser's query:\n" + text_content_for_api

        # Ensure file parts are within limits for this specific message
        # (max_files_per_message applies to files *within* one API message part, not total history files)
        current_api_file_parts = []
        if accept_files:  # Only include files if the model accepts them
            raw_parts_from_node = entry["files"][
                :max_files_per_message
            ]  # entry["files"] are already prepared API parts
            if is_target_provider_gemini:
                for part_in_node in raw_parts_from_node:
                    if isinstance(part_in_node, google_types_module.Part):
                        current_api_file_parts.append(part_in_node)
                    elif (
                        isinstance(part_in_node, dict)
                        and part_in_node.get("type") == "image_url"
                    ):  # Convert OpenAI to Gemini
                        try:
                            header, encoded_data = part_in_node["image_url"][
                                "url"
                            ].split(";base64,", 1)
                            mime_type = header.split(":")[1]
                            img_bytes = base64.b64decode(encoded_data)
                            current_api_file_parts.append(
                                google_types_module.Part.from_bytes(
                                    data=img_bytes, mime_type=mime_type
                                )
                            )
                        except Exception:
                            pass
            else:  # OpenAI format
                for part_in_node in raw_parts_from_node:
                    if isinstance(part_in_node, dict):
                        current_api_file_parts.append(part_in_node)
                    elif isinstance(part_in_node, google_types_module.Part) and hasattr(
                        part_in_node, "inline_data"
                    ):  # Convert Gemini to OpenAI
                        try:
                            if part_in_node.inline_data.mime_type.startswith("image/"):
                                # Offload base64 encoding
                                b64_data = await asyncio.to_thread(
                                    lambda b: base64.b64encode(b).decode("utf-8"),
                                    part_in_node.inline_data.data,
                                )
                                current_api_file_parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{part_in_node.inline_data.mime_type};base64,{b64_data}"
                                        },
                                    }
                                )
                        except Exception:
                            pass

        parts_for_this_api_message = []
        if is_target_provider_gemini:
            if text_content_for_api:
                parts_for_this_api_message.append(
                    google_types_module.Part.from_text(text=text_content_for_api)
                )
            parts_for_this_api_message.extend(current_api_file_parts)
        else:  # OpenAI
            if text_content_for_api:
                parts_for_this_api_message.append(
                    {"type": "text", "text": text_content_for_api}
                )
            parts_for_this_api_message.extend(current_api_file_parts)

        if parts_for_this_api_message:  # Only add if there's something to send
            message_data = {"role": entry["role"]}
            if is_target_provider_gemini:
                message_data["parts"] = (
                    parts_for_this_api_message
                    if isinstance(parts_for_this_api_message, list)
                    else [parts_for_this_api_message]
                )
            else:  # OpenAI
                if message_data["role"] == "model":
                    message_data["role"] = "assistant"  # OpenAI uses "assistant"
                # OpenAI expects 'content' to be a string if only text, or list of parts if multimodal
                message_data["content"] = (
                    parts_for_this_api_message[0]["text"]
                    if len(parts_for_this_api_message) == 1
                    and parts_for_this_api_message[0]["type"] == "text"
                    else parts_for_this_api_message
                )
                if (
                    target_provider_name in providers_supporting_usernames_const
                    and entry["role"] == "user"
                    and entry["user_id"]
                ):
                    message_data["name"] = str(entry["user_id"])
            api_formatted_history.append(message_data)

    return api_formatted_history


# STREAMING_INDICATOR is now imported from .constants
