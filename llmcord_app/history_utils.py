# llmcord_app/history_utils.py
import logging
from typing import Optional, List, Dict, Any # Added List, Dict, Any
import asyncio # Added asyncio
import base64 # Added base64
import httpx # Added httpx
import re # Added re

import discord

from .constants import AT_AI_PATTERN, PROVIDERS_SUPPORTING_USERNAMES, STREAMING_INDICATOR # Added more constants
from . import models # Added models import
# Assuming google.genai.types will be passed as google_types_module
# Assuming extract_text_from_pdf_bytes will be passed as a function

async def find_parent_message(
    message: discord.Message,
    current_bot_user: discord.User, # Pass the bot's user object
    is_dm: bool
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
                    ref_msg = await message.channel.fetch_message(message.reference.message_id)
                if ref_msg and ref_msg.type in (discord.MessageType.default, discord.MessageType.reply):
                    return ref_msg
                else:
                    logging.debug(f"Referenced message {message.reference.message_id} is not usable type {getattr(ref_msg, 'type', 'N/A')}")
            except (discord.NotFound, discord.HTTPException) as e:
                logging.warning(f"Could not fetch referenced message {message.reference.message_id}: {e}")

        # 2. Thread Start: If it's the first user message in a thread (not a reply)
        if isinstance(message.channel, discord.Thread) and message.channel.parent and not message.reference:
            try:
                starter_msg = message.channel.starter_message
                if not starter_msg:
                    starter_msg = await message.channel.parent.fetch_message(message.channel.id)
                if starter_msg and starter_msg.type in (discord.MessageType.default, discord.MessageType.reply):
                    return starter_msg
                else:
                    logging.debug(f"Thread starter message {message.channel.id} is not usable type {getattr(starter_msg, 'type', 'N/A')}")
            except (discord.NotFound, discord.HTTPException, AttributeError) as e:
                logging.warning(f"Could not fetch thread starter message for thread {message.channel.id}: {e}")

        # 3. Automatic Chaining (Only if NOT explicitly triggered)
        if not is_explicit_trigger:
            prev_msg_in_channel = None
            try:
                async for m in message.channel.history(before=message, limit=1):
                    prev_msg_in_channel = m
                    break
            except (discord.Forbidden, discord.HTTPException) as e:
                logging.warning(f"Could not fetch history in channel {message.channel.id}: {e}")

            if prev_msg_in_channel and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply):
                if (is_dm and prev_msg_in_channel.author == current_bot_user) or \
                   (not is_dm and prev_msg_in_channel.author == message.author):
                    return prev_msg_in_channel

        # 4. No logical parent found
        return None

    except Exception as e:
        logging.exception(f"Error determining parent message for {message.id}")
        return None

async def build_message_history(
    new_msg: discord.Message,
    initial_cleaned_content: str,
    combined_context: str,
    max_messages: int,
    max_text: int,
    max_files_per_message: int,
    accept_files: bool,
    use_google_lens: bool,
    is_target_provider_gemini: bool,
    target_provider_name: str,
    target_model_name: str,
    user_warnings: set,
    current_message_url_fetch_results: Optional[List['models.UrlFetchResult']],
    # Client-specific attributes to be passed
    msg_nodes_cache: dict, # This is self.msg_nodes
    bot_user_obj: discord.User, # This is self.user
    httpx_async_client: 'httpx.AsyncClient', # This is self.httpx_client
    # Modules/Constants needed
    models_module: Any, # Pass the models module
    google_types_module: Any, # Pass the google.genai.types module
    extract_text_from_pdf_bytes_func: callable,
    at_ai_pattern_re: Any, # Pass the compiled regex
    providers_supporting_usernames_const: tuple # This is already from .constants
) -> List[Dict[str, Any]]:
    history = []
    curr_msg = new_msg
    is_dm_current_msg_channel = isinstance(new_msg.channel, discord.DMChannel) # is_dm for the current message context

    while curr_msg is not None and len(history) < max_messages:
        if curr_msg.id not in msg_nodes_cache:
            logging.debug(f"Node for message {curr_msg.id} not in cache. Fetching message.")
            try:
                if curr_msg.id != new_msg.id:
                    curr_msg = await new_msg.channel.fetch_message(curr_msg.id)
                    if not curr_msg:
                        logging.warning(f"Failed to fetch message {curr_msg.id} for history building.")
                        user_warnings.add(f"⚠️ Couldn't fetch full history (message {curr_msg.id} missing).")
                        break
            except (discord.NotFound, discord.HTTPException) as fetch_err:
                logging.warning(f"Failed to fetch message {curr_msg.id} for history building: {fetch_err}")
                user_warnings.add(f"⚠️ Couldn't fetch full history (message {curr_msg.id} missing).")
                break
            msg_nodes_cache[curr_msg.id] = models_module.MsgNode()

        curr_node = msg_nodes_cache[curr_msg.id]

        async with curr_node.lock:
            is_current_message_node = (curr_msg.id == new_msg.id)
            current_role = "model" if curr_msg.author == bot_user_obj else "user"

            if is_current_message_node:
                curr_node.external_content = combined_context if combined_context else None
                logging.debug(f"Set external_content for node {curr_msg.id} to {'present' if combined_context else 'None'} for this history build.")

            should_populate_node = (curr_node.text is None) or is_current_message_node

            if should_populate_node:
                curr_node.has_bad_attachments = False
                if is_current_message_node:
                    curr_node.api_file_parts = []

                content_to_store = ""
                if current_role == "model":
                    if curr_node.full_response_text:
                        content_to_store = curr_node.full_response_text
                    else:
                        if curr_msg.embeds and curr_msg.embeds[0].description:
                            content_to_store = curr_msg.embeds[0].description.replace(STREAMING_INDICATOR, "").strip() # STREAMING_INDICATOR needs to be imported or passed
                        else:
                            content_to_store = curr_msg.content
                else: # User message
                    content_to_store = initial_cleaned_content if curr_msg.id == new_msg.id else curr_msg.content
                    is_dm_iter_msg_channel = isinstance(curr_msg.channel, discord.DMChannel)
                    if not is_dm_iter_msg_channel and bot_user_obj.mentioned_in(curr_msg):
                         content_to_store = content_to_store.replace(bot_user_obj.mention, '').strip()
                    if curr_msg.id != new_msg.id: # Don't re-sub @ai for the current message as it's already cleaned
                        content_to_store = at_ai_pattern_re.sub(' ', content_to_store)
                    content_to_store = re.sub(r'\s{2,}', ' ', content_to_store).strip()


                current_attachments = curr_msg.attachments
                MAX_ATTACHMENTS_TO_DOWNLOAD_IN_HISTORY = 5
                attachments_to_fetch = []
                unfetched_unsupported_types = False

                for att_idx, att in enumerate(current_attachments):
                    if len(attachments_to_fetch) >= MAX_ATTACHMENTS_TO_DOWNLOAD_IN_HISTORY:
                        curr_node.has_bad_attachments = True
                        break
                    if att.content_type:
                        is_relevant_for_download = False
                        if att.content_type.startswith("text/"):
                            is_relevant_for_download = True
                        elif att.content_type.startswith("image/"):
                            if accept_files or (curr_msg.id == new_msg.id and use_google_lens):
                                is_relevant_for_download = True
                        elif att.content_type == "application/pdf":
                            if (is_target_provider_gemini and accept_files) or (not is_target_provider_gemini):
                                is_relevant_for_download = True
                        if is_relevant_for_download:
                            attachments_to_fetch.append(att)
                        else:
                            unfetched_unsupported_types = True
                
                if unfetched_unsupported_types:
                    curr_node.has_bad_attachments = True

                attachment_responses = await asyncio.gather(*[httpx_async_client.get(att.url, timeout=15.0) for att in attachments_to_fetch], return_exceptions=True)
                
                text_parts = [content_to_store] if content_to_store else []
                if current_role == "user":
                    text_parts.extend(filter(None, (embed.title for embed in curr_msg.embeds)))
                    text_parts.extend(filter(None, (embed.description for embed in curr_msg.embeds)))

                for att, resp in zip(attachments_to_fetch, attachment_responses):
                    if isinstance(resp, httpx.Response) and resp.status_code == 200 and att.content_type.startswith("text/"):
                        try:
                            text_parts.append(resp.text)
                        except Exception as e:
                            logging.warning(f"Failed to decode text attachment {att.filename} in history: {e}")
                            curr_node.has_bad_attachments = True
                    elif isinstance(resp, Exception) and att.content_type.startswith("text/"):
                        logging.warning(f"Failed to fetch text attachment {att.filename} in history: {resp}")
                        curr_node.has_bad_attachments = True
                
                curr_node.text = "\n".join(filter(None, text_parts))

                if current_role == "user" and not is_target_provider_gemini:
                    pdf_texts_to_append = []
                    for att, resp in zip(attachments_to_fetch, attachment_responses):
                        if att.content_type == "application/pdf":
                            if isinstance(resp, httpx.Response) and resp.status_code == 200:
                                try:
                                    extracted_pdf_text = await extract_text_from_pdf_bytes_func(resp.content)
                                    if extracted_pdf_text:
                                        pdf_texts_to_append.append(f"\n\n--- Content from PDF: {att.filename} ---\n{extracted_pdf_text}\n--- End of PDF: {att.filename} ---")
                                    else:
                                        curr_node.has_bad_attachments = True
                                except Exception as pdf_extract_err:
                                    logging.error(f"Error extracting text from PDF {att.filename}: {pdf_extract_err}")
                                    curr_node.has_bad_attachments = True
                            elif isinstance(resp, Exception):
                                curr_node.has_bad_attachments = True
                    if pdf_texts_to_append:
                        curr_node.text = (curr_node.text or "") + "".join(pdf_texts_to_append)

                api_file_parts = []
                files_processed_for_api_count = 0
                is_lens_trigger_message = curr_msg.id == new_msg.id and use_google_lens
                should_process_files_for_api = (current_role == "user" or is_lens_trigger_message) and (accept_files or is_lens_trigger_message)

                if should_process_files_for_api and not is_lens_trigger_message:
                    for att, resp in zip(attachments_to_fetch, attachment_responses):
                        is_api_relevant_type = False
                        mime_type_for_api = att.content_type
                        file_bytes_for_api = None
                        if att.content_type.startswith("image/"):
                            is_api_relevant_type = True
                            if isinstance(resp, httpx.Response) and resp.status_code == 200: file_bytes_for_api = resp.content
                            else: curr_node.has_bad_attachments = True; continue
                        elif att.content_type == "application/pdf" and is_target_provider_gemini and accept_files:
                            is_api_relevant_type = True
                            mime_type_for_api = "application/pdf"
                            if isinstance(resp, httpx.Response) and resp.status_code == 200: file_bytes_for_api = resp.content
                            else: curr_node.has_bad_attachments = True; continue
                        
                        if not is_api_relevant_type or file_bytes_for_api is None: continue
                        if files_processed_for_api_count >= max_files_per_message: curr_node.has_bad_attachments = True; break
                        try:
                            if is_target_provider_gemini:
                                api_file_parts.append(google_types_module.Part.from_bytes(data=file_bytes_for_api, mime_type=mime_type_for_api))
                            else:
                                api_file_parts.append(dict(type="image_url", image_url=dict(url=f"data:{mime_type_for_api};base64,{base64.b64encode(file_bytes_for_api).decode('utf-8')}")))
                            files_processed_for_api_count += 1
                        except Exception as e: curr_node.has_bad_attachments = True; logging.error(f"Error preparing attachment {att.filename} for API: {e}")

                    if curr_msg.id == new_msg.id and current_message_url_fetch_results:
                        for fetched_url_res in current_message_url_fetch_results:
                            if fetched_url_res.type == "image_url_content" and isinstance(fetched_url_res.content, bytes) and not fetched_url_res.error:
                                if files_processed_for_api_count >= max_files_per_message: curr_node.has_bad_attachments = True; user_warnings.add("⚠️ Max files reached."); break
                                img_bytes = fetched_url_res.content
                                url_lower = fetched_url_res.url.lower()
                                mime_type = "image/png"
                                if url_lower.endswith((".jpg", ".jpeg")): mime_type = "image/jpeg"
                                elif url_lower.endswith(".gif"): mime_type = "image/gif"
                                elif url_lower.endswith(".webp"): mime_type = "image/webp"
                                elif url_lower.endswith(".bmp"): mime_type = "image/bmp"
                                try:
                                    if is_target_provider_gemini:
                                        api_file_parts.append(google_types_module.Part.from_bytes(data=img_bytes, mime_type=mime_type))
                                    else:
                                        api_file_parts.append(dict(type="image_url", image_url=dict(url=f"data:{mime_type};base64,{base64.b64encode(img_bytes).decode('utf-8')}")))
                                    files_processed_for_api_count +=1
                                except Exception as e: curr_node.has_bad_attachments = True; user_warnings.add(f"⚠️ Error processing image URL: {fetched_url_res.url[:50]}..."); logging.error(f"Error preparing image URL {fetched_url_res.url} for API: {e}")
                
                curr_node.api_file_parts = api_file_parts
                curr_node.role = current_role
                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
                
                if curr_node.parent_msg is None and not curr_node.fetch_parent_failed:
                    # Use find_parent_message from the same module
                    parent = await find_parent_message(curr_msg, bot_user_obj, is_dm_current_msg_channel)
                    if parent is None and curr_msg.reference and curr_msg.reference.message_id:
                        curr_node.fetch_parent_failed = True
                    curr_node.parent_msg = parent

            current_text_content = ""
            if curr_node.external_content:
                current_text_content += curr_node.external_content + "\n\nUser's query:\n"
            
            node_text_to_use = curr_node.full_response_text if curr_node.role == "model" and curr_node.full_response_text else (curr_node.text or "")
            current_text_content += node_text_to_use
            current_text_content = current_text_content[:max_text] if current_text_content else ""

            current_api_file_parts = []
            if accept_files:
                raw_parts_from_node = curr_node.api_file_parts[:max_files_per_message]
                if is_target_provider_gemini:
                    for part_in_node in raw_parts_from_node:
                        if isinstance(part_in_node, google_types_module.Part): current_api_file_parts.append(part_in_node)
                        elif isinstance(part_in_node, dict) and part_in_node.get("type") == "image_url":
                            # Convert OpenAI dict to Gemini Part
                            # (Simplified, assumes valid base64 data URL)
                            try:
                                header, encoded_data = part_in_node["image_url"]["url"].split(";base64,",1)
                                mime_type = header.split(":")[1]
                                img_bytes = base64.b64decode(encoded_data)
                                current_api_file_parts.append(google_types_module.Part.from_bytes(data=img_bytes, mime_type=mime_type))
                            except Exception: pass # Log error if needed
                else: # OpenAI format
                    for part_in_node in raw_parts_from_node:
                        if isinstance(part_in_node, dict): current_api_file_parts.append(part_in_node)
                        elif isinstance(part_in_node, google_types_module.Part) and hasattr(part_in_node, 'inline_data'):
                             # Convert Gemini Part to OpenAI dict
                            try:
                                if part_in_node.inline_data.mime_type.startswith("image/"):
                                    b64_data = base64.b64encode(part_in_node.inline_data.data).decode('utf-8')
                                    current_api_file_parts.append({"type": "image_url", "image_url": {"url": f"data:{part_in_node.inline_data.mime_type};base64,{b64_data}"}})
                            except Exception: pass # Log error if needed
            
            parts_for_api = []
            if is_target_provider_gemini:
                if current_text_content: parts_for_api.append(google_types_module.Part.from_text(text=current_text_content))
                parts_for_api.extend(current_api_file_parts)
            else:
                if current_text_content: parts_for_api.append({"type": "text", "text": current_text_content})
                parts_for_api.extend(current_api_file_parts)

            if parts_for_api:
                message_data = {"role": curr_node.role}
                if is_target_provider_gemini:
                    message_data["parts"] = parts_for_api if isinstance(parts_for_api, list) else [parts_for_api]
                else:
                    if message_data["role"] == "model": message_data["role"] = "assistant"
                    message_data["content"] = parts_for_api[0]["text"] if len(parts_for_api) == 1 and parts_for_api[0]["type"] == "text" else parts_for_api
                    if target_provider_name in providers_supporting_usernames_const and curr_node.role == "user" and curr_node.user_id:
                        message_data["name"] = str(curr_node.user_id)
                history.append(message_data)

            if curr_node.text and len(curr_node.text) > max_text: user_warnings.add(f"⚠️ Max {max_text:,} chars/msg node text")
            if curr_node.has_bad_attachments: user_warnings.add("⚠️ Some attachments might not have been processed.")
            if curr_node.fetch_parent_failed: user_warnings.add("⚠️ Couldn't fetch full history")
            if curr_node.parent_msg is not None and len(history) >= max_messages: user_warnings.add(f"⚠️ Only using last {max_messages} messages")

            curr_msg = curr_node.parent_msg
    return history[::-1]

# STREAMING_INDICATOR is now imported from .constants
