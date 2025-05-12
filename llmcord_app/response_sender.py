# llmcord_app/response_sender.py
import asyncio
import logging
from datetime import datetime as dt
from typing import List, Dict, Optional, Any, Set

import discord
from google.genai import types as google_types # For finish_reason comparison

from . import models # Use relative import
from .constants import (
    EMBED_COLOR_COMPLETE, EMBED_COLOR_INCOMPLETE, EMBED_COLOR_ERROR,
    STREAMING_INDICATOR, EDIT_DELAY_SECONDS, MAX_EMBED_DESCRIPTION_LENGTH,
    MAX_PLAIN_TEXT_LENGTH, IMGUR_HEADER, IMGUR_URL_PATTERN, IMGUR_URL_PREFIX,
    AllKeysFailedError
)
from .ui import ResponseActionView
from .llm_handler import generate_response_stream # For type hinting and direct call

# Forward declaration for LLMCordClient to resolve circular import for type hinting if needed
# However, it's better to pass necessary attributes directly if possible.
# class LLMCordClient(discord.Client): ...

async def send_initial_processing_message(
    new_msg: discord.Message,
    use_plain_responses: bool
) -> Optional[discord.Message]:
    """Sends the initial 'Processing request...' message."""
    processing_msg: Optional[discord.Message] = None
    try:
        if use_plain_responses:
            processing_msg = await new_msg.reply("⏳ Processing request...", mention_author=False, suppress_embeds=True)
        else:
            processing_embed = discord.Embed(description="⏳ Processing request...", color=EMBED_COLOR_INCOMPLETE)
            processing_msg = await new_msg.reply(embed=processing_embed, mention_author=False)
    except discord.HTTPException as e:
        logging.warning(f"Failed to send initial 'Processing request...' message: {e}")
    except Exception as e:
        logging.error(f"Unexpected error sending initial 'Processing request...' message: {e}", exc_info=True)
    return processing_msg

async def handle_llm_response_stream(
    client, # Actually an LLMCordClient instance, pass necessary attributes/methods
    new_msg: discord.Message,
    processing_msg: Optional[discord.Message],
    provider: str,
    model_name: str,
    history_for_llm: List[Dict[str, Any]],
    system_prompt_text: Optional[str],
    provider_config: Dict[str, Any],
    extra_api_params: Dict[str, Any],
    initial_user_warnings: Set[str],
    use_plain_responses_config: bool, # Renamed to avoid conflict
    split_limit_config: int # Renamed
):
    """Handles the streaming, editing, and sending of LLM responses."""
    response_msgs: List[discord.Message] = []
    final_text = ""
    llm_call_successful = False
    # final_view = None # This is determined within the loop
    grounding_metadata = None
    edit_task = None
    # client.last_task_time should be managed by the client instance or passed if needed for reset
    # For now, assume it's handled by the caller or we manage a local version if this function becomes very independent.
    # Let's assume client object has 'last_task_time', 'msg_nodes'
    
    # Initial embed for warnings and footer
    base_embed = discord.Embed()
    base_embed.set_footer(text=f"Model: {model_name}") # Use model_name passed in
    for warning in sorted(initial_user_warnings): # Use initial_user_warnings
        base_embed.add_field(name=warning, value="", inline=False)

    try:
        async with new_msg.channel.typing():
            stream_generator = generate_response_stream(
                provider=provider,
                model_name=model_name,
                history_for_llm=history_for_llm,
                system_prompt_text=system_prompt_text,
                provider_config=provider_config,
                extra_params=extra_api_params
            )

            async for text_chunk, finish_reason, chunk_grounding_metadata, error_message in stream_generator:
                if error_message:
                    logging.error(f"LLM stream failed with non-retryable error: {error_message}")
                    await _handle_stream_error(new_msg, processing_msg, response_msgs, error_message, use_plain_responses_config)
                    llm_call_successful = False
                    return llm_call_successful, final_text, response_msgs # Return immediately

                if chunk_grounding_metadata:
                    grounding_metadata = chunk_grounding_metadata
                if text_chunk:
                    final_text += text_chunk

                if not use_plain_responses_config:
                    is_final_chunk = finish_reason is not None
                    if not final_text and not is_final_chunk: continue

                    current_msg_index = (len(final_text) - 1) // split_limit_config if final_text else 0
                    start_next_msg = current_msg_index >= len(response_msgs)
                    
                    ready_to_edit = (
                        (edit_task is None or edit_task.done()) and
                        (dt.now().timestamp() - client.last_task_time >= EDIT_DELAY_SECONDS)
                    )

                    if start_next_msg or ready_to_edit or is_final_chunk:
                        if edit_task is not None and not edit_task.done():
                            try: await edit_task
                            except asyncio.CancelledError: logging.warning("Previous edit task cancelled.")
                            except Exception as e: logging.error(f"Error waiting for previous edit task: {e}")
                            edit_task = None
                        
                        if start_next_msg and response_msgs: # Finalize previous message
                            prev_msg_idx = current_msg_index -1
                            if 0 <= prev_msg_idx < len(response_msgs):
                                prev_text = final_text[prev_msg_idx * split_limit_config : current_msg_index * split_limit_config]
                                prev_text = prev_text[:MAX_EMBED_DESCRIPTION_LENGTH + len(STREAMING_INDICATOR)]
                                temp_prev_embed = discord.Embed.from_dict(response_msgs[prev_msg_idx].embeds[0].to_dict()) if response_msgs[prev_msg_idx].embeds else discord.Embed()
                                temp_prev_embed.description = prev_text or "..."
                                temp_prev_embed.color = EMBED_COLOR_COMPLETE
                                try: await response_msgs[prev_msg_idx].edit(embed=temp_prev_embed, view=None)
                                except discord.HTTPException as e: logging.error(f"Failed to finalize prev msg {prev_msg_idx}: {e}")
                        
                        current_display_text = final_text[current_msg_index * split_limit_config : (current_msg_index + 1) * split_limit_config]
                        current_display_text = current_display_text[:MAX_EMBED_DESCRIPTION_LENGTH]

                        view_to_attach = None
                        is_successful_finish = finish_reason and (finish_reason.lower() in ("stop", "end_turn") or (provider == "google" and finish_reason == str(google_types.FinishReason.FINISH_REASON_UNSPECIFIED)))
                        is_blocked = finish_reason and finish_reason.lower() in ("safety", "recitation", "other")

                        current_segment_embed = discord.Embed()
                        current_segment_embed.set_footer(text=base_embed.footer.text if base_embed.footer else "")
                        for field in base_embed.fields:
                            current_segment_embed.add_field(name=field.name, value=field.value, inline=field.inline)
                        
                        current_segment_embed.description = (current_display_text or "...") if is_final_chunk else ((current_display_text or "...") + STREAMING_INDICATOR)

                        if is_final_chunk and is_blocked:
                            current_segment_embed.description = (current_display_text or "...").replace(STREAMING_INDICATOR, "").strip()
                            if finish_reason.lower() == "safety": current_segment_embed.description += "\n\n⚠️ Response blocked by safety filters."
                            elif finish_reason.lower() == "recitation": current_segment_embed.description += "\n\n⚠️ Response stopped due to recitation."
                            else: current_segment_embed.description += "\n\n⚠️ Response blocked (Reason: Other)."
                            current_segment_embed.description = current_segment_embed.description[:MAX_EMBED_DESCRIPTION_LENGTH]
                            current_segment_embed.color = EMBED_COLOR_ERROR
                        else:
                            current_segment_embed.color = EMBED_COLOR_COMPLETE if is_final_chunk and is_successful_finish else EMBED_COLOR_INCOMPLETE
                            if is_final_chunk and is_successful_finish:
                                has_sources = grounding_metadata and (getattr(grounding_metadata, 'web_search_queries', None) or getattr(grounding_metadata, 'grounding_chunks', None) or getattr(grounding_metadata, 'search_entry_point', None))
                                has_text_content = bool(final_text)
                                if has_sources or has_text_content:
                                    view_to_attach = ResponseActionView(grounding_metadata=grounding_metadata, full_response_text=final_text, model_name=model_name)
                                    if not view_to_attach or len(view_to_attach.children) == 0: view_to_attach = None
                        
                        if start_next_msg:
                            if not response_msgs and processing_msg:
                                await processing_msg.edit(content=None, embed=current_segment_embed, view=view_to_attach)
                                response_msg = processing_msg
                            else:
                                reply_target = new_msg if not response_msgs else response_msgs[-1]
                                response_msg = await reply_target.reply(embed=current_segment_embed, view=view_to_attach, mention_author=False)
                            response_msgs.append(response_msg)
                            client.msg_nodes[response_msg.id] = models.MsgNode(parent_msg=new_msg)
                            if view_to_attach: view_to_attach.message = response_msg
                            await client.msg_nodes[response_msg.id].lock.acquire()
                        elif response_msgs and current_msg_index < len(response_msgs):
                            target_msg = response_msgs[current_msg_index]
                            if target_msg:
                                edit_task = asyncio.create_task(target_msg.edit(embed=current_segment_embed, view=view_to_attach))
                                if view_to_attach: view_to_attach.message = target_msg
                        elif not response_msgs and is_final_chunk: # Short final response
                            if processing_msg:
                                await processing_msg.edit(content=None, embed=current_segment_embed, view=view_to_attach)
                                response_msg = processing_msg
                            else:
                                response_msg = await new_msg.reply(embed=current_segment_embed, view=view_to_attach, mention_author=False)
                            response_msgs.append(response_msg)
                            client.msg_nodes[response_msg.id] = models.MsgNode(parent_msg=new_msg)
                            if view_to_attach: view_to_attach.message = response_msg
                            await client.msg_nodes[response_msg.id].lock.acquire()
                        
                        client.last_task_time = dt.now().timestamp()

                if finish_reason:
                    llm_call_successful = finish_reason.lower() in ("stop", "end_turn") or (provider == "google" and finish_reason == str(google_types.FinishReason.FINISH_REASON_UNSPECIFIED))
                    break
        
        if use_plain_responses_config and llm_call_successful:
            final_messages_content = [final_text[i:i+split_limit_config] for i in range(0, len(final_text), split_limit_config)]
            if not final_messages_content: final_messages_content.append("...")
            
            temp_response_msgs = []
            for i, content_chunk in enumerate(final_messages_content):
                if i == 0 and processing_msg:
                    await processing_msg.edit(content=content_chunk or "...", embed=None, view=None)
                    response_msg = processing_msg
                else:
                    reply_target = new_msg if not temp_response_msgs else temp_response_msgs[-1]
                    response_msg = await reply_target.reply(content=content_chunk or "...", suppress_embeds=True, view=None, mention_author=False)
                temp_response_msgs.append(response_msg)
                client.msg_nodes[response_msg.id] = models.MsgNode(parent_msg=new_msg)
                node = client.msg_nodes[response_msg.id]
                await node.lock.acquire()
                if i == len(final_messages_content) -1:
                    node.full_response_text = final_text
            response_msgs = temp_response_msgs

    except AllKeysFailedError as e:
        logging.error(f"LLM generation failed for message {new_msg.id}: {e}")
        error_text = f"⚠️ All API keys for provider `{e.service_name}` failed."
        # Simplified error reporting (can be expanded as in original)
        last_err_str = str(e.errors[-1]) if e.errors else "Unknown reason."
        error_text += f"\nLast error: `{last_err_str[:100]}{'...' if len(last_err_str) > 100 else ''}`"
        await _handle_llm_exception(new_msg, processing_msg, response_msgs, error_text, use_plain_responses_config, client.config.get("use_plain_responses", False))
        llm_call_successful = False
    except Exception as outer_e:
        logging.exception(f"Unhandled error during message processing for {new_msg.id}.")
        error_text = f"⚠️ An unexpected error occurred: {type(outer_e).__name__}"
        await _handle_llm_exception(new_msg, processing_msg, response_msgs, error_text, use_plain_responses_config, client.config.get("use_plain_responses", False))
        llm_call_successful = False
    
    return llm_call_successful, final_text, response_msgs


async def _handle_stream_error(
    new_msg: discord.Message,
    processing_msg: Optional[discord.Message],
    response_msgs: List[discord.Message],
    error_message: str,
    use_plain_responses_config: bool
):
    """Handles errors that occur mid-stream."""
    if not use_plain_responses_config and response_msgs and response_msgs[-1].embeds:
        try:
            embed = response_msgs[-1].embeds[0]
            embed.description = (embed.description or "").replace(STREAMING_INDICATOR, "").strip()
            embed.description += f"\n\n⚠️ Error: {error_message}"
            embed.description = embed.description[:MAX_EMBED_DESCRIPTION_LENGTH]
            embed.color = EMBED_COLOR_ERROR
            await response_msgs[-1].edit(embed=embed, view=None)
        except Exception as edit_err:
            logging.error(f"Failed to edit message to show stream error: {edit_err}")
            # Fallback reply
            target = processing_msg if processing_msg and processing_msg.id == response_msgs[-1].id else new_msg
            await target.reply(f"⚠️ Error during response generation: {error_message}", mention_author=False)
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
    use_plain_responses_stream: bool, # Whether the stream was plain
    use_plain_initial_status: bool # Whether the initial status message was plain
):
    """Handles exceptions from the LLM call (e.g., AllKeysFailedError)."""
    if processing_msg:
        if use_plain_initial_status:
            await processing_msg.edit(content=error_text, embed=None, view=None)
        else: # Initial status was an embed
            if not use_plain_responses_stream and response_msgs and response_msgs[-1].embeds: # Stream was also embed
                target_edit_msg = response_msgs[-1]
                embed_to_edit = discord.Embed.from_dict(target_edit_msg.embeds[0].to_dict())
                current_desc = embed_to_edit.description or ""
                embed_to_edit.description = current_desc.replace(STREAMING_INDICATOR, "").strip()
                embed_to_edit.description += f"\n\n{error_text}"
                embed_to_edit.description = embed_to_edit.description[:MAX_EMBED_DESCRIPTION_LENGTH]
                embed_to_edit.color = EMBED_COLOR_ERROR
                await target_edit_msg.edit(embed=embed_to_edit, view=None)
            else: # Stream was plain or no stream messages, edit initial embed status
                await processing_msg.edit(content=None, embed=discord.Embed(description=error_text, color=EMBED_COLOR_ERROR), view=None)
    elif not use_plain_responses_stream and response_msgs and response_msgs[-1].embeds: # No processing_msg, but stream had embeds
        target_edit_msg = response_msgs[-1]
        # ... (similar logic as above to edit the last stream message)
        embed_to_edit = discord.Embed.from_dict(target_edit_msg.embeds[0].to_dict())
        current_desc = embed_to_edit.description or ""
        embed_to_edit.description = current_desc.replace(STREAMING_INDICATOR, "").strip()
        embed_to_edit.description += f"\n\n{error_text}"
        embed_to_edit.description = embed_to_edit.description[:MAX_EMBED_DESCRIPTION_LENGTH]
        embed_to_edit.color = EMBED_COLOR_ERROR
        await target_edit_msg.edit(embed=embed_to_edit, view=None)
    else: # Fallback: No processing_msg and no stream to edit, send new reply
        await new_msg.reply(error_text, mention_author=False)


async def resend_imgur_urls(
    new_msg: discord.Message, # Original user message
    response_msgs: List[discord.Message], # List of bot's response messages
    final_text: str
):
    """Checks for Imgur URLs in the final_text and resends them if found."""
    if not final_text: return

    lines = final_text.strip().split('\n')
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
        logging.info(f"Detected Imgur URLs in response to message {new_msg.id}. Resending.")
        formatted_urls = imgur_urls_to_resend
        max_chars = MAX_PLAIN_TEXT_LENGTH
        messages_to_send_content = []
        current_message_content = ""

        for url_str in formatted_urls:
            needed_len = len(url_str) + (2 if current_message_content else 0)
            if len(current_message_content) + needed_len > max_chars:
                if current_message_content:
                    messages_to_send_content.append(current_message_content)
                current_message_content = url_str[:max_chars] if len(url_str) > max_chars else url_str
                if len(url_str) > max_chars: logging.warning(f"Single Imgur URL too long: {url_str}")
            else:
                current_message_content += ("\n\n" if current_message_content else "") + url_str
        
        if current_message_content:
            messages_to_send_content.append(current_message_content)

        reply_target = response_msgs[-1] if response_msgs else new_msg
        last_sent_msg = reply_target
        for i, msg_content in enumerate(messages_to_send_content):
            try:
                target_to_reply_to = last_sent_msg
                if isinstance(target_to_reply_to, discord.Message):
                    sent_msg = await target_to_reply_to.reply(content=msg_content, mention_author=False)
                    last_sent_msg = sent_msg
                else:
                    logging.warning(f"Invalid reply target type: {type(target_to_reply_to)}. Replying to original.")
                    sent_msg = await new_msg.reply(content=msg_content, mention_author=False)
                    last_sent_msg = sent_msg
                await asyncio.sleep(0.1)
            except discord.HTTPException as send_err:
                logging.error(f"Failed to resend Imgur URL chunk {i+1}: {send_err}")
                try: await new_msg.reply(f"(Error sending previous chunk)\n{msg_content}", mention_author=False)
                except discord.HTTPException as fallback_err: logging.error(f"Fallback Imgur resend failed: {fallback_err}")
            except Exception as e:
                 logging.error(f"Unexpected error resending Imgur URL chunk {i+1}: {e}")
