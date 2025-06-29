import asyncio
from datetime import datetime as dt
from typing import Dict, Optional, Any
import discord
from google.genai import types as google_types
from ..core.constants import (
    EMBED_COLOR_COMPLETE,
    EMBED_COLOR_INCOMPLETE,
    EMBED_COLOR_ERROR,
    STREAMING_INDICATOR,
    EDIT_DELAY_SECONDS_CONFIG_KEY,
    MAX_EMBED_DESCRIPTION_LENGTH,
)
from ..core import models
from ..ui.ui import ResponseActionView


class MessageEditor:
    """Handles Discord message creation and editing for LLM responses."""

    def __init__(self, client, app_config: Dict[str, Any]):
        self.client = client
        self.app_config = app_config
        self.edit_task = None

    async def send_initial_processing_message(
        self, new_msg: discord.Message, use_plain_responses: bool
    ) -> Optional[discord.Message]:
        """Sends the initial 'Processing request...' message."""
        processing_msg: Optional[discord.Message] = None
        try:
            if use_plain_responses:
                processing_msg = await new_msg.reply(
                    "⏳ Processing request...",
                    mention_author=False,
                    suppress_embeds=True,
                )
            else:
                processing_embed = discord.Embed(
                    description="⏳ Processing request...", color=EMBED_COLOR_INCOMPLETE
                )
                processing_msg = await new_msg.reply(
                    embed=processing_embed, mention_author=False
                )
        except discord.HTTPException:
            pass
        except Exception:
            pass
        return processing_msg

    async def process_text_stream(
        self,
        stream_generator,
        new_msg,
        processing_msg,
        response_msgs,
        current_params,
        initial_user_warnings,
        use_plain_responses_config,
        split_limit_config,
        custom_search_queries_generated,
        successful_api_results_count,
        deep_search_used,
        attempt_num,
    ):
        """Process text stream and update Discord messages."""
        final_text = ""
        grounding_metadata = None
        success = False
        should_retry = False
        base_embed = discord.Embed()
        for warning_text in sorted(list(initial_user_warnings)):
            base_embed.add_field(name=warning_text, value="", inline=False)
        async for (
            text_chunk,
            finish_reason,
            chunk_grounding_metadata,
            error_message,
            image_data,
            image_mime_type,
        ) in stream_generator:
            # Handle retry signals
            if error_message in [
                "RETRY_WITH_GEMINI_NO_FINISH_REASON",
                "RETRY_WITH_FALLBACK_MODEL_UNPROCESSABLE_ENTITY",
            ]:
                if attempt_num == 0:
                    should_retry = True
                    break
                else:
                    error_message = "Stream error on retry attempt"
            # Check for compression warning message
            if error_message and error_message.startswith("COMPRESSION_INFO:"):
                try:
                    # Extract compression information from the error_message
                    compression_msg = error_message.replace(
                        "COMPRESSION_INFO:", ""
                    ).strip()
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
                                name=field.name, value=field.value, inline=field.inline
                            )
                    # Add the new compression warning
                    warning_field_name = compression_msg
                    new_embed.add_field(name=warning_field_name, value="", inline=False)
                    # Replace the base_embed with our new one
                    base_embed = new_embed
                    # Continue processing, this isn't a fatal error
                    error_message = None
                except Exception:
                    # Continue with normal processing if parsing fails
                    pass
            if error_message:
                return {
                    "success": False,
                    "text": final_text,
                    "response_msgs": response_msgs,
                }
            if chunk_grounding_metadata:
                grounding_metadata = chunk_grounding_metadata
            if text_chunk:
                final_text += (
                    text_chunk if isinstance(text_chunk, str) else "".join(text_chunk)
                )
            # Update messages for streaming
            if not use_plain_responses_config:
                await self._update_embed_messages(
                    new_msg,
                    processing_msg,
                    response_msgs,
                    final_text,
                    split_limit_config,
                    base_embed,
                    finish_reason,
                    current_params,
                    grounding_metadata,
                    custom_search_queries_generated,
                    successful_api_results_count,
                    deep_search_used,
                    attempt_num,
                )
            if finish_reason:
                success = self._is_successful_finish(
                    finish_reason, current_params["provider"]
                )
                break
        if should_retry:
            return {
                "success": False,
                "text": "",
                "response_msgs": response_msgs,
                "should_retry": True,
            }
        # Handle plain text responses
        if use_plain_responses_config and success:
            await self._send_plain_text_response(
                new_msg,
                processing_msg,
                response_msgs,
                final_text,
                split_limit_config,
                current_params,
                grounding_metadata,
                custom_search_queries_generated,
            )
        return {
            "success": success,
            "text": final_text,
            "response_msgs": response_msgs,
            "should_retry": False,
        }

    async def _update_embed_messages(
        self,
        new_msg,
        processing_msg,
        response_msgs,
        final_text,
        split_limit_config,
        base_embed,
        finish_reason,
        current_params,
        grounding_metadata,
        custom_search_queries_generated,
        successful_api_results_count,
        deep_search_used,
        attempt_num,
    ):
        """Update embed messages during streaming."""
        if not final_text and not finish_reason:
            return
        current_msg_idx = (
            (len(final_text) - 1) // split_limit_config if final_text else 0
        )
        start_next_msg = current_msg_idx >= len(response_msgs)
        ready_to_edit = (self.edit_task is None or self.edit_task.done()) and (
            dt.now().timestamp() - self.client.last_task_time
            >= self.client.config.get(EDIT_DELAY_SECONDS_CONFIG_KEY, 1.0)
        )
        is_final_chunk = finish_reason is not None
        if start_next_msg or ready_to_edit or is_final_chunk:
            if self.edit_task and not self.edit_task.done():
                try:
                    await self.edit_task
                except (asyncio.CancelledError, Exception):
                    pass
                self.edit_task = None
            # Create embed for current segment
            current_embed = await self._create_segment_embed(
                final_text,
                current_msg_idx,
                split_limit_config,
                base_embed,
                finish_reason,
                current_params,
                grounding_metadata,
                custom_search_queries_generated,
                successful_api_results_count,
                deep_search_used,
                attempt_num,
                is_final_chunk,
            )
            # Create view for final successful chunk
            view_to_attach = None
            if is_final_chunk:
                is_successful = self._is_successful_finish(
                    finish_reason, current_params["provider"]
                )
                if is_successful:
                    view_to_attach = self._create_response_action_view(
                        new_msg,
                        final_text,
                        current_params,
                        grounding_metadata,
                        custom_search_queries_generated,
                        successful_api_results_count,
                    )
            # Send or edit message
            await self._send_or_edit_message(
                new_msg,
                processing_msg,
                response_msgs,
                current_embed,
                current_msg_idx,
                start_next_msg,
                is_final_chunk,
                final_text,
                view_to_attach,
            )
            self.client.last_task_time = dt.now().timestamp()

    async def _create_segment_embed(
        self,
        final_text,
        current_msg_idx,
        split_limit_config,
        base_embed,
        finish_reason,
        current_params,
        grounding_metadata,
        custom_search_queries_generated,
        successful_api_results_count,
        deep_search_used,
        attempt_num,
        is_final_chunk,
    ):
        """Create embed for current message segment."""
        current_display_text = final_text[
            current_msg_idx * split_limit_config : (current_msg_idx + 1)
            * split_limit_config
        ][:MAX_EMBED_DESCRIPTION_LENGTH]
        current_embed = discord.Embed()
        for field in base_embed.fields:
            current_embed.add_field(
                name=field.name, value=field.value, inline=field.inline
            )
        stripped_text = current_display_text.strip() if current_display_text else ""
        effective_text = stripped_text or "..."
        current_embed.description = (
            effective_text
            if is_final_chunk
            else f"{effective_text}{STREAMING_INDICATOR}"
        )
        is_successful = (
            self._is_successful_finish(finish_reason, current_params["provider"])
            if finish_reason
            else False
        )
        is_blocked = finish_reason and finish_reason.lower() in (
            "safety",
            "recitation",
            "other",
        )
        if is_final_chunk and is_blocked:
            block_msg = self._get_block_message(finish_reason)
            current_embed.description = (
                f"{stripped_text}\n\n{block_msg}" if stripped_text else block_msg
            )
            current_embed.description = current_embed.description[
                :MAX_EMBED_DESCRIPTION_LENGTH
            ]
            current_embed.color = EMBED_COLOR_ERROR
        else:
            current_embed.color = (
                EMBED_COLOR_COMPLETE
                if is_final_chunk and is_successful
                else EMBED_COLOR_INCOMPLETE
            )
            if is_final_chunk and is_successful:
                footer_text = self._create_footer_text(
                    current_params,
                    attempt_num,
                    grounding_metadata,
                    custom_search_queries_generated,
                    successful_api_results_count,
                    deep_search_used,
                )
                current_embed.set_footer(text=footer_text)
        return current_embed

    async def _send_or_edit_message(
        self,
        new_msg,
        processing_msg,
        response_msgs,
        current_embed,
        current_msg_idx,
        start_next_msg,
        is_final_chunk,
        final_text,
        view_to_attach,
    ):
        """Send new message or edit existing one."""
        target_msg_for_node_update = None
        if start_next_msg:
            if not response_msgs and processing_msg:
                await processing_msg.edit(
                    content=None, embed=current_embed, view=view_to_attach
                )
                response_msg = processing_msg
                target_msg_for_node_update = response_msg
                if view_to_attach:
                    view_to_attach.message = response_msg
            else:
                reply_target = new_msg if not response_msgs else response_msgs[-1]
                response_msg = await reply_target.reply(
                    embed=current_embed, view=view_to_attach, mention_author=False
                )
                target_msg_for_node_update = response_msg
                if view_to_attach:
                    view_to_attach.message = response_msg
            response_msgs.append(response_msg)
            if response_msg.id not in self.client.msg_nodes:
                self.client.msg_nodes[response_msg.id] = models.MsgNode(
                    parent_msg=new_msg
                )
        elif response_msgs and current_msg_idx < len(response_msgs):
            target_msg = response_msgs[current_msg_idx]
            if target_msg:
                self.edit_task = asyncio.create_task(
                    target_msg.edit(embed=current_embed, view=view_to_attach)
                )
                if view_to_attach:
                    view_to_attach.message = target_msg
                target_msg_for_node_update = target_msg
        elif not response_msgs and is_final_chunk:
            # Short final response, no prior stream msgs
            if processing_msg:
                await processing_msg.edit(
                    content=None, embed=current_embed, view=view_to_attach
                )
                response_msg = processing_msg
                target_msg_for_node_update = response_msg
                if view_to_attach:
                    view_to_attach.message = response_msg
            else:
                response_msg = await new_msg.reply(
                    embed=current_embed, view=view_to_attach, mention_author=False
                )
                target_msg_for_node_update = response_msg
                if view_to_attach:
                    view_to_attach.message = response_msg
            response_msgs.append(response_msg)
            if response_msg.id not in self.client.msg_nodes:
                self.client.msg_nodes[response_msg.id] = models.MsgNode(
                    parent_msg=new_msg
                )
        # Update full response text in node if this is the final chunk
        if (
            is_final_chunk
            and target_msg_for_node_update
            and target_msg_for_node_update.id in self.client.msg_nodes
        ):
            self.client.msg_nodes[
                target_msg_for_node_update.id
            ].full_response_text = final_text

    async def _send_plain_text_response(
        self,
        new_msg,
        processing_msg,
        response_msgs,
        final_text,
        split_limit_config,
        current_params,
        grounding_metadata,
        custom_search_queries_generated,
    ):
        """Send response as plain text messages."""
        final_messages_content = [
            final_text[i : i + split_limit_config]
            for i in range(0, len(final_text), split_limit_config)
        ]
        if not final_messages_content:
            final_messages_content.append("...")
        temp_response_msgs = []
        start_index = 0
        if processing_msg and (
            not response_msgs or processing_msg.id == response_msgs[0].id
        ):
            await processing_msg.edit(
                content=final_messages_content[0] or "...", embed=None, view=None
            )
            temp_response_msgs.append(processing_msg)
            if processing_msg.id not in self.client.msg_nodes:
                self.client.msg_nodes[processing_msg.id] = models.MsgNode(
                    parent_msg=new_msg, full_response_text=final_text
                )
            start_index = 1
        reply_target = temp_response_msgs[-1] if temp_response_msgs else new_msg
        for i in range(start_index, len(final_messages_content)):
            content_chunk = final_messages_content[i]
            # Create view for the final message chunk
            view_to_attach = None
            if i == len(final_messages_content) - 1:  # Last chunk
                # For plain text, we don't have grounding metadata, but we can still offer text file download
                view_to_attach = ResponseActionView(
                    grounding_metadata=grounding_metadata,
                    full_response_text=final_text,
                    model_name=current_params["model_name"],
                    app_config=self.app_config,
                    original_user_message=new_msg,
                    internet_used=custom_search_queries_generated,
                )
            response_msg = await reply_target.reply(
                content=content_chunk or "...",
                suppress_embeds=True,
                view=view_to_attach,
                mention_author=False,
            )
            temp_response_msgs.append(response_msg)
            if view_to_attach:
                view_to_attach.message = response_msg
            self.client.msg_nodes[response_msg.id] = models.MsgNode(parent_msg=new_msg)
            if i == len(final_messages_content) - 1:
                self.client.msg_nodes[response_msg.id].full_response_text = final_text
            reply_target = response_msg
        response_msgs[:] = temp_response_msgs

    def _is_successful_finish(self, finish_reason: str, provider: str) -> bool:
        """Check if finish reason indicates successful completion."""
        if not finish_reason:
            return False
        successful_reasons = {
            "stop",
            "end_turn",
            "content_filter",
            "length",
            "max_tokens",
        }
        if finish_reason.lower() in successful_reasons:
            return True
        if provider == "google" and finish_reason == str(
            google_types.FinishReason.FINISH_REASON_UNSPECIFIED
        ):
            return True
        return False

    def _get_block_message(self, finish_reason: str) -> str:
        """Get appropriate message for blocked responses."""
        reason_lower = finish_reason.lower()
        if reason_lower == "safety":
            return "⚠️ Response blocked by safety filters."
        elif reason_lower == "recitation":
            return "⚠️ Response stopped due to recitation."
        else:
            return f"⚠️ Response blocked (Reason: {finish_reason})."

    def _create_footer_text(
        self,
        current_params,
        attempt_num,
        grounding_metadata,
        custom_search_queries_generated,
        successful_api_results_count,
        deep_search_used,
    ) -> str:
        """Create footer text for final message."""
        footer_parts = [f"Model: {current_params['model_name']}"]
        if attempt_num == 1:
            footer_parts.append("(Retried)")
        internet_info = []
        if deep_search_used:
            internet_info.append("Deepsearch was used")
        elif grounding_metadata:
            internet_info.append("Internet used")
        elif custom_search_queries_generated:
            internet_info.extend(
                [
                    "Internet used",
                    f"{successful_api_results_count} search result{'s' if successful_api_results_count != 1 else ''} processed",
                ]
            )
        else:
            internet_info.append("Internet not used")
        if internet_info:
            footer_parts.append(" | ".join(internet_info))
        return " | ".join(footer_parts)

    def _create_response_action_view(
        self,
        new_msg,
        final_text,
        current_params,
        grounding_metadata,
        custom_search_queries_generated,
        successful_api_results_count,
    ):
        """Create a ResponseActionView for a final successful chunk."""
        # Determine if grounding metadata has sources
        has_sources = grounding_metadata and (
            getattr(grounding_metadata, "web_search_queries", None)
            or getattr(grounding_metadata, "grounding_chunks", None)
            or getattr(grounding_metadata, "search_entry_point", None)
        )
        has_text_content = bool(final_text)
        # Only create view if there's content to show or sources to display
        if has_sources or has_text_content:
            # Determine whether the response used the internet
            internet_used_flag = (
                has_sources
                if current_params["provider"]
                == "google"  # Gemini grounding implies web usage
                else custom_search_queries_generated
            )
            return ResponseActionView(
                grounding_metadata=grounding_metadata,
                full_response_text=final_text,
                model_name=current_params["model_name"],
                app_config=self.app_config,
                original_user_message=new_msg,
                internet_used=internet_used_flag,
            )
        return None
