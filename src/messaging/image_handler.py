import asyncio
import io
import re
from typing import List, Dict, Optional, Any
import discord
from ..core.constants import (
    IMGUR_HEADER,
    IMGUR_URL_PATTERN,
    IMGUR_URL_PREFIX,
    MAX_PLAIN_TEXT_LENGTH,
)
from ..core import models
class ImageHandler:
    """Handles image generation and Imgur URL processing."""
    def __init__(self, client, app_config: Dict[str, Any]):
        self.client = client
        self.app_config = app_config
    async def process_image_stream(
        self, stream_generator, new_msg, processing_msg, response_msgs, final_text
    ):
        """Process image generation stream."""
        accumulated_image_data = None
        accumulated_image_mime_type = None
        success = False
        async for (text_chunk, finish_reason, chunk_grounding_metadata,
                   error_message, image_data, image_mime_type) in stream_generator:
            if error_message:
                return {'success': False, 'text': final_text, 'response_msgs': response_msgs}
            if text_chunk:
                final_text += text_chunk if isinstance(text_chunk, str) else "".join(text_chunk)
            if image_data and image_mime_type:
                accumulated_image_data = image_data
                accumulated_image_mime_type = image_mime_type
            if finish_reason:
                success = self._is_successful_finish(finish_reason)
                break
        if success and accumulated_image_data:
            await self._send_generated_image(
                new_msg, processing_msg, response_msgs, 
                accumulated_image_data, accumulated_image_mime_type, final_text
            )
        return {
            'success': success,
            'text': final_text,
            'response_msgs': response_msgs,
            'should_retry': False
        }
    async def _send_generated_image(
        self, new_msg, processing_msg, response_msgs, 
        image_data, image_mime_type, final_text
    ):
        """Send the generated image as a Discord message."""
        try:
            # Create a file from the image data
            image_file = io.BytesIO(image_data)
            # Determine file extension from mime type
            file_extension = "png"  # default
            if "jpeg" in image_mime_type or "jpg" in image_mime_type:
                file_extension = "jpg"
            elif "webp" in image_mime_type:
                file_extension = "webp"
            elif "gif" in image_mime_type:
                file_extension = "gif"
            filename = f"generated_image.{file_extension}"
            discord_file = discord.File(fp=image_file, filename=filename)
            # Send the image as a reply with any text content
            content_to_send = final_text.strip() if final_text.strip() else None
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
                reply_target = response_msgs[-1] if response_msgs else new_msg
                response_msg = await reply_target.reply(
                    content=content_to_send,
                    file=discord_file,
                    mention_author=False,
                )
            response_msgs.append(response_msg)
            # Update msg_nodes cache
            if response_msg.id not in self.client.msg_nodes:
                self.client.msg_nodes[response_msg.id] = models.MsgNode(parent_msg=new_msg)
                self.client.msg_nodes[response_msg.id].full_response_text = final_text
        except Exception as e:
            # Fall back to text-only response
            if processing_msg:
                await processing_msg.edit(
                    content=f"Generated image but failed to send: {str(e)}",
                    embed=None,
                    view=None,
                )
    async def resend_imgur_urls(
        self,
        new_msg: discord.Message,
        response_msgs: List[discord.Message],
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
            await self._send_imgur_urls(new_msg, response_msgs, imgur_urls_to_resend)
    async def _send_imgur_urls(
        self, new_msg: discord.Message, response_msgs: List[discord.Message], 
        imgur_urls: List[str]
    ):
        """Send Imgur URLs as separate messages."""
        max_chars = MAX_PLAIN_TEXT_LENGTH
        messages_to_send_content = []
        current_message_content = ""
        for url_str in imgur_urls:
            needed_len = len(url_str) + (2 if current_message_content else 0)
            if len(current_message_content) + needed_len > max_chars:
                if current_message_content:
                    messages_to_send_content.append(current_message_content)
                current_message_content = url_str[:max_chars] if len(url_str) > max_chars else url_str
                if len(url_str) > max_chars:
                    break
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
                    sent_msg = await target_to_reply_to.reply(
                        content=msg_content, mention_author=False
                    )
                    last_sent_msg = sent_msg
                else:
                    sent_msg = await new_msg.reply(content=msg_content, mention_author=False)
                    last_sent_msg = sent_msg
                await asyncio.sleep(0.1)
            except discord.HTTPException as send_err:
                try:
                    await new_msg.reply(
                        f"(Error sending previous chunk)\n{msg_content}",
                        mention_author=False,
                    )
                except discord.HTTPException as fallback_err:
                    pass
            except Exception as e:
                pass
    def _is_successful_finish(self, finish_reason: str) -> bool:
        """Check if finish reason indicates successful completion."""
        if not finish_reason:
            return False
        successful_reasons = {"stop", "end_turn", "content_filter", "length", "max_tokens"}
        return finish_reason.lower() in successful_reasons 