import discord
from typing import Optional
from ..core.utils import extract_text_from_docx_bytes
from ..bot.prompt_enhancer import execute_enhance_prompt_logic
class PrefixCommandHandler:
    """Handles prefix-based commands like !enhanceprompt."""
    pass
    ENHANCE_CMD_PREFIX = "!enhanceprompt"
    pass
    @classmethod
    async def handle_enhance_prompt_command(
        cls, 
        message: discord.Message, 
        bot_client
    ) -> bool:
        """
        Handle the !enhanceprompt prefix command.
        pass
        Returns:
            True if the command was handled, False otherwise
        """
        if not message.content.startswith(cls.ENHANCE_CMD_PREFIX):
            return False
            pass
        # Extract text after the command prefix
        text_content = message.content[len(cls.ENHANCE_CMD_PREFIX):].strip()
        pass
        prompt_text = ""
        processed_from_file = False
        attachment_filename = None
        initial_status_message = "⏳ Enhancing your prompt..."
        pass
        # Process attachments if present
        if message.attachments:
            prompt_text, processed_from_file, attachment_filename, initial_status_message = (
                await cls._process_attachments(message.attachments, cls.ENHANCE_CMD_PREFIX)
            if prompt_text is None:  # Error occurred
                return True  # Command was handled (even if it failed)
        pass
        # Use text content if no file was processed
        if not processed_from_file:
            prompt_text = text_content
        pass
        # Validate that we have a prompt
        if not prompt_text.strip():
            await cls._send_prompt_required_message(message, processed_from_file)
            return True
        pass
        # Send initial status message
        try:
            await message.reply(initial_status_message, mention_author=False)
        except discord.HTTPException as e:
            pass
        pass
        # Execute the enhancement logic
        await execute_enhance_prompt_logic(message, prompt_text, bot_client)
        return True
    pass
    @staticmethod
    async def _process_attachments(attachments, command_prefix):
        """Process message attachments for text extraction."""
        prompt_text = ""
        processed_from_file = False
        attachment_filename = None
        initial_status_message = "⏳ Enhancing your prompt..."
        pass
        for attachment in attachments:
            # Check for text files
            is_text_content_type = (
                attachment.content_type and 
                attachment.content_type.startswith("text/")
            is_common_text_extension = attachment.filename.lower().endswith((
                ".txt", ".md", ".py", ".js", ".html", ".css", 
                ".json", ".xml", ".csv"
            ))
            pass
            # Check for DOCX files
            is_docx_file = (
                attachment.content_type == 
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or
                attachment.filename.lower().endswith(".docx")
            pass
            if is_text_content_type or (
                attachment.content_type == "application/octet-stream" and 
                is_common_text_extension
            ):
                try:
                    file_bytes = await attachment.read()
                    prompt_text = file_bytes.decode("utf-8", errors="replace")
                    attachment_filename = attachment.filename
                    processed_from_file = True
                    initial_status_message = (
                        f"⏳ Enhancing prompt from file "
                        f"`{discord.utils.escape_markdown(attachment_filename)}`..."
                    break
                except Exception as e:
                    # Return None to indicate error
                    return None, False, None, None
                    pass
            elif is_docx_file:
                try:
                    file_bytes = await attachment.read()
                    prompt_text = await extract_text_from_docx_bytes(file_bytes)
                    if prompt_text:
                        attachment_filename = attachment.filename
                        processed_from_file = True
                        initial_status_message = (
                            f"⏳ Enhancing prompt from DOCX file "
                            f"`{discord.utils.escape_markdown(attachment_filename)}`..."
                        break
                    else:
                except Exception as e:
                    # Return None to indicate error
                    return None, False, None, None
        pass
        return prompt_text, processed_from_file, attachment_filename, initial_status_message
    pass
    @staticmethod
    async def _send_prompt_required_message(message: discord.Message, had_attachments: bool):
        """Send appropriate error message when no prompt is provided."""
        if had_attachments:
            error_msg = (
                f"No suitable text file found in attachments for `{PrefixCommandHandler.ENHANCE_CMD_PREFIX}`. "
                "Please provide a prompt as text or attach a recognized text file "
                "(.txt, .py, .md, etc.) or DOCX file."
        else:
            error_msg = (
                f"Please provide a prompt to enhance (either as text after "
                f"`{PrefixCommandHandler.ENHANCE_CMD_PREFIX}` or as a text file attachment)."
        pass
        await message.reply(error_msg, mention_author=False) 