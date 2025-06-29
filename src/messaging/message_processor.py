import asyncio
import time
from datetime import datetime
from typing import Dict, Set, Tuple, Optional, Any
import discord
from ..core.constants import (
    MAX_EMBED_DESCRIPTION_LENGTH,
    MAX_PLAIN_TEXT_LENGTH,
    EMBED_COLOR_INCOMPLETE,
    GEMINI_USE_THINKING_BUDGET_CONFIG_KEY,
    GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY,
    MAX_MESSAGE_NODES_CONFIG_KEY,
from ..core import models
from ..core.rate_limiter import check_and_perform_global_reset
from ..core.config_loader import get_max_text_for_model
from ..core.utils import (
    extract_urls_with_indices,
    is_image_url,
    extract_text_from_pdf_bytes, 
    extract_text_from_docx_bytes,
from ..llm.model_selector import determine_final_model
from .message_parser import (
    should_process_message,
    clean_message_content, 
    check_google_lens_trigger,
from ..content.processor import process_content_and_grounding
from ..bot.permissions import is_message_allowed
from ..services.prompt_utils import prepare_system_prompt
from .response_sender import (
    send_initial_processing_message,
    handle_llm_response_stream,
from .history_utils import build_message_history
from ..research.deep_research import perform_deep_research_async
class MessageProcessor:
    """Handles Discord message processing logic."""
    pass
    def __init__(self, bot_client):
        self.bot = bot_client
        self.config = bot_client.config
    pass
    async def process_message(self, message: discord.Message) -> None:
        """
        Main message processing entry point.
        pass
        Args:
            message: The Discord message to process
        """
        start_time = time.time()
        pass
        # Basic validation
        if message.author.bot or message.author == self.bot.user:
            return
        pass
        is_dm = isinstance(message.channel, discord.DMChannel)
        allow_dms = self.config.get("allow_dms", True)
        pass
        # Check if should process message
        should_process, original_content = should_process_message(
            message, self.bot.user, allow_dms, is_dm
        pass
        if not should_process:
            # Handle prefix commands
            if await self._handle_prefix_commands(message, is_dm):
                return
            return
        pass
        # Process the message
        await self._process_regular_message(message, original_content, is_dm, start_time)
    pass
    async def _handle_prefix_commands(self, message: discord.Message, is_dm: bool) -> bool:
        """
        Handle prefix commands like !enhanceprompt.
        pass
        Returns:
            bool: True if a prefix command was handled, False otherwise
        """
        ENHANCE_CMD_PREFIX = "!enhanceprompt"
        pass
        if not message.content.startswith(ENHANCE_CMD_PREFIX):
            return False
        pass
        # Check permissions
        if not is_message_allowed(message, self.config, is_dm):
            logger.warning(
                f"Blocked {ENHANCE_CMD_PREFIX} from user {message.author.id} "
                f"in channel {message.channel.id} due to permissions."
            return True
        pass
        await self._handle_enhance_prompt_command(message, ENHANCE_CMD_PREFIX)
        return True
    pass
    async def _handle_enhance_prompt_command(self, message: discord.Message, cmd_prefix: str):
        """Handle the !enhanceprompt prefix command."""
        from ..bot.prompt_enhancer import execute_enhance_prompt_logic
        pass
        # Extract text after command prefix
        text_content = message.content[len(cmd_prefix):].strip()
        pass
        prompt_text = ""
        processed_from_file = False
        attachment_filename = None
        initial_status = "⏳ Enhancing your prompt..."
        pass
        # Process attachments if present
        if message.attachments:
            prompt_text, processed_from_file, attachment_filename, initial_status = (
                await self._process_attachment_for_enhance_command(message.attachments)
            if prompt_text is None:  # Error occurred
                return
        pass
        # Use text content if no file processed
        if not processed_from_file:
            prompt_text = text_content
        pass
        # Validate we have content to enhance
        if not prompt_text.strip():
            error_msg = self._get_enhance_command_error_message(
                message.attachments, processed_from_file, cmd_prefix
            await message.reply(error_msg, mention_author=False)
            return
        pass
        # Send initial status and execute enhancement
        try:
            await message.reply(initial_status, mention_author=False)
        except discord.HTTPException as e:
            pass
        pass
        await execute_enhance_prompt_logic(message, prompt_text, self.bot)
    pass
    async def _process_attachment_for_enhance_command(self, attachments) -> Tuple[Optional[str], bool, Optional[str], str]:
        """
        Process attachments for enhance command.
        pass
        Returns:
            Tuple of (prompt_text, processed_from_file, attachment_filename, initial_status)
            Returns (None, False, None, "") if error occurred
        """
        for attachment in attachments:
            # Check for text files
            is_text_file = (
                (attachment.content_type and attachment.content_type.startswith("text/")) or
                (attachment.content_type == "application/octet-stream" and 
                 attachment.filename.lower().endswith(('.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv')))
            pass
            # Check for DOCX files  
            is_docx_file = (
                attachment.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or
                attachment.filename.lower().endswith(".docx")
            pass
            if is_text_file:
                try:
                    file_bytes = await attachment.read()
                    prompt_text = file_bytes.decode("utf-8", errors="replace")
                    initial_status = f"⏳ Enhancing prompt from file `{discord.utils.escape_markdown(attachment.filename)}`..."
                    return prompt_text, True, attachment.filename, initial_status
                except Exception as e:
                    await message.reply(f"Sorry, I couldn't read the file: {attachment.filename}.", mention_author=False)
                    return None, False, None, ""
            pass
            elif is_docx_file:
                try:
                    file_bytes = await attachment.read()
                    prompt_text = await extract_text_from_docx_bytes(file_bytes)
                    if prompt_text:
                        initial_status = f"⏳ Enhancing prompt from DOCX file `{discord.utils.escape_markdown(attachment.filename)}`..."
                        return prompt_text, True, attachment.filename, initial_status
                    else:
                except Exception as e:
                    await message.reply(f"Sorry, I couldn't read the DOCX file: {attachment.filename}.", mention_author=False)
                    return None, False, None, ""
        pass
        return "", False, None, "⏳ Enhancing your prompt..."
    pass
    def _get_enhance_command_error_message(self, attachments, processed_from_file: bool, cmd_prefix: str) -> str:
        """Get appropriate error message for enhance command."""
        if attachments and not processed_from_file:
            return (
                f"No suitable text file found in attachments for `{cmd_prefix}`. "
                "Please provide a prompt as text or attach a recognized text file "
                "(.txt, .py, .md, etc.) or DOCX file."
        else:
            return (
                f"Please provide a prompt to enhance (either as text after `{cmd_prefix}` "
                "or as a text file attachment)."
    pass
    async def _process_regular_message(
        self, 
        message: discord.Message, 
        original_content: str, 
        is_dm: bool, 
        start_time: float
    ) -> None:
        """Process a regular bot message (not a prefix command)."""
        # Check global rate limits
        await check_and_perform_global_reset(self.config)
        pass
        # Check permissions
        if not is_message_allowed(message, self.config, is_dm):
            logger.warning(
                f"Blocked message from user {message.author.id} "
                f"in channel {message.channel.id} due to permissions."
            return
        pass
        # Send initial processing message
        use_plain_responses = self.config.get("use_plain_responses", False)
        processing_msg = await send_initial_processing_message(message, use_plain_responses)
        pass
        # Parse and clean message content
        cleaned_content = clean_message_content(
            original_content,
            self.bot.user.mention if self.bot.user else None,
            is_dm,
        pass
        # Process image attachments and Google Lens
        image_attachments = [
            att for att in message.attachments 
            if att.content_type and att.content_type.startswith("image/")
        ]
        pass
        user_warnings: Set[str] = set()
        use_google_lens, cleaned_content, lens_warning = check_google_lens_trigger(
            cleaned_content, image_attachments, self.config
        pass
        if lens_warning:
            user_warnings.add(lens_warning)
        if use_google_lens:
            pass
        pass
        # Check for image URLs in text
        has_potential_image_urls = self._check_for_image_urls(message, cleaned_content)
        pass
        # Handle deep search queries
        if cleaned_content.lower().strip().startswith("deepsearch"):
            await self._handle_deep_search(message, cleaned_content, processing_msg, use_plain_responses)
            return
        pass
        # Process regular message flow
        await self._handle_regular_flow(
            message, cleaned_content, image_attachments, has_potential_image_urls,
            user_warnings, use_google_lens, processing_msg, start_time
    pass
    def _check_for_image_urls(self, message: discord.Message, cleaned_content: str) -> bool:
        """Check for potential image URLs in message text and replied messages."""
        # Check current message content
        if cleaned_content:
            urls_in_text = extract_urls_with_indices(cleaned_content)
            if any(is_image_url(url_info[0]) for url_info in urls_in_text):
                return True
        pass
        # Check replied-to message
        if message.reference and message.reference.message_id:
            try:
                referenced_msg = message.reference.cached_message
                if not referenced_msg:
                    # This would normally be an async call, but we're keeping the structure simple
                    pass  # In real implementation, we'd await the fetch
                pass
                if referenced_msg and referenced_msg.content:
                    urls_in_replied = extract_urls_with_indices(referenced_msg.content)
                    if any(is_image_url(url_info[0]) for url_info in urls_in_replied):
                        return True
            except Exception:
                pass  # Continue without referenced message if error
        pass
        return False
    pass
    async def _handle_deep_search(
        self, 
        message: discord.Message, 
        cleaned_content: str, 
        processing_msg: Optional[discord.Message],
        use_plain_responses: bool
    ) -> None:
        """Handle deep search queries."""
        # Extract topic and add current date
        current_date = datetime.utcnow().strftime("%Y-%m-%d")
        topic_raw = cleaned_content[len("deepsearch"):].strip() or "general"
        topic = f"{topic_raw} {current_date}"
        pass
        # Update status message
        try:
            status_text = "⏳ Performing deep search, this may take a while..."
            if processing_msg:
                pass
                if use_plain_responses:
                    await processing_msg.edit(content=status_text)
                else:
                    embed = discord.Embed(
                        description=status_text,
                        color=EMBED_COLOR_INCOMPLETE,
                    await processing_msg.edit(embed=embed)
        except discord.HTTPException as e:
            pass
        pass
        # Perform deep research
        try:
            deep_research_output = await perform_deep_research_async(topic, self.config)
            pass
            # Build enhanced prompt for LLM
            user_query_with_date = f"{message.content.strip()} {current_date}"
            enhanced_content = (
                "Answer the query based on the deep research report.\n\n"
                f"user query:\n{user_query_with_date}\n\n"
                "deep research output:\n"
                f"{deep_research_output}"
            pass
            # Process with LLM (simplified flow for deep search)
            await self._process_deep_search_response(message, enhanced_content)
            pass
        except Exception as e:
            error_msg = "Sorry, there was an error performing the deep search."
            if processing_msg:
                pass
                if use_plain_responses:
                    await processing_msg.edit(content=error_msg)
                else:
                    error_embed = discord.Embed(description=error_msg, color=0xff0000)
                    await processing_msg.edit(embed=error_embed)
    pass
    async def _process_deep_search_response(self, message: discord.Message, enhanced_content: str):
        """Process the deep search response through the LLM."""
        # This would continue with the normal LLM flow
        # For now, just log that we would process it
        # Implementation would continue with model selection and response handling
    pass
    async def _handle_regular_flow(
        self,
        message: discord.Message,
        cleaned_content: str,
        image_attachments: list,
        has_potential_image_urls: bool,
        user_warnings: Set[str],
        use_google_lens: bool,
        processing_msg: Optional[discord.Message],
        start_time: float
    ):
        """Handle the regular message processing flow."""
        # This would contain the rest of the regular message processing logic
        # Including model selection, content processing, and response handling
        # Implementation would continue with the existing logic from bot.py 