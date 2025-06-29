import logging
import time
from typing import Dict

import discord
from discord import app_commands

# Use google.genai.types

from ..core import models
from ..core.rate_limiter import check_and_perform_global_reset, close_all_db_managers
from ..messaging.message_router import MessageRouter
from ..messaging.prefix_commands import PrefixCommandHandler
from ..messaging.deep_search_handler import DeepSearchHandler
from ..messaging.standard_message_handler import StandardMessageHandler
from ..messaging.response_sender import send_initial_processing_message
from .commands import (
    set_model_command,
    set_system_prompt_command,
    setgeminithinking,
    help_command,
    load_all_preferences,
    enhance_prompt_command,
)


# --- Discord Client Setup ---
class LLMCordClient(discord.Client):
    def __init__(
        self,
        *,
        intents: discord.Intents,
        activity: discord.CustomActivity,
        config: Dict,
    ):
        super().__init__(intents=intents, activity=activity)
        self.tree = app_commands.CommandTree(self)
        self.msg_nodes: Dict[int, models.MsgNode] = {}  # Message cache
        self.last_task_time: float = 0  # For stream editing delay
        self.config = config  # Store loaded config

        # Use the **shared** HTTPX client for the whole application
        from ..core.http_client import get_httpx_client

        self.httpx_client = get_httpx_client(
            self.config
        )  # HTTP client for attachments/web

        # Initialize content fetcher modules that need config
        self._initialize_content_fetchers()

    def _initialize_content_fetchers(self):
        """Initialize content fetcher modules with config."""
        from ..content.fetchers.youtube import (
            initialize_ytt_api,
            initialize_youtube_data_api,
        )
        from ..content.fetchers.reddit import initialize_reddit_client

        initialize_ytt_api(self.config.get("proxy_config"))
        initialize_youtube_data_api(self.config.get("youtube_api_key"))
        initialize_reddit_client(
            client_id=self.config.get("reddit_client_id"),
            client_secret=self.config.get("reddit_client_secret"),
            user_agent=self.config.get("reddit_user_agent"),
        )

    async def setup_hook(self):
        """Sync slash commands and load preferences when the bot is ready."""
        await load_all_preferences()

        # Register slash commands
        commands_to_register = [
            ("model", "Set your preferred LLM provider and model.", set_model_command),
            (
                "systemprompt",
                "Set your custom system prompt for the bot.",
                set_system_prompt_command,
            ),
            (
                "setgeminithinking",
                "Toggle usage of the 'thinkingBudget' parameter for Gemini models.",
                setgeminithinking,
            ),
            (
                "help",
                "Displays all available commands and how to use them.",
                help_command,
            ),
            (
                "enhanceprompt",
                "Enhances a given prompt using an LLM.",
                enhance_prompt_command,
            ),
        ]

        for name, description, callback in commands_to_register:
            self.tree.add_command(
                app_commands.Command(
                    name=name, description=description, callback=callback
                )
            )

        await self.tree.sync()
        logging.info(f"Synced slash commands for {self.user}.")

    async def on_ready(self):
        """Called when the bot is ready and logged in."""
        logging.info(f"Logged in as {self.user}")
        print(f"✅ LLMCord bot is now running as {self.user}!")
        await check_and_perform_global_reset(self.config)

    async def on_message(self, new_msg: discord.Message):
        """Handles incoming messages."""
        start_time = time.time()

        # Check if we should handle this message
        should_handle, original_content, is_dm = MessageRouter.should_handle_message(
            new_msg, self.user, self.config
        )

        if not should_handle:
            # Handle prefix commands even if not processing other messages
            if MessageRouter.is_prefix_command(
                new_msg, PrefixCommandHandler.ENHANCE_CMD_PREFIX
            ):
                if MessageRouter.check_permissions(new_msg, self.config, is_dm):
                    await PrefixCommandHandler.handle_enhance_prompt_command(
                        new_msg, self
                    )
            return

        # Check global reset and permissions
        await check_and_perform_global_reset(self.config)
        if not MessageRouter.check_permissions(new_msg, self.config, is_dm):
            return

        # Send initial processing message
        use_plain_for_initial_status = self.config.get("use_plain_responses", False)
        processing_msg = await send_initial_processing_message(
            new_msg, use_plain_for_initial_status
        )

        # Route to appropriate handler
        try:
            if await self._handle_special_messages(
                new_msg, original_content, processing_msg, start_time
            ):
                return

            # Handle standard messages
            await StandardMessageHandler.process_standard_message(
                new_msg, original_content, processing_msg, self, start_time
            )

        except Exception as e:
            logging.error(f"Error processing message {new_msg.id}: {e}", exc_info=True)
            try:
                await new_msg.channel.send(
                    f"❌ An error occurred while processing your message: {str(e)}"
                )
            except Exception:
                pass  # Ignore errors when sending error messages

    async def _handle_special_messages(
        self,
        new_msg: discord.Message,
        original_content: str,
        processing_msg: discord.Message,
        start_time: float,
    ) -> bool:
        """
        Handle special message types (deep search, etc.).

        Returns:
            True if message was handled, False if it should continue to standard processing
        """
        from ..messaging.message_parser import clean_message_content

        cleaned_content = clean_message_content(
            original_content,
            self.user.mention if self.user else None,
            isinstance(new_msg.channel, discord.DMChannel),
        )

        # Handle deep search queries
        if MessageRouter.is_deep_search_query(cleaned_content):
            await self._handle_deep_search(
                new_msg, cleaned_content, processing_msg, start_time
            )
            return True

        return False

    async def _handle_deep_search(
        self,
        new_msg: discord.Message,
        cleaned_content: str,
        processing_msg: discord.Message,
        start_time: float,
    ) -> None:
        """Handle deep search queries."""
        use_plain_for_initial_status = self.config.get("use_plain_responses", False)

        # Process deep search query
        modified_content, topic = await DeepSearchHandler.process_deep_search_query(
            new_msg,
            cleaned_content,
            processing_msg,
            use_plain_for_initial_status,
            self.config,
        )

        # Create a modified message object for processing
        modified_msg = self._create_modified_message(new_msg, modified_content)

        # Process through standard handler with modified content and disabled grounding
        await StandardMessageHandler.process_standard_message(
            modified_msg,
            modified_content,
            processing_msg,
            self,
            start_time,
            disable_grounding=True,  # Disable grounding/tools for deep search
            is_deep_search=True,  # Mark as deep search for proper parameter passing
        )

    def _create_modified_message(self, original_msg: discord.Message, new_content: str):
        """Create a message-like object with modified content for deep search processing."""

        class ModifiedMessage:
            def __init__(self, original, content):
                # Copy essential attributes
                for attr in [
                    "author",
                    "channel",
                    "guild",
                    "id",
                    "attachments",
                    "reference",
                    "created_at",
                    "edited_at",
                    "mention_everyone",
                    "mentions",
                    "channel_mentions",
                    "role_mentions",
                    "pinned",
                    "flags",
                    "type",
                ]:
                    setattr(self, attr, getattr(original, attr, None))

                self.content = content
                self.clean_content = content

                # Set optional attributes with defaults
                optional_attrs = [
                    "embeds",
                    "reactions",
                    "webhook_id",
                    "application_id",
                    "activity",
                    "application",
                    "stickers",
                    "components",
                    "thread",
                    "interaction",
                    "role_subscription",
                    "resolved",
                    "position",
                    "poll",
                    "call",
                    "system_content",
                ]
                for attr in optional_attrs:
                    setattr(
                        self,
                        attr,
                        getattr(original, attr, [] if attr.endswith("s") else None),
                    )

            def reply(self, *args, **kwargs):
                return self.channel.send(*args, **kwargs)

        return ModifiedMessage(original_msg, new_content)

    async def close(self):
        """Clean up resources when the bot is shutting down."""
        logging.info("Closing HTTPX client...")
        from ..core.http_client import close_httpx_client

        await close_httpx_client()

        # Close Reddit client
        from ..content.fetchers.reddit import reddit_client_instance

        if reddit_client_instance:
            logging.info("Closing Reddit client...")
            try:
                await reddit_client_instance.close()
            except Exception as e:
                logging.error(f"Error closing Reddit client: {e}")

        logging.info("Closing database connections...")
        close_all_db_managers()
        await super().close()

    async def retry_with_modified_content(
        self, original_message: discord.Message, content_prefix: str
    ):
        """
        Retry processing a message with additional content prepended.

        Args:
            original_message: The original Discord message to retry
            content_prefix: Text to prepend to the original message content
        """
        try:
            modified_content = (
                f"user query: {original_message.content}\n\n{content_prefix}"
            )
            retry_message = self._create_modified_message(
                original_message, modified_content
            )
            await self.on_message(retry_message)
        except Exception as e:
            logging.error(f"Error in retry_with_modified_content: {e}", exc_info=True)
            try:
                await original_message.channel.send(
                    f"❌ Failed to retry the request: {str(e)}"
                )
            except Exception:
                pass
