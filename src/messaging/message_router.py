import logging
import discord
from typing import Optional, Tuple

from .message_parser import should_process_message
from ..bot.permissions import is_message_allowed


class MessageRouter:
    """Handles message routing and initial validation logic."""

    @staticmethod
    def should_handle_message(
        message: discord.Message, bot_user: discord.User, config: dict
    ) -> Tuple[bool, Optional[str], bool]:
        """
        Determine if the bot should handle this message and what type of handling.

        Returns:
            Tuple of (should_handle, original_content_for_processing, is_dm)
        """
        if message.author.bot or message.author == bot_user:
            return False, None, False

        is_dm = isinstance(message.channel, discord.DMChannel)
        allow_dms = config.get("allow_dms", True)

        should_process, original_content = should_process_message(
            message, bot_user, allow_dms, is_dm
        )

        return should_process, original_content, is_dm

    @staticmethod
    def check_permissions(message: discord.Message, config: dict, is_dm: bool) -> bool:
        """Check if message is allowed based on permissions."""
        if not is_message_allowed(message, config, is_dm):
            logging.warning(
                f"Blocked message from user {message.author.id} "
                f"in channel {message.channel.id} due to permissions."
            )
            return False
        return True

    @staticmethod
    def is_prefix_command(message: discord.Message, prefix: str) -> bool:
        """Check if message is a prefix command."""
        return message.content.startswith(prefix)

    @staticmethod
    def is_deep_search_query(cleaned_content: str) -> bool:
        """Check if message is a deep search query."""
        return cleaned_content.lower().strip().startswith("deepsearch")
