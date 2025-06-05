import re
from typing import Tuple, Optional, List, Dict, Any
import discord

from ..core.constants import AT_AI_PATTERN, GOOGLE_LENS_PATTERN


def should_process_message(
    message: discord.Message,
    bot_user: discord.User,
    allow_dms: bool,
    is_dm_channel: bool,
) -> Tuple[bool, str]:
    """
    Determines if the bot should process this message and returns the original content.
    """
    original_content = message.content
    mentions_bot = bot_user.mentioned_in(message)
    contains_at_ai = AT_AI_PATTERN.search(original_content) is not None

    if is_dm_channel:
        if not allow_dms:
            return False, original_content  # DMs not allowed

        # In DMs, process unless it's a reply *not* to the bot and doesn't trigger
        is_reply_to_other_user_in_dm = (
            message.reference
            and message.reference.resolved
            and message.reference.resolved.author != bot_user
        )

        if is_reply_to_other_user_in_dm and not (mentions_bot or contains_at_ai):
            return False, original_content
        return True, original_content

    # In guild channels, only process if explicitly triggered
    if mentions_bot or contains_at_ai:
        return True, original_content

    return False, original_content


def clean_message_content(
    content: str, bot_user_mention: Optional[str], is_dm: bool
) -> str:
    """
    Cleans the message content by removing bot mentions and "at ai" patterns.
    """
    cleaned = content
    if not is_dm and bot_user_mention and bot_user_mention in cleaned:
        cleaned = cleaned.replace(bot_user_mention, "").strip()
    cleaned = AT_AI_PATTERN.sub(" ", cleaned).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def check_google_lens_trigger(
    cleaned_content: str,
    image_attachments: List[discord.Attachment],
    config: Dict[str, Any],
) -> Tuple[bool, str, Optional[str]]:
    """
    Checks for Google Lens trigger, validates config, and returns updated content.
    Returns: (use_google_lens, updated_cleaned_content, warning_message)
    """
    use_lens = False
    warning = None
    updated_content = cleaned_content

    if GOOGLE_LENS_PATTERN.match(cleaned_content) and image_attachments:
        if not config.get("serpapi_api_keys"):
            warning = "⚠️ Google Lens requested but requires SerpAPI key configuration."
        else:
            use_lens = True
            updated_content = GOOGLE_LENS_PATTERN.sub("", cleaned_content).strip()
    return use_lens, updated_content, warning
