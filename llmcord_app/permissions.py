# llmcord_app/permissions.py
import discord
from typing import Dict


def is_message_allowed(message: discord.Message, config: Dict, is_dm: bool) -> bool:
    """Checks if the user and channel are allowed based on config."""
    permissions = config.get("permissions", {})
    user_perms = permissions.get("users", {"allowed_ids": [], "blocked_ids": []})
    role_perms = permissions.get("roles", {"allowed_ids": [], "blocked_ids": []})
    channel_perms = permissions.get("channels", {"allowed_ids": [], "blocked_ids": []})

    # Ensure lists exist and are lists
    allowed_user_ids = user_perms.get("allowed_ids", []) or []
    blocked_user_ids = user_perms.get("blocked_ids", []) or []
    allowed_role_ids = role_perms.get("allowed_ids", []) or []
    blocked_role_ids = role_perms.get("blocked_ids", []) or []
    allowed_channel_ids = channel_perms.get("allowed_ids", []) or []
    blocked_channel_ids = channel_perms.get("blocked_ids", []) or []

    role_ids = set(role.id for role in getattr(message.author, "roles", []))
    # For channel_ids, ensure we are checking the correct attributes.
    # message.channel.id is always present.
    # message.channel.category_id is for text channels within categories.
    # message.channel.parent_id is for threads, pointing to the parent channel.
    current_channel_ids = {message.channel.id}
    if hasattr(message.channel, "category_id") and message.channel.category_id:
        current_channel_ids.add(message.channel.category_id)
    if isinstance(message.channel, discord.Thread) and message.channel.parent_id:
        current_channel_ids.add(message.channel.parent_id)
        # Also check the category of the parent channel if the thread is in one
        if (
            hasattr(message.channel.parent, "category_id")
            and message.channel.parent.category_id
        ):
            current_channel_ids.add(message.channel.parent.category_id)

    # User check
    allow_all_users = (
        not allowed_user_ids and not allowed_role_ids
    )  # True if both allowed lists are empty

    user_is_explicitly_allowed = message.author.id in allowed_user_ids or any(
        id in allowed_role_ids for id in role_ids
    )
    is_good_user = allow_all_users or user_is_explicitly_allowed
    is_bad_user = (message.author.id in blocked_user_ids) or any(
        id in blocked_role_ids for id in role_ids
    )

    if (
        is_bad_user or not is_good_user
    ):  # If blocked OR (not allow_all_users AND not user_is_explicitly_allowed)
        return False

    # Channel check (only if not DM)
    if not is_dm:
        allow_all_channels = (
            not allowed_channel_ids
        )  # True if allowed_channel_ids is empty

        channel_is_explicitly_allowed = any(
            id in allowed_channel_ids for id in current_channel_ids
        )
        is_good_channel = allow_all_channels or channel_is_explicitly_allowed
        is_bad_channel = any(id in blocked_channel_ids for id in current_channel_ids)

        if (
            is_bad_channel or not is_good_channel
        ):  # If blocked OR (not allow_all_channels AND not channel_is_explicitly_allowed)
            return False

    return True  # Allowed if checks pass
