from typing import List, Dict, Optional, Any
import discord
from ..core.constants import (
    EMBED_COLOR_ERROR,
    STREAMING_INDICATOR,
    MAX_EMBED_DESCRIPTION_LENGTH,
    AllKeysFailedError,
)
class StreamErrorHandler:
    """Handles various error scenarios during streaming."""
    pass
    def __init__(self, client, app_config: Dict[str, Any]):
        self.client = client
        self.app_config = app_config
    async def handle_stream_error(
        self,
        new_msg: discord.Message,
        processing_msg: Optional[discord.Message],
        response_msgs: List[discord.Message],
        error_message: str,
        use_plain_responses_config: bool,
    ):
        """Handles errors that occur mid-stream."""
        if not use_plain_responses_config and response_msgs and response_msgs[-1].embeds:
            try:
                embed = response_msgs[-1].embeds[0]
                embed.description = (
                    (embed.description or "").replace(STREAMING_INDICATOR, "").strip()
                )
                embed.description += f"\n\n⚠️ Error: {error_message}"
                embed.description = embed.description[:MAX_EMBED_DESCRIPTION_LENGTH]
                embed.color = EMBED_COLOR_ERROR
                await response_msgs[-1].edit(embed=embed, view=None)
            except Exception as edit_err:
                # Fallback reply
                target = (
                    processing_msg
                    if processing_msg and processing_msg.id == response_msgs[-1].id
                    else new_msg
                )
                await target.reply(
                    f"⚠️ Error during response generation: {error_message}",
                    mention_author=False,
                )
        else:
            error_text_plain = f"⚠️ Error during response generation: {error_message}"
            if processing_msg:
                await processing_msg.edit(content=error_text_plain, embed=None, view=None)
            else:
                await new_msg.reply(error_text_plain, mention_author=False)
    async def handle_llm_exception(
        self,
        new_msg: discord.Message,
        processing_msg: Optional[discord.Message],
        response_msgs: List[discord.Message],
        error_text: str,
        use_plain_responses_stream: bool,  # Whether the stream was plain
        use_plain_initial_status: bool,  # Whether the initial status message was plain
    ):
        """Handles exceptions from the LLM call (e.g., AllKeysFailedError)."""
        if processing_msg:
            if use_plain_initial_status:
                await processing_msg.edit(content=error_text, embed=None, view=None)
            else:  # Initial status was an embed
                if (
                    not use_plain_responses_stream
                    and response_msgs
                    and response_msgs[-1].embeds
                ):  # Stream was also embed
                    target_edit_msg = response_msgs[-1]
                    embed_to_edit = discord.Embed.from_dict(
                        target_edit_msg.embeds[0].to_dict()
                    )
                    current_desc = embed_to_edit.description or ""
                    embed_to_edit.description = current_desc.replace(
                        STREAMING_INDICATOR, ""
                    ).strip()
                    embed_to_edit.description += f"\n\n{error_text}"
                    embed_to_edit.description = embed_to_edit.description[
                        :MAX_EMBED_DESCRIPTION_LENGTH
                    ]
                    embed_to_edit.color = EMBED_COLOR_ERROR
                    await target_edit_msg.edit(embed=embed_to_edit, view=None)
                else:  # Stream was plain or no stream messages, edit initial embed status
                    await processing_msg.edit(
                        content=None,
                        embed=discord.Embed(
                            description=error_text, color=EMBED_COLOR_ERROR
                        ),
                        view=None,
                    )
        elif (
            not use_plain_responses_stream and response_msgs and response_msgs[-1].embeds
        ):  # No processing_msg, but stream had embeds
            target_edit_msg = response_msgs[-1]
            embed_to_edit = discord.Embed.from_dict(target_edit_msg.embeds[0].to_dict())
            current_desc = embed_to_edit.description or ""
            embed_to_edit.description = current_desc.replace(
                STREAMING_INDICATOR, ""
            ).strip()
            embed_to_edit.description += f"\n\n{error_text}"
            embed_to_edit.description = embed_to_edit.description[
                :MAX_EMBED_DESCRIPTION_LENGTH
            ]
            embed_to_edit.color = EMBED_COLOR_ERROR
            await target_edit_msg.edit(embed=embed_to_edit, view=None)
        else:  # Fallback: No processing_msg and no stream to edit, send new reply
            await new_msg.reply(error_text, mention_author=False)
    async def handle_all_keys_failed(
        self,
        new_msg: discord.Message,
        processing_msg: Optional[discord.Message],
        response_msgs: List[discord.Message],
        error: AllKeysFailedError,
        use_plain_responses_config: bool,
    ):
        """Handle AllKeysFailedError specifically."""
        error_text = f"⚠️ All API keys for provider `{error.service_name}` failed."
        last_err_str = str(error.errors[-1]) if error.errors else "Unknown reason."
        error_text += f"\nLast error: `{last_err_str[:100]}{'...' if len(last_err_str) > 100 else ''}`"
        pass
        await self.handle_llm_exception(
            new_msg,
            processing_msg,
            response_msgs,
            error_text,
            use_plain_responses_config,
            self.client.config.get("use_plain_responses", False),
        )
    async def handle_unexpected_error(
        self,
        new_msg: discord.Message,
        processing_msg: Optional[discord.Message],
        response_msgs: List[discord.Message],
        error: Exception,
        use_plain_responses_config: bool,
    ):
        """Handle unexpected errors during processing."""
        error_text = f"⚠️ An unexpected error occurred: {type(error).__name__}"
        pass
        await self.handle_llm_exception(
            new_msg,
            processing_msg,
            response_msgs,
            error_text,
            use_plain_responses_config,
            self.client.config.get("use_plain_responses", False),
        ) 