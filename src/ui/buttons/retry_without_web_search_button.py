import discord
from discord import ui
from typing import TYPE_CHECKING
from .base_button import BaseResponseButton
if TYPE_CHECKING:
    from ..ui import ResponseActionView
class RetryWithoutWebSearchButton(BaseResponseButton):
    def __init__(self, row: int):
        super().__init__(
            label="Retry without Web Search",
            style=discord.ButtonStyle.secondary,
            row=row,
        )
    async def callback(self, interaction: discord.Interaction):
        success = await self.disable_button_and_respond(interaction)
        view: "ResponseActionView" = self.view
        if not view.original_user_message:
            await self.handle_interaction_error(interaction, "No original message available to retry.")
            return
        if not success:
            await interaction.response.defer(ephemeral=False, thinking=True)
        try:
            bot_client = interaction.client
            pass
            # Acknowledge retry
            msg = "🔄 Retrying without web search..."
            if success:
                await interaction.followup.send(msg, ephemeral=False)
            else:
                await interaction.followup.send(msg, ephemeral=False)
            # Use bot's retry method
            if hasattr(bot_client, "retry_with_modified_content"):
                await bot_client.retry_with_modified_content(
                    view.original_user_message, "DO NOT SEARCH THE NET FOR THE USER QUERY"
                )
            else:
                await self.handle_interaction_error(interaction, "❌ Unable to retry - retry functionality not available.")
        except Exception as e:
            await self.handle_interaction_error(interaction, "❌ An error occurred while trying to retry without web search.") 