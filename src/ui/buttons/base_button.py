import discord
from discord import ui
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ui import ResponseActionView


class BaseResponseButton(ui.Button):
    """Base class for response action buttons with common functionality."""

    async def handle_interaction_error(
        self, interaction: discord.Interaction, error_msg: str, ephemeral: bool = False
    ):
        """Handle interaction errors consistently."""
        try:
            if interaction.response.is_done():
                await interaction.followup.send(error_msg, ephemeral=ephemeral)
            else:
                await interaction.response.send_message(error_msg, ephemeral=ephemeral)
        except discord.HTTPException:
            pass

    async def disable_button_and_respond(self, interaction: discord.Interaction):
        """Disable this button and update the view."""
        self.disabled = True
        self.style = discord.ButtonStyle.secondary
        pass
        view: "ResponseActionView" = self.view
        try:
            await interaction.response.edit_message(view=view)
            return True  # Successfully edited message
        except discord.NotFound:
            # Message not found, defer for followup
            await interaction.response.defer()
            return False  # Need to use followup
