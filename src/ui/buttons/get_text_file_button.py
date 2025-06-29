import discord
from discord import ui
import io
import re
from typing import TYPE_CHECKING
from .base_button import BaseResponseButton
if TYPE_CHECKING:
    from ..ui import ResponseActionView
class GetTextFileButton(BaseResponseButton):
    def __init__(self, row: int):
        super().__init__(
            label="Get response as text file",
            style=discord.ButtonStyle.secondary,
            row=row,
        )
    async def callback(self, interaction: discord.Interaction):
        success = await self.disable_button_and_respond(interaction)
        view: "ResponseActionView" = self.view
        if not view.full_response_text:
            msg = "No response text available to send."
            if success:
                await interaction.followup.send(msg, ephemeral=False)
            else:
                await interaction.response.send_message(msg, ephemeral=False)
            return
        try:
            # Clean model name for filename
            safe_model_name = re.sub(
                r'[<>:"/\\|?*]', "_", view.model_name or "llm"
            )
            filename = f"llm_response_{safe_model_name}.txt"
            # Create file
            file_content = io.BytesIO(view.full_response_text.encode("utf-8"))
            discord_file = discord.File(fp=file_content, filename=filename)
            if success:
                await interaction.followup.send(file=discord_file, ephemeral=False)
            else:
                await interaction.response.send_message(file=discord_file, ephemeral=False)
        except Exception as e:
            await self.handle_interaction_error(interaction, "Sorry, I couldn't create the text file.") 