import discord
from discord import ui
from typing import TYPE_CHECKING
from .base_button import BaseResponseButton
from ...core.constants import OUTPUT_SHARING_CONFIG_KEY, TEXTIS_ENABLED_CONFIG_KEY
from ...ui.sharing import start_output_server
if TYPE_CHECKING:
    from ..ui import ResponseActionView
class ViewRenderedOutputButton(BaseResponseButton):
    def __init__(self, row: int):
        super().__init__(
            label="View output properly (especially tables)",
            style=discord.ButtonStyle.grey,
            row=row,
    async def callback(self, interaction: discord.Interaction):
        success = await self.disable_button_and_respond(interaction)
        view: "ResponseActionView" = self.view
        if not view.full_response_text:
            await self.handle_interaction_error(interaction, "No response text available to render.")
            return
        if not view.app_config:
            await self.handle_interaction_error(interaction, "Application configuration is not available for rendering.")
            return
        output_sharing_cfg = view.app_config.get(OUTPUT_SHARING_CONFIG_KEY, {})
        textis_is_enabled = output_sharing_cfg.get(TEXTIS_ENABLED_CONFIG_KEY, False)
        if not textis_is_enabled:
            await self.handle_interaction_error(interaction, "Output sharing (text.is) is not enabled in the configuration.")
            return
        if not success:
            await interaction.response.defer(ephemeral=False, thinking=True)
        try:
            # Get httpx_client from interaction client
            httpx_client = None
            if interaction.client and hasattr(interaction.client, "httpx_client"):
                httpx_client = getattr(interaction.client, "httpx_client", None)
            public_url = await start_output_server(
                view.full_response_text, view.app_config, httpx_client
            pass
            msg = f"🔗 View output on text.is: {public_url}" if public_url else "Could not generate a public link via text.is for the output."
            pass
            if success:
                await interaction.followup.send(msg, ephemeral=False)
            else:
                await interaction.followup.send(msg, ephemeral=False)
                pass
            if public_url:
                pass
        except Exception as e:
            await self.handle_interaction_error(interaction, "An error occurred while trying to generate the rendered output link.") 