import discord
from discord import ui
import json
from typing import TYPE_CHECKING
from .base_button import BaseResponseButton
from ...core.constants import EMBED_COLOR_COMPLETE, MAX_EMBED_FIELD_VALUE_LENGTH
from ...core.utils import add_field_safely
if TYPE_CHECKING:
    from ..ui import ResponseActionView
class ShowSourcesButton(BaseResponseButton):
    def __init__(self):
        super().__init__(
            label="Show Sources", style=discord.ButtonStyle.grey, row=0
        )
    async def callback(self, interaction: discord.Interaction):
        success = await self.disable_button_and_respond(interaction)
        pass
        view: "ResponseActionView" = self.view
        pass
        if not view.grounding_metadata:
            msg = "No grounding metadata available."
            if success:
                await interaction.followup.send(msg, ephemeral=False)
            else:
                await interaction.response.send_message(msg, ephemeral=False)
            return
        embeds_to_send = []
        current_embed = discord.Embed(
            title="Grounding Sources", color=EMBED_COLOR_COMPLETE
        )
        current_embed.description = None
        # Extract data safely
        queries = getattr(view.grounding_metadata, "web_search_queries", None)
        chunks = getattr(view.grounding_metadata, "grounding_chunks", None)
        # Add search queries field
        if queries:
            query_text = "\n".join(f"- `{q}`" for q in queries)
            query_field_name = "Search Queries Used by Model"
            query_field_value = query_text[:MAX_EMBED_FIELD_VALUE_LENGTH]
            if len(query_text) > MAX_EMBED_FIELD_VALUE_LENGTH:
                query_field_value = (
                    query_text[: MAX_EMBED_FIELD_VALUE_LENGTH - 4] + "\n..."
                )
            current_embed = add_field_safely(
                current_embed,
                query_field_name,
                query_field_value,
                False,
                embeds_to_send,
                current_embed,
                EMBED_COLOR_COMPLETE,
            )
        # Process and add sources consulted fields
        if chunks:
            current_field_value = ""
            current_field_name = "Sources Consulted by Model"
            sources_added_count = 0
            for i, chunk in enumerate(chunks):
                web_chunk = getattr(chunk, "web", None)
                if (
                    web_chunk
                    and hasattr(web_chunk, "title")
                    and hasattr(web_chunk, "uri")
                ):
                    title = web_chunk.title or "Source"
                    uri = web_chunk.uri
                    if (
                        not uri
                        or not isinstance(uri, str)
                        or not uri.startswith(("http://", "https://"))
                    ):
                        continue
                    escaped_title = discord.utils.escape_markdown(title)
                    source_line = f"- [{escaped_title}]({uri})\n"
                    sources_added_count += 1
                    if (
                        len(current_field_value) + len(source_line)
                        > MAX_EMBED_FIELD_VALUE_LENGTH
                    ):
                        if current_field_value:
                            current_embed = add_field_safely(
                                current_embed,
                                current_field_name,
                                current_field_value,
                                False,
                                embeds_to_send,
                                current_embed,
                                EMBED_COLOR_COMPLETE,
                            )
                            current_field_name = "Sources Consulted (cont.)"
                            current_field_value = ""
                        if len(source_line) > MAX_EMBED_FIELD_VALUE_LENGTH:
                            source_line = (
                                source_line[: MAX_EMBED_FIELD_VALUE_LENGTH - 4]
                                + "...\n"
                            )
                        current_field_value = source_line
                    else:
                        current_field_value += source_line
            if current_field_value:
                current_embed = add_field_safely(
                    current_embed,
                    current_field_name,
                    current_field_value,
                    False,
                    embeds_to_send,
                    current_embed,
                    EMBED_COLOR_COMPLETE,
                )
            if sources_added_count == 0 and not queries:
                current_embed.description = (
                    current_embed.description or ""
                ) + "\nNo web sources or search queries found in metadata."
        elif not queries:
            current_embed.description = (
                current_embed.description or ""
            ) + "\nNo grounding source information found in metadata."
        # Finalize and send
        if current_embed.fields or current_embed.description:
            embeds_to_send.append(current_embed)
        if not embeds_to_send:
            try:
                metadata_str = "Could not serialize metadata."
                if hasattr(view.grounding_metadata, "model_dump"):
                    metadata_str = json.dumps(
                        view.grounding_metadata.model_dump(mode="json"), indent=2
                    )
                elif view.grounding_metadata:
                    metadata_str = str(view.grounding_metadata)
                msg = f"Could not extract specific sources. Raw metadata:\n```json\n{metadata_str[:1900]}\n```"
                if success:
                    await interaction.followup.send(msg, ephemeral=False)
                else:
                    await interaction.response.send_message(msg, ephemeral=False)
            except Exception as e:
                await self.handle_interaction_error(interaction, "No grounding source information could be extracted.")
            return
        try:
            # Send embeds
            if success:
                await interaction.followup.send(embed=embeds_to_send[0], ephemeral=False)
                for embed in embeds_to_send[1:]:
                    await interaction.followup.send(embed=embed, ephemeral=False)
            else:
                await interaction.response.send_message(embed=embeds_to_send[0], ephemeral=False)
                for embed in embeds_to_send[1:]:
                    await interaction.followup.send(embed=embed, ephemeral=False)
        except discord.HTTPException as e:
            await self.handle_interaction_error(interaction, "Failed to send sources as embeds (likely too large).")
        except Exception as e:
            await self.handle_interaction_error(interaction, "An unexpected error occurred while sending sources.") 