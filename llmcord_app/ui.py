import discord
from discord import ui
import logging
import io
import re
import json
from typing import Optional, Any, Dict

from .constants import (
    EMBED_COLOR_COMPLETE,
    MAX_EMBED_FIELD_VALUE_LENGTH,
    OUTPUT_SHARING_CONFIG_KEY,
    TEXTIS_ENABLED_CONFIG_KEY,  # Changed from NGROK_ENABLED_CONFIG_KEY
)
from .utils import add_field_safely
from .output_server import start_output_server


class ResponseActionView(ui.View):
    """A view combining 'Show Sources' and 'Get response as text file' buttons."""

    def __init__(
        self,
        *,
        grounding_metadata: Optional[Any] = None,
        full_response_text: Optional[str] = None,
        model_name: Optional[str] = None,
        app_config: Optional[Dict[str, Any]] = None,  # Added app_config
        timeout=3600,  # 1 hour
    ):
        super().__init__(timeout=timeout)
        self.grounding_metadata = grounding_metadata
        self.full_response_text = full_response_text
        self.model_name = model_name or "llm"  # Default filename model name
        self.app_config = app_config  # Store app_config
        self.message: Optional[discord.Message] = (
            None  # Will be set after sending the message
        )

        # Conditionally add buttons
        has_sources_button = False
        # Check if grounding_metadata exists and has relevant attributes
        if self.grounding_metadata and (
            (
                hasattr(self.grounding_metadata, "web_search_queries")
                and self.grounding_metadata.web_search_queries
            )
            or (
                hasattr(self.grounding_metadata, "grounding_chunks")
                and self.grounding_metadata.grounding_chunks
            )
            or (
                hasattr(self.grounding_metadata, "search_entry_point")
                and self.grounding_metadata.search_entry_point
            )  # Keep this check, even if we don't display rendered_content
        ):
            self.add_item(self.ShowSourcesButton())
            has_sources_button = True

        # Determine row for GetTextFileButton
        text_file_button_row = 1 if has_sources_button else 0
        if self.full_response_text:
            self.add_item(self.GetTextFileButton(row=text_file_button_row))

        # Conditionally add "View Rendered Output" button
        output_sharing_cfg = (
            self.app_config.get(OUTPUT_SHARING_CONFIG_KEY, {})
            if self.app_config
            else {}
        )
        textis_is_enabled = output_sharing_cfg.get(
            TEXTIS_ENABLED_CONFIG_KEY, False
        )  # Changed variable name and key

        if self.full_response_text and textis_is_enabled:  # Changed variable name
            # Determine row for ViewRenderedOutputButton
            # If sources button exists, and text file button also exists, this goes to row 2
            # If only sources button exists, this goes to row 1 (same as text file would)
            # If only text file button exists (no sources), this goes to row 1
            # If neither sources nor text file button exists, this goes to row 0
            rendered_output_button_row = 0
            if has_sources_button:
                rendered_output_button_row = 1
                if self.full_response_text:  # if text file button is also present
                    rendered_output_button_row = 2
            elif self.full_response_text:  # only text file button is present
                rendered_output_button_row = 1

            self.add_item(self.ViewRenderedOutputButton(row=rendered_output_button_row))

    async def on_timeout(self):
        """Disables all buttons when the view times out."""
        if self.message:
            try:
                # Check if the view hasn't already been replaced or removed
                # Fetch the message again to ensure we have the latest state
                message = await self.message.channel.fetch_message(self.message.id)
                if (
                    hasattr(message, "view") and message.view is self
                ):  # Only edit if this view is still attached
                    for item in self.children:
                        if isinstance(item, ui.Button):
                            item.disabled = True
                    await message.edit(view=self)
                    logging.debug(
                        f"View timed out for message {self.message.id}. Buttons disabled."
                    )
                else:
                    logging.debug(
                        f"View for message {self.message.id} already replaced or removed. Skipping timeout disable."
                    )
            except discord.NotFound:
                logging.warning(
                    f"Message {self.message.id} not found when trying to disable view on timeout."
                )
            except discord.HTTPException as e:
                logging.error(
                    f"Failed to edit message {self.message.id} on view timeout: {e}"
                )
        self.stop()  # Stop the view regardless of whether the message was edited

    # Inner class for the Show Sources button - MODIFIED FOR SPLITTING
    class ShowSourcesButton(ui.Button):
        def __init__(self):
            super().__init__(
                label="Show Sources", style=discord.ButtonStyle.grey, row=0
            )

        async def callback(self, interaction: discord.Interaction):
            view: "ResponseActionView" = self.view  # Type hint for clarity
            if not view.grounding_metadata:
                await interaction.response.send_message(
                    "No grounding metadata available.", ephemeral=False
                )
                return

            embeds_to_send = []
            current_embed = discord.Embed(
                title="Grounding Sources", color=EMBED_COLOR_COMPLETE
            )
            current_embed.description = None  # Ensure description starts empty

            # --- Extract Data Safely ---
            queries = getattr(view.grounding_metadata, "web_search_queries", None)
            chunks = getattr(view.grounding_metadata, "grounding_chunks", None)

            # --- Add Search Queries Field ---
            if queries:
                query_text = "\n".join(f"- `{q}`" for q in queries)
                query_field_name = "Search Queries Used by Model"
                query_field_value = query_text[:MAX_EMBED_FIELD_VALUE_LENGTH]
                if len(query_text) > MAX_EMBED_FIELD_VALUE_LENGTH:
                    query_field_value = (
                        query_text[: MAX_EMBED_FIELD_VALUE_LENGTH - 4] + "\n..."
                    )
                    logging.warning("Search query list truncated for embed field.")
                current_embed = add_field_safely(
                    current_embed,
                    query_field_name,
                    query_field_value,
                    False,
                    embeds_to_send,
                    current_embed,
                    EMBED_COLOR_COMPLETE,
                )

            # --- Process and Add Sources Consulted Field(s) ---
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
                            logging.warning(
                                f"Invalid or missing URI in grounding chunk: {uri}"
                            )
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
                                current_field_name = "Sources Consulted (cont.)"  # type: ignore
                                current_field_value = ""

                            if len(source_line) > MAX_EMBED_FIELD_VALUE_LENGTH:
                                source_line = (
                                    source_line[: MAX_EMBED_FIELD_VALUE_LENGTH - 4]
                                    + "...\n"
                                )
                                logging.warning(f"Single source line truncated: {uri}")

                            current_field_value = source_line  # type: ignore
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

            # --- Finalize and Send ---
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
                    await interaction.response.send_message(
                        f"Could not extract specific sources. Raw metadata:\n```json\n{metadata_str[:1900]}\n```",
                        ephemeral=False,
                    )
                except Exception as e:
                    logging.error(f"Error sending raw metadata: {e}")
                    await interaction.response.send_message(
                        "No grounding source information could be extracted.",
                        ephemeral=False,
                    )
                return

            try:
                await interaction.response.send_message(
                    embed=embeds_to_send[0],
                    ephemeral=False,
                )
                for embed in embeds_to_send[1:]:
                    await interaction.followup.send(embed=embed, ephemeral=False)

            except discord.HTTPException as e:
                logging.error(
                    f"HTTPException sending source embeds (might be too large even after split): {e}"
                )
                # Use followup because initial response was already sent
                await interaction.followup.send(
                    "Failed to send sources as embeds (likely too large).",
                    ephemeral=False,
                )
            except Exception as e:
                logging.error(f"Unexpected error sending source embeds: {e}")
                try:
                    # Use followup because initial response was already sent
                    await interaction.followup.send(
                        "An unexpected error occurred while sending sources.",
                        ephemeral=False,
                    )
                except discord.HTTPException:
                    logging.error("Failed to send followup error message for sources.")

    class GetTextFileButton(ui.Button):
        def __init__(self, row: int):
            super().__init__(
                label="Get response as text file",
                style=discord.ButtonStyle.secondary,
                row=row,
            )

        async def callback(self, interaction: discord.Interaction):
            # Access parent view's data
            view: "ResponseActionView" = self.view
            if not view.full_response_text:
                await interaction.response.send_message(
                    "No response text available to send.",
                    ephemeral=False,
                )
                return

            try:
                # Clean model name for filename
                safe_model_name = re.sub(
                    r'[<>:"/\\|?*]', "_", view.model_name or "llm"
                )  # Replace invalid chars
                filename = f"llm_response_{safe_model_name}.txt"

                # Create a file-like object from the string
                file_content = io.BytesIO(view.full_response_text.encode("utf-8"))
                discord_file = discord.File(fp=file_content, filename=filename)

                await interaction.response.send_message(
                    file=discord_file,
                    ephemeral=False,
                )
            except Exception as e:
                logging.error(f"Error creating or sending text file: {e}")
                await interaction.response.send_message(
                    "Sorry, I couldn't create the text file.",
                    ephemeral=False,
                )

    class ViewRenderedOutputButton(ui.Button):
        def __init__(self, row: int):
            super().__init__(
                label="View output properly (especially tables)",
                style=discord.ButtonStyle.grey,
                row=row,
            )

        async def callback(self, interaction: discord.Interaction):
            view: "ResponseActionView" = self.view
            if not view.full_response_text:
                await interaction.response.send_message(
                    "No response text available to render.", ephemeral=False
                )
                return
            if not view.app_config:
                await interaction.response.send_message(
                    "Application configuration is not available for rendering.",
                    ephemeral=False,
                )
                return

            output_sharing_cfg = view.app_config.get(OUTPUT_SHARING_CONFIG_KEY, {})
            textis_is_enabled = output_sharing_cfg.get(
                TEXTIS_ENABLED_CONFIG_KEY, False
            )  # Changed variable name and key

            if not textis_is_enabled:  # Changed variable name
                await interaction.response.send_message(
                    "Output sharing (text.is) is not enabled in the configuration.",  # Changed message
                    ephemeral=False,
                )
                return

            await interaction.response.defer(
                ephemeral=False, thinking=True
            )  # Defer while processing

            try:
                public_url = await start_output_server(
                    view.full_response_text, view.app_config
                )
                if public_url:
                    await interaction.followup.send(
                        f"🔗 View output on text.is: {public_url}",  # Changed message
                        ephemeral=False,
                    )
                    logging.info(
                        f"Sent text.is public URL via button: {public_url}"
                    )  # Changed log
                else:
                    await interaction.followup.send(
                        "Could not generate a public link via text.is for the output.",  # Changed message
                        ephemeral=False,
                    )
            except Exception as e:
                logging.error(
                    f"Error starting or managing output server via button: {e}",
                    exc_info=True,
                )
                await interaction.followup.send(
                    "An error occurred while trying to generate the rendered output link.",
                    ephemeral=False,
                )
