import discord
from datetime import datetime
from typing import Tuple
from ..core.constants import EMBED_COLOR_INCOMPLETE
from ..research.deep_research import perform_deep_research_async


class DeepSearchHandler:
    """Handles deep search query processing."""

    @staticmethod
    async def process_deep_search_query(
        message: discord.Message,
        cleaned_content: str,
        processing_msg: discord.Message,
        use_plain_for_initial_status: bool,
        config: dict,
    ) -> Tuple[str, str]:
        """
        Process a deep search query.

        Returns:
            Tuple of (modified_cleaned_content, topic_for_research)
        """
        # Get current date in ISO format (UTC) to append to the query/topic
        current_date_str = datetime.utcnow().strftime("%Y-%m-%d")

        # Extract the actual topic after the keyword and append the date
        topic_for_research_raw = (
            cleaned_content[len("deepsearch") :].strip() or "general"
        )
        topic_for_research = f"{topic_for_research_raw} {current_date_str}"

        # Update initial status message to indicate deep search is running
        await DeepSearchHandler._update_processing_message(
            processing_msg, use_plain_for_initial_status
        )

        # Run the asynchronous deep research workflow
        deep_research_output = await perform_deep_research_async(
            topic_for_research, config
        )

        # Also append the date to the user query that will be sent to the LLM
        user_query_with_date = f"{message.content.strip()} {current_date_str}"

        # Build the new prompt expected by the downstream LLM
        modified_cleaned_content = (
            "Answer the query based on the deep research report.\n\n"
            f"user query:\n{user_query_with_date}\n\n"
            "deep research output:\n"
            f"{deep_research_output}"
        )

        return modified_cleaned_content, topic_for_research

    @staticmethod
    async def _update_processing_message(
        processing_msg: discord.Message, use_plain_for_initial_status: bool
    ):
        """Update the processing message to indicate deep search is running."""
        try:
            deepsearch_status_text = (
                "⏳ Performing deep search, this may take a while..."
            )

            if processing_msg:
                if use_plain_for_initial_status:
                    await processing_msg.edit(content=deepsearch_status_text)
                else:
                    deepsearch_embed = discord.Embed(
                        description=deepsearch_status_text,
                        color=EMBED_COLOR_INCOMPLETE,
                    )
                    await processing_msg.edit(embed=deepsearch_embed)
        except discord.HTTPException:
            pass
        except Exception:
            pass
