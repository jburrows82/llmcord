import logging
from typing import List, Dict, Optional, Any, Set

import discord

from ..content.table_renderer import process_and_send_table_images
from .message_editor import MessageEditor
from .stream_processor import StreamProcessor
from .image_handler import ImageHandler
from .utils import safe_typing

# Forward declaration for LLMCordClient to resolve circular import for type hinting if needed
# However, it's better to pass necessary attributes directly if possible.


async def send_initial_processing_message(
    new_msg: discord.Message, use_plain_responses: bool
) -> Optional[discord.Message]:
    """Sends the initial 'Processing request...' message."""
    message_editor = MessageEditor(None, {})
    return await message_editor.send_initial_processing_message(
        new_msg, use_plain_responses
    )


async def handle_llm_response_stream(
    client,
    new_msg: discord.Message,
    processing_msg: Optional[discord.Message],
    provider: str,
    model_name: str,
    history_for_llm: List[Dict[str, Any]],
    system_prompt_text: Optional[str],
    provider_config: Dict[str, Any],
    extra_api_params: Dict[str, Any],
    app_config: Dict[str, Any],
    initial_user_warnings: Set[str],
    use_plain_responses_config: bool,
    split_limit_config: int,
    custom_search_queries_generated: bool,
    successful_api_results_count: int,
    deep_search_used: bool = False,
):
    """Handles the streaming, editing, and sending of LLM responses."""

    # Initialize processors
    stream_processor = StreamProcessor(client, app_config)
    image_handler = ImageHandler(client, app_config)

    async with safe_typing(new_msg.channel):
        # Process the stream
        (
            llm_call_successful_final,
            final_text_to_return,
            response_msgs,
        ) = await stream_processor.process_stream(
            new_msg=new_msg,
            processing_msg=processing_msg,
            provider=provider,
            model_name=model_name,
            history_for_llm=history_for_llm,
            system_prompt_text=system_prompt_text,
            provider_config=provider_config,
            extra_api_params=extra_api_params,
            initial_user_warnings=initial_user_warnings,
            use_plain_responses_config=use_plain_responses_config,
            split_limit_config=split_limit_config,
            custom_search_queries_generated=custom_search_queries_generated,
            successful_api_results_count=successful_api_results_count,
            deep_search_used=deep_search_used,
        )

    # Handle post-processing
    await _handle_post_processing(
        client,
        llm_call_successful_final,
        final_text_to_return,
        response_msgs,
        new_msg,
        app_config,
        image_handler,
    )

    return llm_call_successful_final, final_text_to_return, response_msgs


async def _handle_post_processing(
    client,
    llm_call_successful_final: bool,
    final_text_to_return: str,
    response_msgs: List[discord.Message],
    new_msg: discord.Message,
    app_config: Dict[str, Any],
    image_handler: ImageHandler,
):
    """Handle post-processing tasks like table rendering and imgur URL resending."""

    if not llm_call_successful_final or not final_text_to_return:
        return

    # Process and send table images if the LLM call was successful and feature is enabled
    if app_config.get("auto_render_markdown_tables", True):
        try:
            await process_and_send_table_images(
                final_text_to_return, response_msgs, new_msg
            )
        except Exception as e:
            logging.error(f"Error processing table images: {e}")
            # Don't let table processing errors affect the main response

    # Handle Imgur URL resending
    try:
        await image_handler.resend_imgur_urls(
            new_msg, response_msgs, final_text_to_return
        )
    except Exception as e:
        logging.error(f"Error processing Imgur URLs: {e}")


# Legacy function aliases for backward compatibility
async def resend_imgur_urls(
    new_msg: discord.Message,
    response_msgs: List[discord.Message],
    final_text: str,
):
    """Legacy function - redirects to ImageHandler."""
    image_handler = ImageHandler(None, {})
    await image_handler.resend_imgur_urls(new_msg, response_msgs, final_text)
