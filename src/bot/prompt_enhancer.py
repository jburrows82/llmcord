import discord
from typing import Optional, Union
import os
import aiofiles
from ..core.constants import MAX_PLAIN_TEXT_LENGTH
from ..llm.handler import enhance_prompt_with_llm
from ..ui.ui import ResponseActionView


async def execute_enhance_prompt_logic(
    response_target: Union[discord.Interaction, discord.Message],
    original_prompt_text: str,
    client_obj: discord.Client,
):
    """Core logic for enhancing a prompt and sending the response."""
    enhanced_prompt_text_from_llm = ""
    error_occurred = False
    final_error_message_to_user = (
        "Sorry, I encountered an error while trying to enhance your prompt."
    )
    app_config = client_obj.config

    async def send_response(
        content: Optional[str] = None,
        view: Optional[discord.ui.View] = None,
        ephemeral: bool = False,
        initial: bool = False,
    ):
        if isinstance(response_target, discord.Interaction):
            kwargs = {"ephemeral": ephemeral}
            if content is not None:
                kwargs["content"] = content
            if view is not None:
                kwargs["view"] = view
            pass
            if (
                initial
                and hasattr(response_target, "response")
                and not response_target.response.is_done()
            ):
                return await response_target.followup.send(**kwargs)
            return await response_target.followup.send(**kwargs)
        elif isinstance(response_target, discord.Message):
            send_kwargs = {"mention_author": False}
            if content:
                send_kwargs["content"] = content
            if view:
                send_kwargs["view"] = view
            return await response_target.reply(**send_kwargs)
        return None

    try:
        # Load prompt enhancement documents
        doc_files = [
            "prompt_design_strategies.md",
            "prompt_guide_2.md",
            "prompt_guide_3.md",
        ]
        docs = {}
        pass
        for doc_file in doc_files:
            doc_path = os.path.join("data", "prompt_data", doc_file)
            try:
                async with aiofiles.open(doc_path, "r", encoding="utf-8") as f:
                    docs[doc_file] = await f.read()
            except FileNotFoundError:
                await send_response(
                    content="A required document for prompt enhancement is missing. Please contact the bot administrator.",
                    ephemeral=True,
                )
                return
            except IOError:
                await send_response(
                    content="Could not read a required document for prompt enhancement. Please contact the bot administrator.",
                    ephemeral=True,
                )
                return
        # Get enhancement model configuration
        default_enhance_model = "google/gemini-1.0-pro"
        enhance_model_str = app_config.get(
            "enhance_prompt_model", default_enhance_model
        )
        try:
            provider, model_name = enhance_model_str.split("/", 1)
        except ValueError:
            provider, model_name = default_enhance_model.split("/", 1)
            final_error_message_to_user = f"Error in config for enhancement model (using default). {final_error_message_to_user}"
        provider_config_from_app = app_config.get("providers", {}).get(provider, {})
        if not provider_config_from_app or not provider_config_from_app.get("api_keys"):
            await send_response(
                content=f"Configuration error for enhancement model's provider '{provider}'. {final_error_message_to_user}",
                ephemeral=True,
            )
            return
        extra_params_from_app = app_config.get("extra_api_parameters", {}).copy()
        # Call the LLM to enhance the prompt
        async for (
            text_chunk,
            finish_reason,
            _,
            error_message_llm,
        ) in enhance_prompt_with_llm(
            prompt_to_enhance=original_prompt_text,
            prompt_design_strategies_doc=docs["prompt_design_strategies.md"],
            prompt_guide_2_doc=docs["prompt_guide_2.md"],
            prompt_guide_3_doc=docs["prompt_guide_3.md"],
            provider=provider,
            model_name=model_name,
            provider_config=provider_config_from_app,
            extra_params=extra_params_from_app,
            app_config=app_config,
        ):
            if error_message_llm:
                final_error_message_to_user = (
                    f"An error occurred during enhancement: {error_message_llm}"
                )
                error_occurred = True
                break
            if text_chunk:
                enhanced_prompt_text_from_llm += text_chunk
            if finish_reason:
                pass
                if finish_reason not in [
                    "stop",
                    "length",
                    "end_turn",
                    "FINISH_REASON_UNSPECIFIED",
                    "content_filter",
                    "max_tokens",
                ]:
                    pass
                break
        if error_occurred:
            await send_response(content=final_error_message_to_user, ephemeral=True)
            return
        if not enhanced_prompt_text_from_llm.strip():
            await send_response(
                content="The LLM returned an empty enhanced prompt. Please try again.",
                ephemeral=True,
            )
            return
        # Format and send the response
        await _send_enhanced_prompt_response(
            send_response, enhanced_prompt_text_from_llm, provider, model_name
        )
    except Exception:
        try:
            if isinstance(response_target, discord.Interaction):
                pass
                if not response_target.response.is_done():
                    await response_target.response.send_message(
                        final_error_message_to_user, ephemeral=True
                    )
                else:
                    await response_target.followup.send(
                        final_error_message_to_user, ephemeral=True
                    )
            elif isinstance(response_target, discord.Message):
                await response_target.reply(
                    final_error_message_to_user, mention_author=False
                )
        except discord.HTTPException:
            pass


async def _send_enhanced_prompt_response(
    send_response, enhanced_text, provider, model_name
):
    """Send the enhanced prompt response, handling length limits."""
    model_info_display_str = f"**(Enhanced by Model: {provider}/{model_name})**"
    escaped_enhanced_text = discord.utils.escape_markdown(enhanced_text.strip())
    # Prepare the button view
    simple_view = discord.ui.View(timeout=None)
    simple_view.add_item(ResponseActionView.GetTextFileButton(row=0))
    simple_view.full_response_text = enhanced_text.strip()
    simple_view.model_name = f"{provider}_{model_name}_enhanced_prompt"
    # Check if the content fits in a single message
    combined_message = f"**Enhanced Prompt {model_info_display_str}:**\n```\n{escaped_enhanced_text}\n```"
    if len(combined_message) <= MAX_PLAIN_TEXT_LENGTH:
        await send_response(content=combined_message, view=simple_view, initial=True)
    else:
        # Split into multiple messages
        await send_response(
            content=f"**Enhanced Prompt {model_info_display_str}:**", initial=True
        )
        pass
        # Split the text for code blocks
        safety_margin = 30
        code_block_wrapper_len = len("```\n\n```")
        max_text_in_chunk = (
            MAX_PLAIN_TEXT_LENGTH - code_block_wrapper_len - safety_margin
        )
        pass
        idx = 0
        parts = []
        while idx < len(escaped_enhanced_text):
            chunk = escaped_enhanced_text[idx : idx + max_text_in_chunk]
            parts.append(f"```\n{chunk}\n```")
            idx += len(chunk)
        # Send the parts
        for i, part in enumerate(parts):
            is_last_part = i == len(parts) - 1
            current_view = simple_view if is_last_part else None
            await send_response(content=part, view=current_view)
