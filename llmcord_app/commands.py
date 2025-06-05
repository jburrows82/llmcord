import discord
from discord import app_commands
from discord.app_commands import Choice
from typing import List, Dict, Optional, Any, Union
import logging
import json
import os
import aiofiles
import aiofiles.os as aio_os

from .constants import (
    AVAILABLE_MODELS,
    USER_SYSTEM_PROMPTS_FILENAME,
    USER_GEMINI_THINKING_BUDGET_PREFS_FILENAME,
    USER_MODEL_PREFS_FILENAME,
    MAX_PLAIN_TEXT_LENGTH,  # Discord message character limit
)
from .llm_handler import enhance_prompt_with_llm
from .ui import ResponseActionView  # For the button

logger = logging.getLogger(__name__)


# --- ADDED: Helper functions for loading and saving user preferences ---
async def _load_user_preferences(filename: str) -> Dict[int, Any]:
    """Loads user preferences from a JSON file asynchronously."""
    if not await aio_os.path.exists(filename):
        logger.info(
            f"Preference file '{filename}' not found. Starting with empty preferences."
        )
        return {}
    try:
        async with aiofiles.open(filename, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)
            # Convert string keys from JSON back to integers
            return {int(k): v for k, v in data.items()}
    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON from '{filename}'. Starting with empty preferences."
        )
        # Optionally, create a backup of the corrupted file here
        # await aio_os.rename(filename, filename + ".corrupted")
        return {}
    except IOError as e:
        logger.error(
            f"IOError reading from '{filename}': {e}. Starting with empty preferences."
        )
        return {}
    except Exception as e:
        logger.error(
            f"Unexpected error loading preferences from '{filename}': {e}. Starting with empty preferences."
        )
        return {}


async def _save_user_preferences(filename: str, data: Dict[int, Any]):
    """Saves user preferences to a JSON file asynchronously."""
    try:
        # Ensure the directory exists
        directory = os.path.dirname(filename)
        if directory and not await aio_os.path.exists(directory):
            await aio_os.makedirs(directory)

        async with aiofiles.open(filename, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=4))
        logger.debug(f"Saved user preferences to '{filename}'.")
    except IOError as e:
        logger.error(f"IOError writing to '{filename}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving preferences to '{filename}': {e}")


# These dictionaries will store user preferences.
# They will be loaded asynchronously during bot setup.
user_model_preferences: Dict[int, str] = {}
user_system_prompt_preferences: Dict[int, Optional[str]] = {}
user_gemini_thinking_budget_preferences: Dict[int, bool] = {}


# --- ADDED: Function to load all preferences ---
async def load_all_preferences():
    """Loads all user preferences from their respective files."""
    global \
        user_model_preferences, \
        user_system_prompt_preferences, \
        user_gemini_thinking_budget_preferences

    loaded_model_prefs = await _load_user_preferences(USER_MODEL_PREFS_FILENAME)
    if loaded_model_prefs:  # Only update if loading was successful and returned data
        user_model_preferences.update(loaded_model_prefs)

    loaded_system_prompts = await _load_user_preferences(USER_SYSTEM_PROMPTS_FILENAME)
    if loaded_system_prompts:
        user_system_prompt_preferences.update(loaded_system_prompts)

    loaded_gemini_prefs = await _load_user_preferences(
        USER_GEMINI_THINKING_BUDGET_PREFS_FILENAME
    )
    if loaded_gemini_prefs:
        user_gemini_thinking_budget_preferences.update(loaded_gemini_prefs)

    logger.info("User preferences loaded.")


# --- Slash Command Autocomplete Functions ---
async def model_autocomplete(
    interaction: discord.Interaction, current: str
) -> List[Choice[str]]:
    """Autocompletes the model argument with combined provider/model_name."""
    choices = []
    for provider_name, models in AVAILABLE_MODELS.items():
        for model_name in models:
            full_model_name = f"{provider_name}/{model_name}"
            if current.lower() in full_model_name.lower():
                choices.append(Choice(name=full_model_name, value=full_model_name))
    return choices[:25]  # Limit to 25 choices


# --- Slash Command Definition ---
# Note: The command registration (@discord_client.tree.command) happens in bot.py
# This file just defines the command function and its logic.


@app_commands.autocomplete(model_full_name=model_autocomplete)
@app_commands.describe(
    model_full_name="The LLM provider and model (e.g., google/gemini-2.0-flash, openai/gpt-4.1)."
)
async def set_model_command(interaction: discord.Interaction, model_full_name: str):
    """
    Sets the user's preferred LLM (provider and model) for future interactions.
    """
    global user_model_preferences

    try:
        provider, model_name = model_full_name.split("/", 1)
    except ValueError:
        await interaction.response.send_message(
            f"Invalid model format: `{model_full_name}`. Please use the format `provider/model_name` (e.g., `openai/gpt-4.1`).",
            ephemeral=False,
        )
        return

    # Validate provider
    if provider not in AVAILABLE_MODELS:
        await interaction.response.send_message(
            f"Invalid provider: `{provider}`. Please choose from the suggestions.",
            ephemeral=False,
        )
        return

    # Validate model against the selected provider
    if model_name not in AVAILABLE_MODELS.get(provider, []):
        await interaction.response.send_message(
            f"Invalid model: `{model_name}` for provider `{provider}`. Please choose from the suggestions.",
            ephemeral=False,
        )
        return

    user_id = interaction.user.id
    user_model_preferences[user_id] = model_full_name
    logger.info(
        f"User {user_id} ({interaction.user.name}) set model preference to: {model_full_name}"
    )

    # --- ADDED: Save model preferences ---
    await _save_user_preferences(USER_MODEL_PREFS_FILENAME, user_model_preferences)

    await interaction.response.send_message(
        f"Your LLM model has been set to `{model_full_name}`.", ephemeral=False
    )


def get_user_model_preference(user_id: int, default_model: str) -> str:
    """Gets the user's model preference or the default."""
    return user_model_preferences.get(user_id, default_model)


# --- ADDED: Slash Command for Setting System Prompt ---
@app_commands.describe(
    prompt="Your custom system prompt. Use 'reset' to use the default prompt from config.yaml."
)
async def set_system_prompt_command(interaction: discord.Interaction, prompt: str):
    """
    Sets your custom system prompt for the bot.
    This prompt will be used to guide the AI's responses for you.
    To revert to the default system prompt, use 'reset' as the prompt.
    """
    global user_system_prompt_preferences
    user_id = interaction.user.id

    try:
        if prompt.lower() == "reset":
            user_system_prompt_preferences[user_id] = (
                None  # None signifies using the default
            )
            logger.info(
                f"User {user_id} ({interaction.user.name}) reset their system prompt to default."
            )
            await interaction.response.send_message(
                "Your system prompt has been reset to the default.", ephemeral=False
            )
        else:
            modified_prompt = f"{prompt}\nDon't use LaTeX unless told by the user"
            user_system_prompt_preferences[user_id] = modified_prompt
            logger.info(
                f'User {user_id} ({interaction.user.name}) set system prompt to: "{modified_prompt[:100]}{"..." if len(modified_prompt) > 100 else ""}"'
            )
            await interaction.response.send_message(
                f'Your system prompt has been set to: "{modified_prompt[:200]}{"..." if len(modified_prompt) > 200 else ""}"',
                ephemeral=False,
            )

        # --- ADDED: Save preferences after modification ---
        await _save_user_preferences(
            USER_SYSTEM_PROMPTS_FILENAME, user_system_prompt_preferences
        )

    except Exception as e:
        logger.exception(
            f"Error in set_system_prompt_command for user {user_id} (Interaction ID: {interaction.id}): {e}"
        )
        try:
            # Try to send an error message if the interaction hasn't been responded to yet.
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "An error occurred while setting your system prompt. Please check the bot logs for more details.",
                    ephemeral=False,
                )
            else:
                # If already responded (e.g., error happened during _save_user_preferences, though less likely to be caught here)
                # or if interaction is too old, try a followup.
                await interaction.followup.send(
                    "An error occurred after the initial response while processing your system prompt. Please check the bot logs.",
                    ephemeral=False,
                )
        except discord.HTTPException as http_err:
            logger.error(
                f"Failed to send error message followup for set_system_prompt_command (Interaction ID: {interaction.id}): {http_err}"
            )
        # The command will now complete from Discord's perspective, showing the error message.


def get_user_system_prompt_preference(
    user_id: int, default_prompt: Optional[str]
) -> Optional[str]:
    """
    Gets the user's system prompt preference.
    Returns the user-set prompt if available, otherwise the default_prompt.
    If the user explicitly reset to default, it returns default_prompt.
    """
    user_specific_prompt = user_system_prompt_preferences.get(user_id)
    if user_specific_prompt is None and user_id in user_system_prompt_preferences:
        # User explicitly set to reset, so use default
        return default_prompt
    elif user_specific_prompt is not None:
        # User has a custom prompt set
        return user_specific_prompt
    else:
        # User has not set any preference, use default
        return default_prompt


# --- ADDED: Slash Command for Setting Gemini Thinking Budget Usage ---
@app_commands.describe(
    enabled="Set to 'True' to use the thinking budget for Gemini, 'False' to disable it for your interactions."
)
async def setgeminithinking(interaction: discord.Interaction, enabled: bool):
    """
    Sets your preference for using the 'thinkingBudget' parameter with Gemini models.
    This can potentially improve response quality for complex queries but may increase latency.
    The actual budget value is set globally in config.yaml.
    """
    global user_gemini_thinking_budget_preferences
    user_id = interaction.user.id

    try:
        user_gemini_thinking_budget_preferences[user_id] = enabled
        logger.info(
            f"User {user_id} ({interaction.user.name}) set Gemini thinking budget usage to: {enabled}"
        )
        status_message = "enabled" if enabled else "disabled"
        await interaction.response.send_message(
            f"Your preference for Gemini 'thinkingBudget' has been set to **{status_message}**.",
            ephemeral=False,
        )

        await _save_user_preferences(
            USER_GEMINI_THINKING_BUDGET_PREFS_FILENAME,
            user_gemini_thinking_budget_preferences,
        )

    except Exception as e:
        logger.exception(
            f"Error in setgeminithinking command for user {user_id} (Interaction ID: {interaction.id}): {e}"
        )
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "An error occurred while setting your Gemini thinking budget preference. Please check the bot logs.",
                    ephemeral=False,
                )
            else:
                await interaction.followup.send(
                    "An error occurred after the initial response while processing your Gemini thinking budget preference.",
                    ephemeral=False,
                )
        except discord.HTTPException as http_err:
            logger.error(
                f"Failed to send error message followup for setgeminithinking (Interaction ID: {interaction.id}): {http_err}"
            )


def get_user_gemini_thinking_budget_preference(
    user_id: int, default_enabled: bool
) -> bool:
    """
    Gets the user's preference for using the Gemini thinking budget.
    Returns the user-set preference if available, otherwise the default_enabled value from config.
    """
    return user_gemini_thinking_budget_preferences.get(user_id, default_enabled)


# --- ADDED: Slash Command for Help ---
async def help_command(interaction: discord.Interaction):
    """Displays all available commands and how to use them."""
    embed = discord.Embed(
        title="LLMCord Bot Help",
        description="Here are the available commands:",
        color=discord.Color.blue(),
    )

    commands_info = [
        {
            "name": "/model `model_full_name`",
            "value": "Sets your preferred LLM provider and model (e.g., `google/gemini-2.0-flash`, `openai/gpt-4.1`). This preference is saved for your future messages.",
            "inline": False,
        },
        {
            "name": "/systemprompt `prompt`",
            "value": "Sets your custom system prompt. Use `reset` to revert to the default prompt from `config.yaml`. This guides the AI's responses for you.",
            "inline": False,
        },
        {
            "name": "/setgeminithinking `enabled`",
            "value": "Sets your preference for using the 'thinkingBudget' parameter with Gemini models (`True` or `False`). This can potentially improve response quality but may increase latency. The actual budget value is set globally in `config.yaml`.",
            "inline": False,
        },
        {
            "name": "/help",
            "value": "Displays this help message, showing all available commands and their usage.",
            "inline": False,
        },
    ]

    for cmd_info in commands_info:
        embed.add_field(
            name=cmd_info["name"], value=cmd_info["value"], inline=cmd_info["inline"]
        )

    embed.set_footer(
        text="Use these commands to customize your interaction with the bot."
    )

    try:
        await interaction.response.send_message(embed=embed, ephemeral=False)
    except discord.HTTPException as e:
        logger.error(f"Failed to send help message: {e}")
        # Fallback if embed fails or is too large, though unlikely for this content
        await interaction.response.send_message(
            "Could not display help information. Please check bot logs.",
            ephemeral=True,
        )


# --- ADDED: Slash Command for Enhancing Prompts ---


async def _execute_enhance_prompt_logic(
    response_target: Union[discord.Interaction, discord.Message],
    original_prompt_text: str,
    client_obj: discord.Client,
):
    """
    Core logic for enhancing a prompt and sending the response.
    Can be called from a slash command (Interaction) or a prefix command (Message).
    """
    enhanced_prompt_text_from_llm = ""
    error_occurred = False
    final_error_message_to_user = (
        "Sorry, I encountered an error while trying to enhance your prompt."
    )
    prompt_design_strategies_doc = ""
    prompt_guide_2_doc = ""

    user_obj = (
        response_target.user
        if isinstance(response_target, discord.Interaction)
        else response_target.author
    )
    app_config = client_obj.config  # Assumes client_obj has a .config attribute

    # Determine how to send messages
    async def send_response(
        content: Optional[str] = None,
        view: Optional[discord.ui.View] = None,
        ephemeral: bool = False,
        initial: bool = False,
    ):
        if isinstance(response_target, discord.Interaction):
            _kwargs = {"ephemeral": ephemeral}
            if content is not None:
                _kwargs["content"] = content
            if view is not None:
                _kwargs["view"] = view

            # The original conditional for 'initial' existed, though both branches made the same type of call.
            # We'll maintain the conditional structure but use the corrected kwargs.
            if (
                initial
                and hasattr(response_target, "response")
                and not response_target.response.is_done()
            ):
                return await response_target.followup.send(**_kwargs)
            return await response_target.followup.send(**_kwargs)
        elif isinstance(response_target, discord.Message):
            # For message commands, typically reply or send to channel.
            # Let's use reply for subsequent messages if an initial one was sent, else channel.send or reply.
            # For simplicity now, always reply.
            send_kwargs = {"mention_author": False}
            if content:
                send_kwargs["content"] = content
            if view:
                send_kwargs["view"] = view
            # ephemeral not directly supported for message.reply, user sees it.
            return await response_target.reply(**send_kwargs)
        return None

    try:
        # Load prompt enhancement documents
        doc_path_base = "prompt_data"
        strategies_doc_path = os.path.join(doc_path_base, "prompt_design_strategies.md")
        guide_2_doc_path = os.path.join(doc_path_base, "prompt_guide_2.md")
        guide_3_doc_path = os.path.join(doc_path_base, "prompt_guide_3.md")

        try:
            async with aiofiles.open(strategies_doc_path, "r", encoding="utf-8") as f:
                prompt_design_strategies_doc = await f.read()
            async with aiofiles.open(guide_2_doc_path, "r", encoding="utf-8") as f:
                prompt_guide_2_doc = await f.read()
            async with aiofiles.open(guide_3_doc_path, "r", encoding="utf-8") as f:
                prompt_guide_3_doc = await f.read()
        except FileNotFoundError as fnf_err:
            logger.error(f"Prompt enhancement document not found: {fnf_err}")
            await send_response(
                content="A required document for prompt enhancement is missing. Please contact the bot administrator.",
                ephemeral=True,
            )
            return
        except IOError as io_err:
            logger.error(f"IOError reading prompt enhancement document: {io_err}")
            await send_response(
                content="Could not read a required document for prompt enhancement. Please contact the bot administrator.",
                ephemeral=True,
            )
            return

        default_enhance_model = "google/gemini-1.0-pro"
        enhance_model_str = app_config.get(
            "enhance_prompt_model", default_enhance_model
        )

        try:
            provider, model_name = enhance_model_str.split("/", 1)
        except ValueError:
            logger.error(
                f"Invalid format for enhance_prompt_model: '{enhance_model_str}'. Using default: {default_enhance_model}"
            )
            provider, model_name = default_enhance_model.split("/", 1)
            final_error_message_to_user = f"Error in config for enhancement model (using default). {final_error_message_to_user}"

        provider_config_from_app = app_config.get("providers", {}).get(provider, {})
        if not provider_config_from_app or not provider_config_from_app.get("api_keys"):
            logger.error(
                f"No API keys or config for provider '{provider}' (enhancement model)."
            )
            await send_response(
                content=f"Configuration error for enhancement model's provider '{provider}'. {final_error_message_to_user}",
                ephemeral=True,
            )
            return

        extra_params_from_app = app_config.get("extra_api_parameters", {}).copy()

        async for (
            text_chunk,
            finish_reason,
            _,
            error_message_llm,
        ) in enhance_prompt_with_llm(
            prompt_to_enhance=original_prompt_text,
            prompt_design_strategies_doc=prompt_design_strategies_doc,
            prompt_guide_2_doc=prompt_guide_2_doc,
            prompt_guide_3_doc=prompt_guide_3_doc,
            provider=provider,
            model_name=model_name,
            provider_config=provider_config_from_app,
            extra_params=extra_params_from_app,
            app_config=app_config,
        ):
            if error_message_llm:
                logger.error(f"LLM enhancement stream error: {error_message_llm}")
                final_error_message_to_user = (
                    f"An error occurred during enhancement: {error_message_llm}"
                )
                error_occurred = True
                break
            if text_chunk:
                enhanced_prompt_text_from_llm += text_chunk
            if finish_reason:
                logger.info(f"LLM enhancement finished. Reason: {finish_reason}")
                if finish_reason not in [
                    "stop",
                    "length",
                    "end_turn",
                    "FINISH_REASON_UNSPECIFIED",
                ]:
                    logger.warning(
                        f"Unexpected finish reason from LLM: {finish_reason}"
                    )
                break

        if error_occurred:
            await send_response(content=final_error_message_to_user, ephemeral=True)
            return

        if not enhanced_prompt_text_from_llm.strip():
            logger.warning("LLM returned an empty enhanced prompt.")
            await send_response(
                content="The LLM returned an empty enhanced prompt. Please try again.",
                ephemeral=True,
            )
            return

        # --- Response Formatting and Sending ---
        model_info_display_str = f"**(Enhanced by Model: {provider}/{model_name})**"
        full_enhanced_prompt_text_for_file = (
            enhanced_prompt_text_from_llm.strip()
        )  # Raw for file

        escaped_full_enhanced_text_for_display = discord.utils.escape_markdown(
            full_enhanced_prompt_text_for_file
        )

        # Prepare the button view
        simple_view_for_button = discord.ui.View(timeout=None)
        simple_view_for_button.add_item(ResponseActionView.GetTextFileButton(row=0))
        simple_view_for_button.full_response_text = full_enhanced_prompt_text_for_file
        simple_view_for_button.model_name = f"{provider}_{model_name}_enhanced_prompt"

        sent_message_for_button_hook: Optional[discord.Message] = None

        # Check if the combined initial parts + header + full enhanced text fits
        combined_message_for_single_send = (
            f"**Enhanced Prompt {model_info_display_str}:**\n"
            f"```\n{escaped_full_enhanced_text_for_display}\n```"
        )

        if len(combined_message_for_single_send) <= MAX_PLAIN_TEXT_LENGTH:
            sent_message_for_button_hook = await send_response(
                content=combined_message_for_single_send,
                view=simple_view_for_button,
                initial=True,
            )
        else:
            # Content is too long, split it
            messages_to_send_parts_list = []

            # Define a helper for splitting text meant for code blocks
            def split_text_for_code_blocks(
                text_content: str,
            ) -> List[str]:
                parts = []
                safety_margin = 30
                code_block_wrapper_len = len("```\n\n```")
                max_text_in_chunk = (
                    MAX_PLAIN_TEXT_LENGTH - code_block_wrapper_len - safety_margin
                )

                if max_text_in_chunk <= 10:
                    logger.warning(
                        f"Cannot effectively split text for code blocks, max_text_in_chunk: {max_text_in_chunk}. Sending truncated."
                    )
                    if text_content:
                        parts.append(
                            f"```\n{text_content[: max_text_in_chunk - len('...')] if max_text_in_chunk > len('...') else ''}...\n```"
                        )
                    else:
                        parts.append("```\n...\n```")
                    return parts

                idx = 0
                while idx < len(text_content):
                    chunk = text_content[idx : idx + max_text_in_chunk]
                    parts.append(f"```\n{chunk}\n```")
                    idx += len(chunk)
                return parts

            # Only Enhanced Prompt Header and Content
            messages_to_send_parts_list.append(
                f"**Enhanced Prompt {model_info_display_str}:**"
            )
            messages_to_send_parts_list.extend(
                split_text_for_code_blocks(escaped_full_enhanced_text_for_display)
            )

            # Sending logic
            last_sent_msg_obj = None
            for i, part_content_split in enumerate(messages_to_send_parts_list):
                is_last_part_split = i == len(messages_to_send_parts_list) - 1
                current_view_for_this_part = (
                    simple_view_for_button if is_last_part_split else None
                )

                send_kwargs_split = {}
                if current_view_for_this_part:
                    send_kwargs_split["view"] = current_view_for_this_part

                if not part_content_split.strip():
                    logger.warning(
                        f"Skipping empty message part at index {i} during enhance prompt split."
                    )
                    if (
                        is_last_part_split
                        and last_sent_msg_obj
                        and not last_sent_msg_obj.view
                    ):
                        try:
                            await last_sent_msg_obj.edit(view=simple_view_for_button)
                            sent_message_for_button_hook = last_sent_msg_obj
                        except discord.HTTPException as e_edit:
                            logger.error(
                                f"Failed to attach view to previous message on empty last part: {e_edit}"
                            )
                    continue

                last_sent_msg_obj = await send_response(
                    content=part_content_split,
                    **send_kwargs_split,
                    initial=(
                        i == 0 and isinstance(response_target, discord.Interaction)
                    ),
                )

                if is_last_part_split:
                    sent_message_for_button_hook = last_sent_msg_obj

        # Hook the sent message to the view if necessary
        if sent_message_for_button_hook and simple_view_for_button.children:
            # The view itself (simple_view_for_button) is what the button's callback will access via self.view
            # If the button's logic needs self.view.message, it should be set on simple_view_for_button
            if hasattr(simple_view_for_button, "message"):
                simple_view_for_button.message = sent_message_for_button_hook
            # For the GetTextFileButton, it accesses self.view.full_response_text and self.view.model_name,
            # which are already set on simple_view_for_button directly.

        logger.info(
            f'User {user_obj.id} ({user_obj.name}) successfully used enhanceprompt for: "{original_prompt_text[:50]}..." with model {provider}/{model_name}'
        )

    except Exception as e:
        logger.error(
            f"Critical error in _execute_enhance_prompt_logic for user {user_obj.id}: {e}",
            exc_info=True,
        )
        try:
            if isinstance(response_target, discord.Interaction):
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
        except discord.HTTPException as http_err:
            logger.error(
                f"Failed to send error message for _execute_enhance_prompt_logic: {http_err}"
            )


@app_commands.describe(prompt="The prompt you want to enhance.")
async def enhance_prompt_command(interaction: discord.Interaction, prompt: str):
    """
    Enhances a given prompt using an LLM based on predefined strategies.
    """
    if not hasattr(interaction.client, "config"):  # Basic check before defer
        logger.error(
            "Client object does not have a 'config' attribute for enhance_prompt_command."
        )
        await interaction.response.send_message(
            "Bot configuration error.", ephemeral=True
        )
        return

    await interaction.response.defer(ephemeral=False, thinking=True)
    await _execute_enhance_prompt_logic(interaction, prompt, interaction.client)
