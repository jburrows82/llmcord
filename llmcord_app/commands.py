import discord
from discord import app_commands
from discord.app_commands import Choice
from typing import List, Dict, Optional, Any
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
)

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
