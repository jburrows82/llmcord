import discord
from discord import app_commands
from discord.app_commands import Choice
from typing import List
import logging

from ..core.constants import AVAILABLE_MODELS
from .user_preferences import (
    load_all_preferences,
    get_user_model_preference,
    get_user_system_prompt_preference,
    get_user_gemini_thinking_budget_preference,
    save_model_preference,
    save_system_prompt_preference,
    save_gemini_thinking_preference,
)
from .prompt_enhancer import execute_enhance_prompt_logic

logger = logging.getLogger(__name__)


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
    return choices[:25]


@app_commands.autocomplete(model_full_name=model_autocomplete)
@app_commands.describe(
    model_full_name="The LLM provider and model (e.g., google/gemini-2.0-flash, openai/gpt-4.1)."
)
async def set_model_command(interaction: discord.Interaction, model_full_name: str):
    """Sets the user's preferred LLM (provider and model) for future interactions."""
    try:
        provider, model_name = model_full_name.split("/", 1)
    except ValueError:
        await interaction.response.send_message(
            f"Invalid model format: `{model_full_name}`. Please use the format `provider/model_name` (e.g., `openai/gpt-4.1`).",
            ephemeral=False,
        )
        return

    if provider not in AVAILABLE_MODELS:
        await interaction.response.send_message(
            f"Invalid provider: `{provider}`. Please choose from the suggestions.",
            ephemeral=False,
        )
        return

    if model_name not in AVAILABLE_MODELS.get(provider, []):
        await interaction.response.send_message(
            f"Invalid model: `{model_name}` for provider `{provider}`. Please choose from the suggestions.",
            ephemeral=False,
        )
        return

    user_id = interaction.user.id
    await save_model_preference(user_id, model_full_name)

    logger.info(
        f"User {user_id} ({interaction.user.name}) set model preference to: {model_full_name}"
    )
    await interaction.response.send_message(
        f"Your LLM model has been set to `{model_full_name}`.", ephemeral=False
    )


@app_commands.describe(
    prompt="Your custom system prompt. Use 'reset' to use the default prompt from config.yaml."
)
async def set_system_prompt_command(interaction: discord.Interaction, prompt: str):
    """
    Sets your custom system prompt for the bot.
    This prompt will be used to guide the AI's responses for you.
    To revert to the default system prompt, use 'reset' as the prompt.
    """
    user_id = interaction.user.id

    try:
        if prompt.lower() == "reset":
            await save_system_prompt_preference(user_id, None)
            logger.info(
                f"User {user_id} ({interaction.user.name}) reset their system prompt to default."
            )
            await interaction.response.send_message(
                "Your system prompt has been reset to the default.", ephemeral=False
            )
        else:
            modified_prompt = f"{prompt}\nDon't use LaTeX unless told by the user"
            await save_system_prompt_preference(user_id, modified_prompt)
            logger.info(
                f'User {user_id} ({interaction.user.name}) set system prompt to: "{modified_prompt[:100]}{"..." if len(modified_prompt) > 100 else ""}"'
            )
            await interaction.response.send_message(
                f'Your system prompt has been set to: "{modified_prompt[:200]}{"..." if len(modified_prompt) > 200 else ""}"',
                ephemeral=False,
            )

    except Exception as e:
        logger.exception(f"Error in set_system_prompt_command for user {user_id}: {e}")
        await _send_error_response(
            interaction, "An error occurred while setting your system prompt."
        )


@app_commands.describe(
    enabled="Set to 'True' to use the thinking budget for Gemini, 'False' to disable it for your interactions."
)
async def setgeminithinking(interaction: discord.Interaction, enabled: bool):
    """
    Sets your preference for using the 'thinkingBudget' parameter with Gemini models.
    This can potentially improve response quality for complex queries but may increase latency.
    The actual budget value is set globally in config.yaml.
    """
    user_id = interaction.user.id

    try:
        await save_gemini_thinking_preference(user_id, enabled)
        logger.info(
            f"User {user_id} ({interaction.user.name}) set Gemini thinking budget usage to: {enabled}"
        )

        status_message = "enabled" if enabled else "disabled"
        await interaction.response.send_message(
            f"Your preference for Gemini 'thinkingBudget' has been set to **{status_message}**.",
            ephemeral=False,
        )

    except Exception as e:
        logger.exception(f"Error in setgeminithinking command for user {user_id}: {e}")
        await _send_error_response(
            interaction,
            "An error occurred while setting your Gemini thinking budget preference.",
        )


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
        await interaction.response.send_message(
            "Could not display help information. Please check bot logs.", ephemeral=True
        )


@app_commands.describe(prompt="The prompt you want to enhance.")
async def enhance_prompt_command(interaction: discord.Interaction, prompt: str):
    """Enhances a given prompt using an LLM based on predefined strategies."""
    if not hasattr(interaction.client, "config"):
        logger.error(
            "Client object does not have a 'config' attribute for enhance_prompt_command."
        )
        await interaction.response.send_message(
            "Bot configuration error.", ephemeral=True
        )
        return

    await interaction.response.defer(ephemeral=False, thinking=True)
    await execute_enhance_prompt_logic(interaction, prompt, interaction.client)


async def _send_error_response(interaction: discord.Interaction, message: str):
    """Helper function to send error responses consistently."""
    try:
        if not interaction.response.is_done():
            await interaction.response.send_message(message, ephemeral=False)
        else:
            await interaction.followup.send(message, ephemeral=False)
    except discord.HTTPException as http_err:
        logger.error(f"Failed to send error message: {http_err}")


# Export the preference getter functions for use by other modules
__all__ = [
    "load_all_preferences",
    "get_user_model_preference",
    "get_user_system_prompt_preference",
    "get_user_gemini_thinking_budget_preference",
    "set_model_command",
    "set_system_prompt_command",
    "setgeminithinking",
    "help_command",
    "enhance_prompt_command",
]
