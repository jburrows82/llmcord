# llmcord_app/commands.py
import discord
from discord import app_commands
from discord.app_commands import Choice
from typing import List, Dict
import logging # <-- Add this import

from .constants import AVAILABLE_MODELS

# This dictionary will store user preferences {user_id: "provider/model_name"}
# It should be managed by the bot instance or a dedicated state manager in a real app.
# For this refactor, we'll keep it simple as a module-level dict.
user_model_preferences: Dict[int, str] = {}

# Get a logger for this module
logger = logging.getLogger(__name__) # <-- Add this line

# --- Slash Command Autocomplete Functions ---
async def provider_autocomplete(interaction: discord.Interaction, current: str) -> List[Choice[str]]:
    """Autocompletes the provider argument."""
    providers = list(AVAILABLE_MODELS.keys())
    return [
        Choice(name=provider, value=provider)
        for provider in providers if current.lower() in provider.lower()
    ][:25] # Limit to 25 choices

async def model_autocomplete(interaction: discord.Interaction, current: str) -> List[Choice[str]]:
    """Autocompletes the model argument based on the selected provider."""
    # Access the provider value entered by the user so far
    provider = interaction.namespace.provider # Correct way to access other options

    if not provider or provider not in AVAILABLE_MODELS:
        # If provider is empty or invalid, return no model suggestions
        return []

    models = AVAILABLE_MODELS.get(provider, []) # Get models for the valid provider
    return [
        Choice(name=model, value=model)
        for model in models if current.lower() in model.lower()
    ][:25] # Limit to 25 choices


# --- Slash Command Definition ---
# Note: The command registration (@discord_client.tree.command) happens in bot.py
# This file just defines the command function and its logic.

@app_commands.autocomplete(provider=provider_autocomplete)
@app_commands.autocomplete(model=model_autocomplete)
@app_commands.describe(provider="The LLM provider (e.g., google, openai).", model="The specific model name (e.g., gemini-2.0-flash, gpt-4.1).")
async def set_model_command(interaction: discord.Interaction, provider: str, model: str):
    """
    Sets the user's preferred LLM provider and model for future interactions.
    """
    global user_model_preferences

    # Validate provider
    if provider not in AVAILABLE_MODELS:
        await interaction.response.send_message(f"Invalid provider: `{provider}`. Please choose from the suggestions.", ephemeral=False)
        return

    # Validate model against the selected provider
    if model not in AVAILABLE_MODELS.get(provider, []):
        await interaction.response.send_message(f"Invalid model: `{model}` for provider `{provider}`. Please choose from the suggestions.", ephemeral=False)
        return

    # Store the preference
    user_id = interaction.user.id
    model_preference = f"{provider}/{model}"
    user_model_preferences[user_id] = model_preference
    # Use the logger obtained earlier
    logger.info(f"User {user_id} ({interaction.user.name}) set model preference to: {model_preference}") # <-- Corrected line

    await interaction.response.send_message(f"Your LLM model has been set to `{model_preference}`.", ephemeral=False)

def get_user_model_preference(user_id: int, default_model: str) -> str:
    """Gets the user's model preference or the default."""
    return user_model_preferences.get(user_id, default_model)