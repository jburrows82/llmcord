import logging
from typing import Tuple, Dict, Any, List, Set
import discord

from ..core.constants import (
    AVAILABLE_MODELS,
    DEEP_SEARCH_KEYWORDS,
    VISION_MODEL_TAGS,
    FALLBACK_VISION_MODEL_CONFIG_KEY,
    DEEP_SEARCH_MODEL_CONFIG_KEY,
)
from ..bot.commands import get_user_model_preference

# Define a return type for clarity, though it's a bit complex
ModelSelectionResult = Tuple[
    str,  # final_provider_slash_model
    str,  # provider
    str,  # model_name
    Dict[str, Any],  # provider_config
    List[str],  # all_api_keys
    bool,  # is_gemini
    bool,  # is_grok_model
    bool,  # keys_required
    bool,  # accept_files (vision capability)
]


def determine_final_model(
    user_id: int,
    initial_cleaned_content: str,  # Content after basic cleaning (mentions, "at ai")
    image_attachments: List[discord.Attachment],
    has_potential_image_urls_in_text: bool,
    config: Dict[str, Any],
    user_warnings: Set[str],
) -> ModelSelectionResult:
    """
    Determines the final LLM provider and model to use based on user preferences,
    keywords, and image presence.
    """
    default_model_str = config.get(
        "model", "google/gemini-2.5-flash-preview-05-20"
    )  # More robust default
    provider_slash_model = get_user_model_preference(user_id, default_model_str)
    final_provider_slash_model = provider_slash_model

    # --- Override Model based on Keywords (e.g., deepsearch) ---
    # Check if 'googlelens' keyword was already processed and removed from initial_cleaned_content
    # For this function, we assume initial_cleaned_content is *before* Google Lens keyword removal
    # if Google Lens is active, deep search override might be skipped by the caller.
    # Here, we just check for deep search keywords.

    if any(
        keyword in initial_cleaned_content.lower() for keyword in DEEP_SEARCH_KEYWORDS
    ):
        target_model_str = config.get(
            DEEP_SEARCH_MODEL_CONFIG_KEY,
            "x-ai/grok-1",  # Default deep search model
        )
        try:
            target_provider, target_model_name = target_model_str.split("/", 1)
            provider_exists = target_provider in config.get("providers", {})
            keys_exist = bool(
                config.get("providers", {}).get(target_provider, {}).get("api_keys")
            )
            is_target_available_in_constants = (
                target_provider in AVAILABLE_MODELS
                and target_model_name in AVAILABLE_MODELS.get(target_provider, [])
            )

            if provider_exists and keys_exist and is_target_available_in_constants:
                final_provider_slash_model = target_model_str
                logging.info(
                    f"Keywords {DEEP_SEARCH_KEYWORDS} detected. Overriding model to {final_provider_slash_model} for user {user_id}."
                )
            else:
                reason = (
                    "config or keys missing"
                    if not (provider_exists and keys_exist)
                    else "not in AVAILABLE_MODELS"
                )
                logging.warning(
                    f"Keywords {DEEP_SEARCH_KEYWORDS} detected, but cannot use '{target_model_str}' ({reason}). Using original: {provider_slash_model}"
                )
                user_warnings.add(
                    f"⚠️ Deep search requested, but model '{target_model_str}' unavailable ({reason})."
                )
        except ValueError:
            logging.error(
                f"Invalid format for DEEP_SEARCH_MODEL_CONFIG_KEY: '{target_model_str}'"
            )
            user_warnings.add(
                f"⚠️ Deep search model misconfigured. Using '{provider_slash_model}'."
            )

    # --- Validate Final Model Selection (after potential keyword override) ---
    try:
        provider, model_name = final_provider_slash_model.split("/", 1)
        if not (
            provider in AVAILABLE_MODELS
            and model_name in AVAILABLE_MODELS.get(provider, [])
            and provider in config.get("providers", {})  # Ensure provider is configured
        ):
            logging.warning(
                f"Final model '{final_provider_slash_model}' is invalid/unavailable or provider not configured. Falling back to default: {default_model_str}"
            )
            user_warnings.add(
                f"⚠️ Model '{final_provider_slash_model}' unavailable. Using default."
            )
            final_provider_slash_model = default_model_str
            provider, model_name = final_provider_slash_model.split("/", 1)
    except ValueError:
        logging.error(
            f"Invalid model format for final selection '{final_provider_slash_model}'. Using hardcoded default."
        )
        user_warnings.add(
            f"⚠️ Invalid model format '{final_provider_slash_model}'. Using default."
        )
        final_provider_slash_model = (
            "google/gemini-2.5-flash-preview-05-20"  # Hard fallback
        )
        provider, model_name = final_provider_slash_model.split("/", 1)

    logging.info(
        f"Model selected before vision check for user {user_id}: '{final_provider_slash_model}'"
    )

    # --- Get Config for the Selected Provider ---
    provider_config = config.get("providers", {}).get(provider, {})
    if not isinstance(
        provider_config, dict
    ):  # Should not happen if validation above worked
        logging.error(
            f"CRITICAL: Configuration for provider '{provider}' is invalid. This should have been caught."
        )
        # This is a critical state, potentially raise an error or use a hard fallback
        # For now, let's assume a very basic fallback to prevent outright crash
        provider_config = {}  # Minimal config
        user_warnings.add(
            f"⚠️ Critical config error for provider '{provider}'. Features may be limited."
        )

    all_api_keys = provider_config.get("api_keys", [])
    is_gemini = provider == "google"
    is_grok_model = provider == "x-ai"  # Assuming x-ai is the provider for grok models
    keys_required = provider not in ["ollama", "lmstudio", "vllm", "oobabooga", "jan"]

    if keys_required and not all_api_keys:
        logging.error(
            f"No API keys configured for the selected provider '{provider}' which requires them."
        )
        user_warnings.add(
            f"⚠️ No API keys for model '{final_provider_slash_model}'. It may not work."
        )
        # Depending on strictness, could raise error or try to continue if a keyless model was intended

    # --- Determine Vision Capability and Handle Fallback ---
    current_model_accepts_files = any(
        tag in model_name.lower() for tag in VISION_MODEL_TAGS
    )
    if provider == "openai" and provider_config.get("disable_vision", False):
        current_model_accepts_files = False
        logging.info(
            f"Vision explicitly disabled for OpenAI model '{model_name}' via config."
        )

    if (
        image_attachments or has_potential_image_urls_in_text
    ) and not current_model_accepts_files:
        original_model_for_warning = final_provider_slash_model
        fallback_vision_model_str = config.get(
            FALLBACK_VISION_MODEL_CONFIG_KEY, "google/gemini-2.5-flash-preview-05-20"
        )
        logging.info(
            f"Query has images, but current model '{final_provider_slash_model}' does not support vision. Attempting to switch to fallback '{fallback_vision_model_str}'."
        )

        try:
            fb_provider, fb_model_name = fallback_vision_model_str.split("/", 1)
            fb_provider_config = config.get("providers", {}).get(fb_provider, {})
            fb_keys_exist = bool(fb_provider_config.get("api_keys"))
            fb_keys_required = fb_provider not in [
                "ollama",
                "lmstudio",
                "vllm",
                "oobabooga",
                "jan",
            ]

            is_fb_model_available = (
                fb_provider in AVAILABLE_MODELS
                and fb_model_name in AVAILABLE_MODELS.get(fb_provider, [])
                and fb_provider in config.get("providers", {})  # Provider configured
            )
            fb_model_accepts_files = any(
                tag in fb_model_name.lower() for tag in VISION_MODEL_TAGS
            )
            if fb_provider == "openai" and fb_provider_config.get(
                "disable_vision", False
            ):
                fb_model_accepts_files = False

            if (
                is_fb_model_available
                and fb_model_accepts_files
                and (not fb_keys_required or fb_keys_exist)
            ):
                user_warnings.add(
                    f"⚠️ Images detected. Switched from '{original_model_for_warning}' to vision model '{fallback_vision_model_str}'."
                )
                final_provider_slash_model = fallback_vision_model_str
                provider = fb_provider
                model_name = fb_model_name
                provider_config = fb_provider_config
                all_api_keys = fb_provider_config.get("api_keys", [])
                is_gemini = provider == "google"
                is_grok_model = provider == "x-ai"
                keys_required = fb_keys_required
                current_model_accepts_files = True  # Now it does
            else:
                reason = (
                    "not configured/keys missing"
                    if not (
                        fb_provider_config and (not fb_keys_required or fb_keys_exist)
                    )
                    else "not vision capable or unavailable"
                )
                user_warnings.add(
                    f"⚠️ Fallback vision model '{fallback_vision_model_str}' unavailable ({reason}). Images may not be processed."
                )
                logging.warning(
                    f"Could not switch to fallback vision model '{fallback_vision_model_str}' ({reason}). Sticking with '{original_model_for_warning}'."
                )
        except ValueError:
            logging.error(
                f"Invalid format for FALLBACK_VISION_MODEL_CONFIG_KEY: '{fallback_vision_model_str}'."
            )
            user_warnings.add(
                "⚠️ Fallback vision model misconfigured. Images may not be processed."
            )

    logging.info(
        f"Final model determined for user {user_id}: '{final_provider_slash_model}', Accepts files: {current_model_accepts_files}"
    )

    return (
        final_provider_slash_model,
        provider,
        model_name,
        provider_config,
        all_api_keys,
        is_gemini,
        is_grok_model,
        keys_required,
        current_model_accepts_files,
    )
