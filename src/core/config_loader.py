import yaml
import logging
import aiofiles
from typing import Dict, Any, Optional

from .config_validator import ConfigValidator
from .config_defaults import (
    ALT_SEARCH_SECTION_KEY,
    ALT_SEARCH_ENABLED_KEY,
    ALT_SEARCH_PROMPT_KEY,
    DEFAULT_ALT_SEARCH_PROMPT_TEMPLATE,
)
from .constants import (
    MAX_TEXT_LIMITS_CONFIG_KEY,
    DEFAULT_MAX_TEXT_KEY,
    MODEL_SPECIFIC_MAX_TEXT_KEY,
    MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY,
    MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY,
)


logger = logging.getLogger(__name__)


class ConfigLoader:
    """Handles loading and processing of configuration files."""

    @staticmethod
    async def load_config(
        filename: str = "config/config.yaml",
    ) -> Optional[Dict[str, Any]]:
        """
        Load, validate, and return the configuration from a YAML file.

        Args:
            filename: Path to the configuration file

        Returns:
            Configuration dictionary or None if loading failed
        """
        try:
            # Load raw YAML data
            config_data = await ConfigLoader._load_yaml_file(filename)
            if config_data is None:
                return None

            # Validate and normalize configuration - this now handles ALL validation
            config_data = ConfigValidator.validate_and_normalize_config(
                config_data, filename
            )

            # Load additional configuration sections that need special handling
            ConfigLoader._load_alternative_search_config(config_data)

            return config_data

        except Exception as e:
            logger.critical(f"Unexpected error loading config: {e}", exc_info=True)
            return None

    @staticmethod
    async def _load_yaml_file(filename: str) -> Optional[Dict[str, Any]]:
        """Load and parse YAML file."""
        try:
            async with aiofiles.open(filename, "r", encoding="utf-8") as file:
                content = await file.read()
                config_data = yaml.safe_load(content)

                if not isinstance(config_data, dict):
                    logger.error(
                        f"CRITICAL: {filename} is not a valid YAML dictionary."
                    )
                    return None

                return config_data

        except FileNotFoundError:
            logger.critical(
                f"CRITICAL: {filename} not found. Please copy config-example.yaml "
                f"to config.yaml and configure it."
            )
            return None
        except yaml.YAMLError as e:
            logger.critical(f"CRITICAL: Error parsing {filename}: {e}")
            return None

    @staticmethod
    def _load_alternative_search_config(config_data: Dict[str, Any]):
        """Load alternative search query generation configuration with full validation."""
        if ALT_SEARCH_SECTION_KEY not in config_data:
            config_data[ALT_SEARCH_SECTION_KEY] = {}
            logger.info(
                f"'{ALT_SEARCH_SECTION_KEY}' section not found. Creating with defaults."
            )

        alt_search_config = config_data[ALT_SEARCH_SECTION_KEY]
        if not isinstance(alt_search_config, dict):
            logger.warning(
                f"'{ALT_SEARCH_SECTION_KEY}' section is not a dictionary. Using defaults."
            )
            alt_search_config = {}
            config_data[ALT_SEARCH_SECTION_KEY] = alt_search_config

        # Validate enabled state
        if ALT_SEARCH_ENABLED_KEY not in alt_search_config:
            alt_search_config[ALT_SEARCH_ENABLED_KEY] = False
            logger.info(
                f"'{ALT_SEARCH_ENABLED_KEY}' not found in '{ALT_SEARCH_SECTION_KEY}'. Defaulting to False."
            )
        elif not isinstance(alt_search_config[ALT_SEARCH_ENABLED_KEY], bool):
            logger.warning(
                f"'{ALT_SEARCH_ENABLED_KEY}' in '{ALT_SEARCH_SECTION_KEY}' is not a boolean. Defaulting to False."
            )
            alt_search_config[ALT_SEARCH_ENABLED_KEY] = False

        # Validate prompt template
        if (
            ALT_SEARCH_PROMPT_KEY not in alt_search_config
            or not alt_search_config.get(ALT_SEARCH_PROMPT_KEY)
            or not isinstance(alt_search_config[ALT_SEARCH_PROMPT_KEY], str)
        ):
            alt_search_config[ALT_SEARCH_PROMPT_KEY] = (
                DEFAULT_ALT_SEARCH_PROMPT_TEMPLATE
            )
            logger.info(
                f"'{ALT_SEARCH_PROMPT_KEY}' not found, empty, or invalid in '{ALT_SEARCH_SECTION_KEY}'. Using default template."
            )


# For backward compatibility, provide the same interface as the original config.py
async def get_config(filename: str = "config/config.yaml") -> Optional[Dict[str, Any]]:
    """
    Load configuration (backward compatibility function).

    Args:
        filename: Path to the configuration file

    Returns:
        Configuration dictionary or None if loading failed
    """
    return await ConfigLoader.load_config(filename)


def get_max_text_for_model(config: Dict[str, Any], model_identifier: str) -> int:
    """
    Get the maximum text limit for a specific model.

    Args:
        config: Configuration dictionary
        model_identifier: Model identifier string

    Returns:
        Maximum text limit for the model
    """
    from .config_defaults import (
        DEFAULT_MAX_TEXT_SAFETY_MARGIN,
        DEFAULT_MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN,
        DEFAULT_MAX_TEXT_VALUE,
    )

    max_text_limits = config.get(MAX_TEXT_LIMITS_CONFIG_KEY, {})

    # Get model-specific limit first
    model_specific_limits = max_text_limits.get(MODEL_SPECIFIC_MAX_TEXT_KEY, {})
    if model_identifier in model_specific_limits:
        base_limit = model_specific_limits[model_identifier]
    else:
        # Fall back to default - use the higher 128k value to match original
        base_limit = max_text_limits.get(DEFAULT_MAX_TEXT_KEY, DEFAULT_MAX_TEXT_VALUE)

    # Apply safety margin - now using fixed amount subtraction (like original)
    safety_margin = config.get(
        MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY, DEFAULT_MAX_TEXT_SAFETY_MARGIN
    )
    min_limit_after_margin = config.get(
        MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY,
        DEFAULT_MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN,
    )

    # Calculate limit with fixed amount safety margin (not percentage)
    limit_with_margin = base_limit - safety_margin

    # Ensure minimum limit
    final_limit = max(limit_with_margin, min_limit_after_margin)

    # Log if we hit the minimum cap for debugging
    if final_limit != limit_with_margin:
        logger.info(
            f"Applied safety margin ({safety_margin}) to '{model_identifier}'. "
            f"Raw: {base_limit}, Adjusted: {limit_with_margin}, Final (after min check): {final_limit}"
        )

    return final_limit
