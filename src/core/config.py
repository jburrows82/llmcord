"""
Configuration module for LLMCord.

This module provides the main configuration interface, delegating to specialized modules
for loading, validation, and default values.
"""

import logging
from typing import Dict, Any, Optional

# Import the modular components
from .config_loader import ConfigLoader, get_max_text_for_model
from .constants import (
    MAX_TEXT_LIMITS_CONFIG_KEY,
    DEFAULT_MAX_TEXT_KEY,
    MODEL_SPECIFIC_MAX_TEXT_KEY,
    MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY,
    MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY,
)

logger = logging.getLogger(__name__)


async def get_config(filename: str = "config/config.yaml") -> Optional[Dict[str, Any]]:
    """
    Load and validate configuration from a YAML file.
    
    This is the main entry point for loading configuration. It delegates to the
    ConfigLoader class which handles validation and normalization through the
    modular configuration system.
    
    Args:
        filename: Path to the configuration file (default: "config/config.yaml")
        
    Returns:
        Configuration dictionary or None if loading failed
        
    Example:
        >>> config = await get_config("config/config.yaml")
        >>> if config:
        ...     bot_token = config.get("bot_token")
        ...     model = config.get("model")
    """
    try:
        config_data = await ConfigLoader.load_config(filename)
        
        if config_data is None:
            logger.error("Failed to load configuration")
            return None
            
        logger.info("Configuration loaded successfully")
        return config_data
        
    except Exception as e:
        logger.critical(f"Critical error loading configuration: {e}", exc_info=True)
        return None


# Re-export get_max_text_for_model for backward compatibility
__all__ = ["get_config", "get_max_text_for_model"]
