import logging
from typing import Dict, Any, List

from .constants import (
    MAX_MESSAGE_NODES_CONFIG_KEY,
    EDIT_DELAY_SECONDS_CONFIG_KEY,
    SEARXNG_BASE_URL_CONFIG_KEY,
    SEARXNG_DEFAULT_URL,
    GROUNDING_SYSTEM_PROMPT_CONFIG_KEY,
    GEMINI_USE_THINKING_BUDGET_CONFIG_KEY,
    GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY,
    GEMINI_DEFAULT_USE_THINKING_BUDGET,
    GEMINI_DEFAULT_THINKING_BUDGET_VALUE,
    GEMINI_MIN_THINKING_BUDGET_VALUE,
    GEMINI_MAX_THINKING_BUDGET_VALUE,
    SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY,
    SEARXNG_DEFAULT_URL_CONTENT_MAX_LENGTH,
    SEARXNG_NUM_RESULTS_CONFIG_KEY,
    GEMINI_SAFETY_SETTINGS_CONFIG_KEY,
    # Model configuration keys
    GROUNDING_MODEL_CONFIG_KEY,
    FALLBACK_VISION_MODEL_CONFIG_KEY,
    FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY,
    DEEP_SEARCH_MODEL_CONFIG_KEY,
    FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY,
    DEFAULT_FALLBACK_MODEL_SYSTEM_PROMPT,
    # Grounding model parameters
    GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY,
    DEFAULT_GROUNDING_MODEL_TEMPERATURE,
    GROUNDING_MODEL_TOP_K_CONFIG_KEY,
    DEFAULT_GROUNDING_MODEL_TOP_K,
    GROUNDING_MODEL_TOP_P_CONFIG_KEY,
    DEFAULT_GROUNDING_MODEL_TOP_P,
    GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY,
    GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY,
    GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET,
    GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE,
    # Output sharing
    OUTPUT_SHARING_CONFIG_KEY,
    TEXTIS_ENABLED_CONFIG_KEY,
    URL_SHORTENER_ENABLED_CONFIG_KEY,
    URL_SHORTENER_SERVICE_CONFIG_KEY,
    # URL extractors
    MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY,
    FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY,
    DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR,
    DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR,
    VALID_URL_EXTRACTORS,
    # Jina settings
    JINA_ENGINE_MODE_CONFIG_KEY,
    DEFAULT_JINA_ENGINE_MODE,
    VALID_JINA_ENGINE_MODES,
    JINA_WAIT_FOR_SELECTOR_CONFIG_KEY,
    DEFAULT_JINA_WAIT_FOR_SELECTOR,
    JINA_TIMEOUT_CONFIG_KEY,
    DEFAULT_JINA_TIMEOUT,
    # Crawl4AI settings
    CRAWL4AI_CACHE_MODE_CONFIG_KEY,
    DEFAULT_CRAWL4AI_CACHE_MODE,
    VALID_CRAWL4AI_CACHE_MODES,
    # External Web Content API
    WEB_CONTENT_EXTRACTION_API_ENABLED_CONFIG_KEY,
    WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY,
    WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY,
    WEB_CONTENT_EXTRACTION_API_CACHE_TTL_CONFIG_KEY,
    HTTP_CLIENT_USE_HTTP2_CONFIG_KEY,
    DEFAULT_WEB_CONTENT_EXTRACTION_API_ENABLED,
    DEFAULT_WEB_CONTENT_EXTRACTION_API_URL,
    DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS,
    DEFAULT_WEB_CONTENT_EXTRACTION_API_CACHE_TTL,
    DEFAULT_HTTP_CLIENT_USE_HTTP2,
    # Text limits
    MAX_TEXT_LIMITS_CONFIG_KEY,
    DEFAULT_MAX_TEXT_KEY,
    MODEL_SPECIFIC_MAX_TEXT_KEY,
    MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY,
    MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY,
    # Other keys
    RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY,
    PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY,
    AUTO_RENDER_MARKDOWN_TABLES_CONFIG_KEY,
    DEFAULT_AUTO_RENDER_MARKDOWN_TABLES,
)

from .config_defaults import (
    DEFAULT_GROUNDING_SYSTEM_PROMPT,
    DEFAULT_ALT_SEARCH_PROMPT_TEMPLATE,
    DEFAULT_PROMPT_ENHANCER_SYSTEM_PROMPT,
    DEFAULT_GEMINI_SAFETY_SETTINGS,
    ALT_SEARCH_SECTION_KEY,
    ALT_SEARCH_ENABLED_KEY,
    ALT_SEARCH_PROMPT_KEY,
)


logger = logging.getLogger(__name__)


class ConfigValidator:
    """Handles configuration validation and normalization."""
    
    @staticmethod
    def validate_and_normalize_config(config_data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """
        Validate and normalize the configuration data.
        
        Args:
            config_data: Raw configuration dictionary
            filename: Configuration file name for logging
            
        Returns:
            Validated and normalized configuration dictionary
        """
        if not isinstance(config_data, dict):
            logger.error(f"CRITICAL: {filename} is not a valid YAML dictionary.")
            return {}
        
        # Validate and normalize each section
        ConfigValidator._validate_providers(config_data, filename)
        ConfigValidator._validate_serpapi_keys(config_data, filename)
        ConfigValidator._validate_discord_config(config_data, filename)
        ConfigValidator._validate_numeric_configs(config_data)
        ConfigValidator._validate_permissions(config_data)
        ConfigValidator._validate_searxng_config(config_data)
        ConfigValidator._validate_gemini_config(config_data)
        ConfigValidator._validate_gemini_safety_settings(config_data)
        ConfigValidator._validate_model_selections(config_data)
        ConfigValidator._validate_grounding_model_params(config_data)
        ConfigValidator._validate_output_sharing_config(config_data)
        ConfigValidator._validate_url_extractor_config(config_data)
        ConfigValidator._validate_web_content_api_config(config_data)
        ConfigValidator._validate_text_limits_config(config_data, filename)
        ConfigValidator._validate_misc_config(config_data)
        
        return config_data
    
    @staticmethod
    def _validate_providers(config_data: Dict[str, Any], filename: str):
        """Validate and normalize provider configurations."""
        providers = config_data.get("providers", {})
        if not isinstance(providers, dict):
            logger.warning("Config Warning: 'providers' section is not a dictionary. Treating as empty.")
            providers = {}
            config_data["providers"] = providers
        
        for name, provider_cfg in providers.items():
            if provider_cfg and isinstance(provider_cfg, dict):
                ConfigValidator._normalize_api_keys(provider_cfg, name, filename)
                ConfigValidator._validate_openai_config(provider_cfg, name)
            elif provider_cfg is not None:
                logger.warning(
                    f"Config Warning: Provider '{name}' configuration is not a dictionary. Ignoring provider."
                )
    
    @staticmethod
    def _normalize_api_keys(provider_cfg: Dict[str, Any], provider_name: str, filename: str):
        """Normalize API keys from single key to list format."""
        single_key = provider_cfg.get("api_key")
        key_list = provider_cfg.get("api_keys")
        
        if single_key and not key_list:
            logger.warning(
                f"Config Warning: Provider '{provider_name}' uses deprecated 'api_key'. "
                f"Converting to 'api_keys' list. Please update {filename}."
            )
            provider_cfg["api_keys"] = [single_key]
            del provider_cfg["api_key"]
        elif single_key and key_list:
            logger.warning(
                f"Config Warning: Provider '{provider_name}' has both 'api_key' and 'api_keys'. "
                f"Using 'api_keys'. Please remove 'api_key' from {filename}."
            )
            del provider_cfg["api_key"]
        elif key_list is None:
            # Allow providers without keys (like Ollama)
            provider_cfg["api_keys"] = []
        elif not isinstance(key_list, list):
            logger.error(
                f"Config Error: Provider '{provider_name}' has 'api_keys' but it's not a list. Treating as empty."
            )
            provider_cfg["api_keys"] = []
    
    @staticmethod
    def _validate_openai_config(provider_cfg: Dict[str, Any], provider_name: str):
        """Validate OpenAI-specific configuration."""
        if provider_name == "openai":
            if "disable_vision" not in provider_cfg:
                provider_cfg["disable_vision"] = False
                logger.info("OpenAI 'disable_vision' not set in config. Assuming False.")
            elif not isinstance(provider_cfg["disable_vision"], bool):
                logger.warning("OpenAI 'disable_vision' is not a boolean. Defaulting to False.")
                provider_cfg["disable_vision"] = False
    
    @staticmethod
    def _validate_serpapi_keys(config_data: Dict[str, Any], filename: str):
        """Validate and normalize SerpAPI keys."""
        single_serp_key = config_data.get("serpapi_api_key")
        serp_key_list = config_data.get("serpapi_api_keys")
        
        if single_serp_key and not serp_key_list:
            logger.warning(
                f"Config Warning: Found 'serpapi_api_key'. Converting to 'serpapi_api_keys' list. "
                f"Please update {filename}."
            )
            config_data["serpapi_api_keys"] = [single_serp_key]
            del config_data["serpapi_api_key"]
        elif single_serp_key and serp_key_list:
            logger.warning(
                f"Config Warning: Found both 'serpapi_api_key' and 'serpapi_api_keys'. "
                f"Using 'serpapi_api_keys'. Please remove 'serpapi_api_key' from {filename}."
            )
            del config_data["serpapi_api_key"]
        elif serp_key_list is None:
            config_data["serpapi_api_keys"] = []
        elif not isinstance(serp_key_list, list):
            logger.error("Config Error: Found 'serpapi_api_keys' but it's not a list. Treating as empty.")
            config_data["serpapi_api_keys"] = []
    
    @staticmethod
    def _validate_discord_config(config_data: Dict[str, Any], filename: str):
        """Validate Discord-specific configuration."""
        if not config_data.get("bot_token"):
            logger.error(f"CRITICAL: bot_token is not set in {filename}")
        
        if not config_data.get("client_id"):
            logger.warning(f"client_id not found in {filename}. Cannot generate invite URL.")
    
    @staticmethod
    def _validate_numeric_configs(config_data: Dict[str, Any]):
        """Validate numeric configuration values."""
        # Max message nodes
        ConfigValidator._validate_int_config(
            config_data, MAX_MESSAGE_NODES_CONFIG_KEY, 500, min_value=1
        )
        
        # Edit delay seconds
        ConfigValidator._validate_float_config(
            config_data, EDIT_DELAY_SECONDS_CONFIG_KEY, 1.0, min_value=0.0
        )
        
        # SearxNG number of results
        ConfigValidator._validate_int_config(
            config_data, SEARXNG_NUM_RESULTS_CONFIG_KEY, 5, min_value=1
        )
        
        # SearxNG URL content max length
        ConfigValidator._validate_int_config(
            config_data, SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY, 
            SEARXNG_DEFAULT_URL_CONTENT_MAX_LENGTH, min_value=1
        )
        
        # Rate limit cooldown hours
        ConfigValidator._validate_int_config(
            config_data, RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY, 24, min_value=1
        )
    
    @staticmethod
    def _validate_int_config(config_data: Dict[str, Any], key: str, default: int, min_value: int = None):
        """Validate an integer configuration value."""
        if key not in config_data:
            config_data[key] = default
            logger.info(f"'{key}' not found. Using default: {default}")
        else:
            try:
                val = int(config_data[key])
                if min_value is not None and val < min_value:
                    logger.warning(f"'{key}' ({val}) must be >= {min_value}. Using default: {default}")
                    config_data[key] = default
                else:
                    config_data[key] = val
            except ValueError:
                logger.warning(f"'{key}' is not a valid integer. Using default: {default}")
                config_data[key] = default
    
    @staticmethod
    def _validate_float_config(config_data: Dict[str, Any], key: str, default: float, min_value: float = None):
        """Validate a float configuration value."""
        if key not in config_data:
            config_data[key] = default
            logger.info(f"'{key}' not found. Using default: {default}")
        else:
            try:
                val = float(config_data[key])
                if min_value is not None and val < min_value:
                    logger.warning(f"'{key}' ({val}) must be >= {min_value}. Using default: {default}")
                    config_data[key] = default
                else:
                    config_data[key] = val
            except ValueError:
                logger.warning(f"'{key}' is not a valid float. Using default: {default}")
                config_data[key] = default
    
    @staticmethod
    def _validate_permissions(config_data: Dict[str, Any]):
        """Validate and ensure permissions structure exists."""
        if "permissions" not in config_data:
            config_data["permissions"] = {}
        
        perms = config_data["permissions"]
        for key in ["users", "roles", "channels"]:
            if key not in perms:
                perms[key] = {}
            if "allowed_ids" not in perms[key]:
                perms[key]["allowed_ids"] = []
            if "blocked_ids" not in perms[key]:
                perms[key]["blocked_ids"] = []
    
    @staticmethod
    def _validate_searxng_config(config_data: Dict[str, Any]):
        """Validate SearxNG configuration."""
        if SEARXNG_BASE_URL_CONFIG_KEY not in config_data:
            logger.info(
                f"'{SEARXNG_BASE_URL_CONFIG_KEY}' not found. Using default: {SEARXNG_DEFAULT_URL}"
            )
            config_data[SEARXNG_BASE_URL_CONFIG_KEY] = SEARXNG_DEFAULT_URL
        elif not config_data.get(SEARXNG_BASE_URL_CONFIG_KEY):
            logger.warning(
                f"'{SEARXNG_BASE_URL_CONFIG_KEY}' is empty. Using default: {SEARXNG_DEFAULT_URL}"
            )
            config_data[SEARXNG_BASE_URL_CONFIG_KEY] = SEARXNG_DEFAULT_URL
        
        # Validate grounding system prompt
        if (GROUNDING_SYSTEM_PROMPT_CONFIG_KEY not in config_data or 
            not config_data.get(GROUNDING_SYSTEM_PROMPT_CONFIG_KEY)):
            logger.info(f"'{GROUNDING_SYSTEM_PROMPT_CONFIG_KEY}' not found or empty. Using default.")
            config_data[GROUNDING_SYSTEM_PROMPT_CONFIG_KEY] = DEFAULT_GROUNDING_SYSTEM_PROMPT
    
    @staticmethod
    def _validate_gemini_config(config_data: Dict[str, Any]):
        """Validate Gemini-specific configuration."""
        # Validate thinking budget usage
        if GEMINI_USE_THINKING_BUDGET_CONFIG_KEY not in config_data:
            config_data[GEMINI_USE_THINKING_BUDGET_CONFIG_KEY] = GEMINI_DEFAULT_USE_THINKING_BUDGET
            logger.info(
                f"'{GEMINI_USE_THINKING_BUDGET_CONFIG_KEY}' not found. "
                f"Using default: {GEMINI_DEFAULT_USE_THINKING_BUDGET}"
            )
        elif not isinstance(config_data[GEMINI_USE_THINKING_BUDGET_CONFIG_KEY], bool):
            logger.warning(
                f"'{GEMINI_USE_THINKING_BUDGET_CONFIG_KEY}' is not a boolean. "
                f"Using default: {GEMINI_DEFAULT_USE_THINKING_BUDGET}"
            )
            config_data[GEMINI_USE_THINKING_BUDGET_CONFIG_KEY] = GEMINI_DEFAULT_USE_THINKING_BUDGET
        
        # Validate thinking budget value
        if GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY not in config_data:
            config_data[GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY] = GEMINI_DEFAULT_THINKING_BUDGET_VALUE
            logger.info(
                f"'{GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY}' not found. "
                f"Using default: {GEMINI_DEFAULT_THINKING_BUDGET_VALUE}"
            )
        else:
            try:
                val = int(config_data[GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY])
                if not (GEMINI_MIN_THINKING_BUDGET_VALUE <= val <= GEMINI_MAX_THINKING_BUDGET_VALUE):
                    logger.warning(
                        f"'{GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY}' ({val}) is outside valid range "
                        f"({GEMINI_MIN_THINKING_BUDGET_VALUE}-{GEMINI_MAX_THINKING_BUDGET_VALUE}). "
                        f"Using default: {GEMINI_DEFAULT_THINKING_BUDGET_VALUE}"
                    )
                    config_data[GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY] = GEMINI_DEFAULT_THINKING_BUDGET_VALUE
                else:
                    config_data[GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY] = val
            except ValueError:
                logger.warning(
                    f"'{GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY}' is not a valid integer. "
                    f"Using default: {GEMINI_DEFAULT_THINKING_BUDGET_VALUE}"
                )
                config_data[GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY] = GEMINI_DEFAULT_THINKING_BUDGET_VALUE
    
    @staticmethod
    def _validate_gemini_safety_settings(config_data: Dict[str, Any]):
        """Validate Gemini safety settings."""
        if GEMINI_SAFETY_SETTINGS_CONFIG_KEY not in config_data:
            config_data[GEMINI_SAFETY_SETTINGS_CONFIG_KEY] = DEFAULT_GEMINI_SAFETY_SETTINGS
            logger.info(f"'{GEMINI_SAFETY_SETTINGS_CONFIG_KEY}' not found. Using default safety settings.")
        elif not isinstance(config_data[GEMINI_SAFETY_SETTINGS_CONFIG_KEY], dict):
            logger.warning(
                f"'{GEMINI_SAFETY_SETTINGS_CONFIG_KEY}' is not a dictionary. Using default safety settings."
            )
            config_data[GEMINI_SAFETY_SETTINGS_CONFIG_KEY] = DEFAULT_GEMINI_SAFETY_SETTINGS
    
    @staticmethod
    def _validate_model_selections(config_data: Dict[str, Any]):
        """Validate model selection configuration."""
        # Main model
        if "model" not in config_data or not config_data.get("model"):
            logger.error("'model' not found or empty. This is a required field.")
            config_data["model"] = "google/gemini-2.5-flash"
            logger.warning(f"Using hardcoded default model: {config_data['model']}")
        else:
            try:
                provider, model_name = str(config_data["model"]).split("/", 1)
            except ValueError:
                logger.error(f"Invalid 'model' format: {config_data['model']}. Should be 'provider/model_name'.")
                config_data["model"] = "google/gemini-2.5-flash"
                logger.warning(f"Using hardcoded default model: {config_data['model']}")

        # Other models with defaults
        model_defaults = {
            GROUNDING_MODEL_CONFIG_KEY: "google/gemini-2.5-flash",
            FALLBACK_VISION_MODEL_CONFIG_KEY: "google/gemini-2.5-flash",
            FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY: "google/gemini-2.5-flash",
            DEEP_SEARCH_MODEL_CONFIG_KEY: "x-ai/grok-3",
        }
        
        for key, default in model_defaults.items():
            if not config_data.get(key):
                config_data[key] = default
                logger.info(f"'{key}' not found or empty. Using default: {default}")

        # Fallback model system prompt
        if not config_data.get(FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY):
            logger.info(f"'{FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY}' not found or empty. Using default.")
            config_data[FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY] = DEFAULT_FALLBACK_MODEL_SYSTEM_PROMPT
    
    @staticmethod
    def _validate_grounding_model_params(config_data: Dict[str, Any]):
        """Validate grounding model parameters."""
        # Temperature
        ConfigValidator._validate_float_range_config(
            config_data, GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY, 
            DEFAULT_GROUNDING_MODEL_TEMPERATURE, 0.0, 1.0
        )
        
        # Top K
        ConfigValidator._validate_int_config(
            config_data, GROUNDING_MODEL_TOP_K_CONFIG_KEY, 
            DEFAULT_GROUNDING_MODEL_TOP_K, min_value=1
        )
        
        # Top P
        ConfigValidator._validate_float_range_config(
            config_data, GROUNDING_MODEL_TOP_P_CONFIG_KEY, 
            DEFAULT_GROUNDING_MODEL_TOP_P, 0.0, 1.0
        )
        
        # Thinking budget settings
        ConfigValidator._validate_bool_config(
            config_data, GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY,
            GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET
        )
        
        if GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY in config_data:
            try:
                val = int(config_data[GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY])
                if not (GEMINI_MIN_THINKING_BUDGET_VALUE <= val <= GEMINI_MAX_THINKING_BUDGET_VALUE):
                    logger.warning(f"'{GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY}' ({val}) is outside valid range. Using default: {GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE}")
                    config_data[GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY] = GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE
                else:
                    config_data[GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY] = val
            except ValueError:
                logger.warning(f"'{GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY}' is not a valid integer. Using default: {GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE}")
                config_data[GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY] = GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE
        else:
            config_data[GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY] = GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE
    
    @staticmethod
    def _validate_float_range_config(config_data: Dict[str, Any], key: str, default: float, min_val: float, max_val: float):
        """Validate a float configuration value within a range."""
        if key not in config_data:
            config_data[key] = default
            logger.info(f"'{key}' not found. Using default: {default}")
        else:
            try:
                val = float(config_data[key])
                if not (min_val <= val <= max_val):
                    logger.warning(f"'{key}' ({val}) must be between {min_val} and {max_val}. Using default: {default}")
                    config_data[key] = default
                else:
                    config_data[key] = val
            except ValueError:
                logger.warning(f"'{key}' is not a valid float. Using default: {default}")
                config_data[key] = default
    
    @staticmethod
    def _validate_bool_config(config_data: Dict[str, Any], key: str, default: bool):
        """Validate a boolean configuration value."""
        if key not in config_data:
            config_data[key] = default
            logger.info(f"'{key}' not found. Using default: {default}")
        elif not isinstance(config_data[key], bool):
            logger.warning(f"'{key}' is not a boolean. Using default: {default}")
            config_data[key] = default
    
    @staticmethod
    def _validate_output_sharing_config(config_data: Dict[str, Any]):
        """Validate output sharing configuration."""
        if OUTPUT_SHARING_CONFIG_KEY not in config_data:
            config_data[OUTPUT_SHARING_CONFIG_KEY] = {}
            logger.info(f"'{OUTPUT_SHARING_CONFIG_KEY}' section not found. Using defaults.")

        output_sharing_cfg = config_data[OUTPUT_SHARING_CONFIG_KEY]
        if not isinstance(output_sharing_cfg, dict):
            logger.warning(f"'{OUTPUT_SHARING_CONFIG_KEY}' section is not a dictionary. Using defaults.")
            output_sharing_cfg = {}
            config_data[OUTPUT_SHARING_CONFIG_KEY] = output_sharing_cfg

        # Textis enabled
        if TEXTIS_ENABLED_CONFIG_KEY not in output_sharing_cfg:
            output_sharing_cfg[TEXTIS_ENABLED_CONFIG_KEY] = False
            logger.info(f"'{TEXTIS_ENABLED_CONFIG_KEY}' not found. Defaulting to False.")
        elif not isinstance(output_sharing_cfg[TEXTIS_ENABLED_CONFIG_KEY], bool):
            logger.warning(f"'{TEXTIS_ENABLED_CONFIG_KEY}' is not a boolean. Defaulting to False.")
            output_sharing_cfg[TEXTIS_ENABLED_CONFIG_KEY] = False

        # URL shortener settings
        if URL_SHORTENER_ENABLED_CONFIG_KEY not in output_sharing_cfg:
            output_sharing_cfg[URL_SHORTENER_ENABLED_CONFIG_KEY] = False
            logger.info(f"'{URL_SHORTENER_ENABLED_CONFIG_KEY}' not found. Defaulting to False.")
        elif not isinstance(output_sharing_cfg[URL_SHORTENER_ENABLED_CONFIG_KEY], bool):
            logger.warning(f"'{URL_SHORTENER_ENABLED_CONFIG_KEY}' is not a boolean. Defaulting to False.")
            output_sharing_cfg[URL_SHORTENER_ENABLED_CONFIG_KEY] = False

        if URL_SHORTENER_SERVICE_CONFIG_KEY not in output_sharing_cfg:
            output_sharing_cfg[URL_SHORTENER_SERVICE_CONFIG_KEY] = "tinyurl"
            logger.info(f"'{URL_SHORTENER_SERVICE_CONFIG_KEY}' not found. Defaulting to 'tinyurl'.")
        elif (not isinstance(output_sharing_cfg[URL_SHORTENER_SERVICE_CONFIG_KEY], str) or 
              not output_sharing_cfg[URL_SHORTENER_SERVICE_CONFIG_KEY].strip()):
            logger.warning(f"'{URL_SHORTENER_SERVICE_CONFIG_KEY}' is not a non-empty string. Defaulting to 'tinyurl'.")
            output_sharing_cfg[URL_SHORTENER_SERVICE_CONFIG_KEY] = "tinyurl"
    
    @staticmethod
    def _validate_url_extractor_config(config_data: Dict[str, Any]):
        """Validate URL content extractor configuration."""
        # Main extractor
        if config_data.get(MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY) not in VALID_URL_EXTRACTORS:
            logger.warning(f"Invalid value for '{MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY}'. Using default: {DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR}")
            config_data[MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY] = DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR

        # Fallback extractor
        if config_data.get(FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY) not in VALID_URL_EXTRACTORS:
            logger.warning(f"Invalid value for '{FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY}'. Using default: {DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR}")
            config_data[FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY] = DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR

        # Jina engine mode
        if config_data.get(JINA_ENGINE_MODE_CONFIG_KEY) not in VALID_JINA_ENGINE_MODES:
            logger.warning(f"Invalid value for '{JINA_ENGINE_MODE_CONFIG_KEY}'. Using default: {DEFAULT_JINA_ENGINE_MODE}")
            config_data[JINA_ENGINE_MODE_CONFIG_KEY] = DEFAULT_JINA_ENGINE_MODE

        # Jina wait for selector
        if config_data.get(JINA_WAIT_FOR_SELECTOR_CONFIG_KEY) == "":
            config_data[JINA_WAIT_FOR_SELECTOR_CONFIG_KEY] = None
        elif (config_data.get(JINA_WAIT_FOR_SELECTOR_CONFIG_KEY) is not None and 
              not isinstance(config_data[JINA_WAIT_FOR_SELECTOR_CONFIG_KEY], str)):
            logger.warning(f"'{JINA_WAIT_FOR_SELECTOR_CONFIG_KEY}' is not a string or null. Using default: {DEFAULT_JINA_WAIT_FOR_SELECTOR}")
            config_data[JINA_WAIT_FOR_SELECTOR_CONFIG_KEY] = DEFAULT_JINA_WAIT_FOR_SELECTOR

        # Jina timeout
        if config_data.get(JINA_TIMEOUT_CONFIG_KEY) is not None:
            ConfigValidator._validate_int_config(
                config_data, JINA_TIMEOUT_CONFIG_KEY, DEFAULT_JINA_TIMEOUT, min_value=0
            )

        # Crawl4AI cache mode
        if config_data.get(CRAWL4AI_CACHE_MODE_CONFIG_KEY) not in VALID_CRAWL4AI_CACHE_MODES:
            logger.warning(f"Invalid value for '{CRAWL4AI_CACHE_MODE_CONFIG_KEY}'. Using default: {DEFAULT_CRAWL4AI_CACHE_MODE}")
            config_data[CRAWL4AI_CACHE_MODE_CONFIG_KEY] = DEFAULT_CRAWL4AI_CACHE_MODE
    
    @staticmethod
    def _validate_web_content_api_config(config_data: Dict[str, Any]):
        """Validate external web content extraction API configuration."""
        ConfigValidator._validate_bool_config(
            config_data, WEB_CONTENT_EXTRACTION_API_ENABLED_CONFIG_KEY,
            DEFAULT_WEB_CONTENT_EXTRACTION_API_ENABLED
        )

        if (not isinstance(config_data.get(WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY), str) or 
            not config_data.get(WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY, "").strip()):
            logger.warning(f"'{WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY}' is not a non-empty string. Using default: {DEFAULT_WEB_CONTENT_EXTRACTION_API_URL}")
            config_data[WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY] = DEFAULT_WEB_CONTENT_EXTRACTION_API_URL

        ConfigValidator._validate_int_config(
            config_data, WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY,
            DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS, min_value=1
        )

        ConfigValidator._validate_int_config(
            config_data, WEB_CONTENT_EXTRACTION_API_CACHE_TTL_CONFIG_KEY,
            DEFAULT_WEB_CONTENT_EXTRACTION_API_CACHE_TTL, min_value=1
        )

        ConfigValidator._validate_bool_config(
            config_data, HTTP_CLIENT_USE_HTTP2_CONFIG_KEY,
            DEFAULT_HTTP_CLIENT_USE_HTTP2
        )
    
    @staticmethod
    def _validate_text_limits_config(config_data: Dict[str, Any], filename: str):
        """Validate text limits configuration."""
        if MAX_TEXT_LIMITS_CONFIG_KEY not in config_data:
            logger.warning(f"'{MAX_TEXT_LIMITS_CONFIG_KEY}' section not found in {filename}. Using hardcoded default of 128000 for all models.")
            config_data[MAX_TEXT_LIMITS_CONFIG_KEY] = {
                DEFAULT_MAX_TEXT_KEY: 128000,
                MODEL_SPECIFIC_MAX_TEXT_KEY: {},
            }
            return

        limits_config = config_data[MAX_TEXT_LIMITS_CONFIG_KEY]
        if not isinstance(limits_config, dict):
            logger.warning(f"'{MAX_TEXT_LIMITS_CONFIG_KEY}' is not a dictionary. Using hardcoded default of 128000.")
            config_data[MAX_TEXT_LIMITS_CONFIG_KEY] = {
                DEFAULT_MAX_TEXT_KEY: 128000,
                MODEL_SPECIFIC_MAX_TEXT_KEY: {},
            }
            return

        # Validate default limit
        ConfigValidator._validate_int_config(
            limits_config, DEFAULT_MAX_TEXT_KEY, 128000, min_value=1
        )

        # Validate model-specific limits
        if MODEL_SPECIFIC_MAX_TEXT_KEY not in limits_config:
            limits_config[MODEL_SPECIFIC_MAX_TEXT_KEY] = {}
        elif not isinstance(limits_config[MODEL_SPECIFIC_MAX_TEXT_KEY], dict):
            logger.warning(f"'{MODEL_SPECIFIC_MAX_TEXT_KEY}' is not a dictionary. No model-specific limits will be applied.")
            limits_config[MODEL_SPECIFIC_MAX_TEXT_KEY] = {}
        else:
            model_limits = limits_config[MODEL_SPECIFIC_MAX_TEXT_KEY]
            for model_id, limit_val in list(model_limits.items()):
                try:
                    val = int(limit_val)
                    if val <= 0:
                        logger.warning(f"Max_text for model '{model_id}' ('{val}') must be positive. Removing specific limit.")
                        del model_limits[model_id]
                    else:
                        model_limits[model_id] = val
                except ValueError:
                    logger.warning(f"Max_text for model '{model_id}' ('{limit_val}') is not a valid integer. Removing specific limit.")
                    del model_limits[model_id]

        # Safety margin and minimum limit
        ConfigValidator._validate_int_config(
            config_data, MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY, 5000, min_value=0
        )

        ConfigValidator._validate_int_config(
            config_data, MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY, 1000, min_value=1
        )

        # Remove old max_text if it exists
        if "max_text" in config_data:
            logger.warning(f"Old 'max_text' setting found in {filename}. It is now ignored. Please use '{MAX_TEXT_LIMITS_CONFIG_KEY}' instead.")
    
    @staticmethod
    def _validate_misc_config(config_data: Dict[str, Any]):
        """Validate miscellaneous configuration options."""
        # Prompt enhancer system prompt
        if (not config_data.get(PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY) or 
            not isinstance(config_data[PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY], str)):
            logger.info(f"'{PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY}' not found or invalid. Using default.")
            config_data[PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY] = DEFAULT_PROMPT_ENHANCER_SYSTEM_PROMPT

        # Auto render markdown tables
        ConfigValidator._validate_bool_config(
            config_data, AUTO_RENDER_MARKDOWN_TABLES_CONFIG_KEY,
            DEFAULT_AUTO_RENDER_MARKDOWN_TABLES
        ) 