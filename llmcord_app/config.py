import yaml
import logging
from .constants import (
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
    # Output Sharing Constants
    OUTPUT_SHARING_CONFIG_KEY,
    NGROK_ENABLED_CONFIG_KEY,
    NGROK_AUTHTOKEN_CONFIG_KEY,
    GRIP_PORT_CONFIG_KEY,
    DEFAULT_GRIP_PORT,
    NGROK_STATIC_DOMAIN_CONFIG_KEY,
    CLEANUP_ON_SHUTDOWN_CONFIG_KEY,
    URL_SHORTENER_ENABLED_CONFIG_KEY,
    URL_SHORTENER_SERVICE_CONFIG_KEY,
    FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY,  # <-- ADDED
    DEFAULT_FALLBACK_MODEL_SYSTEM_PROMPT,  # <-- ADDED
)

# --- ADDED DEFAULT GROUNDING PROMPT ---
DEFAULT_GROUNDING_SYSTEM_PROMPT = """
You are an expert at analyzing user queries and conversation history to determine the most effective web search queries that will help answer the user's latest request or continue the conversation meaningfully.
Based on the provided conversation history (especially the last user message), output a list of concise and targeted search queries.
Focus on identifying key entities, concepts, questions, or current events mentioned that would benefit from fresh information from the web.
If the user's query is a direct question, formulate search queries that would find the answer.
If the user is discussing a topic, formulate queries that would find recent developments, facts, or relevant discussions.
Do not generate more than 5 search queries.
Output only the search queries, each on a new line. Do not add any other text, preamble, or explanation.
""".strip()
# --- END DEFAULT GROUNDING PROMPT ---


def get_config(filename="config.yaml"):
    """Loads, validates, and returns the configuration from a YAML file."""
    try:
        with open(filename, "r", encoding="utf-8") as file:  # Specify encoding
            config_data = yaml.safe_load(file)
            if not isinstance(config_data, dict):
                logging.error(f"CRITICAL: {filename} is not a valid YAML dictionary.")
                exit()

            # --- Config Validation & Key Normalization ---
            # Ensure providers have api_keys (plural) as a list
            providers = config_data.get("providers", {})
            if not isinstance(providers, dict):
                logging.warning(
                    "Config Warning: 'providers' section is not a dictionary. Treating as empty."
                )
                providers = {}
                config_data["providers"] = providers  # Fix in loaded data

            for name, provider_cfg in providers.items():
                if provider_cfg and isinstance(
                    provider_cfg, dict
                ):  # Check if provider config exists and is a dict
                    single_key = provider_cfg.get("api_key")
                    key_list = provider_cfg.get("api_keys")

                    if single_key and not key_list:
                        logging.warning(
                            f"Config Warning: Provider '{name}' uses deprecated 'api_key'. Converting to 'api_keys' list. Please update {filename}."
                        )
                        provider_cfg["api_keys"] = [single_key]
                        del provider_cfg["api_key"]
                    elif single_key and key_list:
                        logging.warning(
                            f"Config Warning: Provider '{name}' has both 'api_key' and 'api_keys'. Using 'api_keys'. Please remove 'api_key' from {filename}."
                        )
                        del provider_cfg["api_key"]
                    elif (
                        key_list is None
                    ):  # Handle case where api_keys is explicitly null or missing
                        # Allow providers without keys (like Ollama)
                        provider_cfg["api_keys"] = []
                    elif not isinstance(key_list, list):
                        logging.error(
                            f"Config Error: Provider '{name}' has 'api_keys' but it's not a list. Treating as empty."
                        )
                        provider_cfg["api_keys"] = []

                    # --- ADDED: Validate disable_vision for OpenAI ---
                    if name == "openai":
                        if "disable_vision" not in provider_cfg:
                            provider_cfg["disable_vision"] = False  # Default to False
                            logging.info(
                                "OpenAI 'disable_vision' not set in config. Assuming False."
                            )
                        elif not isinstance(provider_cfg["disable_vision"], bool):
                            logging.warning(
                                "OpenAI 'disable_vision' is not a boolean. Defaulting to False."
                            )
                            provider_cfg["disable_vision"] = False
                    # --- END ADDED ---

                elif provider_cfg is not None:
                    logging.warning(
                        f"Config Warning: Provider '{name}' configuration is not a dictionary. Ignoring provider."
                    )
                    # Optionally remove invalid provider config: del providers[name] or set providers[name] = {}

            # Handle SerpAPI key(s)
            single_serp_key = config_data.get("serpapi_api_key")
            serp_key_list = config_data.get("serpapi_api_keys")
            if single_serp_key and not serp_key_list:
                logging.warning(
                    f"Config Warning: Found 'serpapi_api_key'. Converting to 'serpapi_api_keys' list. Please update {filename}."
                )
                config_data["serpapi_api_keys"] = [single_serp_key]
                del config_data["serpapi_api_key"]
            elif single_serp_key and serp_key_list:
                logging.warning(
                    f"Config Warning: Found both 'serpapi_api_key' and 'serpapi_api_keys'. Using 'serpapi_api_keys'. Please remove 'serpapi_api_key' from {filename}."
                )
                del config_data["serpapi_api_key"]
            elif (
                serp_key_list is None
            ):  # Handle case where serpapi_api_keys is explicitly null or missing
                config_data["serpapi_api_keys"] = []
            elif not isinstance(serp_key_list, list):
                logging.error(
                    "Config Error: Found 'serpapi_api_keys' but it's not a list. Treating as empty."
                )
                config_data["serpapi_api_keys"] = []

            # Basic check for essential Discord config
            if not config_data.get("bot_token"):
                logging.error(f"CRITICAL: bot_token is not set in {filename}")
                # Don't exit here, let the main script handle it after logging
            if not config_data.get("client_id"):
                logging.warning(
                    f"client_id not found in {filename}. Cannot generate invite URL."
                )

            # Ensure permissions structure exists
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

            # Load SearxNG base URL
            if SEARXNG_BASE_URL_CONFIG_KEY not in config_data:
                logging.info(
                    f"'{SEARXNG_BASE_URL_CONFIG_KEY}' not found in {filename}. Using default: {SEARXNG_DEFAULT_URL}"
                )
                config_data[SEARXNG_BASE_URL_CONFIG_KEY] = SEARXNG_DEFAULT_URL
            elif not config_data.get(
                SEARXNG_BASE_URL_CONFIG_KEY
            ):  # Check if it's empty
                logging.warning(
                    f"'{SEARXNG_BASE_URL_CONFIG_KEY}' is empty in {filename}. Using default: {SEARXNG_DEFAULT_URL}"
                )
                config_data[SEARXNG_BASE_URL_CONFIG_KEY] = SEARXNG_DEFAULT_URL

            # --- ADDED: Load Grounding System Prompt ---
            if (
                GROUNDING_SYSTEM_PROMPT_CONFIG_KEY not in config_data
                or not config_data.get(GROUNDING_SYSTEM_PROMPT_CONFIG_KEY)
            ):
                logging.info(
                    f"'{GROUNDING_SYSTEM_PROMPT_CONFIG_KEY}' not found or empty in {filename}. Using default."
                )
                config_data[GROUNDING_SYSTEM_PROMPT_CONFIG_KEY] = (
                    DEFAULT_GROUNDING_SYSTEM_PROMPT
                )
            # --- END ADDED ---

            # --- ADDED: Load SearxNG URL content max length ---
            if SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY not in config_data:
                config_data[SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY] = (
                    SEARXNG_DEFAULT_URL_CONTENT_MAX_LENGTH
                )
                logging.info(
                    f"'{SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY}' not found. Using default: {SEARXNG_DEFAULT_URL_CONTENT_MAX_LENGTH}"
                )
            else:
                try:
                    val = int(config_data[SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY])
                    if val <= 0:
                        logging.warning(
                            f"'{SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY}' ({val}) must be positive. "
                            f"Using default: {SEARXNG_DEFAULT_URL_CONTENT_MAX_LENGTH}"
                        )
                        config_data[SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY] = (
                            SEARXNG_DEFAULT_URL_CONTENT_MAX_LENGTH
                        )
                    else:
                        config_data[SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY] = val
                except ValueError:
                    logging.warning(
                        f"'{SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY}' is not a valid integer. "
                        f"Using default: {SEARXNG_DEFAULT_URL_CONTENT_MAX_LENGTH}"
                    )
                    config_data[SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY] = (
                        SEARXNG_DEFAULT_URL_CONTENT_MAX_LENGTH
                    )
            # --- END ADDED ---

            # --- Load Gemini Thinking Budget Settings ---
            if GEMINI_USE_THINKING_BUDGET_CONFIG_KEY not in config_data:
                config_data[GEMINI_USE_THINKING_BUDGET_CONFIG_KEY] = (
                    GEMINI_DEFAULT_USE_THINKING_BUDGET
                )
                logging.info(
                    f"'{GEMINI_USE_THINKING_BUDGET_CONFIG_KEY}' not found. Using default: {GEMINI_DEFAULT_USE_THINKING_BUDGET}"
                )
            elif not isinstance(
                config_data[GEMINI_USE_THINKING_BUDGET_CONFIG_KEY], bool
            ):
                logging.warning(
                    f"'{GEMINI_USE_THINKING_BUDGET_CONFIG_KEY}' is not a boolean. Using default: {GEMINI_DEFAULT_USE_THINKING_BUDGET}"
                )
                config_data[GEMINI_USE_THINKING_BUDGET_CONFIG_KEY] = (
                    GEMINI_DEFAULT_USE_THINKING_BUDGET
                )

            if GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY not in config_data:
                config_data[GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY] = (
                    GEMINI_DEFAULT_THINKING_BUDGET_VALUE
                )
                logging.info(
                    f"'{GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY}' not found. Using default: {GEMINI_DEFAULT_THINKING_BUDGET_VALUE}"
                )
            else:
                try:
                    val = int(config_data[GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY])
                    if not (
                        GEMINI_MIN_THINKING_BUDGET_VALUE
                        <= val
                        <= GEMINI_MAX_THINKING_BUDGET_VALUE
                    ):
                        logging.warning(
                            f"'{GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY}' ({val}) is outside the valid range "
                            f"({GEMINI_MIN_THINKING_BUDGET_VALUE}-{GEMINI_MAX_THINKING_BUDGET_VALUE}). "
                            f"Using default: {GEMINI_DEFAULT_THINKING_BUDGET_VALUE}"
                        )
                        config_data[GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY] = (
                            GEMINI_DEFAULT_THINKING_BUDGET_VALUE
                        )
                    else:
                        config_data[GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY] = val
                except ValueError:
                    logging.warning(
                        f"'{GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY}' is not a valid integer. "
                        f"Using default: {GEMINI_DEFAULT_THINKING_BUDGET_VALUE}"
                    )
                    config_data[GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY] = (
                        GEMINI_DEFAULT_THINKING_BUDGET_VALUE
                    )
            # --- End Load Gemini Thinking Budget Settings ---

            # --- Load Output Sharing Settings ---
            if OUTPUT_SHARING_CONFIG_KEY not in config_data:
                config_data[OUTPUT_SHARING_CONFIG_KEY] = {}
                logging.info(
                    f"'{OUTPUT_SHARING_CONFIG_KEY}' section not found in {filename}. Using defaults."
                )

            output_sharing_cfg = config_data[OUTPUT_SHARING_CONFIG_KEY]
            if not isinstance(output_sharing_cfg, dict):
                logging.warning(
                    f"'{OUTPUT_SHARING_CONFIG_KEY}' section is not a dictionary. Using defaults for all output sharing settings."
                )
                output_sharing_cfg = {}  # Reset to empty dict to ensure defaults are applied
                config_data[OUTPUT_SHARING_CONFIG_KEY] = output_sharing_cfg

            if NGROK_ENABLED_CONFIG_KEY not in output_sharing_cfg:
                output_sharing_cfg[NGROK_ENABLED_CONFIG_KEY] = False
                logging.info(
                    f"'{NGROK_ENABLED_CONFIG_KEY}' not found in '{OUTPUT_SHARING_CONFIG_KEY}'. Defaulting to False."
                )
            elif not isinstance(output_sharing_cfg[NGROK_ENABLED_CONFIG_KEY], bool):
                logging.warning(
                    f"'{NGROK_ENABLED_CONFIG_KEY}' is not a boolean. Defaulting to False."
                )
                output_sharing_cfg[NGROK_ENABLED_CONFIG_KEY] = False

            if NGROK_AUTHTOKEN_CONFIG_KEY not in output_sharing_cfg:
                output_sharing_cfg[NGROK_AUTHTOKEN_CONFIG_KEY] = (
                    None  # Default to None if not present
                )
                logging.info(
                    f"'{NGROK_AUTHTOKEN_CONFIG_KEY}' not found in '{OUTPUT_SHARING_CONFIG_KEY}'. Defaulting to None."
                )
            # No specific validation for authtoken string format, user responsibility

            if GRIP_PORT_CONFIG_KEY not in output_sharing_cfg:
                output_sharing_cfg[GRIP_PORT_CONFIG_KEY] = DEFAULT_GRIP_PORT
                logging.info(
                    f"'{GRIP_PORT_CONFIG_KEY}' not found in '{OUTPUT_SHARING_CONFIG_KEY}'. Using default: {DEFAULT_GRIP_PORT}"
                )
            else:
                try:
                    port_val = int(output_sharing_cfg[GRIP_PORT_CONFIG_KEY])
                    if not (1024 <= port_val <= 65535):
                        logging.warning(
                            f"'{GRIP_PORT_CONFIG_KEY}' ({port_val}) is outside the valid range (1024-65535). "
                            f"Using default: {DEFAULT_GRIP_PORT}"
                        )
                        output_sharing_cfg[GRIP_PORT_CONFIG_KEY] = DEFAULT_GRIP_PORT
                    else:
                        output_sharing_cfg[GRIP_PORT_CONFIG_KEY] = port_val
                except ValueError:
                    logging.warning(
                        f"'{GRIP_PORT_CONFIG_KEY}' is not a valid integer. "
                        f"Using default: {DEFAULT_GRIP_PORT}"
                    )
                    output_sharing_cfg[GRIP_PORT_CONFIG_KEY] = DEFAULT_GRIP_PORT

            if NGROK_STATIC_DOMAIN_CONFIG_KEY not in output_sharing_cfg:
                output_sharing_cfg[NGROK_STATIC_DOMAIN_CONFIG_KEY] = (
                    None  # Default to None
                )
                logging.info(
                    f"'{NGROK_STATIC_DOMAIN_CONFIG_KEY}' not found in '{OUTPUT_SHARING_CONFIG_KEY}'. Defaulting to None."
                )
            elif (
                output_sharing_cfg[NGROK_STATIC_DOMAIN_CONFIG_KEY] == ""
            ):  # Treat empty string as None
                output_sharing_cfg[NGROK_STATIC_DOMAIN_CONFIG_KEY] = None
            elif not isinstance(
                output_sharing_cfg[NGROK_STATIC_DOMAIN_CONFIG_KEY], (str, type(None))
            ):
                logging.warning(
                    f"'{NGROK_STATIC_DOMAIN_CONFIG_KEY}' is not a string or null. Defaulting to None."
                )
                output_sharing_cfg[NGROK_STATIC_DOMAIN_CONFIG_KEY] = None

            if CLEANUP_ON_SHUTDOWN_CONFIG_KEY not in output_sharing_cfg:
                output_sharing_cfg[CLEANUP_ON_SHUTDOWN_CONFIG_KEY] = (
                    True  # Default to True
                )
                logging.info(
                    f"'{CLEANUP_ON_SHUTDOWN_CONFIG_KEY}' not found in '{OUTPUT_SHARING_CONFIG_KEY}'. Defaulting to True."
                )
            elif not isinstance(
                output_sharing_cfg[CLEANUP_ON_SHUTDOWN_CONFIG_KEY], bool
            ):
                logging.warning(
                    f"'{CLEANUP_ON_SHUTDOWN_CONFIG_KEY}' is not a boolean. Defaulting to True."
                )
                output_sharing_cfg[CLEANUP_ON_SHUTDOWN_CONFIG_KEY] = True

            if URL_SHORTENER_ENABLED_CONFIG_KEY not in output_sharing_cfg:
                output_sharing_cfg[URL_SHORTENER_ENABLED_CONFIG_KEY] = (
                    False  # Default to False
                )
                logging.info(
                    f"'{URL_SHORTENER_ENABLED_CONFIG_KEY}' not found in '{OUTPUT_SHARING_CONFIG_KEY}'. Defaulting to False."
                )
            elif not isinstance(
                output_sharing_cfg[URL_SHORTENER_ENABLED_CONFIG_KEY], bool
            ):
                logging.warning(
                    f"'{URL_SHORTENER_ENABLED_CONFIG_KEY}' is not a boolean. Defaulting to False."
                )
                output_sharing_cfg[URL_SHORTENER_ENABLED_CONFIG_KEY] = False

            if URL_SHORTENER_SERVICE_CONFIG_KEY not in output_sharing_cfg:
                output_sharing_cfg[URL_SHORTENER_SERVICE_CONFIG_KEY] = (
                    "tinyurl"  # Default to "tinyurl"
                )
                logging.info(
                    f"'{URL_SHORTENER_SERVICE_CONFIG_KEY}' not found in '{OUTPUT_SHARING_CONFIG_KEY}'. Defaulting to 'tinyurl'."
                )
            elif (
                not isinstance(
                    output_sharing_cfg[URL_SHORTENER_SERVICE_CONFIG_KEY], str
                )
                or not output_sharing_cfg[URL_SHORTENER_SERVICE_CONFIG_KEY].strip()
            ):
                logging.warning(
                    f"'{URL_SHORTENER_SERVICE_CONFIG_KEY}' is not a non-empty string. Defaulting to 'tinyurl'."
                )
                output_sharing_cfg[URL_SHORTENER_SERVICE_CONFIG_KEY] = "tinyurl"
            # --- End Load Output Sharing Settings ---

            # --- Load Fallback Model System Prompt ---
            if (
                FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY not in config_data
                or not config_data.get(FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY)
            ):
                logging.info(
                    f"'{FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY}' not found or empty in {filename}. Using default."
                )
                config_data[FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY] = (
                    DEFAULT_FALLBACK_MODEL_SYSTEM_PROMPT
                )
            # --- End Load Fallback Model System Prompt ---

            return config_data

    except FileNotFoundError:
        logging.error(
            f"CRITICAL: {filename} not found. Please copy config-example.yaml to {filename} and configure it."
        )
        exit()
    except yaml.YAMLError as e:
        logging.error(f"CRITICAL: Error parsing {filename}: {e}")
        exit()
    except Exception as e:
        logging.error(f"CRITICAL: Unexpected error loading config from {filename}: {e}")
        exit()
