import yaml
import logging
import aiofiles
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
    TEXTIS_ENABLED_CONFIG_KEY,  # Renamed from NGROK_ENABLED_CONFIG_KEY
    # NGROK_AUTHTOKEN_CONFIG_KEY, # Removed
    # GRIP_PORT_CONFIG_KEY, # Removed
    # DEFAULT_GRIP_PORT, # Removed
    # NGROK_STATIC_DOMAIN_CONFIG_KEY, # Removed
    # CLEANUP_ON_SHUTDOWN_CONFIG_KEY, # Removed
    URL_SHORTENER_ENABLED_CONFIG_KEY,
    URL_SHORTENER_SERVICE_CONFIG_KEY,
    FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY,
    DEFAULT_FALLBACK_MODEL_SYSTEM_PROMPT,
    # General URL Extractors
    MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY,
    FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY,
    DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR,
    DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR,
    VALID_URL_EXTRACTORS,
    # Jina settings
    JINA_ENGINE_MODE_CONFIG_KEY,
    DEFAULT_JINA_ENGINE_MODE,
    VALID_JINA_ENGINE_MODES,
    JINA_WAIT_FOR_SELECTOR_CONFIG_KEY,  # Added
    DEFAULT_JINA_WAIT_FOR_SELECTOR,  # Added
    JINA_TIMEOUT_CONFIG_KEY,  # Added
    DEFAULT_JINA_TIMEOUT,  # Added
    # New config keys
    MAX_MESSAGE_NODES_CONFIG_KEY,
    MAX_TEXT_LIMITS_CONFIG_KEY,  # New
    DEFAULT_MAX_TEXT_KEY,  # New
    MODEL_SPECIFIC_MAX_TEXT_KEY,  # New
    MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY,  # New
    MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY,  # New
    EDIT_DELAY_SECONDS_CONFIG_KEY,
    RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY,
    SEARXNG_NUM_RESULTS_CONFIG_KEY,
    GROUNDING_MODEL_CONFIG_KEY,
    FALLBACK_VISION_MODEL_CONFIG_KEY,
    FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY,
    DEEP_SEARCH_MODEL_CONFIG_KEY,
    GEMINI_SAFETY_SETTINGS_CONFIG_KEY,
    # Crawl4AI settings
    CRAWL4AI_CACHE_MODE_CONFIG_KEY,  # Added
    DEFAULT_CRAWL4AI_CACHE_MODE,  # Added
    VALID_CRAWL4AI_CACHE_MODES,  # Added
    # External Web Content API
    WEB_CONTENT_EXTRACTION_API_ENABLED_CONFIG_KEY,
    WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY,
    WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY,
    DEFAULT_WEB_CONTENT_EXTRACTION_API_ENABLED,
    DEFAULT_WEB_CONTENT_EXTRACTION_API_URL,
    DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS,
    # Grounding Model Parameters
    GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY,
    DEFAULT_GROUNDING_MODEL_TEMPERATURE,
    GROUNDING_MODEL_TOP_K_CONFIG_KEY,
    DEFAULT_GROUNDING_MODEL_TOP_K,
    GROUNDING_MODEL_TOP_P_CONFIG_KEY,
    DEFAULT_GROUNDING_MODEL_TOP_P,
    # Grounding Model Thinking Budget
    GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY,
    GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY,
    GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET,
    GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE,
    PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY,  # New
)

# Alternative Search Query Generation Config Keys
ALT_SEARCH_SECTION_KEY = "alternative_search_query_generation"
ALT_SEARCH_ENABLED_KEY = "enabled"
ALT_SEARCH_PROMPT_KEY = "search_query_generation_prompt_template"

DEFAULT_ALT_SEARCH_PROMPT_TEMPLATE = """
<task>
Analyze the latest query to determine if it requires web search. Consider the chat history for context.
</task>

<criteria>
Web search IS required when the query asks about:
- Current events, news, or recent developments
- Real-time information (prices, weather, stock data)
- Specific facts that may have changed recently
- Information about new products, services, or updates
- People, places, or organizations where current status matters

Web search is NOT required when the query asks about:
- General knowledge or established facts
- Conceptual explanations or definitions
- Personal opinions or advice
- Mathematical calculations or logic problems
- Analysis of provided information
</criteria>

<instructions>
1. Analyze the latest query in the context of the chat history
2. Determine if web search is required based on the criteria above
3. If web search is required, generate specific search queries that would find the needed information
4. For queries with multiple distinct subjects, create separate search queries for each
5. Return your response in the exact JSON format shown in the examples
</instructions>

<examples>
<example>
<chat_history>
User: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence...
</chat_history>
<latest_query>Can you explain deep learning?</latest_query>
<output>
{"web_search_required": false}
</output>
</example>

<example>
<chat_history>
User: I'm interested in electric vehicles
Assistant: Electric vehicles are becoming increasingly popular...
</chat_history>
<latest_query>What are Tesla's latest Model 3 prices and what new features did Apple announce for iPhone 15?</latest_query>
<output>
{
    "web_search_required": true,
    "search_queries": [
    "Tesla Model 3 prices 2024",
    "Apple iPhone 15 new features announcement"
    ]
}
</output>
</example>

<example>
<chat_history>
User: Tell me about climate change
Assistant: Climate change refers to long-term shifts in global temperatures...
</chat_history>
<latest_query>What were the key outcomes of the latest UN climate summit?</latest_query>
<output>
{
    "web_search_required": true,
    "search_queries": ["UN climate summit latest outcomes 2024"]
}
</output>
</example>
</examples>

<chat_history>
{chat_history}
</chat_history>

<latest_query>
{latest_query}
</latest_query>

<output_format>
Return ONLY valid JSON in one of these formats:
- If no search needed: {"web_search_required": false}
- If search needed: {"web_search_required": true, "search_queries": ["query1", "query2", ...]}
</output_format>
""".strip()
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

DEFAULT_PROMPT_ENHANCER_SYSTEM_PROMPT = """
You are an expert prompt engineer. Your task is to refine a user's input to make it a more effective prompt for a large language model.
Follow the provided prompt design strategies and guides to improve the user's original prompt.
Output *only* the improved prompt, without any preamble, explanation, or markdown formatting.
""".strip()


async def get_config(filename="config/config.yaml"):
    """Loads, validates, and returns the configuration from a YAML file asynchronously."""
    try:
        async with aiofiles.open(
            filename, "r", encoding="utf-8"
        ) as file:  # Specify encoding
            content = await file.read()
            config_data = yaml.safe_load(content)
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

                elif provider_cfg is not None:
                    logging.warning(
                        f"Config Warning: Provider '{name}' configuration is not a dictionary. Ignoring provider."
                    )

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

            # Load max_message_node_cache
            if MAX_MESSAGE_NODES_CONFIG_KEY not in config_data:
                config_data[MAX_MESSAGE_NODES_CONFIG_KEY] = 500  # Default value
                logging.info(
                    f"'{MAX_MESSAGE_NODES_CONFIG_KEY}' not found. Using default: 500"
                )
            else:
                try:
                    val = int(config_data[MAX_MESSAGE_NODES_CONFIG_KEY])
                    if val <= 0:
                        logging.warning(
                            f"'{MAX_MESSAGE_NODES_CONFIG_KEY}' ({val}) must be positive. Using default: 500"
                        )
                        config_data[MAX_MESSAGE_NODES_CONFIG_KEY] = 500
                    else:
                        config_data[MAX_MESSAGE_NODES_CONFIG_KEY] = val
                except ValueError:
                    logging.warning(
                        f"'{MAX_MESSAGE_NODES_CONFIG_KEY}' is not a valid integer. Using default: 500"
                    )
                    config_data[MAX_MESSAGE_NODES_CONFIG_KEY] = 500

            # Load edit_delay_seconds
            if EDIT_DELAY_SECONDS_CONFIG_KEY not in config_data:
                config_data[EDIT_DELAY_SECONDS_CONFIG_KEY] = 1.0  # Default value
                logging.info(
                    f"'{EDIT_DELAY_SECONDS_CONFIG_KEY}' not found. Using default: 1.0"
                )
            else:
                try:
                    val = float(config_data[EDIT_DELAY_SECONDS_CONFIG_KEY])
                    if val < 0:  # Can be 0 for no delay
                        logging.warning(
                            f"'{EDIT_DELAY_SECONDS_CONFIG_KEY}' ({val}) cannot be negative. Using default: 1.0"
                        )
                        config_data[EDIT_DELAY_SECONDS_CONFIG_KEY] = 1.0
                    else:
                        config_data[EDIT_DELAY_SECONDS_CONFIG_KEY] = val
                except ValueError:
                    logging.warning(
                        f"'{EDIT_DELAY_SECONDS_CONFIG_KEY}' is not a valid float. Using default: 1.0"
                    )
                    config_data[EDIT_DELAY_SECONDS_CONFIG_KEY] = 1.0

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

            # --- Load SearxNG Number of Results ---
            if SEARXNG_NUM_RESULTS_CONFIG_KEY not in config_data:
                config_data[SEARXNG_NUM_RESULTS_CONFIG_KEY] = 5  # Default value
                logging.info(
                    f"'{SEARXNG_NUM_RESULTS_CONFIG_KEY}' not found. Using default: 5"
                )
            else:
                try:
                    val = int(config_data[SEARXNG_NUM_RESULTS_CONFIG_KEY])
                    if val <= 0:
                        logging.warning(
                            f"'{SEARXNG_NUM_RESULTS_CONFIG_KEY}' ({val}) must be positive. Using default: 5"
                        )
                        config_data[SEARXNG_NUM_RESULTS_CONFIG_KEY] = 5
                    else:
                        config_data[SEARXNG_NUM_RESULTS_CONFIG_KEY] = val
                except ValueError:
                    logging.warning(
                        f"'{SEARXNG_NUM_RESULTS_CONFIG_KEY}' is not a valid integer. Using default: 5"
                    )
                    config_data[SEARXNG_NUM_RESULTS_CONFIG_KEY] = 5

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

            # --- Load Gemini Safety Settings ---
            if GEMINI_SAFETY_SETTINGS_CONFIG_KEY not in config_data:
                config_data[GEMINI_SAFETY_SETTINGS_CONFIG_KEY] = {
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                }  # Default value
                logging.info(
                    f"'{GEMINI_SAFETY_SETTINGS_CONFIG_KEY}' not found. Using default safety settings."
                )
            elif not isinstance(config_data[GEMINI_SAFETY_SETTINGS_CONFIG_KEY], dict):
                logging.warning(
                    f"'{GEMINI_SAFETY_SETTINGS_CONFIG_KEY}' is not a dictionary. Using default safety settings."
                )
                config_data[GEMINI_SAFETY_SETTINGS_CONFIG_KEY] = {
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                }
            # Further validation for specific keys and values within the dict could be added here
            # For example, checking if the harm categories and thresholds are valid.
            # For now, we assume the user provides them correctly as strings.

            # --- Load Model Selections ---
            if "model" not in config_data or not config_data.get("model"):
                logging.error(
                    f"'model' not found or empty in {filename}. This is a required field."
                )
                # Potentially exit or raise error, but for now, let main script handle it if bot_token is also missing.

                config_data["model"] = (
                    "google/gemini-2.5-flash-preview-05-20"  # A sensible default
                )
                logging.warning(
                    f"Using hardcoded default model: {config_data['model']}"
                )
            else:
                # Basic validation for provider/model format
                try:
                    provider, model_name = str(config_data["model"]).split("/", 1)
                    # Further validation against AVAILABLE_MODELS could be done here if AVAILABLE_MODELS is imported
                except ValueError:
                    logging.error(
                        f"Invalid 'model' format: {config_data['model']}. Should be 'provider/model_name'."
                    )
                    config_data["model"] = "google/gemini-2.5-flash-preview-05-20"
                    logging.warning(
                        f"Using hardcoded default model: {config_data['model']}"
                    )

            if GROUNDING_MODEL_CONFIG_KEY not in config_data or not config_data.get(
                GROUNDING_MODEL_CONFIG_KEY
            ):
                config_data[GROUNDING_MODEL_CONFIG_KEY] = (
                    "google/gemini-2.5-flash-preview-05-20"  # Default
                )
                logging.info(
                    f"'{GROUNDING_MODEL_CONFIG_KEY}' not found or empty. Using default: {config_data[GROUNDING_MODEL_CONFIG_KEY]}"
                )

            # --- Load Grounding Model Parameters ---
            # Temperature
            if GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY not in config_data:
                config_data[GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY] = (
                    DEFAULT_GROUNDING_MODEL_TEMPERATURE
                )
                logging.info(
                    f"'{GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY}' not found. "
                    f"Using default: {DEFAULT_GROUNDING_MODEL_TEMPERATURE}"
                )
            else:
                try:
                    val = float(config_data[GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY])
                    if not (0.0 <= val <= 1.0):
                        logging.warning(
                            f"'{GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY}' ({val}) must be between 0.0 and 1.0. "
                            f"Using default: {DEFAULT_GROUNDING_MODEL_TEMPERATURE}"
                        )
                        config_data[GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY] = (
                            DEFAULT_GROUNDING_MODEL_TEMPERATURE
                        )
                    else:
                        config_data[GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY] = val
                except ValueError:
                    logging.warning(
                        f"'{GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY}' is not a valid float. "
                        f"Using default: {DEFAULT_GROUNDING_MODEL_TEMPERATURE}"
                    )
                    config_data[GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY] = (
                        DEFAULT_GROUNDING_MODEL_TEMPERATURE
                    )

            # Top K
            if GROUNDING_MODEL_TOP_K_CONFIG_KEY not in config_data:
                config_data[GROUNDING_MODEL_TOP_K_CONFIG_KEY] = (
                    DEFAULT_GROUNDING_MODEL_TOP_K
                )
                logging.info(
                    f"'{GROUNDING_MODEL_TOP_K_CONFIG_KEY}' not found. "
                    f"Using default: {DEFAULT_GROUNDING_MODEL_TOP_K}"
                )
            else:
                try:
                    val = int(config_data[GROUNDING_MODEL_TOP_K_CONFIG_KEY])
                    if val <= 0:
                        logging.warning(
                            f"'{GROUNDING_MODEL_TOP_K_CONFIG_KEY}' ({val}) must be a positive integer. "
                            f"Using default: {DEFAULT_GROUNDING_MODEL_TOP_K}"
                        )
                        config_data[GROUNDING_MODEL_TOP_K_CONFIG_KEY] = (
                            DEFAULT_GROUNDING_MODEL_TOP_K
                        )
                    else:
                        config_data[GROUNDING_MODEL_TOP_K_CONFIG_KEY] = val
                except ValueError:
                    logging.warning(
                        f"'{GROUNDING_MODEL_TOP_K_CONFIG_KEY}' is not a valid integer. "
                        f"Using default: {DEFAULT_GROUNDING_MODEL_TOP_K}"
                    )
                    config_data[GROUNDING_MODEL_TOP_K_CONFIG_KEY] = (
                        DEFAULT_GROUNDING_MODEL_TOP_K
                    )

            # Top P
            if GROUNDING_MODEL_TOP_P_CONFIG_KEY not in config_data:
                config_data[GROUNDING_MODEL_TOP_P_CONFIG_KEY] = (
                    DEFAULT_GROUNDING_MODEL_TOP_P
                )
                logging.info(
                    f"'{GROUNDING_MODEL_TOP_P_CONFIG_KEY}' not found. "
                    f"Using default: {DEFAULT_GROUNDING_MODEL_TOP_P}"
                )
            else:
                try:
                    val = float(config_data[GROUNDING_MODEL_TOP_P_CONFIG_KEY])
                    if not (0.0 <= val <= 1.0):
                        logging.warning(
                            f"'{GROUNDING_MODEL_TOP_P_CONFIG_KEY}' ({val}) must be between 0.0 and 1.0. "
                            f"Using default: {DEFAULT_GROUNDING_MODEL_TOP_P}"
                        )
                        config_data[GROUNDING_MODEL_TOP_P_CONFIG_KEY] = (
                            DEFAULT_GROUNDING_MODEL_TOP_P
                        )
                    else:
                        config_data[GROUNDING_MODEL_TOP_P_CONFIG_KEY] = val
                except ValueError:
                    logging.warning(
                        f"'{GROUNDING_MODEL_TOP_P_CONFIG_KEY}' is not a valid float. "
                        f"Using default: {DEFAULT_GROUNDING_MODEL_TOP_P}"
                    )
                    config_data[GROUNDING_MODEL_TOP_P_CONFIG_KEY] = (
                        DEFAULT_GROUNDING_MODEL_TOP_P
                    )

            # --- Load Grounding Model Thinking Budget Settings ---
            if GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY not in config_data:
                config_data[GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY] = (
                    GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET
                )
                logging.info(
                    f"'{GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY}' not found. "
                    f"Using default: {GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET}"
                )
            elif not isinstance(
                config_data[GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY], bool
            ):
                logging.warning(
                    f"'{GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY}' is not a boolean. "
                    f"Using default: {GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET}"
                )
                config_data[GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY] = (
                    GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET
                )

            if GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY not in config_data:
                config_data[GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY] = (
                    GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE
                )
                logging.info(
                    f"'{GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY}' not found. "
                    f"Using default: {GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE}"
                )
            else:
                try:
                    val = int(
                        config_data[GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY]
                    )
                    if not (
                        GEMINI_MIN_THINKING_BUDGET_VALUE  # Shared min/max with general Gemini budget
                        <= val
                        <= GEMINI_MAX_THINKING_BUDGET_VALUE
                    ):
                        logging.warning(
                            f"'{GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY}' ({val}) is outside the valid range "
                            f"({GEMINI_MIN_THINKING_BUDGET_VALUE}-{GEMINI_MAX_THINKING_BUDGET_VALUE}). "
                            f"Using default: {GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE}"
                        )
                        config_data[
                            GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY
                        ] = GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE
                    else:
                        config_data[
                            GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY
                        ] = val
                except ValueError:
                    logging.warning(
                        f"'{GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY}' is not a valid integer. "
                        f"Using default: {GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE}"
                    )
                    config_data[GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY] = (
                        GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE
                    )

            if (
                FALLBACK_VISION_MODEL_CONFIG_KEY not in config_data
                or not config_data.get(FALLBACK_VISION_MODEL_CONFIG_KEY)
            ):
                config_data[FALLBACK_VISION_MODEL_CONFIG_KEY] = (
                    "google/gemini-2.5-flash-preview-05-20"  # Default
                )
                logging.info(
                    f"'{FALLBACK_VISION_MODEL_CONFIG_KEY}' not found or empty. Using default: {config_data[FALLBACK_VISION_MODEL_CONFIG_KEY]}"
                )

            if (
                FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY not in config_data
                or not config_data.get(FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY)
            ):
                config_data[FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY] = (
                    "google/gemini-2.5-flash-preview-05-20"  # Default
                )
                logging.info(
                    f"'{FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY}' not found or empty. Using default: {config_data[FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY]}"
                )

            if DEEP_SEARCH_MODEL_CONFIG_KEY not in config_data or not config_data.get(
                DEEP_SEARCH_MODEL_CONFIG_KEY
            ):
                config_data[DEEP_SEARCH_MODEL_CONFIG_KEY] = "x-ai/grok-3"  # Default
                logging.info(
                    f"'{DEEP_SEARCH_MODEL_CONFIG_KEY}' not found or empty. Using default: {config_data[DEEP_SEARCH_MODEL_CONFIG_KEY]}"
                )

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

            if TEXTIS_ENABLED_CONFIG_KEY not in output_sharing_cfg:
                output_sharing_cfg[TEXTIS_ENABLED_CONFIG_KEY] = False
                logging.info(
                    f"'{TEXTIS_ENABLED_CONFIG_KEY}' not found in '{OUTPUT_SHARING_CONFIG_KEY}'. Defaulting to False."
                )
            elif not isinstance(output_sharing_cfg[TEXTIS_ENABLED_CONFIG_KEY], bool):
                logging.warning(
                    f"'{TEXTIS_ENABLED_CONFIG_KEY}' is not a boolean. Defaulting to False."
                )
                output_sharing_cfg[TEXTIS_ENABLED_CONFIG_KEY] = False
            # NGROK_AUTHTOKEN_CONFIG_KEY, GRIP_PORT_CONFIG_KEY, NGROK_STATIC_DOMAIN_CONFIG_KEY, CLEANUP_ON_SHUTDOWN_CONFIG_KEY removed

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

            # --- Load Rate Limit Cooldown Hours ---
            if RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY not in config_data:
                config_data[RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY] = 24  # Default value
                logging.info(
                    f"'{RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY}' not found. Using default: 24 hours"
                )
            else:
                try:
                    val = int(config_data[RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY])
                    if val <= 0:
                        logging.warning(
                            f"'{RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY}' ({val}) must be positive. Using default: 24"
                        )
                        config_data[RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY] = 24
                    else:
                        config_data[RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY] = val
                except ValueError:
                    logging.warning(
                        f"'{RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY}' is not a valid integer. Using default: 24"
                    )
                    config_data[RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY] = 24

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

            # --- Load General URL Content Extractor Settings ---
            if MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY not in config_data:
                config_data[MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY] = (
                    DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR
                )
                logging.info(
                    f"'{MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY}' not found. Using default: {DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR}"
                )
            elif (
                config_data[MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY]
                not in VALID_URL_EXTRACTORS
            ):
                logging.warning(
                    f"Invalid value for '{MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY}': {config_data[MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY]}. Using default: {DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR}"
                )
                config_data[MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY] = (
                    DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR
                )

            if FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY not in config_data:
                config_data[FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY] = (
                    DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR
                )
                logging.info(
                    f"'{FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY}' not found. Using default: {DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR}"
                )
            elif (
                config_data[FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY]
                not in VALID_URL_EXTRACTORS
            ):
                logging.warning(
                    f"Invalid value for '{FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY}': {config_data[FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY]}. Using default: {DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR}"
                )
                config_data[FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY] = (
                    DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR
                )

            # --- Load Jina Engine Mode ---
            if JINA_ENGINE_MODE_CONFIG_KEY not in config_data:
                config_data[JINA_ENGINE_MODE_CONFIG_KEY] = DEFAULT_JINA_ENGINE_MODE
                logging.info(
                    f"'{JINA_ENGINE_MODE_CONFIG_KEY}' not found. Using default: {DEFAULT_JINA_ENGINE_MODE}"
                )
            elif (
                config_data[JINA_ENGINE_MODE_CONFIG_KEY] not in VALID_JINA_ENGINE_MODES
            ):
                logging.warning(
                    f"Invalid value for '{JINA_ENGINE_MODE_CONFIG_KEY}': {config_data[JINA_ENGINE_MODE_CONFIG_KEY]}. Using default: {DEFAULT_JINA_ENGINE_MODE}"
                )
                config_data[JINA_ENGINE_MODE_CONFIG_KEY] = DEFAULT_JINA_ENGINE_MODE

            # --- Load Jina Wait For Selector ---
            if JINA_WAIT_FOR_SELECTOR_CONFIG_KEY not in config_data:
                config_data[JINA_WAIT_FOR_SELECTOR_CONFIG_KEY] = (
                    DEFAULT_JINA_WAIT_FOR_SELECTOR
                )
                logging.info(
                    f"'{JINA_WAIT_FOR_SELECTOR_CONFIG_KEY}' not found. Using default: {DEFAULT_JINA_WAIT_FOR_SELECTOR}"
                )
            elif config_data[
                JINA_WAIT_FOR_SELECTOR_CONFIG_KEY
            ] is not None and not isinstance(
                config_data[JINA_WAIT_FOR_SELECTOR_CONFIG_KEY], str
            ):
                logging.warning(
                    f"'{JINA_WAIT_FOR_SELECTOR_CONFIG_KEY}' is not a string or null. Using default: {DEFAULT_JINA_WAIT_FOR_SELECTOR}"
                )
                config_data[JINA_WAIT_FOR_SELECTOR_CONFIG_KEY] = (
                    DEFAULT_JINA_WAIT_FOR_SELECTOR
                )
            elif (
                config_data[JINA_WAIT_FOR_SELECTOR_CONFIG_KEY] == ""
            ):  # Treat empty string as None (no selector)
                config_data[JINA_WAIT_FOR_SELECTOR_CONFIG_KEY] = None

            # --- Load Jina Timeout ---
            if JINA_TIMEOUT_CONFIG_KEY not in config_data:
                config_data[JINA_TIMEOUT_CONFIG_KEY] = DEFAULT_JINA_TIMEOUT
                logging.info(
                    f"'{JINA_TIMEOUT_CONFIG_KEY}' not found. Using default: {DEFAULT_JINA_TIMEOUT}"
                )
            elif config_data[JINA_TIMEOUT_CONFIG_KEY] is not None:
                try:
                    val = int(config_data[JINA_TIMEOUT_CONFIG_KEY])
                    if (
                        val < 0
                    ):  # Timeout cannot be negative, 0 might be valid for some APIs to mean "no explicit timeout" or "immediate"
                        logging.warning(
                            f"'{JINA_TIMEOUT_CONFIG_KEY}' ({val}) cannot be negative. Using default: {DEFAULT_JINA_TIMEOUT}"
                        )
                        config_data[JINA_TIMEOUT_CONFIG_KEY] = DEFAULT_JINA_TIMEOUT
                    else:
                        config_data[JINA_TIMEOUT_CONFIG_KEY] = val
                except ValueError:
                    logging.warning(
                        f"'{JINA_TIMEOUT_CONFIG_KEY}' is not a valid integer. Using default: {DEFAULT_JINA_TIMEOUT}"
                    )
                    config_data[JINA_TIMEOUT_CONFIG_KEY] = DEFAULT_JINA_TIMEOUT
            # If it's None, it remains None (DEFAULT_JINA_TIMEOUT)

            # --- Load Crawl4AI Cache Mode ---
            if CRAWL4AI_CACHE_MODE_CONFIG_KEY not in config_data:
                config_data[CRAWL4AI_CACHE_MODE_CONFIG_KEY] = (
                    DEFAULT_CRAWL4AI_CACHE_MODE
                )
                logging.info(
                    f"'{CRAWL4AI_CACHE_MODE_CONFIG_KEY}' not found. Using default: {DEFAULT_CRAWL4AI_CACHE_MODE}"
                )
            elif (
                config_data[CRAWL4AI_CACHE_MODE_CONFIG_KEY]
                not in VALID_CRAWL4AI_CACHE_MODES
            ):
                logging.warning(
                    f"Invalid value for '{CRAWL4AI_CACHE_MODE_CONFIG_KEY}': {config_data[CRAWL4AI_CACHE_MODE_CONFIG_KEY]}. Using default: {DEFAULT_CRAWL4AI_CACHE_MODE}"
                )
                config_data[CRAWL4AI_CACHE_MODE_CONFIG_KEY] = (
                    DEFAULT_CRAWL4AI_CACHE_MODE
                )

            # --- Load External Web Content API Settings ---
            if WEB_CONTENT_EXTRACTION_API_ENABLED_CONFIG_KEY not in config_data:
                config_data[WEB_CONTENT_EXTRACTION_API_ENABLED_CONFIG_KEY] = (
                    DEFAULT_WEB_CONTENT_EXTRACTION_API_ENABLED
                )
                logging.info(
                    f"'{WEB_CONTENT_EXTRACTION_API_ENABLED_CONFIG_KEY}' not found. "
                    f"Using default: {DEFAULT_WEB_CONTENT_EXTRACTION_API_ENABLED}"
                )
            elif not isinstance(
                config_data[WEB_CONTENT_EXTRACTION_API_ENABLED_CONFIG_KEY], bool
            ):
                logging.warning(
                    f"'{WEB_CONTENT_EXTRACTION_API_ENABLED_CONFIG_KEY}' is not a boolean. "
                    f"Using default: {DEFAULT_WEB_CONTENT_EXTRACTION_API_ENABLED}"
                )
                config_data[WEB_CONTENT_EXTRACTION_API_ENABLED_CONFIG_KEY] = (
                    DEFAULT_WEB_CONTENT_EXTRACTION_API_ENABLED
                )

            if WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY not in config_data:
                config_data[WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY] = (
                    DEFAULT_WEB_CONTENT_EXTRACTION_API_URL
                )
                logging.info(
                    f"'{WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY}' not found. "
                    f"Using default: {DEFAULT_WEB_CONTENT_EXTRACTION_API_URL}"
                )
            elif (
                not isinstance(
                    config_data[WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY], str
                )
                or not config_data[WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY].strip()
            ):
                logging.warning(
                    f"'{WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY}' is not a non-empty string. "
                    f"Using default: {DEFAULT_WEB_CONTENT_EXTRACTION_API_URL}"
                )
                config_data[WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY] = (
                    DEFAULT_WEB_CONTENT_EXTRACTION_API_URL
                )

            if WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY not in config_data:
                config_data[WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY] = (
                    DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS
                )
                logging.info(
                    f"'{WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY}' not found. "
                    f"Using default: {DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS}"
                )
            else:
                try:
                    val = int(
                        config_data[WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY]
                    )
                    if val <= 0:
                        logging.warning(
                            f"'{WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY}' ({val}) must be positive. "
                            f"Using default: {DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS}"
                        )
                        config_data[
                            WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY
                        ] = DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS
                    else:
                        config_data[
                            WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY
                        ] = val
                except ValueError:
                    logging.warning(
                        f"'{WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY}' is not a valid integer. "
                        f"Using default: {DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS}"
                    )
                    config_data[WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY] = (
                        DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS
                    )

            # --- Load Alternative Search Query Generation Settings ---
            alt_search_config = config_data.get(ALT_SEARCH_SECTION_KEY, {})
            if not isinstance(alt_search_config, dict):
                logging.warning(
                    f"'{ALT_SEARCH_SECTION_KEY}' section is not a dictionary. Using defaults."
                )
                alt_search_config = {}
            config_data[ALT_SEARCH_SECTION_KEY] = (
                alt_search_config  # Ensure the section exists in config_data
            )

            # Enabled
            if ALT_SEARCH_ENABLED_KEY not in alt_search_config:
                alt_search_config[ALT_SEARCH_ENABLED_KEY] = False
                logging.info(
                    f"'{ALT_SEARCH_ENABLED_KEY}' not found in '{ALT_SEARCH_SECTION_KEY}'. Defaulting to False."
                )
            elif not isinstance(alt_search_config[ALT_SEARCH_ENABLED_KEY], bool):
                logging.warning(
                    f"'{ALT_SEARCH_ENABLED_KEY}' in '{ALT_SEARCH_SECTION_KEY}' is not a boolean. Defaulting to False."
                )
                alt_search_config[ALT_SEARCH_ENABLED_KEY] = False

            # Search Query Generation Prompt Template
            if (
                ALT_SEARCH_PROMPT_KEY not in alt_search_config
                or not alt_search_config.get(ALT_SEARCH_PROMPT_KEY)
            ):
                alt_search_config[ALT_SEARCH_PROMPT_KEY] = (
                    DEFAULT_ALT_SEARCH_PROMPT_TEMPLATE
                )
                logging.info(
                    f"'{ALT_SEARCH_PROMPT_KEY}' not found or empty in '{ALT_SEARCH_SECTION_KEY}'. Using default template."
                )
            elif not isinstance(alt_search_config[ALT_SEARCH_PROMPT_KEY], str):
                logging.warning(
                    f"'{ALT_SEARCH_PROMPT_KEY}' in '{ALT_SEARCH_SECTION_KEY}' is not a string. Using default template."
                )
                alt_search_config[ALT_SEARCH_PROMPT_KEY] = (
                    DEFAULT_ALT_SEARCH_PROMPT_TEMPLATE
                )

            # --- Load Prompt Enhancer System Prompt ---
            if (
                PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY not in config_data
                or not config_data.get(PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY)
            ):
                logging.info(
                    f"'{PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY}' not found or empty in {filename}. Using default."
                )
                config_data[PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY] = (
                    DEFAULT_PROMPT_ENHANCER_SYSTEM_PROMPT
                )
            elif not isinstance(
                config_data[PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY], str
            ):
                logging.warning(
                    f"'{PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY}' is not a string. Using default."
                )
                config_data[PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY] = (
                    DEFAULT_PROMPT_ENHANCER_SYSTEM_PROMPT
                )

            # --- Load Max Text Limits ---
            if MAX_TEXT_LIMITS_CONFIG_KEY not in config_data:
                logging.warning(
                    f"'{MAX_TEXT_LIMITS_CONFIG_KEY}' section not found in {filename}. "
                    f"Using hardcoded default of 128000 for all models."
                )
                config_data[MAX_TEXT_LIMITS_CONFIG_KEY] = {
                    DEFAULT_MAX_TEXT_KEY: 128000,
                    MODEL_SPECIFIC_MAX_TEXT_KEY: {},
                }
            else:
                limits_config = config_data[MAX_TEXT_LIMITS_CONFIG_KEY]
                if not isinstance(limits_config, dict):
                    logging.warning(
                        f"'{MAX_TEXT_LIMITS_CONFIG_KEY}' is not a dictionary. "
                        f"Using hardcoded default of 128000."
                    )
                    config_data[MAX_TEXT_LIMITS_CONFIG_KEY] = {
                        DEFAULT_MAX_TEXT_KEY: 128000,
                        MODEL_SPECIFIC_MAX_TEXT_KEY: {},
                    }
                    limits_config = config_data[MAX_TEXT_LIMITS_CONFIG_KEY]

                if DEFAULT_MAX_TEXT_KEY not in limits_config:
                    logging.warning(
                        f"'{DEFAULT_MAX_TEXT_KEY}' not found in '{MAX_TEXT_LIMITS_CONFIG_KEY}'. "
                        f"Using 128000 as default."
                    )
                    limits_config[DEFAULT_MAX_TEXT_KEY] = 128000
                else:
                    try:
                        default_val = int(limits_config[DEFAULT_MAX_TEXT_KEY])
                        if default_val <= 0:
                            logging.warning(
                                f"Default max_text '{default_val}' must be positive. Using 128000."
                            )
                            limits_config[DEFAULT_MAX_TEXT_KEY] = 128000
                        else:
                            limits_config[DEFAULT_MAX_TEXT_KEY] = default_val
                    except ValueError:
                        logging.warning(
                            f"Default max_text '{limits_config[DEFAULT_MAX_TEXT_KEY]}' is not a valid integer. Using 128000."
                        )
                        limits_config[DEFAULT_MAX_TEXT_KEY] = 128000

                if MODEL_SPECIFIC_MAX_TEXT_KEY not in limits_config:
                    limits_config[MODEL_SPECIFIC_MAX_TEXT_KEY] = {}
                elif not isinstance(limits_config[MODEL_SPECIFIC_MAX_TEXT_KEY], dict):
                    logging.warning(
                        f"'{MODEL_SPECIFIC_MAX_TEXT_KEY}' in '{MAX_TEXT_LIMITS_CONFIG_KEY}' is not a dictionary. "
                        f"No model-specific limits will be applied."
                    )
                    limits_config[MODEL_SPECIFIC_MAX_TEXT_KEY] = {}
                else:
                    # Validate model-specific limits
                    model_limits = limits_config[MODEL_SPECIFIC_MAX_TEXT_KEY]
                    for model_id, limit_val in list(
                        model_limits.items()
                    ):  # Iterate over a copy for safe deletion
                        try:
                            val = int(limit_val)
                            if val <= 0:
                                logging.warning(
                                    f"Max_text for model '{model_id}' ('{val}') must be positive. Removing specific limit."
                                )
                                del model_limits[model_id]
                            else:
                                model_limits[model_id] = val
                        except ValueError:
                            logging.warning(
                                f"Max_text for model '{model_id}' ('{limit_val}') is not a valid integer. Removing specific limit."
                            )
                            del model_limits[model_id]

            # Remove old top-level max_text if it exists, to avoid confusion
            if "max_text" in config_data:
                logging.warning(
                    f"Old 'max_text' setting found in {filename}. It is now ignored. "
                    f"Please use '{MAX_TEXT_LIMITS_CONFIG_KEY}' instead."
                )
                # del config_data["max_text"] # Optionally remove it from the loaded config

            # --- Load Max Text Safety Margin ---
            if MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY not in config_data:
                config_data[MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY] = 5000  # Default value
                logging.info(
                    f"'{MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY}' not found. Using default: 5000"
                )
            else:
                try:
                    val = int(config_data[MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY])
                    if val < 0:  # Can be 0 for no margin
                        logging.warning(
                            f"'{MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY}' ({val}) cannot be negative. Using default: 5000"
                        )
                        config_data[MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY] = 5000
                    else:
                        config_data[MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY] = val
                except ValueError:
                    logging.warning(
                        f"'{MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY}' is not a valid integer. Using default: 5000"
                    )
                    config_data[MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY] = 5000

            # --- Load Min Token Limit After Safety Margin ---
            if MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY not in config_data:
                config_data[MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY] = (
                    1000  # Default value
                )
                logging.info(
                    f"'{MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY}' not found. Using default: 1000"
                )
            else:
                try:
                    val = int(
                        config_data[MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY]
                    )
                    if val <= 0:
                        logging.warning(
                            f"'{MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY}' ({val}) must be positive. Using default: 1000"
                        )
                        config_data[MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY] = (
                            1000
                        )
                    else:
                        config_data[MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY] = (
                            val
                        )
                except ValueError:
                    logging.warning(
                        f"'{MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY}' is not a valid integer. Using default: 1000"
                    )
                    config_data[MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY] = 1000

            return config_data

    except FileNotFoundError:
        logging.error(
            f"CRITICAL: {filename} not found. Please copy config-example.yaml to {filename} and configure it."
        )
        # Instead of exit(), raise an exception or return None to be handled by the caller
        raise  # Or return None / raise specific error
    except yaml.YAMLError as e:
        logging.error(f"CRITICAL: Error parsing {filename}: {e}")
        raise  # Or return None / raise specific error
    except Exception as e:
        logging.error(f"CRITICAL: Unexpected error loading config from {filename}: {e}")
        raise  # Or return None / raise specific error


def get_max_text_for_model(config: dict, model_identifier: str) -> int:
    """
    Retrieves the max_text token limit for a given model_identifier,
    applies a configurable safety margin, and falls back to the default limit if needed.
    """
    safety_margin = config.get(MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY, 5000)
    min_limit_after_margin = config.get(
        MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY, 1000
    )

    limits_config = config.get(MAX_TEXT_LIMITS_CONFIG_KEY)
    raw_limit = 0

    if not limits_config or not isinstance(limits_config, dict):
        logging.warning(
            f"'{MAX_TEXT_LIMITS_CONFIG_KEY}' not found or invalid in config. Using hardcoded default of 128000 before safety margin."
        )
        raw_limit = 128000  # Hardcoded fallback
    else:
        model_specific_limits = limits_config.get(MODEL_SPECIFIC_MAX_TEXT_KEY, {})
        if not isinstance(model_specific_limits, dict):
            model_specific_limits = {}

        if model_identifier in model_specific_limits:
            specific_limit_val = model_specific_limits[model_identifier]
            if isinstance(specific_limit_val, int) and specific_limit_val > 0:
                raw_limit = specific_limit_val
            else:
                logging.warning(
                    f"Invalid specific limit for '{model_identifier}': {specific_limit_val}. Falling back to default."
                )
                # Fall through to use default if specific is invalid

        if raw_limit == 0:  # If no valid specific limit was found
            default_limit_val = limits_config.get(
                DEFAULT_MAX_TEXT_KEY, 128000
            )  # Default to 128k if key missing
            if isinstance(default_limit_val, int) and default_limit_val > 0:
                raw_limit = default_limit_val
            else:
                logging.warning(
                    f"Invalid default limit: {default_limit_val}. Using hardcoded default of 128000 before safety margin."
                )
                raw_limit = 128000  # Hardcoded fallback

    # Apply safety margin
    adjusted_limit = raw_limit - safety_margin

    # Ensure the limit doesn't go below the minimum
    final_limit = max(adjusted_limit, min_limit_after_margin)

    if (
        final_limit != raw_limit - safety_margin
    ):  # Log if minimum cap was hit or if it's different from simple subtraction
        logging.info(
            f"Applied safety margin ({safety_margin}) to '{model_identifier}'. Raw: {raw_limit}, Adjusted (raw - margin): {adjusted_limit}, Final (after min check with {min_limit_after_margin}): {final_limit}"
        )
    else:
        logging.debug(
            f"Applied safety margin ({safety_margin}) to '{model_identifier}'. Raw: {raw_limit}, Final: {final_limit}"
        )

    return final_limit
