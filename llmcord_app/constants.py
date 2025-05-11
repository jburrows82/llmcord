# llmcord_app/constants.py
import re
import discord
from google.genai import types as google_types # Use google.genai.types

# --- LLM & Vision ---
VISION_MODEL_TAGS = ("gpt-4", "o3", "o4", "claude-3", "gemini", "gemma", "llama", "pixtral", "mistral-small", "vision", "vl", "flash", "grok")
PROVIDERS_SUPPORTING_USERNAMES = ("openai") # x-ai might support it too, needs verification

# --- Discord Embeds & UI ---
EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()
EMBED_COLOR_ERROR = discord.Color.red()
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1.0 # Use float for time delays

# --- Limits ---
MAX_MESSAGE_NODES = 500 # Max messages stored in cache
MAX_EMBED_FIELD_VALUE_LENGTH = 1024
MAX_EMBED_FIELDS = 25
MAX_EMBED_DESCRIPTION_LENGTH = 4096 - len(STREAMING_INDICATOR) # Adjusted for indicator
MAX_EMBED_TOTAL_SIZE = 5900 # Safety margin below Discord's 6000 limit
MAX_PLAIN_TEXT_LENGTH = 2000 # Discord message character limit

# --- URL Regex Patterns ---
GENERAL_URL_PATTERN = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
YOUTUBE_URL_PATTERN = re.compile(r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11}))')
REDDIT_URL_PATTERN = re.compile(r'(https?://(?:www\.)?reddit\.com/r/[a-zA-Z0-9_]+/comments/([a-zA-Z0-9]+))')
AT_AI_PATTERN = re.compile(r'\bat ai\b', re.IGNORECASE)
GOOGLE_LENS_KEYWORD = "googlelens"
GOOGLE_LENS_PATTERN = re.compile(rf'^{GOOGLE_LENS_KEYWORD}\s+', re.IGNORECASE)
# --- ADDED CONSTANTS ---
IMGUR_HEADER = "--- Generated Images ---"
IMGUR_URL_PREFIX = "https://i.imgur.com/"
IMGUR_URL_PATTERN = re.compile(r"^(https://i\.imgur\.com/[a-zA-Z0-9]+\.(?:jpeg|jpg|png|gif))$")
# --- END ADDED CONSTANTS ---

# --- SearxNG and Grounding ---
SEARXNG_BASE_URL_CONFIG_KEY = "searxng_base_url"
SEARXNG_DEFAULT_URL = "http://localhost:18088" # Default if not in config
SEARXNG_NUM_RESULTS = 5
GROUNDING_MODEL_PROVIDER = "google"
GROUNDING_MODEL_NAME = "gemini-2.0-flash"
# --- ADDED CONSTANT ---
GROUNDING_SYSTEM_PROMPT_CONFIG_KEY = "grounding_system_prompt"
# --- END ADDED CONSTANT ---
# --- ADDED CONSTANTS FOR SEARXNG URL CONTENT LENGTH ---
SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY = "searxng_url_content_max_length"
SEARXNG_DEFAULT_URL_CONTENT_MAX_LENGTH = 20000
# --- END ADDED CONSTANTS ---
# --- End SearxNG and Grounding ---

# --- Gemini Thinking Budget ---
GEMINI_USE_THINKING_BUDGET_CONFIG_KEY = "gemini_use_thinking_budget"
GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY = "gemini_thinking_budget_value"
GEMINI_DEFAULT_USE_THINKING_BUDGET = False
GEMINI_DEFAULT_THINKING_BUDGET_VALUE = 0
GEMINI_MIN_THINKING_BUDGET_VALUE = 0
GEMINI_MAX_THINKING_BUDGET_VALUE = 24576
USER_GEMINI_THINKING_BUDGET_PREFS_FILENAME = "user_gemini_thinking_budget_prefs.json"
USER_SYSTEM_PROMPTS_FILENAME = "user_system_prompts.json"
USER_MODEL_PREFS_FILENAME = "user_model_prefs.json"
# --- End Gemini Thinking Budget ---

# --- Rate Limiting ---
RATE_LIMIT_COOLDOWN_SECONDS = 24 * 60 * 60  # 24 hours
GLOBAL_RESET_FILE = "last_reset_timestamp.txt"
DB_FOLDER = "ratelimit_dbs"

# --- Custom Google Lens (Playwright) ---
# Selectors (These might change if Google updates their site)
LENS_ICON_SELECTOR = '[aria-label="Search by image"]'
PASTE_LINK_INPUT_SELECTOR = 'input[placeholder="Paste image link"]'
SEE_EXACT_MATCHES_SELECTOR = 'div.ndigne.ZwRhJd.RiJqbb:has-text("See exact matches")'
EXACT_MATCH_RESULT_SELECTOR = 'div.ZhosBf.T7iOye.MBI8Pd.dctkEf'
INITIAL_RESULTS_WAIT_SELECTOR = "div#rso"
ORIGINAL_RESULT_SPAN_SELECTOR = 'span.Yt787'
# Timeouts
CUSTOM_LENS_DEFAULT_TIMEOUT = 60000 # milliseconds
CUSTOM_LENS_SHORT_TIMEOUT = 5000 # milliseconds

# --- Model Selection ---
# This defines which models are *known* to the bot for autocomplete/validation
# The actual availability depends on the user's config.yaml

# --- ADDED: Fallback vision model if selected model cannot handle images ---
FALLBACK_VISION_MODEL_PROVIDER_SLASH_MODEL = "google/gemini-2.0-flash"
# --- END ADDED ---

AVAILABLE_MODELS = {
    "google": [
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro-exp-03-25", # Note: Experimental
        "gemini-2.0-flash"
        # Add other Gemini models as needed/supported
    ],
    "openai": [
        "gpt-4.1"
        # Add other OpenAI models as needed/supported
    ],
    "x-ai": [
        "grok-3"
        # Add other xAI models as needed/supported
    ]
}
# Keywords for Model Override
DEEP_SEARCH_KEYWORDS = ["deepsearch", "deepersearch"]
DEEP_SEARCH_MODEL = "x-ai/grok-3" # The model to use when keywords are detected

# --- Gemini Safety Settings ---
# Use google.genai.types (imported as google_types)
GEMINI_SAFETY_SETTINGS_DICT = {
    google_types.HarmCategory.HARM_CATEGORY_HARASSMENT: google_types.HarmBlockThreshold.BLOCK_NONE,
    google_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: google_types.HarmBlockThreshold.BLOCK_NONE,
    google_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: google_types.HarmBlockThreshold.BLOCK_NONE,
    google_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: google_types.HarmBlockThreshold.BLOCK_NONE,
    # google_types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: google_types.HarmBlockThreshold.BLOCK_NONE, # Uncomment if supported and desired
}

# --- Custom Exceptions ---
class AllKeysFailedError(Exception):
    """Custom exception raised when all API keys for a service have failed."""
    def __init__(self, service_name, errors):
        self.service_name = service_name
        self.errors = errors # List of errors encountered for each key
        super().__init__(f"All API keys failed for service '{service_name}'. Last error: {errors[-1] if errors else 'None'}")
