import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime as dt, timedelta, timezone
import logging
import os
import random
import sqlite3
import sys
import time
from typing import Literal, Optional, List, Dict, Any, Tuple, Union, Set
import io
import re
import traceback
import urllib.parse
import json
import copy

import asyncpraw
import discord
from discord import ui
import httpx
# Import specific OpenAI errors and base APIError
from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError, APIConnectionError, BadRequestError
# Import the new google-genai library and its types
from google import genai as google_genai
from google.genai import types as google_types
# Import specific Google API core exceptions
from google.api_core import exceptions as google_api_exceptions
from googleapiclient.discovery import build as build_google_api_client
from googleapiclient.errors import HttpError
from asyncprawcore.exceptions import NotFound, Redirect, Forbidden, RequestException as AsyncPrawRequestException
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from serpapi import GoogleSearch # Import SerpAPI client
from serpapi.serp_api_client_exception import SerpApiClientException # Import SerpAPI exception
import yaml
from bs4 import BeautifulSoup # Added for general URL scraping

# Import Playwright for custom Google Lens
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
except ImportError:
    logging.error("Playwright not found. Custom Google Lens implementation will not work.")
    logging.error("Install it using: pip install playwright")
    logging.error("Then install browsers: python -m playwright install chrome")
    # Define dummy classes/functions if playwright is missing to avoid immediate crashes
    class PlaywrightTimeoutError(Exception): pass
    def sync_playwright():
        class DummyContextManager:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        return DummyContextManager()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# --- Constants and Configuration ---
VISION_MODEL_TAGS = ("gpt-4", "o3", "o4", "claude-3", "gemini", "gemma", "llama", "pixtral", "mistral-small", "vision", "vl", "flash")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()
EMBED_COLOR_ERROR = discord.Color.red()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

# Gemini safety settings (BLOCK_NONE for all categories)
GEMINI_SAFETY_SETTINGS_DICT = {
    google_types.HarmCategory.HARM_CATEGORY_HARASSMENT: google_types.HarmBlockThreshold.BLOCK_NONE,
    google_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: google_types.HarmBlockThreshold.BLOCK_NONE,
    google_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: google_types.HarmBlockThreshold.BLOCK_NONE,
    google_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: google_types.HarmBlockThreshold.BLOCK_NONE,
}
MAX_MESSAGE_NODES = 500 # Max nodes in runtime cache (DB handles long-term history)
MAX_EMBED_FIELD_VALUE_LENGTH = 1024
MAX_EMBED_FIELDS = 25
MAX_EMBED_DESCRIPTION_LENGTH = 4096 - len(STREAMING_INDICATOR)
MAX_URL_CONTENT_LENGTH = 100000 # Limit for scraped web content per URL
MAX_SERPAPI_RESULTS_DISPLAY = 200 # Limit number of visual matches shown per image

# --- URL Regex Patterns ---
GENERAL_URL_PATTERN = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
YOUTUBE_URL_PATTERN = re.compile(r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11}))')
REDDIT_URL_PATTERN = re.compile(r'(https?://(?:www\.)?reddit\.com/r/[a-zA-Z0-9_]+/comments/([a-zA-Z0-9]+))')
AT_AI_PATTERN = re.compile(r'\bat ai\b', re.IGNORECASE)
GOOGLE_LENS_KEYWORD = "googlelens"
GOOGLE_LENS_PATTERN = re.compile(rf'^{GOOGLE_LENS_KEYWORD}\s+', re.IGNORECASE)

# --- Rate Limiting Constants ---
RATE_LIMIT_COOLDOWN_SECONDS = 24 * 60 * 60  # 24 hours
GLOBAL_RESET_FILE = "last_reset_timestamp.txt"
DB_FOLDER = "ratelimit_dbs"

# --- Database Constants ---
HISTORY_DB_FILE = "llmcord_history.db"

# --- Custom Google Lens (Playwright) Configuration ---
LENS_ICON_SELECTOR = '[aria-label="Search by image"]'
PASTE_LINK_INPUT_SELECTOR = 'input[placeholder="Paste image link"]'
SEE_EXACT_MATCHES_SELECTOR = 'div.ndigne.ZwRhJd.RiJqbb:has-text("See exact matches")'
EXACT_MATCH_RESULT_SELECTOR = 'div.ZhosBf.T7iOye.MBI8Pd.dctkEf'
INITIAL_RESULTS_WAIT_SELECTOR = "div#rso"
ORIGINAL_RESULT_SPAN_SELECTOR = 'span.Yt787'
CUSTOM_LENS_DEFAULT_TIMEOUT = 60000
CUSTOM_LENS_SHORT_TIMEOUT = 5000

# --- Custom Exceptions ---
class AllKeysFailedError(Exception):
    """Custom exception raised when all API keys for a service have failed."""
    def __init__(self, service_name, errors):
        self.service_name = service_name
        self.errors = errors # List of errors encountered for each key
        super().__init__(f"All API keys failed for service '{service_name}'. Last error: {errors[-1] if errors else 'None'}")

# --- Rate Limit Database Manager ---
class RateLimitDBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._connect()

    def _connect(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.conn = sqlite3.connect(self.db_path)
            self._create_table()
        except sqlite3.Error as e:
            logging.error(f"Error connecting to rate limit database {self.db_path}: {e}")
            self.conn = None

    def _create_table(self):
        if not self.conn: return
        try:
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS rate_limited_keys (
                        api_key TEXT PRIMARY KEY,
                        ratelimit_timestamp REAL NOT NULL
                    )
                """)
        except sqlite3.Error as e:
            logging.error(f"Error creating rate limit table in {self.db_path}: {e}")

    def add_key(self, api_key: str):
        if not self.conn:
            logging.error(f"Cannot add key, no connection to rate limit DB {self.db_path}")
            return
        timestamp = time.time()
        try:
            with self.conn:
                self.conn.execute("""
                    INSERT OR REPLACE INTO rate_limited_keys (api_key, ratelimit_timestamp)
                    VALUES (?, ?)
                """, (api_key, timestamp))
            logging.info(f"Key ending with ...{api_key[-4:]} marked as rate-limited in {os.path.basename(self.db_path)}.")
        except sqlite3.Error as e:
            logging.error(f"Error adding key {api_key[-4:]} to rate limit DB {self.db_path}: {e}")

    def get_limited_keys(self, cooldown_seconds: int) -> Set[str]:
        if not self.conn:
            logging.error(f"Cannot get limited keys, no connection to rate limit DB {self.db_path}")
            return set()
        cutoff_time = time.time() - cooldown_seconds
        limited_keys = set()
        try:
            with self.conn:
                cursor = self.conn.execute("""
                    SELECT api_key FROM rate_limited_keys WHERE ratelimit_timestamp >= ?
                """, (cutoff_time,))
                limited_keys = {row[0] for row in cursor.fetchall()}
        except sqlite3.Error as e:
            logging.error(f"Error getting limited keys from rate limit DB {self.db_path}: {e}")
        return limited_keys

    def reset_db(self):
        if not self.conn:
            logging.error(f"Cannot reset rate limit DB, no connection to {self.db_path}")
            return
        try:
            with self.conn:
                self.conn.execute("DELETE FROM rate_limited_keys")
            logging.info(f"Rate limit database {os.path.basename(self.db_path)} reset.")
        except sqlite3.Error as e:
            logging.error(f"Error resetting rate limit database {self.db_path}: {e}")

    def close(self):
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                logging.debug(f"Closed rate limit database connection: {self.db_path}")
            except sqlite3.Error as e:
                logging.error(f"Error closing rate limit database {self.db_path}: {e}")

# --- History Database Manager ---
class HistoryDBManager:
    def __init__(self, db_path: str = HISTORY_DB_FILE):
        self.db_path = db_path
        self.conn = None
        self._connect()

    def _connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row # Access columns by name
            self._create_table()
        except sqlite3.Error as e:
            logging.error(f"Error connecting to history database {self.db_path}: {e}")
            self.conn = None

    def _create_table(self):
        if not self.conn: return
        try:
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS message_history (
                        message_id INTEGER PRIMARY KEY,
                        channel_id INTEGER NOT NULL,
                        author_id INTEGER NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                        timestamp REAL NOT NULL,
                        original_content TEXT,
                        image_urls TEXT, -- JSON list of strings
                        fetched_url_data TEXT, -- JSON list of UrlFetchResult-like dicts
                        llm_response_content TEXT, -- Final LLM text output for assistant messages
                        parent_message_id INTEGER -- Can be NULL
                    )
                """)
                # Add index for faster parent lookups
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_parent_message_id ON message_history (parent_message_id)")
        except sqlite3.Error as e:
            logging.error(f"Error creating history table in {self.db_path}: {e}")

    def save_message(self, message_id: int, channel_id: int, author_id: int, role: str, timestamp: float,
                     original_content: Optional[str], image_urls: Optional[List[str]],
                     fetched_url_data: Optional[List[Dict]], llm_response_content: Optional[str],
                     parent_message_id: Optional[int]):
        if not self.conn:
            logging.error(f"Cannot save message {message_id}, no connection to history DB {self.db_path}")
            return

        image_urls_json = json.dumps(image_urls) if image_urls else None
        fetched_url_data_json = json.dumps(fetched_url_data) if fetched_url_data else None

        try:
            with self.conn:
                self.conn.execute("""
                    INSERT OR REPLACE INTO message_history (
                        message_id, channel_id, author_id, role, timestamp, original_content,
                        image_urls, fetched_url_data, llm_response_content, parent_message_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (message_id, channel_id, author_id, role, timestamp, original_content,
                      image_urls_json, fetched_url_data_json, llm_response_content, parent_message_id))
            logging.debug(f"Saved message {message_id} (Role: {role}) to history DB.")
        except sqlite3.Error as e:
            logging.error(f"Error saving message {message_id} to history DB {self.db_path}: {e}")

    def load_message(self, message_id: int) -> Optional[sqlite3.Row]:
        if not self.conn:
            logging.error(f"Cannot load message {message_id}, no connection to history DB {self.db_path}")
            return None
        try:
            cursor = self.conn.execute("SELECT * FROM message_history WHERE message_id = ?", (message_id,))
            row = cursor.fetchone()
            return row
        except sqlite3.Error as e:
            logging.error(f"Error loading message {message_id} from history DB {self.db_path}: {e}")
            return None

    def load_conversation_history(self, start_message_id: int, max_messages: int) -> List[Dict[str, Any]]:
        """Loads conversation history by traversing parent IDs from the database."""
        if not self.conn:
            logging.error(f"Cannot load history, no connection to history DB {self.db_path}")
            return []

        history = []
        current_id: Optional[int] = start_message_id
        visited_ids = set() # Prevent infinite loops

        while current_id is not None and len(history) < max_messages and current_id not in visited_ids:
            visited_ids.add(current_id)
            row = self.load_message(current_id)
            if row:
                # Convert row to a dictionary for easier handling
                message_data = dict(row)
                # Parse JSON fields back into Python objects
                try:
                    message_data['image_urls'] = json.loads(row['image_urls']) if row['image_urls'] else []
                except (json.JSONDecodeError, TypeError) as e:
                    logging.warning(f"Could not parse image_urls JSON for message {current_id}: {e}")
                    message_data['image_urls'] = []
                try:
                    message_data['fetched_url_data'] = json.loads(row['fetched_url_data']) if row['fetched_url_data'] else []
                except (json.JSONDecodeError, TypeError) as e:
                    logging.warning(f"Could not parse fetched_url_data JSON for message {current_id}: {e}")
                    message_data['fetched_url_data'] = []

                history.append(message_data)
                current_id = row['parent_message_id'] # Move to the parent
            else:
                logging.warning(f"Message {current_id} not found in history DB while traversing.")
                break # Stop if a message in the chain is missing

        # The history is loaded in reverse chronological order (newest to oldest)
        # It will be reversed again before sending to the LLM
        return history

    def close(self):
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                logging.debug(f"Closed history database connection: {self.db_path}")
            except sqlite3.Error as e:
                logging.error(f"Error closing history database {self.db_path}: {e}")

# --- Global Rate Limit Management ---
db_managers: Dict[str, RateLimitDBManager] = {}

def get_db_manager(service_name: str) -> RateLimitDBManager:
    """Gets or creates a DB manager for a specific service."""
    global db_managers
    if service_name not in db_managers:
        db_path = os.path.join(DB_FOLDER, f"ratelimit_{service_name.lower().replace('-', '_')}.db")
        db_managers[service_name] = RateLimitDBManager(db_path)
    if db_managers[service_name].conn is None:
         db_managers[service_name]._connect()
    return db_managers[service_name]

def check_and_perform_global_reset():
    """Checks if 24 hours have passed since the last reset and resets all DBs if so."""
    now = time.time()
    last_reset_time = 0.0
    try:
        if os.path.exists(GLOBAL_RESET_FILE):
            with open(GLOBAL_RESET_FILE, 'r') as f:
                last_reset_time = float(f.read().strip())
    except (IOError, ValueError) as e:
        logging.warning(f"Could not read last reset timestamp: {e}. Forcing reset.")
        last_reset_time = 0.0

    if now - last_reset_time >= RATE_LIMIT_COOLDOWN_SECONDS:
        logging.info("Performing global 24-hour rate limit database reset.")
        services_in_config = set()
        providers = cfg.get("providers", {})
        for provider_name, provider_cfg in providers.items():
            if provider_cfg and provider_cfg.get("api_keys"):
                services_in_config.add(provider_name)
        if cfg.get("serpapi_api_keys"):
            services_in_config.add("serpapi")

        for service_name in services_in_config:
             manager = get_db_manager(service_name)
             manager.reset_db()

        for manager in db_managers.values():
            manager.reset_db()

        try:
            with open(GLOBAL_RESET_FILE, 'w') as f:
                f.write(str(now))
        except IOError as e:
            logging.error(f"Could not write last reset timestamp: {e}")
    else:
        logging.info(f"Time since last global reset: {timedelta(seconds=now - last_reset_time)}. Next reset in approx {timedelta(seconds=RATE_LIMIT_COOLDOWN_SECONDS - (now - last_reset_time))}.")

# --- API Key Selection and Retry Logic ---
async def get_available_keys(service_name: str, all_keys: List[str]) -> List[str]:
    """Gets available (non-rate-limited) keys for a service."""
    if not all_keys:
        return []

    db_manager = get_db_manager(service_name)
    limited_keys = db_manager.get_limited_keys(RATE_LIMIT_COOLDOWN_SECONDS)
    available_keys = [key for key in all_keys if key not in limited_keys]

    if not available_keys:
        logging.warning(f"All keys for service '{service_name}' are currently rate-limited. Resetting DB and using full list for this attempt.")
        db_manager.reset_db()
        return all_keys

    return available_keys

# --- Initialization ---
ytt_api = YouTubeTranscriptApi() # Initialize youtube-transcript-api client
history_db = HistoryDBManager() # Initialize history database manager

def get_config(filename="config.yaml"):
    try:
        with open(filename, "r") as file:
            config_data = yaml.safe_load(file)
            # --- Config Validation & Key Normalization ---
            providers = config_data.get("providers", {})
            for name, provider_cfg in providers.items():
                if provider_cfg:
                    single_key = provider_cfg.get("api_key")
                    key_list = provider_cfg.get("api_keys")
                    if single_key and not key_list:
                        logging.warning(f"Config Warning: Provider '{name}' uses deprecated 'api_key'. Converting to 'api_keys' list. Please update config.yaml.")
                        provider_cfg["api_keys"] = [single_key]
                        del provider_cfg["api_key"]
                    elif single_key and key_list:
                         logging.warning(f"Config Warning: Provider '{name}' has both 'api_key' and 'api_keys'. Using 'api_keys'. Please remove 'api_key' from config.yaml.")
                         del provider_cfg["api_key"]
                    elif key_list is None:
                         provider_cfg["api_keys"] = []
                    elif not isinstance(key_list, list):
                         logging.error(f"Config Error: Provider '{name}' has 'api_keys' but it's not a list. Treating as empty.")
                         provider_cfg["api_keys"] = []

            single_serp_key = config_data.get("serpapi_api_key")
            serp_key_list = config_data.get("serpapi_api_keys")
            if single_serp_key and not serp_key_list:
                logging.warning("Config Warning: Found 'serpapi_api_key'. Converting to 'serpapi_api_keys' list. Please update config.yaml.")
                config_data["serpapi_api_keys"] = [single_serp_key]
                del config_data["serpapi_api_key"]
            elif single_serp_key and serp_key_list:
                 logging.warning("Config Warning: Found both 'serpapi_api_key' and 'serpapi_api_keys'. Using 'serpapi_api_keys'. Please remove 'serpapi_api_key' from config.yaml.")
                 del config_data["serpapi_api_key"]
            elif serp_key_list is None:
                 config_data["serpapi_api_keys"] = []
            elif not isinstance(serp_key_list, list):
                 logging.error("Config Error: Found 'serpapi_api_keys' but it's not a list. Treating as empty.")
                 config_data["serpapi_api_keys"] = []

            custom_lens_cfg = config_data.get("custom_google_lens_config")
            if custom_lens_cfg:
                if not isinstance(custom_lens_cfg, dict):
                    logging.error("Config Error: 'custom_google_lens_config' must be a dictionary. Disabling custom Lens.")
                    config_data["custom_google_lens_config"] = None
                else:
                    if not custom_lens_cfg.get("user_data_dir"):
                        logging.warning("Config Warning: 'user_data_dir' missing in 'custom_google_lens_config'. Custom Lens may not work.")
                    if not custom_lens_cfg.get("profile_directory_name"):
                        logging.warning("Config Warning: 'profile_directory_name' missing in 'custom_google_lens_config'. Custom Lens may not work.")
            else:
                 logging.info("Optional 'custom_google_lens_config' not found in config.yaml. SerpAPI will be used for Google Lens.")

            return config_data

    except FileNotFoundError:
        logging.error(f"CRITICAL: {filename} not found. Please copy config-example.yaml to {filename} and configure it.")
        exit()
    except yaml.YAMLError as e:
        logging.error(f"CRITICAL: Error parsing {filename}: {e}")
        exit()

cfg = get_config()
check_and_perform_global_reset() # Perform initial check/reset after loading config

youtube_api_key = cfg.get("youtube_api_key")
reddit_client_id = cfg.get("reddit_client_id")
reddit_client_secret = cfg.get("reddit_client_secret")
reddit_user_agent = cfg.get("reddit_user_agent")
custom_google_lens_config = cfg.get("custom_google_lens_config")

if not cfg.get("bot_token"):
    logging.error("CRITICAL: bot_token is not set in config.yaml")
    exit()

if client_id := cfg.get("client_id"):
    logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")
else:
    logging.warning("client_id not found in config.yaml. Cannot generate invite URL.")

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(cfg.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_client = discord.Client(intents=intents, activity=activity)

httpx_client = httpx.AsyncClient(timeout=20.0, follow_redirects=True)

msg_nodes = {} # Still used for runtime cache/locking
last_task_time = 0

# --- Data Classes ---
@dataclass
class MsgNode: # Represents runtime state, not DB schema
    text: Optional[str] = None # Original text for user, LLM response for assistant
    images: list = field(default_factory=list) # List of image parts (base64 dict or google Part)
    role: Literal["user", "assistant", "model"] = "assistant"
    user_id: Optional[int] = None
    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False
    parent_msg_id: Optional[int] = None # Store parent ID found during runtime processing
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    full_response_text: Optional[str] = None # Final LLM response text
    fetched_url_data: List[Dict] = field(default_factory=list) # Store fetched data before saving to DB

@dataclass
class UrlFetchResult:
    url: str
    content: Optional[Union[str, Dict[str, Any]]]
    error: Optional[str] = None
    type: Literal["youtube", "reddit", "general", "google_lens_custom", "google_lens_serpapi", "google_lens_fallback_failed"] = "general"
    original_index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Converts the result to a dictionary suitable for JSON serialization."""
        return {
            "url": self.url,
            "content": self.content,
            "error": self.error,
            "type": self.type,
            "original_index": self.original_index,
        }

# --- Discord UI ---
class ResponseActionView(ui.View):
    """A view combining 'Show Sources' and 'Get response as text file' buttons."""
    def __init__(self, *,
                 grounding_metadata: Optional[Any] = None,
                 full_response_text: Optional[str] = None,
                 model_name: Optional[str] = None,
                 timeout=300):
        super().__init__(timeout=timeout)
        self.grounding_metadata = grounding_metadata
        self.full_response_text = full_response_text
        self.model_name = model_name or "llm" # Default filename model name
        self.message = None # Will be set after sending the message

        # Conditionally add buttons
        has_sources_button = False
        if self.grounding_metadata and (getattr(self.grounding_metadata, 'web_search_queries', None) or getattr(self.grounding_metadata, 'grounding_chunks', None)):
            self.add_item(self.ShowSourcesButton())
            has_sources_button = True

        if self.full_response_text:
            row = 1 if has_sources_button else 0
            self.add_item(self.GetTextFileButton(row=row))

    # Inner class for the Show Sources button
    class ShowSourcesButton(ui.Button):
        def __init__(self):
            super().__init__(label="Show Sources", style=discord.ButtonStyle.grey, row=0)

        async def callback(self, interaction: discord.Interaction):
            view: 'ResponseActionView' = self.view
            if not view.grounding_metadata:
                await interaction.response.send_message("No grounding metadata available.", ephemeral=True)
                return

            embed = discord.Embed(title="Grounding Sources", color=EMBED_COLOR_COMPLETE)
            field_count = 0

            if queries := getattr(view.grounding_metadata, 'web_search_queries', None):
                query_text = "\n".join(f"- `{q}`" for q in queries)
                if len(query_text) <= MAX_EMBED_FIELD_VALUE_LENGTH and field_count < MAX_EMBED_FIELDS:
                    embed.add_field(name="Search Queries Used", value=query_text, inline=False)
                    field_count += 1
                else:
                    logging.warning("Search query list too long for embed field.")

            if chunks := getattr(view.grounding_metadata, 'grounding_chunks', None):
                current_field_value = ""
                field_title = "Sources Consulted"
                sources_added = 0

                for chunk in chunks:
                    web_chunk = getattr(chunk, 'web', None)
                    if web_chunk and hasattr(web_chunk, 'title') and hasattr(web_chunk, 'uri'):
                        title = web_chunk.title or "Source"
                        uri = web_chunk.uri
                        source_line = f"- [{title}]({uri})\n"

                        if len(current_field_value) + len(source_line) > MAX_EMBED_FIELD_VALUE_LENGTH:
                            if current_field_value and field_count < MAX_EMBED_FIELDS:
                                embed.add_field(name=field_title, value=current_field_value, inline=False)
                                field_count += 1
                                field_title = "Sources Consulted (cont.)"
                            elif field_count >= MAX_EMBED_FIELDS:
                                logging.warning("Max embed fields reached while adding sources.")
                                break

                            if len(source_line) <= MAX_EMBED_FIELD_VALUE_LENGTH:
                                current_field_value = source_line
                            else:
                                truncated_line = source_line[:MAX_EMBED_FIELD_VALUE_LENGTH-4] + "...\n"
                                current_field_value = truncated_line
                                logging.warning(f"Single source line truncated: {source_line}")
                        else:
                            current_field_value += source_line
                        sources_added += 1

                if current_field_value and field_count < MAX_EMBED_FIELDS:
                    embed.add_field(name=field_title, value=current_field_value, inline=False)
                    field_count += 1

                if sources_added == 0 and not embed.fields:
                    embed.description = "No web sources found in metadata."

            if not embed.fields and not embed.description:
                try:
                    metadata_str = str(view.grounding_metadata)
                    embed.description = f"```json\n{metadata_str[:MAX_EMBED_DESCRIPTION_LENGTH-10]}\n```"
                except Exception:
                    embed.description = "Could not display raw grounding metadata."

            if not embed.fields and not embed.description and not embed.title:
                await interaction.response.send_message("Could not extract source information.", ephemeral=True)
            else:
                await interaction.response.send_message(embed=embed, ephemeral=True)

    # Inner class for the Get Text File button
    class GetTextFileButton(ui.Button):
        def __init__(self, row: int):
            super().__init__(label="Get response as a text file", style=discord.ButtonStyle.green, row=row)

        async def callback(self, interaction: discord.Interaction):
            view: 'ResponseActionView' = self.view
            if not view.full_response_text:
                await interaction.response.send_message("No response text available to send.", ephemeral=True)
                return

            try:
                safe_model_name = re.sub(r'[<>:"/\\|?*]', '_', view.model_name)
                filename = f"llm_response_{safe_model_name}.txt"
                file_content = io.BytesIO(view.full_response_text.encode('utf-8'))
                discord_file = discord.File(fp=file_content, filename=filename)
                await interaction.response.send_message(file=discord_file, ephemeral=True)
            except Exception as e:
                logging.error(f"Error creating or sending text file: {e}")
                await interaction.response.send_message("Sorry, I couldn't create the text file.", ephemeral=True)


# --- Helper Functions ---
def extract_urls_with_indices(text: str) -> List[Tuple[str, int]]:
    """Extracts all URLs from text along with their start index."""
    return [(match.group(0), match.start()) for match in GENERAL_URL_PATTERN.finditer(text)]

def get_domain(url: str) -> Optional[str]:
    """Extracts the domain name from a URL."""
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return None

def is_youtube_url(url: str) -> bool:
    """Checks if a URL is a YouTube URL."""
    domain = get_domain(url)
    return domain in ('youtube.com', 'www.youtube.com', 'youtu.be') and YOUTUBE_URL_PATTERN.search(url) is not None

def is_reddit_url(url: str) -> bool:
    """Checks if a URL is a Reddit submission URL."""
    domain = get_domain(url)
    return domain in ('reddit.com', 'www.reddit.com') and REDDIT_URL_PATTERN.search(url) is not None

def extract_video_id(url: str) -> Optional[str]:
    """Extracts the video ID from a YouTube URL."""
    match = YOUTUBE_URL_PATTERN.search(url)
    return match.group(2) if match else None

def extract_reddit_submission_id(url: str) -> Optional[str]:
    """Extracts the submission ID from a Reddit URL."""
    match = REDDIT_URL_PATTERN.search(url)
    return match.group(2) if match else None

def format_fetched_data_for_llm(fetched_data: List[Dict]) -> str:
    """Formats the fetched URL/Lens data (from DB) into a string for the LLM context."""
    if not fetched_data:
        return ""

    google_lens_parts = []
    other_url_parts = []
    other_url_counter = 1

    # Sort by original index to maintain order
    fetched_data.sort(key=lambda r: r.get('original_index', -1))

    for result in fetched_data:
        content = result.get('content')
        result_type = result.get('type')
        url = result.get('url', 'N/A')
        index = result.get('original_index', -1)

        if not content: # Skip items with no content (likely errors)
            continue

        if result_type == "google_lens_custom":
            header = f"Custom Google Lens implementation results for image {index + 1}:\n"
            google_lens_parts.append(header + str(content))
        elif result_type == "google_lens_serpapi":
            header = f"SerpAPI Google Lens fallback results for image {index + 1}:\n"
            google_lens_parts.append(header + str(content))
        elif result_type == "youtube":
            content_str = f"\nurl {other_url_counter}: {url}\n"
            content_str += f"url {other_url_counter} content:\n"
            if isinstance(content, dict):
                content_str += f"  title: {content.get('title', 'N/A')}\n"
                content_str += f"  channel: {content.get('channel_name', 'N/A')}\n"
                desc = content.get('description', 'N/A')
                content_str += f"  description: {desc[:500]}{'...' if len(desc) > 500 else ''}\n"
                transcript = content.get('transcript')
                if transcript:
                    content_str += f"  transcript: {transcript[:MAX_URL_CONTENT_LENGTH]}{'...' if len(transcript) > MAX_URL_CONTENT_LENGTH else ''}\n"
                comments = content.get("comments")
                if comments:
                    content_str += f"  top comments:\n" + "\n".join([f"    - {c[:150]}{'...' if len(c) > 150 else ''}" for c in comments[:5]]) + "\n"
            other_url_parts.append(content_str)
            other_url_counter += 1
        elif result_type == "reddit":
            content_str = f"\nurl {other_url_counter}: {url}\n"
            content_str += f"url {other_url_counter} content:\n"
            if isinstance(content, dict):
                content_str += f"  title: {content.get('title', 'N/A')}\n"
                selftext = content.get('selftext')
                if selftext:
                    content_str += f"  content: {selftext[:MAX_URL_CONTENT_LENGTH]}{'...' if len(selftext) > MAX_URL_CONTENT_LENGTH else ''}\n"
                comments = content.get("comments")
                if comments:
                    content_str += f"  top comments:\n" + "\n".join([f"    - {c[:150]}{'...' if len(c) > 150 else ''}" for c in comments[:5]]) + "\n"
            other_url_parts.append(content_str)
            other_url_counter += 1
        elif result_type == "general":
            content_str = f"\nurl {other_url_counter}: {url}\n"
            content_str += f"url {other_url_counter} content:\n"
            if isinstance(content, str):
                content_str += f"  {content}\n"
            other_url_parts.append(content_str)
            other_url_counter += 1

    combined_context = ""
    if google_lens_parts or other_url_parts:
        combined_context = "Answer the user's query based on the following:\n\n"
        if google_lens_parts:
            combined_context += "\n\n".join(google_lens_parts) + "\n\n"
        if other_url_parts:
            combined_context += "".join(other_url_parts)

    return combined_context.strip()


# --- Content Fetching Functions --- (Mostly unchanged, added logging)

async def get_transcript(video_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetches the transcript for a video ID using youtube-transcript-api."""
    logging.info(f"Fetching transcript for YouTube video ID: {video_id}")
    try:
        transcript_list = await asyncio.to_thread(ytt_api.list_transcripts, video_id)
        transcript = None
        priorities = [
            (transcript_list.find_manually_created_transcript, ['en']),
            (transcript_list.find_generated_transcript, ['en']),
            (transcript_list.find_manually_created_transcript, [lang.language_code for lang in transcript_list]),
            (transcript_list.find_generated_transcript, [lang.language_code for lang in transcript_list]),
        ]
        for find_method, langs in priorities:
            if not langs: continue
            try:
                transcript = await asyncio.to_thread(find_method, langs)
                if transcript:
                    logging.info(f"Found transcript for {video_id} (Lang: {transcript.language_code}, Generated: {transcript.is_generated})")
                    break
            except NoTranscriptFound: continue
            except Exception as e:
                logging.warning(f"Error during transcript find method {find_method.__name__} for {video_id}: {e}")
                continue

        if transcript:
            # youtube-transcript-api returns list of dicts
            fetched_transcript_list = await asyncio.to_thread(transcript.fetch)
            full_transcript = " ".join([entry.text for entry in fetched_transcript_list]) # never make this full_transcript = " ".join([entry['text'] for entry in fetched_transcript_list]) because 'FetchedTranscriptSnippet' object is not subscriptable 
            logging.info(f"Successfully fetched transcript for {video_id} (Length: {len(full_transcript)})")
            return full_transcript, None
        else:
            logging.warning(f"No suitable transcript found for {video_id}")
            return None, "No suitable transcript found."
    except TranscriptsDisabled:
        logging.warning(f"Transcripts disabled for YouTube video ID: {video_id}")
        return None, "Transcripts are disabled for this video."
    except NoTranscriptFound:
        logging.warning(f"No transcripts listed for YouTube video ID: {video_id}")
        return None, "No transcripts listed for this video."
    except Exception as e:
        logging.error(f"Error fetching transcript for {video_id}: {type(e).__name__}: {e}")
        return None, f"An error occurred fetching transcript: {type(e).__name__}"

async def get_youtube_video_details(video_id: str, api_key: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetches video title, description, channel name, and comments using YouTube Data API."""
    if not api_key:
        logging.warning(f"Cannot fetch YouTube details for {video_id}: API key not configured.")
        return None, "YouTube API key not configured."

    logging.info(f"Fetching details for YouTube video ID: {video_id}")
    details = {}
    error_messages = []

    try:
        youtube = await asyncio.to_thread(build_google_api_client, 'youtube', 'v3', developerKey=api_key)

        # Get video details
        try:
            video_request = youtube.videos().list(part="snippet", id=video_id)
            video_response = await asyncio.to_thread(video_request.execute)
            if video_response.get("items"):
                snippet = video_response["items"][0]["snippet"]
                details["title"] = snippet.get("title", "N/A")
                details["description"] = snippet.get("description", "N/A")
                details["channel_name"] = snippet.get("channelTitle", "N/A")
                logging.info(f"Fetched video details for {video_id}")
            else:
                 error_messages.append("Video details not found.")
                 logging.warning(f"Video details not found for {video_id}")
        except HttpError as e:
            error_reason = getattr(e, 'reason', str(e))
            status_code = getattr(e.resp, 'status', 'Unknown')
            logging.warning(f"YouTube Data API error getting video details for {video_id} (Status: {status_code}): {error_reason}")
            error_messages.append(f"API error getting video details ({status_code}).")
        except Exception as e:
            logging.exception(f"Unexpected error getting YouTube video details for {video_id}")
            error_messages.append(f"Unexpected error getting video details: {type(e).__name__}")

        # Get top comments
        try:
            comment_request = youtube.commentThreads().list(
                part="snippet", videoId=video_id, order="relevance", maxResults=10, textFormat="plainText"
            )
            comment_response = await asyncio.to_thread(comment_request.execute)
            details["comments"] = [
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for item in comment_response.get("items", [])
            ]
            logging.info(f"Fetched {len(details.get('comments',[]))} comments for {video_id}")
        except HttpError as e:
             error_reason = getattr(e, 'reason', str(e))
             status_code = getattr(e.resp, 'status', 'Unknown')
             if status_code == 403 and 'commentsDisabled' in str(e):
                 logging.info(f"Comments disabled for YouTube video {video_id}")
                 error_messages.append("Comments disabled.")
             else:
                 logging.warning(f"YouTube Data API error getting comments for {video_id} (Status: {status_code}): {error_reason}")
                 error_messages.append(f"API error getting comments ({status_code}).")
        except Exception as e:
            logging.exception(f"Unexpected error getting YouTube comments for {video_id}")
            error_messages.append(f"Unexpected error getting comments: {type(e).__name__}")

        final_details = details if details else None
        final_error = " ".join(error_messages) if error_messages else None
        return final_details, final_error

    except HttpError as e:
        error_reason = getattr(e, 'reason', str(e))
        status_code = getattr(e.resp, 'status', 'Unknown')
        logging.error(f"YouTube Data API Build/Auth error (Status: {status_code}): {error_reason}")
        if status_code == 403:
             if "quotaExceeded" in str(e): return None, "YouTube API quota exceeded."
             elif "accessNotConfigured" in str(e): return None, "YouTube API access not configured or key invalid."
             else: return None, f"YouTube API permission error: {error_reason}"
        else: return None, f"YouTube Data API HTTP error: {error_reason}"
    except Exception as e:
        logging.exception(f"Unexpected error initializing YouTube client or fetching details for {video_id}")
        return None, f"An unexpected error occurred: {e}"


async def fetch_youtube_data(url: str, index: int, api_key: Optional[str]) -> UrlFetchResult:
    """Fetches transcript and details for a single YouTube URL."""
    video_id = extract_video_id(url)
    if not video_id:
        return UrlFetchResult(url=url, content=None, error="Could not extract video ID.", type="youtube", original_index=index)

    transcript_task = asyncio.create_task(get_transcript(video_id))
    details_task = asyncio.create_task(get_youtube_video_details(video_id, api_key))

    transcript, transcript_error = await transcript_task
    details, details_error = await details_task

    combined_content = {}
    errors = [err for err in [transcript_error, details_error] if err]

    if details: combined_content.update(details)
    if transcript: combined_content["transcript"] = transcript
    if not combined_content and not errors: errors.append("No content fetched.")

    return UrlFetchResult(
        url=url,
        content=combined_content if combined_content else None,
        error=" ".join(errors) if errors else None,
        type="youtube",
        original_index=index
    )

async def fetch_reddit_data(url: str, submission_id: str, index: int, client_id: str, client_secret: str, user_agent: str) -> UrlFetchResult:
    """Fetches content for a single Reddit submission URL."""
    if not all([client_id, client_secret, user_agent]):
        logging.warning(f"Cannot fetch Reddit data for {url}: API credentials not configured.")
        return UrlFetchResult(url=url, content=None, error="Reddit API credentials not configured.", type="reddit", original_index=index)

    logging.info(f"Fetching data for Reddit submission ID: {submission_id}")
    reddit = None
    try:
        reddit = asyncpraw.Reddit(
            client_id=client_id, client_secret=client_secret, user_agent=user_agent, read_only=True
        )
        submission = await reddit.submission(id=submission_id)
        await submission.load()

        content_data = {"title": submission.title}
        if submission.selftext: content_data["selftext"] = submission.selftext

        top_comments_text = []
        comment_limit = 10
        await submission.comments.replace_more(limit=0)
        comment_count = 0
        for top_level_comment in submission.comments.list():
            if comment_count >= comment_limit: break
            if hasattr(top_level_comment, 'body') and top_level_comment.body and top_level_comment.body not in ('[deleted]', '[removed]'):
                comment_body_cleaned = top_level_comment.body.replace('\n', ' ').replace('\r', '')
                top_comments_text.append(comment_body_cleaned)
                comment_count += 1
        if top_comments_text: content_data["comments"] = top_comments_text

        logging.info(f"Successfully fetched Reddit data for {url} (Comments: {len(top_comments_text)})")
        return UrlFetchResult(url=url, content=content_data, type="reddit", original_index=index)

    except (NotFound, Redirect):
        logging.warning(f"Reddit submission not found or invalid URL: {url}")
        return UrlFetchResult(url=url, content=None, error="Submission not found or invalid URL.", type="reddit", original_index=index)
    except Forbidden as e:
         logging.warning(f"Reddit API Forbidden error for {url}: {e}")
         return UrlFetchResult(url=url, content=None, error="Reddit API access forbidden.", type="reddit", original_index=index)
    except AsyncPrawRequestException as e:
        logging.warning(f"Reddit API Request error for {url}: {e}")
        return UrlFetchResult(url=url, content=None, error=f"Reddit API request error: {type(e).__name__}", type="reddit", original_index=index)
    except Exception as e:
        logging.exception(f"Unexpected error fetching Reddit content for {url}")
        return UrlFetchResult(url=url, content=None, error=f"Unexpected error: {type(e).__name__}", type="reddit", original_index=index)
    finally:
        if reddit: await reddit.close()


async def fetch_general_url_content(url: str, index: int) -> UrlFetchResult:
    """Fetches and extracts text content from a general URL using BeautifulSoup."""
    logging.info(f"Fetching general URL content: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        async with httpx_client.stream("GET", url, headers=headers, timeout=15.0) as response:
            if response.status_code != 200:
                 if response.status_code >= 300 and response.status_code < 400 and len(response.history) > 5:
                     logging.warning(f"Too many redirects for URL {url}")
                     return UrlFetchResult(url=url, content=None, error=f"Too many redirects ({response.status_code}).", type="general", original_index=index)
                 logging.warning(f"HTTP status {response.status_code} for URL {url}")
                 return UrlFetchResult(url=url, content=None, error=f"HTTP status {response.status_code}.", type="general", original_index=index)

            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                logging.warning(f"Unsupported content type '{content_type}' for URL {url}")
                return UrlFetchResult(url=url, content=None, error=f"Unsupported content type: {content_type}", type="general", original_index=index)

            html_content = ""
            try:
                async for chunk in response.aiter_bytes():
                    html_content += chunk.decode(response.encoding or 'utf-8', errors='replace')
                    if len(html_content) > 5 * 1024 * 1024:
                        logging.warning(f"HTML content truncated for URL {url} due to size limit.")
                        html_content = html_content[:5*1024*1024] + "..."
                        break
            except httpx.ReadTimeout:
                 logging.warning(f"Timeout reading content for URL {url}")
                 return UrlFetchResult(url=url, content=None, error="Timeout while reading content.", type="general", original_index=index)
            except Exception as e:
                 logging.warning(f"Error decoding content for {url}: {e}")
                 return UrlFetchResult(url=url, content=None, error=f"Content decoding error: {type(e).__name__}", type="general", original_index=index)

        soup = BeautifulSoup(html_content, 'html.parser')
        for script_or_style in soup(["script", "style"]): script_or_style.decompose()
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        text = main_content.get_text(separator=' ', strip=True) if main_content else soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            logging.warning(f"No text content found for URL {url}")
            return UrlFetchResult(url=url, content=None, error="No text content found.", type="general", original_index=index)

        content = text[:MAX_URL_CONTENT_LENGTH] + ('...' if len(text) > MAX_URL_CONTENT_LENGTH else '')
        logging.info(f"Successfully fetched general URL content for {url} (Length: {len(content)})")
        return UrlFetchResult(url=url, content=content, type="general", original_index=index)

    except httpx.RequestError as e:
        logging.warning(f"HTTPX RequestError fetching {url}: {type(e).__name__}")
        return UrlFetchResult(url=url, content=None, error=f"Request failed: {type(e).__name__}", type="general", original_index=index)
    except Exception as e:
        logging.exception(f"Unexpected error fetching general URL {url}")
        return UrlFetchResult(url=url, content=None, error=f"Unexpected error: {type(e).__name__}", type="general", original_index=index)

# --- Custom Google Lens (Playwright) Implementation ---
def _custom_get_google_lens_results_sync(image_url: str, user_data_dir: str, profile_directory_name: str):
    """Synchronous wrapper for the Playwright Google Lens logic."""
    try: from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    except ImportError:
        logging.error("Playwright library not found. Cannot run custom Google Lens.")
        return None

    if not user_data_dir or not profile_directory_name:
        logging.error("Custom Google Lens: Chrome user_data_dir or profile_directory_name not provided.")
        return None
    if not os.path.exists(user_data_dir):
        logging.error(f"Custom Google Lens: Chrome user data directory not found at: {user_data_dir}")
        return None
    profile_path = os.path.join(user_data_dir, profile_directory_name)
    if not os.path.exists(profile_path):
        logging.error(f"Custom Google Lens: Specific profile directory not found at: {profile_path}")
        logging.error(f"Please ensure '{profile_directory_name}' is the correct folder name inside '{user_data_dir}'.")
        return None

    results = []
    context = None
    logging.info(f"Custom Google Lens: Launching Chrome using profile: '{profile_directory_name}'")
    logging.info(f"Custom Google Lens: User Data Directory: {user_data_dir}")
    logging.info(f"Custom Google Lens: Searching for image URL: {image_url}")
    logging.info("Custom Google Lens: INFO: Ensure Google Chrome using this specific profile is completely closed (check Task Manager).")

    launch_args = ['--no-first-run', '--no-default-browser-check', f"--profile-directory={profile_directory_name}"]

    with sync_playwright() as p:
        try:
            context = p.chromium.launch_persistent_context(
                user_data_dir, headless=False, channel="chrome", args=launch_args, slow_mo=50
            )
            page = context.new_page()
            page.set_default_timeout(CUSTOM_LENS_DEFAULT_TIMEOUT)
            logging.info("Custom Google Lens: Navigating to google.com...")
            page.goto("https://www.google.com/")
            logging.info("Custom Google Lens: Waiting for and clicking the Lens icon...")
            lens_icon = page.locator(LENS_ICON_SELECTOR)
            lens_icon.wait_for(state="visible")
            lens_icon.click()
            logging.info("Custom Google Lens: Waiting for the image link input field...")
            paste_link_input = page.locator(PASTE_LINK_INPUT_SELECTOR)
            paste_link_input.wait_for(state="visible")
            logging.info("Custom Google Lens: Pasting the image URL...")
            paste_link_input.fill(image_url)
            logging.info("Custom Google Lens: Submitting the search (pressing Enter)...")
            page.wait_for_timeout(200)
            paste_link_input.press("Enter")
            logging.info("Custom Google Lens: Waiting for initial Lens results page to load...")
            try:
                 page.wait_for_selector(INITIAL_RESULTS_WAIT_SELECTOR, state="attached", timeout=CUSTOM_LENS_DEFAULT_TIMEOUT)
                 logging.info("Custom Google Lens: Initial results container found.")
            except PlaywrightTimeoutError:
                 logging.warning(f"Custom Google Lens: Timed out waiting for initial results container ('{INITIAL_RESULTS_WAIT_SELECTOR}').")
                 current_url = page.url
                 if "google.com/search?" not in current_url or ("lens" not in current_url and "source=lns" not in current_url):
                     logging.warning(f"Custom Google Lens: Current URL doesn't look like a Lens results page: {current_url}")
                 pass

            logging.info(f"Custom Google Lens: Checking for '{SEE_EXACT_MATCHES_SELECTOR}'...")
            see_exact_matches_button = page.locator(SEE_EXACT_MATCHES_SELECTOR)
            final_result_selector = None
            result_elements = []

            try:
                if see_exact_matches_button.is_visible(timeout=CUSTOM_LENS_SHORT_TIMEOUT):
                    logging.info("Custom Google Lens: Found 'See exact matches', clicking it...")
                    try:
                        see_exact_matches_button.click()
                        logging.info("Custom Google Lens: Waiting for exact match results to load (waiting for last element)...")
                        page.locator(EXACT_MATCH_RESULT_SELECTOR).last.wait_for(state="visible", timeout=CUSTOM_LENS_DEFAULT_TIMEOUT)
                        logging.info(f"Custom Google Lens: Exact match results likely loaded. Using selector: '{EXACT_MATCH_RESULT_SELECTOR}'")
                        final_result_selector = EXACT_MATCH_RESULT_SELECTOR
                    except PlaywrightTimeoutError: logging.warning(f"Custom Google Lens: Clicked 'See exact matches' but timed out waiting for the last element of '{EXACT_MATCH_RESULT_SELECTOR}'.")
                    except Exception as click_err: logging.error(f"Custom Google Lens: Error clicking 'See exact matches' or waiting after click: {click_err}")
                else:
                    logging.info("Custom Google Lens: 'See exact matches' not found or not visible within timeout.")
                    logging.info(f"Custom Google Lens: Looking for general results using selector: '{ORIGINAL_RESULT_SPAN_SELECTOR}' (waiting for last element)...")
                    try:
                        page.locator(ORIGINAL_RESULT_SPAN_SELECTOR).last.wait_for(state="visible", timeout=CUSTOM_LENS_DEFAULT_TIMEOUT)
                        logging.info(f"Custom Google Lens: General results likely loaded. Using selector: '{ORIGINAL_RESULT_SPAN_SELECTOR}'")
                        final_result_selector = ORIGINAL_RESULT_SPAN_SELECTOR
                    except PlaywrightTimeoutError: logging.warning(f"Custom Google Lens: Fallback check for the last original result ('{ORIGINAL_RESULT_SPAN_SELECTOR}') also timed out.")
            except PlaywrightTimeoutError:
                 logging.warning(f"Custom Google Lens: Timeout checking visibility for '{SEE_EXACT_MATCHES_SELECTOR}'. Assuming it's not present.")
                 logging.info(f"Custom Google Lens: Looking for general results using selector: '{ORIGINAL_RESULT_SPAN_SELECTOR}' (waiting for last element)...")
                 try:
                     page.locator(ORIGINAL_RESULT_SPAN_SELECTOR).last.wait_for(state="visible", timeout=CUSTOM_LENS_DEFAULT_TIMEOUT)
                     logging.info(f"Custom Google Lens: General results likely loaded. Using selector: '{ORIGINAL_RESULT_SPAN_SELECTOR}'")
                     final_result_selector = ORIGINAL_RESULT_SPAN_SELECTOR
                 except PlaywrightTimeoutError: logging.warning(f"Custom Google Lens: Fallback check for the last original result ('{ORIGINAL_RESULT_SPAN_SELECTOR}') also timed out.")

            if final_result_selector:
                logging.info(f"Custom Google Lens: Extracting text using final selector: '{final_result_selector}'...")
                page.wait_for_timeout(500)
                result_elements = page.locator(final_result_selector).all()
                if not result_elements: logging.info("Custom Google Lens: No result elements found matching the final selector.")
                for i, element in enumerate(result_elements):
                    try:
                        text = element.text_content()
                        if text: results.append(' '.join(text.split()))
                        else: logging.warning(f"Custom Google Lens: Found element {i+1} but it has no text content.")
                    except Exception as e: logging.error(f"Custom Google Lens: Error extracting text from element {i+1}: {e}")
            else: logging.info("Custom Google Lens: No suitable result selector was determined. Skipping extraction.")
            logging.info("Custom Google Lens: Finished extracting results.")

        except PlaywrightTimeoutError as e:
            logging.error(f"Custom Google Lens: ERROR: A timeout occurred: {e}")
            try:
                if 'page' in locals() and page and not page.is_closed():
                    screenshot_path = "error_screenshot_timeout.png"
                    page.screenshot(path=screenshot_path)
                    logging.info(f"Custom Google Lens: Screenshot saved as {screenshot_path}")
            except Exception as screen_err: logging.error(f"Custom Google Lens: Could not take screenshot on timeout error: {screen_err}")
            return None
        except Exception as e:
            logging.error(f"Custom Google Lens: An unexpected error occurred: {e}")
            if "Target page, context or browser has been closed" in str(e):
                 logging.error("Custom Google Lens: ERROR DETAILS: This 'TargetClosedError' usually means Google Chrome was already running with the specified profile.")
                 logging.error(f"Profile Folder: '{profile_directory_name}' within '{user_data_dir}'")
                 logging.error("Please ensure ALL Chrome processes using this profile are closed (check Task Manager) before running the script again.")
            else: logging.exception("Custom Google Lens: Unexpected error details:")
            try:
                 if 'page' in locals() and page and not page.is_closed():
                    screenshot_path = "error_screenshot_unexpected.png"
                    page.screenshot(path=screenshot_path)
                    logging.info(f"Custom Google Lens: Screenshot saved as {screenshot_path}")
            except Exception as screen_err: logging.error(f"Custom Google Lens: Could not take screenshot on unexpected error: {screen_err}")
            return None
        finally:
            if context:
                logging.info("Custom Google Lens: Closing browser context...")
                try:
                    if context.pages: context.close()
                except Exception as close_err: logging.warning(f"Custom Google Lens: Note: Error during context close: {close_err}")
    return results

async def fetch_google_lens_serpapi_fallback(image_url: str, index: int) -> UrlFetchResult:
    """Fetches Google Lens results using SerpAPI with key rotation and retry (FALLBACK ONLY)."""
    service_name = "serpapi"
    all_keys = cfg.get("serpapi_api_keys", [])
    if not all_keys:
        logging.warning(f"Cannot perform SerpAPI fallback for image {index+1}: No keys configured.")
        return UrlFetchResult(url=image_url, content=None, error="SerpAPI keys not configured for fallback.", type="google_lens_serpapi", original_index=index)

    available_keys = await get_available_keys(service_name, all_keys)
    random.shuffle(available_keys)
    db_manager = get_db_manager(service_name)
    encountered_errors = []

    for key_index, api_key in enumerate(available_keys):
        params = {"engine": "google_lens", "url": image_url, "api_key": api_key, "safe": "off"}
        logging.info(f"Attempting SerpAPI Google Lens fallback request for image {index+1} with key ...{api_key[-4:]} ({key_index+1}/{len(available_keys)})")
        try:
            search = GoogleSearch(params)
            results = await asyncio.to_thread(search.get_dict)
            if "error" in results:
                error_msg = results["error"]
                logging.warning(f"SerpAPI fallback error for image {index+1} (key ...{api_key[-4:]}): {error_msg}")
                encountered_errors.append(f"Key ...{api_key[-4:]}: {error_msg}")
                if "rate limit" in error_msg.lower() or "quota" in error_msg.lower() or "plan limit" in error_msg.lower() or "ran out of searches" in error_msg.lower():
                    db_manager.add_key(api_key)
                    continue
                elif "invalid api key" in error_msg.lower():
                    return UrlFetchResult(url=image_url, content=None, error=f"SerpAPI Error: Invalid API Key (...{api_key[-4:]})", type="google_lens_serpapi", original_index=index)
                else: continue
            if results.get("search_metadata", {}).get("status", "").lower() == "error":
                 error_msg = results.get("search_metadata", {}).get("error", "Unknown search error")
                 logging.warning(f"SerpAPI fallback search error for image {index+1} (key ...{api_key[-4:]}): {error_msg}")
                 encountered_errors.append(f"Key ...{api_key[-4:]}: {error_msg}")
                 return UrlFetchResult(url=image_url, content=None, error=f"SerpAPI Search Error: {error_msg}", type="google_lens_serpapi", original_index=index)

            visual_matches = results.get("visual_matches", [])
            if not visual_matches:
                logging.info(f"SerpAPI fallback for image {index+1} found no visual matches.")
                return UrlFetchResult(url=image_url, content="No visual matches found (SerpAPI fallback).", type="google_lens_serpapi", original_index=index)

            formatted_results = []
            for i, match in enumerate(visual_matches[:MAX_SERPAPI_RESULTS_DISPLAY]):
                title, link, source = match.get("title", "N/A"), match.get("link", "#"), match.get("source", "")
                result_line = f"- [{title}]({link})" + (f" (Source: {source})" if source else "")
                formatted_results.append(result_line)
            content_str = "\n".join(formatted_results)
            if len(visual_matches) > MAX_SERPAPI_RESULTS_DISPLAY: content_str += f"\n- ... (and {len(visual_matches) - MAX_SERPAPI_RESULTS_DISPLAY} more)"

            logging.info(f"SerpAPI Google Lens fallback request successful for image {index+1} with key ...{api_key[-4:]}")
            return UrlFetchResult(url=image_url, content=content_str, type="google_lens_serpapi", original_index=index)

        except SerpApiClientException as e:
            logging.warning(f"SerpAPI client exception during fallback for image {index+1} (key ...{api_key[-4:]}): {e}")
            encountered_errors.append(f"Key ...{api_key[-4:]}: Client Error - {e}")
            if "429" in str(e) or "rate limit" in str(e).lower(): db_manager.add_key(api_key)
            continue
        except Exception as e:
            logging.exception(f"Unexpected error during SerpAPI fallback for image {index+1} (key ...{api_key[-4:]})")
            encountered_errors.append(f"Key ...{api_key[-4:]}: Unexpected Error - {type(e).__name__}")
            continue

    logging.error(f"All SerpAPI keys failed during fallback for Google Lens request for image {index+1}.")
    final_error_msg = "All SerpAPI keys failed during fallback." + (f" Last error: {encountered_errors[-1]}" if encountered_errors else "")
    return UrlFetchResult(url=image_url, content=None, error=final_error_msg, type="google_lens_serpapi", original_index=index)


async def process_google_lens_image(image_url: str, index: int) -> UrlFetchResult:
    """Processes a Google Lens request, trying custom first, then falling back to SerpAPI."""
    custom_config = cfg.get("custom_google_lens_config")
    custom_results = None
    custom_error = None
    custom_impl_attempted = False

    if custom_config and custom_config.get("user_data_dir") and custom_config.get("profile_directory_name"):
        user_data_dir, profile_name = custom_config["user_data_dir"], custom_config["profile_directory_name"]
        logging.info(f"Attempting Google Lens request for image {index+1} using custom implementation (Profile: {profile_name})")
        custom_impl_attempted = True
        try:
            custom_results = await asyncio.to_thread(_custom_get_google_lens_results_sync, image_url, user_data_dir, profile_name)
            if custom_results is not None:
                logging.info(f"Custom Google Lens implementation successful for image {index+1}.")
                if not custom_results: content_str = "No visual matches found (custom implementation)."
                else:
                    formatted_results = [f"- {res[:200]}{'...' if len(res) > 200 else ''}" for i, res in enumerate(custom_results[:MAX_SERPAPI_RESULTS_DISPLAY])]
                    content_str = "\n".join(formatted_results)
                    if len(custom_results) > MAX_SERPAPI_RESULTS_DISPLAY: content_str += f"\n- ... (and {len(custom_results) - MAX_SERPAPI_RESULTS_DISPLAY} more)"
                return UrlFetchResult(url=image_url, content=content_str, type="google_lens_custom", original_index=index)
            else:
                custom_error = "Custom implementation failed (returned None)."
                logging.warning(f"Custom Google Lens implementation failed for image {index+1} (returned None). Falling back to SerpAPI.")
        except Exception as e:
            custom_error = f"Custom implementation raised an exception: {type(e).__name__}: {e}"
            logging.exception(f"Custom Google Lens implementation failed for image {index+1} with exception. Falling back to SerpAPI.")
    else:
        custom_error = "Custom Google Lens implementation not configured."
        logging.info("Custom Google Lens implementation not configured. Falling back to SerpAPI.")

    logging.info(f"Falling back to SerpAPI for Google Lens request for image {index+1}.")
    serpapi_result = await fetch_google_lens_serpapi_fallback(image_url, index)

    if serpapi_result.error:
        if custom_impl_attempted and custom_error: final_error_msg = f"Custom Lens failed ({custom_error}). Fallback SerpAPI also failed: {serpapi_result.error}"
        elif custom_error: final_error_msg = f"SerpAPI fallback failed: {serpapi_result.error}"
        else: final_error_msg = f"SerpAPI fallback failed: {serpapi_result.error}"
        return UrlFetchResult(url=image_url, content=None, error=final_error_msg, type="google_lens_fallback_failed", original_index=index)
    else:
        serpapi_result.type = "google_lens_serpapi"
        return serpapi_result


# --- Payload Logging Helper ---
def _sanitize_for_logging(data: Any) -> Any:
    """Recursively sanitizes data structures for JSON logging."""
    if isinstance(data, dict):
        copied_dict = copy.deepcopy(data)
        return {key: _sanitize_for_logging(value) for key, value in copied_dict.items()}
    elif isinstance(data, list):
        copied_list = copy.deepcopy(data)
        return [_sanitize_for_logging(item) for item in copied_list]
    elif isinstance(data, google_types.Part):
        part_dict = {"type": "google_types.Part"}
        if data.text is not None: part_dict["text"] = data.text
        if data.inline_data is not None: part_dict["inline_data"] = {"mime_type": data.inline_data.mime_type, "data": f"<bytes len={len(data.inline_data.data)}>"}
        if data.file_data is not None: part_dict["file_data"] = {"mime_type": data.file_data.mime_type, "file_uri": data.file_data.file_uri}
        if data.function_call is not None: part_dict["function_call"] = {"name": data.function_call.name, "args": _sanitize_for_logging(data.function_call.args)}
        if data.function_response is not None: part_dict["function_response"] = {"name": data.function_response.name, "response": _sanitize_for_logging(data.function_response.response)}
        return part_dict
    elif isinstance(data, google_types.Content):
        return {"role": data.role, "parts": _sanitize_for_logging(data.parts)}
    elif isinstance(data, google_types.Tool):
        tool_dict = {"type": "google_types.Tool"}
        if data.function_declarations: tool_dict["function_declarations"] = [{"name": f.name, "description": f.description} for f in data.function_declarations]
        if data.google_search_retrieval: tool_dict["google_search_retrieval"] = _sanitize_for_logging(data.google_search_retrieval)
        if data.code_execution: tool_dict["code_execution"] = {}
        if data.google_search: tool_dict["google_search"] = {}
        return tool_dict
    elif isinstance(data, google_types.SafetySetting):
        return {"category": data.category.name, "threshold": data.threshold.name}
    elif isinstance(data, google_types.GenerateContentConfig):
        config_dict = {"type": "google_types.GenerateContentConfig"}
        for attr in ['temperature', 'top_p', 'top_k', 'candidate_count', 'max_output_tokens', 'stop_sequences', 'response_mime_type', 'response_schema', 'safety_settings', 'tools', 'tool_config', 'system_instruction', 'thinking_config']:
            if hasattr(data, attr):
                value = getattr(data, attr)
                if value is not None: config_dict[attr] = _sanitize_for_logging(value)
        return config_dict
    elif isinstance(data, bytes): return f"<bytes len={len(data)}>"
    elif isinstance(data, dict) and data.get("type") == "image_url": return data
    elif hasattr(data, '__dict__') or hasattr(data, '__slots__'):
        try: return _sanitize_for_logging(vars(data))
        except TypeError: return f"<{type(data).__name__} object>"
    else: return data

def log_llm_payload(provider: str, model_name: str, args: Dict[str, Any]):
    """Logs the sanitized payload being sent to the LLM."""
    try:
        payload_to_log = {"provider": provider, "model": model_name, **args}
        sanitized_payload = _sanitize_for_logging(payload_to_log)
        json_payload = json.dumps(sanitized_payload, indent=2, ensure_ascii=False)
        logging.info(f"\n--- LLM Payload Sent ---\n{json_payload}\n------------------------")
    except Exception as e:
        logging.error(f"Error logging LLM payload: {e}")
        logging.info(f"--- LLM Payload Sent (Raw Args Fallback) ---\nProvider: {provider}\nModel: {model_name}\nArgs: {args}\n------------------------")


# --- Discord Event Handler ---
@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time, cfg, youtube_api_key, reddit_client_id, reddit_client_secret, reddit_user_agent, custom_google_lens_config, history_db

    if new_msg.author.bot: return

    is_dm = new_msg.channel.type == discord.ChannelType.private
    allow_dms = cfg.get("allow_dms", True)

    should_process = False
    mentions_bot = False
    contains_at_ai = False
    original_content_for_processing = new_msg.content

    if is_dm:
        if allow_dms:
            should_process = True
            mentions_bot = discord_client.user in new_msg.mentions
            contains_at_ai = AT_AI_PATTERN.search(original_content_for_processing) is not None
        else: return
    else:
        mentions_bot = discord_client.user in new_msg.mentions
        contains_at_ai = AT_AI_PATTERN.search(original_content_for_processing) is not None
        if mentions_bot or contains_at_ai: should_process = True

    if not should_process: return

    # --- Reload config & Check Global Reset ---
    cfg = get_config()
    check_and_perform_global_reset()
    youtube_api_key = cfg.get("youtube_api_key")
    reddit_client_id = cfg.get("reddit_client_id")
    reddit_client_secret = cfg.get("reddit_client_secret")
    reddit_user_agent = cfg.get("reddit_user_agent")
    custom_google_lens_config = cfg.get("custom_google_lens_config")

    # --- Permissions Check ---
    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))
    permissions = cfg.get("permissions", {})
    user_perms = permissions.get("users", {"allowed_ids": [], "blocked_ids": []})
    role_perms = permissions.get("roles", {"allowed_ids": [], "blocked_ids": []})
    channel_perms = permissions.get("channels", {"allowed_ids": [], "blocked_ids": []})
    allowed_user_ids, blocked_user_ids = user_perms.get("allowed_ids", []), user_perms.get("blocked_ids", [])
    allowed_role_ids, blocked_role_ids = role_perms.get("allowed_ids", []), role_perms.get("blocked_ids", [])
    allowed_channel_ids, blocked_channel_ids = channel_perms.get("allowed_ids", []), channel_perms.get("blocked_ids", [])

    allow_all_users = not allowed_user_ids and not allowed_role_ids
    is_good_user = allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)
    allow_all_channels = not allowed_channel_ids
    is_good_channel = allow_dms if is_dm else (allow_all_channels or any(id in allowed_channel_ids for id in channel_ids))
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or (not is_dm and is_bad_channel):
        logging.warning(f"Blocked message from user {new_msg.author.id} in channel {new_msg.channel.id} due to permissions.")
        return

    # --- LLM Provider/Model Selection ---
    provider_slash_model = cfg.get("model", "openai/gpt-4.1")
    try: provider, model_name = provider_slash_model.split("/", 1)
    except ValueError:
        logging.error(f"Invalid model format in config: '{provider_slash_model}'.")
        await new_msg.reply(f"⚠️ Invalid model format in config: `{provider_slash_model}`", mention_author = False)
        return

    provider_config = cfg.get("providers", {}).get(provider, {})
    all_api_keys = provider_config.get("api_keys", [])
    base_url = provider_config.get("base_url")
    is_gemini = provider == "google"
    keys_required = provider not in ["ollama", "lmstudio", "vllm", "oobabooga", "jan"]

    if keys_required and not all_api_keys:
         logging.error(f"No API keys configured for provider '{provider}' in config.yaml.")
         await new_msg.reply(f"⚠️ No API keys configured for provider `{provider}`.", mention_author = False)
         return

    # --- Configuration Values ---
    accept_images = any(x in model_name.lower() for x in VISION_MODEL_TAGS)
    max_text = cfg.get("max_text", 100000)
    max_images = cfg.get("max_images", 5) if accept_images else 0
    max_messages = cfg.get("max_messages", 25)
    use_plain_responses = cfg.get("use_plain_responses", False)
    split_limit = MAX_EMBED_DESCRIPTION_LENGTH if not use_plain_responses else 2000

    # --- Clean Content and Check for Google Lens ---
    cleaned_content = original_content_for_processing
    if not is_dm and discord_client.user.mentioned_in(new_msg):
        cleaned_content = cleaned_content.replace(discord_client.user.mention, '').strip()
    cleaned_content = AT_AI_PATTERN.sub(' ', cleaned_content)
    cleaned_content = re.sub(r'\s{2,}', ' ', cleaned_content).strip()

    use_google_lens = False
    image_attachments = [att for att in new_msg.attachments if att.content_type and att.content_type.startswith("image/")]
    user_warnings = set()

    if GOOGLE_LENS_PATTERN.match(cleaned_content) and image_attachments:
        use_google_lens = True
        cleaned_content = GOOGLE_LENS_PATTERN.sub('', cleaned_content).strip()
        logging.info(f"Google Lens keyword detected for message {new_msg.id}")
        custom_lens_ok = custom_google_lens_config and custom_google_lens_config.get("user_data_dir") and custom_google_lens_config.get("profile_directory_name")
        serpapi_keys_ok = bool(cfg.get("serpapi_api_keys"))
        if not custom_lens_ok and not serpapi_keys_ok:
             logging.warning("Google Lens requested but neither custom implementation nor SerpAPI keys are configured.")
             user_warnings.add("⚠️ Google Lens requested but requires configuration (custom or SerpAPI).")

    # --- URL Extraction and Task Creation ---
    all_urls_with_indices = extract_urls_with_indices(cleaned_content)
    fetch_tasks = []
    processed_urls = set()
    fetched_url_results_for_db: List[Dict] = [] # Store results for DB

    if use_google_lens:
        custom_lens_ok = custom_google_lens_config and custom_google_lens_config.get("user_data_dir") and custom_google_lens_config.get("profile_directory_name")
        serpapi_keys_ok = bool(cfg.get("serpapi_api_keys"))
        if custom_lens_ok or serpapi_keys_ok:
            for i, attachment in enumerate(image_attachments):
                fetch_tasks.append(process_google_lens_image(attachment.url, i))
        # Warning already added if neither configured

    for url, index in all_urls_with_indices:
        if url in processed_urls: continue
        processed_urls.add(url)
        if is_youtube_url(url): fetch_tasks.append(fetch_youtube_data(url, index, youtube_api_key))
        elif is_reddit_url(url):
            sub_id = extract_reddit_submission_id(url)
            if sub_id: fetch_tasks.append(fetch_reddit_data(url, sub_id, index, reddit_client_id, reddit_client_secret, reddit_user_agent))
            else: user_warnings.add(f"⚠️ Could not extract submission ID from Reddit URL: {url[:50]}...")
        else: fetch_tasks.append(fetch_general_url_content(url, index))

    # --- Fetch External Content Concurrently ---
    if fetch_tasks:
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Unhandled exception during URL fetch: {result}")
                user_warnings.add("⚠️ Unhandled error fetching URL content")
            elif isinstance(result, UrlFetchResult):
                fetched_url_results_for_db.append(result.to_dict()) # Store dict representation
                if result.error:
                    short_url = result.url[:40] + "..." if len(result.url) > 40 else result.url
                    if result.type != "google_lens_fallback_failed":
                        user_warnings.add(f"⚠️ Error fetching {result.type} URL ({short_url}): {result.error}")
            else:
                 logging.error(f"Unexpected result type from URL fetch: {type(result)}")

    # --- Determine Parent Message ID for DB ---
    parent_message_id_for_db = None
    try:
        is_explicit_trigger = mentions_bot or contains_at_ai
        if new_msg.reference and new_msg.reference.message_id:
            parent_message_id_for_db = new_msg.reference.message_id
        elif new_msg.channel.type == discord.ChannelType.public_thread and not new_msg.reference:
            # Try fetching starter message ID if available
            try:
                starter_message = new_msg.channel.starter_message or await new_msg.channel.parent.fetch_message(new_msg.channel.id)
                if starter_message:
                    parent_message_id_for_db = starter_message.id
                else: # Fallback to thread ID itself if starter message fetch fails
                    parent_message_id_for_db = new_msg.channel.id
            except (discord.NotFound, discord.HTTPException, AttributeError):
                 logging.warning(f"Could not fetch starter message for thread {new_msg.channel.id}, using thread ID as parent.")
                 parent_message_id_for_db = new_msg.channel.id # Use thread ID as parent
        elif not is_explicit_trigger:
            prev_msg_in_channel = None
            async for m in new_msg.channel.history(before=new_msg, limit=1):
                prev_msg_in_channel = m
                break
            if prev_msg_in_channel and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply):
                if (is_dm and prev_msg_in_channel.author == discord_client.user) or \
                   (not is_dm and prev_msg_in_channel.author == new_msg.author):
                    parent_message_id_for_db = prev_msg_in_channel.id
    except Exception as e:
        logging.exception(f"Error determining parent message ID for DB for {new_msg.id}")
        user_warnings.add("⚠️ Couldn't determine parent message")


    # --- Save User Message to DB ---
    user_image_urls = [att.url for att in image_attachments] # Store URLs, not base64
    history_db.save_message(
        message_id=new_msg.id,
        channel_id=new_msg.channel.id,
        author_id=new_msg.author.id,
        role='user',
        timestamp=new_msg.created_at.timestamp(),
        original_content=cleaned_content, # Store the cleaned user text
        image_urls=user_image_urls,
        fetched_url_data=fetched_url_results_for_db, # Store the fetched data
        llm_response_content=None, # No LLM response for user message
        parent_message_id=parent_message_id_for_db
    )

    # --- Build Message History from DB ---
    history_from_db = history_db.load_conversation_history(new_msg.id, max_messages)
    history_for_llm = []
    processed_db_ids = set() # Track IDs processed from DB to avoid duplicates if fallback occurs

    for db_msg_data in history_from_db:
        message_id = db_msg_data['message_id']
        processed_db_ids.add(message_id)

        role = db_msg_data['role']
        author_id = db_msg_data['author_id']
        llm_response = db_msg_data['llm_response_content']
        original_content = db_msg_data['original_content']
        image_urls = db_msg_data['image_urls'] or []
        fetched_data = db_msg_data['fetched_url_data'] or []

        # Reconstruct the content that was sent to/received from the LLM
        content_parts = []
        text_content = ""

        if role == 'user':
            # Format fetched data and prepend to original content
            formatted_fetched_context = format_fetched_data_for_llm(fetched_data)
            text_content = original_content or ""
            if formatted_fetched_context:
                text_content = f"{formatted_fetched_context}\n\nUser's query:\n{text_content}"
        elif role == 'assistant':
            text_content = llm_response or "" # Use the stored LLM response

        # Add text part
        if text_content:
            text_content = text_content[:max_text] # Apply max_text limit
            if is_gemini:
                content_parts.append(google_types.Part.from_text(text=text_content))
            else:
                content_parts.append({"type": "text", "text": text_content})

        # Add image parts (re-fetch and encode if needed, or handle differently)
        if accept_images and image_urls:
            image_count = 0
            for img_url in image_urls:
                if image_count >= max_images:
                    user_warnings.add(f"⚠️ Max {max_images} images/msg")
                    break
                if is_gemini:
                    # Fetch image bytes for Gemini
                    try:
                        async with httpx_client.stream("GET", img_url, timeout=10.0) as img_resp:
                            if img_resp.status_code == 200:
                                img_bytes = await img_resp.aread()
                                mime_type = img_resp.headers.get("content-type", "image/jpeg") # Guess mime type
                                content_parts.append(google_types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
                                image_count += 1
                            else:
                                logging.warning(f"Failed to fetch image {img_url} for history: Status {img_resp.status_code}")
                                user_warnings.add(f"⚠️ Couldn't fetch image: {img_url[:40]}...")
                    except Exception as img_err:
                        logging.warning(f"Error fetching image {img_url} for history: {img_err}")
                        user_warnings.add(f"⚠️ Error fetching image: {img_url[:40]}...")
                else: # OpenAI compatible (pass URL directly)
                    content_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                    image_count += 1

        # Construct the message dictionary for the LLM API
        if content_parts:
            message_dict = {"role": "model" if role == 'assistant' and is_gemini else role} # Use 'model' for Gemini assistant
            if is_gemini:
                message_dict["parts"] = content_parts
            else:
                if message_dict["role"] == "model": message_dict["role"] = "assistant" # OpenAI uses 'assistant'
                message_dict["content"] = content_parts
                if provider in PROVIDERS_SUPPORTING_USERNAMES and role == 'user':
                    message_dict["name"] = str(author_id)
            history_for_llm.append(message_dict)

    # --- Fallback to API Fetching if DB History is Incomplete ---
    # (This section is complex and requires careful merging of DB and API data.
    # For now, we rely solely on the DB history loaded.)
    if len(history_for_llm) < max_messages and history_from_db:
        last_db_msg = history_from_db[-1]
        parent_id_from_db = last_db_msg.get('parent_message_id')
        if parent_id_from_db:
            logging.info(f"DB history shorter than max_messages ({len(history_for_llm)}/{max_messages}). Fetching older messages via API if needed is not fully implemented here.")
            user_warnings.add(f"⚠️ History might be incomplete (loaded {len(history_for_llm)} from DB).")


    # --- Final History Preparation ---
    history_for_llm = history_for_llm[::-1] # Reverse to chronological order for LLM

    # Add warning if max messages reached
    if len(history_from_db) >= max_messages:
         user_warnings.add(f"⚠️ Only using last {max_messages} messages from history.")

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, history length: {len(history_for_llm)}, google_lens: {use_google_lens}):\n{new_msg.content}")

    # --- System Prompt ---
    system_prompt_text = None
    if system_prompt := cfg.get("system_prompt"):
        system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
        if not is_gemini and provider in PROVIDERS_SUPPORTING_USERNAMES:
            system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")
        system_prompt_text = "\n".join([system_prompt] + system_prompt_extras)

    # --- Generate and Send Response with Retry ---
    response_msgs = []
    final_text = ""
    llm_call_successful = False
    llm_errors = []
    final_view = None
    grounding_metadata = None
    edit_task = None

    embed = discord.Embed()
    embed.set_footer(text=f"Model: {provider_slash_model}")
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    try:
        available_llm_keys = await get_available_keys(provider, all_api_keys)
        random.shuffle(available_llm_keys)
        llm_db_manager = get_db_manager(provider)

        if keys_required and not available_llm_keys:
            logging.error(f"No available (non-rate-limited) API keys for provider '{provider}'.")
            await new_msg.reply(f"⚠️ No available API keys for provider `{provider}` right now.", mention_author = False)
            return

        keys_to_loop = available_llm_keys if keys_required else ["dummy_key"]

        for key_index, current_api_key in enumerate(keys_to_loop):
            key_display = f"...{current_api_key[-4:]}" if current_api_key != "dummy_key" else "N/A (keyless)"
            logging.info(f"Attempting LLM request with provider '{provider}' using key {key_display} ({key_index+1}/{len(keys_to_loop)})")

            response_contents = []
            finish_reason = None
            grounding_metadata = None
            llm_client = None
            api_config = None
            api_content_kwargs = {}
            payload_args_for_logging = {}

            try:
                if is_gemini:
                    if current_api_key == "dummy_key": raise ValueError("Gemini requires an API key.")
                    llm_client = google_genai.Client(api_key=current_api_key)
                    gemini_contents = []
                    for msg in history_for_llm:
                        role = msg["role"]
                        parts = msg.get("parts", [])
                        if not isinstance(parts, list):
                             parts = [google_types.Part.from_text(text=str(parts))] if parts else []
                        gemini_contents.append(google_types.Content(role=role, parts=parts))
                    api_content_kwargs["contents"] = gemini_contents
                    gemini_extra_params = cfg.get("extra_api_parameters", {}).copy()
                    if "max_tokens" in gemini_extra_params: gemini_extra_params["max_output_tokens"] = gemini_extra_params.pop("max_tokens")
                    gemini_safety_settings_list = [google_types.SafetySetting(category=c, threshold=t) for c, t in GEMINI_SAFETY_SETTINGS_DICT.items()]
                    api_config = google_types.GenerateContentConfig(
                        **gemini_extra_params,
                        safety_settings=gemini_safety_settings_list,
                        tools=[google_types.Tool(google_search=google_types.GoogleSearch())]
                    )
                    if system_prompt_text: api_config.system_instruction = google_types.Part.from_text(text=system_prompt_text)
                    payload_args_for_logging = {"contents": api_content_kwargs["contents"], "config": api_config}
                else:
                    api_key_to_use = current_api_key if current_api_key != "dummy_key" else None
                    llm_client = AsyncOpenAI(base_url=base_url, api_key=api_key_to_use)
                    openai_messages = history_for_llm[:]
                    if system_prompt_text: openai_messages.insert(0, dict(role="system", content=system_prompt_text))
                    api_content_kwargs["messages"] = openai_messages
                    api_config = cfg.get("extra_api_parameters", {}).copy()
                    api_config["stream"] = True
                    payload_args_for_logging = {"messages": api_content_kwargs["messages"], "stream": True, **api_config}

                log_llm_payload(provider, model_name, payload_args_for_logging)

                async with new_msg.channel.typing():
                    stream_response = None
                    if is_gemini:
                        if not llm_client: raise ValueError("Gemini client not initialized.")
                        stream_response = await llm_client.aio.models.generate_content_stream(
                            model=model_name, contents=api_content_kwargs["contents"], config=api_config
                        )
                    else:
                        if not llm_client: raise ValueError("OpenAI client not initialized.")
                        stream_response = await llm_client.chat.completions.create(
                            model=model_name, messages=api_content_kwargs["messages"], **api_config
                        )

                    async for chunk in stream_response:
                        new_content_chunk = ""
                        chunk_finish_reason = None
                        chunk_grounding_metadata = None
                        try:
                            if is_gemini:
                                if hasattr(chunk, 'text') and chunk.text: new_content_chunk = chunk.text
                                if hasattr(chunk, 'candidates') and chunk.candidates:
                                     candidate = chunk.candidates[0]
                                     if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                                          reason_map = {google_types.FinishReason.STOP: "stop", google_types.FinishReason.MAX_TOKENS: "length", google_types.FinishReason.SAFETY: "safety", google_types.FinishReason.RECITATION: "recitation"}
                                          chunk_finish_reason = reason_map.get(candidate.finish_reason, "stop" if candidate.finish_reason in (google_types.FinishReason.FINISH_REASON_UNSPECIFIED, google_types.FinishReason.STOP) else str(candidate.finish_reason))
                                     if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata: chunk_grounding_metadata = candidate.grounding_metadata
                            else:
                                if chunk.choices:
                                    delta = chunk.choices[0].delta
                                    chunk_finish_reason = chunk.choices[0].finish_reason
                                    if delta and delta.content: new_content_chunk = delta.content

                            if chunk_finish_reason: finish_reason = chunk_finish_reason
                            if chunk_grounding_metadata: grounding_metadata = chunk_grounding_metadata

                            if finish_reason and finish_reason.lower() == "safety":
                                logging.warning(f"Response Blocked (finish_reason=SAFETY) with key {key_display}")
                                llm_errors.append(f"Key {key_display}: Response Blocked (Safety)")
                                llm_call_successful = False
                                if not use_plain_responses and response_msgs:
                                    try:
                                        current_desc = response_msgs[-1].embeds[0].description if response_msgs[-1].embeds else ""
                                        embed.description = current_desc.replace(STREAMING_INDICATOR, "").strip() + "\n\n⚠️ Response blocked by safety filters."
                                        embed.description = embed.description[:MAX_EMBED_DESCRIPTION_LENGTH]
                                        embed.color = EMBED_COLOR_ERROR
                                        if edit_task and not edit_task.done(): await edit_task
                                        await response_msgs[-1].edit(embed=embed, view=None)
                                    except Exception as edit_err: logging.error(f"Failed to edit message to show safety block: {edit_err}")
                                break # Break inner stream loop

                            if new_content_chunk: response_contents.append(new_content_chunk)

                            if not use_plain_responses:
                                current_full_text = "".join(response_contents)
                                if not current_full_text and not finish_reason: continue
                                view_to_attach = None
                                is_final_chunk = finish_reason is not None
                                if is_final_chunk:
                                    has_sources = grounding_metadata and (getattr(grounding_metadata, 'web_search_queries', None) or getattr(grounding_metadata, 'grounding_chunks', None))
                                    has_text = bool(current_full_text)
                                    if has_sources or has_text:
                                        view_to_attach = ResponseActionView(grounding_metadata=grounding_metadata, full_response_text=current_full_text, model_name=provider_slash_model)
                                        if not view_to_attach or len(view_to_attach.children) == 0: view_to_attach = None

                                current_msg_index = (len(current_full_text) - 1) // split_limit if current_full_text else 0
                                start_next_msg = current_msg_index >= len(response_msgs)
                                ready_to_edit = (edit_task is None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS

                                if start_next_msg or ready_to_edit or is_final_chunk:
                                    if edit_task is not None: await edit_task
                                    if start_next_msg and response_msgs:
                                        prev_msg_index = current_msg_index - 1
                                        prev_msg_text = current_full_text[prev_msg_index * split_limit : current_msg_index * split_limit][:MAX_EMBED_DESCRIPTION_LENGTH + len(STREAMING_INDICATOR)]
                                        embed.description = prev_msg_text or "..."
                                        embed.color = EMBED_COLOR_COMPLETE
                                        try: await response_msgs[prev_msg_index].edit(embed=embed, view=None)
                                        except discord.HTTPException as e: logging.error(f"Failed to finalize previous message {prev_msg_index}: {e}")

                                    current_display_text = current_full_text[current_msg_index * split_limit : (current_msg_index + 1) * split_limit][:MAX_EMBED_DESCRIPTION_LENGTH]
                                    is_successful_finish = finish_reason and finish_reason.lower() in ("stop", "end_turn")
                                    embed.description = (current_display_text or "...") if is_final_chunk else ((current_display_text or "...") + STREAMING_INDICATOR)
                                    embed.color = EMBED_COLOR_COMPLETE if is_final_chunk and is_successful_finish else EMBED_COLOR_INCOMPLETE

                                    if start_next_msg:
                                        reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                                        if key_index > 0:
                                            logging.info(f"Clearing previous response messages due to retry (Key index: {key_index})")
                                            for old_msg in response_msgs:
                                                try: await old_msg.delete()
                                                except discord.HTTPException: pass
                                            response_msgs = []
                                        response_msg = await reply_to_msg.reply(embed=embed, view=view_to_attach, mention_author = False)
                                        response_msgs.append(response_msg)
                                        # Don't add to msg_nodes here, save to DB later
                                    elif response_msgs and current_msg_index < len(response_msgs):
                                        edit_task = asyncio.create_task(response_msgs[current_msg_index].edit(embed=embed, view=view_to_attach))
                                    elif not response_msgs and is_final_chunk:
                                         reply_to_msg = new_msg
                                         response_msg = await reply_to_msg.reply(embed=embed, view=view_to_attach, mention_author = False)
                                         response_msgs.append(response_msg)
                                         # Don't add to msg_nodes here, save to DB later
                                    last_task_time = dt.now().timestamp()

                            if finish_reason: break # Exit inner stream loop

                        except APIConnectionError as stream_err:
                            logging.warning(f"Connection error during streaming with key {key_display}: {stream_err}")
                            llm_errors.append(f"Key {key_display}: Stream Connection Error - {stream_err}")
                            break
                        except APIError as stream_err:
                            logging.warning(f"API error during streaming with key {key_display}: {stream_err}")
                            llm_errors.append(f"Key {key_display}: Stream API Error - {stream_err}")
                            if isinstance(stream_err, RateLimitError):
                                if current_api_key != "dummy_key": llm_db_manager.add_key(current_api_key)
                            break
                        except google_api_exceptions.GoogleAPIError as stream_err:
                            logging.warning(f"Google API error during streaming with key {key_display}: {stream_err}")
                            llm_errors.append(f"Key {key_display}: Stream Google API Error - {stream_err}")
                            if isinstance(stream_err, google_api_exceptions.ResourceExhausted):
                                if current_api_key != "dummy_key": llm_db_manager.add_key(current_api_key)
                            break
                        except Exception as stream_err:
                            logging.exception(f"Unexpected error during streaming with key {key_display}")
                            llm_errors.append(f"Key {key_display}: Unexpected Stream Error - {type(stream_err).__name__}")
                            break

                    if finish_reason and finish_reason.lower() == "safety": break # Break outer retry loop

                    if finish_reason and finish_reason.lower() != "safety":
                        llm_call_successful = True
                        logging.info(f"LLM request successful with key {key_display}")
                        break

            except (RateLimitError, google_api_exceptions.ResourceExhausted) as e:
                logging.warning(f"Rate limit hit for provider '{provider}' with key {key_display}. Error: {e}")
                if current_api_key != "dummy_key": llm_db_manager.add_key(current_api_key)
                llm_errors.append(f"Key {key_display}: Rate Limited")
                continue
            except (AuthenticationError, google_api_exceptions.PermissionDenied) as e:
                logging.error(f"Authentication failed for provider '{provider}' with key {key_display}. Error: {e}")
                llm_errors.append(f"Key {key_display}: Authentication Failed")
                llm_call_successful = False; break
            except (APIConnectionError, google_api_exceptions.ServiceUnavailable, google_api_exceptions.DeadlineExceeded) as e:
                logging.warning(f"Connection/Service error for provider '{provider}' with key {key_display}. Error: {e}")
                llm_errors.append(f"Key {key_display}: Connection/Service Error - {type(e).__name__}")
                continue
            except (BadRequestError, google_api_exceptions.InvalidArgument) as e:
                 logging.error(f"Bad request error for provider '{provider}' with key {key_display}. Error: {e}")
                 llm_errors.append(f"Key {key_display}: Bad Request - {e}")
                 llm_call_successful = False; break
            except APIError as e:
                logging.exception(f"OpenAI API Error for key {key_display}")
                llm_errors.append(f"Key {key_display}: API Error - {type(e).__name__}: {e}")
                continue
            except google_api_exceptions.GoogleAPICallError as e:
                logging.exception(f"Google API Call Error for key {key_display}")
                llm_errors.append(f"Key {key_display}: Google API Error - {type(e).__name__}: {e}")
                continue
            except Exception as e:
                logging.exception(f"Unexpected error during LLM call with key {key_display}")
                llm_errors.append(f"Key {key_display}: Unexpected Error - {type(e).__name__}: {e}")
                continue

        # --- Post-Retry Loop Processing ---
        if not llm_call_successful:
            logging.error(f"All LLM API keys failed for provider '{provider}'. Errors: {llm_errors}")
            error_message = f"⚠️ All API keys for provider `{provider}` failed."
            if llm_errors:
                last_error_short = str(llm_errors[-1])
                error_message += f"\nLast error: `{last_error_short[:100]}{'...' if len(last_error_short) > 100 else ''}`"
            if not use_plain_responses and response_msgs:
                 try:
                     current_desc = response_msgs[-1].embeds[0].description if response_msgs[-1].embeds else ""
                     embed.description = current_desc.replace(STREAMING_INDICATOR, "").strip() + f"\n\n{error_message}"
                     embed.description = embed.description[:MAX_EMBED_DESCRIPTION_LENGTH]
                     embed.color = EMBED_COLOR_ERROR
                     await response_msgs[-1].edit(embed=embed, view=None)
                 except Exception as edit_err:
                     logging.error(f"Failed to edit message to show final error: {edit_err}")
                     await new_msg.reply(error_message, mention_author = False)
            else: await new_msg.reply(error_message, mention_author = False)

        else: # If successful
            final_text = "".join(response_contents)
            final_view = None
            if not use_plain_responses:
                has_sources = grounding_metadata and (getattr(grounding_metadata, 'web_search_queries', None) or getattr(grounding_metadata, 'grounding_chunks', None))
                has_text = bool(final_text)
                if has_sources or has_text:
                    final_view = ResponseActionView(grounding_metadata=grounding_metadata, full_response_text=final_text, model_name=provider_slash_model)
                    if not final_view or len(final_view.children) == 0: final_view = None

            if use_plain_responses:
                 final_messages_content = [final_text[i:i+2000] for i in range(0, len(final_text), 2000)]
                 if not final_messages_content: final_messages_content.append("...")
                 temp_response_msgs = []
                 for i, content in enumerate(final_messages_content):
                     reply_to_msg = new_msg if not temp_response_msgs else temp_response_msgs[-1]
                     response_msg = await reply_to_msg.reply(content=content or "...", suppress_embeds=True, view=None, mention_author = False)
                     temp_response_msgs.append(response_msg)
                     # Save assistant message to DB
                     history_db.save_message(
                         message_id=response_msg.id, channel_id=response_msg.channel.id, author_id=discord_client.user.id,
                         role='assistant', timestamp=response_msg.created_at.timestamp(), original_content=None,
                         image_urls=None, fetched_url_data=None,
                         llm_response_content=final_text if i == len(final_messages_content) - 1 else content, # Store full text in last segment
                         parent_message_id=reply_to_msg.id
                     )
                 response_msgs = temp_response_msgs

            elif not use_plain_responses and response_msgs:
                 if edit_task is not None and not edit_task.done(): await edit_task
                 final_msg_index = len(response_msgs) - 1
                 final_msg_text = final_text[final_msg_index * split_limit : (final_msg_index + 1) * split_limit][:MAX_EMBED_DESCRIPTION_LENGTH + len(STREAMING_INDICATOR)]
                 embed.description = final_msg_text or "..."
                 embed.color = EMBED_COLOR_COMPLETE
                 try:
                     last_msg = response_msgs[final_msg_index]
                     needs_edit = False
                     current_description = last_msg.embeds[0].description if last_msg.embeds else ""
                     current_color = last_msg.embeds[0].color if last_msg.embeds else None
                     current_view_exists = bool(last_msg.components)
                     if (final_view and not current_view_exists) or (not final_view and current_view_exists): needs_edit = True
                     elif current_description != embed.description or current_color != embed.color: needs_edit = True
                     elif not last_msg.embeds: needs_edit = True
                     if needs_edit: await last_msg.edit(embed=embed, view=final_view)

                     # Save/Update assistant message in DB (store full response in last message)
                     for i, msg in enumerate(response_msgs):
                         segment_text = final_text[i * split_limit : (i + 1) * split_limit]
                         is_last = (i == len(response_msgs) - 1)
                         history_db.save_message(
                             message_id=msg.id, channel_id=msg.channel.id, author_id=discord_client.user.id,
                             role='assistant', timestamp=msg.created_at.timestamp(), original_content=None,
                             image_urls=None, fetched_url_data=None,
                             llm_response_content=final_text if is_last else segment_text, # Store full text in last segment
                             parent_message_id=new_msg.id if i == 0 else response_msgs[i-1].id
                         )

                 except discord.HTTPException as e: logging.error(f"Failed final edit on message {final_msg_index}: {e}")
                 except IndexError: logging.error(f"IndexError during final edit for index {final_msg_index}, response_msgs len: {len(response_msgs)}")

            elif not use_plain_responses and not response_msgs: # Handle empty successful response
                 embed.description = "..."
                 embed.color = EMBED_COLOR_COMPLETE
                 response_msg = await new_msg.reply(embed=embed, view=final_view, mention_author = False)
                 response_msgs.append(response_msg)
                 # Save assistant message to DB
                 history_db.save_message(
                     message_id=response_msg.id, channel_id=response_msg.channel.id, author_id=discord_client.user.id,
                     role='assistant', timestamp=response_msg.created_at.timestamp(), original_content=None,
                     image_urls=None, fetched_url_data=None, llm_response_content=final_text or "...",
                     parent_message_id=new_msg.id
                 )

    except Exception as outer_e:
        logging.exception("Unhandled error during message processing.")
        try: await new_msg.reply(f"⚠️ An unexpected error occurred: {type(outer_e).__name__}", mention_author = False)
        except discord.HTTPException: pass

    finally: # --- Runtime Cache Management ---
        if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
            nodes_to_delete = sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]
            logging.info(f"Runtime cache limit reached ({num_nodes}/{MAX_MESSAGE_NODES}). Removing {len(nodes_to_delete)} oldest nodes.")
            for msg_id in nodes_to_delete:
                node_to_delete = msg_nodes.get(msg_id)
                if node_to_delete and hasattr(node_to_delete, 'lock') and node_to_delete.lock.locked():
                    try: node_to_delete.lock.release()
                    except RuntimeError: pass
                msg_nodes.pop(msg_id, None)

# --- Main Function ---
async def main():
    bot_token = cfg.get("bot_token")
    if not bot_token:
        logging.critical("bot_token not found in config.yaml. Exiting.")
        return
    try:
        await discord_client.start(bot_token)
    except discord.LoginFailure:
        logging.critical("Failed to log in. Please check your bot_token in config.yaml.")
    except Exception as e:
        logging.critical(f"Error starting Discord client: {e}")
    finally:
        logging.info("Closing database connections...")
        history_db.close() # Close history DB
        for manager in db_managers.values(): manager.close() # Close rate limit DBs
        logging.info("Database connections closed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")