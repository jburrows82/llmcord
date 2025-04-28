import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime as dt, timedelta, timezone
import logging # Keep standard logging
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
import copy # Added for deep copying payloads for printing

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
    stream=sys.stdout # Explicitly set stream to stdout for terminal output
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
# Use google.genai.types (imported as google_types)
GEMINI_SAFETY_SETTINGS_DICT = {
    google_types.HarmCategory.HARM_CATEGORY_HARASSMENT: google_types.HarmBlockThreshold.BLOCK_NONE,
    google_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: google_types.HarmBlockThreshold.BLOCK_NONE,
    google_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: google_types.HarmBlockThreshold.BLOCK_NONE,
    google_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: google_types.HarmBlockThreshold.BLOCK_NONE,
    # google_types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: google_types.HarmBlockThreshold.BLOCK_NONE, # Uncomment if supported and desired
}
MAX_MESSAGE_NODES = 500
MAX_EMBED_FIELD_VALUE_LENGTH = 1024
MAX_EMBED_FIELDS = 25
MAX_EMBED_DESCRIPTION_LENGTH = 4096 - len(STREAMING_INDICATOR)
MAX_EMBED_TOTAL_SIZE = 5900 # Safety margin below 6000

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

# --- Custom Google Lens (Playwright) Configuration ---
# Selectors (These might change if Google updates their site)
LENS_ICON_SELECTOR = '[aria-label="Search by image"]'
PASTE_LINK_INPUT_SELECTOR = 'input[placeholder="Paste image link"]'
SEE_EXACT_MATCHES_SELECTOR = 'div.ndigne.ZwRhJd.RiJqbb:has-text("See exact matches")'
EXACT_MATCH_RESULT_SELECTOR = 'div.ZhosBf.T7iOye.MBI8Pd.dctkEf'
INITIAL_RESULTS_WAIT_SELECTOR = "div#rso"
ORIGINAL_RESULT_SPAN_SELECTOR = 'span.Yt787'
# Timeouts
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
            logging.error(f"Error connecting to database {self.db_path}: {e}")
            self.conn = None # Ensure conn is None if connection fails

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
            logging.error(f"Error creating table in {self.db_path}: {e}")

    def add_key(self, api_key: str):
        if not self.conn:
            logging.error(f"Cannot add key, no connection to {self.db_path}")
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
            logging.error(f"Error adding key {api_key[-4:]} to {self.db_path}: {e}")

    def get_limited_keys(self, cooldown_seconds: int) -> Set[str]:
        if not self.conn:
            logging.error(f"Cannot get limited keys, no connection to {self.db_path}")
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
            logging.error(f"Error getting limited keys from {self.db_path}: {e}")
        return limited_keys

    def reset_db(self):
        if not self.conn:
            logging.error(f"Cannot reset DB, no connection to {self.db_path}")
            return
        try:
            with self.conn:
                self.conn.execute("DELETE FROM rate_limited_keys")
            logging.info(f"Rate limit database {os.path.basename(self.db_path)} reset.")
        except sqlite3.Error as e:
            logging.error(f"Error resetting database {self.db_path}: {e}")

    def close(self):
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                logging.debug(f"Closed database connection: {self.db_path}")
            except sqlite3.Error as e:
                logging.error(f"Error closing database {self.db_path}: {e}")

# --- Global Rate Limit Management ---
db_managers: Dict[str, RateLimitDBManager] = {}

def get_db_manager(service_name: str) -> RateLimitDBManager:
    """Gets or creates a DB manager for a specific service."""
    global db_managers
    if service_name not in db_managers:
        db_path = os.path.join(DB_FOLDER, f"ratelimit_{service_name.lower().replace('-', '_')}.db")
        db_managers[service_name] = RateLimitDBManager(db_path)
    # Ensure connection is alive (e.g., if it failed initially)
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
        last_reset_time = 0.0 # Force reset if file is invalid

    if now - last_reset_time >= RATE_LIMIT_COOLDOWN_SECONDS:
        logging.info("Performing global 24-hour rate limit database reset.")
        # Ensure all potential DB managers are instantiated before resetting
        services_in_config = set()
        providers = cfg.get("providers", {})
        for provider_name, provider_cfg in providers.items():
            if provider_cfg and provider_cfg.get("api_keys"): # Check for list of keys
                services_in_config.add(provider_name)
        if cfg.get("serpapi_api_keys"): # Check SerpAPI separately
            services_in_config.add("serpapi")
        # Add other services with keys here if needed

        for service_name in services_in_config:
             manager = get_db_manager(service_name)
             manager.reset_db()

        # Also reset any managers that might exist but weren't in config scan (less likely)
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
        return all_keys # Return the full list after reset

    return available_keys

# --- Initialization ---
ytt_api = YouTubeTranscriptApi() # Initialize youtube-transcript-api client

def get_config(filename="config.yaml"):
    try:
        with open(filename, "r") as file:
            config_data = yaml.safe_load(file)
            # --- Config Validation & Key Normalization ---
            # Ensure providers have api_keys (plural) as a list
            providers = config_data.get("providers", {})
            for name, provider_cfg in providers.items():
                if provider_cfg: # Check if provider config exists
                    single_key = provider_cfg.get("api_key")
                    key_list = provider_cfg.get("api_keys")

                    if single_key and not key_list:
                        logging.warning(f"Config Warning: Provider '{name}' uses deprecated 'api_key'. Converting to 'api_keys' list. Please update config.yaml.")
                        provider_cfg["api_keys"] = [single_key]
                        del provider_cfg["api_key"]
                    elif single_key and key_list:
                         logging.warning(f"Config Warning: Provider '{name}' has both 'api_key' and 'api_keys'. Using 'api_keys'. Please remove 'api_key' from config.yaml.")
                         del provider_cfg["api_key"]
                    elif key_list is None: # Handle case where api_keys is explicitly null or missing
                         # Allow providers without keys (like Ollama)
                         provider_cfg["api_keys"] = []
                    elif not isinstance(key_list, list):
                         logging.error(f"Config Error: Provider '{name}' has 'api_keys' but it's not a list. Treating as empty.")
                         provider_cfg["api_keys"] = []

            # Handle SerpAPI key(s)
            single_serp_key = config_data.get("serpapi_api_key")
            serp_key_list = config_data.get("serpapi_api_keys")
            if single_serp_key and not serp_key_list:
                logging.warning("Config Warning: Found 'serpapi_api_key'. Converting to 'serpapi_api_keys' list. Please update config.yaml.")
                config_data["serpapi_api_keys"] = [single_serp_key]
                del config_data["serpapi_api_key"]
            elif single_serp_key and serp_key_list:
                 logging.warning("Config Warning: Found both 'serpapi_api_key' and 'serpapi_api_keys'. Using 'serpapi_api_keys'. Please remove 'serpapi_api_key' from config.yaml.")
                 del config_data["serpapi_api_key"]
            elif serp_key_list is None: # Handle case where serpapi_api_keys is explicitly null or missing
                 config_data["serpapi_api_keys"] = []
            elif not isinstance(serp_key_list, list):
                 logging.error("Config Error: Found 'serpapi_api_keys' but it's not a list. Treating as empty.")
                 config_data["serpapi_api_keys"] = []

            # Validate custom Google Lens config (optional)
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

youtube_api_key = cfg.get("youtube_api_key") # YouTube still uses single key for now
reddit_client_id = cfg.get("reddit_client_id")
reddit_client_secret = cfg.get("reddit_client_secret")
reddit_user_agent = cfg.get("reddit_user_agent")
# SerpAPI keys are now handled by the fallback logic
custom_google_lens_config = cfg.get("custom_google_lens_config") # Load custom lens config

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

msg_nodes = {}
last_task_time = 0

# --- Data Classes ---
@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)
    role: Literal["user", "assistant", "model"] = "assistant" # Added 'model' for Gemini
    user_id: Optional[int] = None
    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False
    parent_msg: Optional[discord.Message] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    full_response_text: Optional[str] = None # Added field for full response text
    external_content: Optional[str] = None # Added field for fetched external content

@dataclass
class UrlFetchResult:
    url: str
    content: Optional[Union[str, Dict[str, Any]]] # This is the required argument
    error: Optional[str] = None
    type: Literal["youtube", "reddit", "general", "google_lens_custom", "google_lens_serpapi", "google_lens_fallback_failed"] = "general"
    original_index: int = -1

# --- Helper Function for Embed Size Calculation ---
def calculate_embed_size(embed: discord.Embed) -> int:
    """Calculates the approximate total size of an embed's text content."""
    size = 0
    if embed.title:
        size += len(embed.title)
    if embed.description:
        size += len(embed.description)
    if embed.footer and embed.footer.text:
        size += len(embed.footer.text)
    if embed.author and embed.author.name:
        size += len(embed.author.name)
    # Add sizes of fields
    for field in embed.fields:
        size += len(field.name) + len(field.value)
    # Add a small buffer for overhead
    size += 100
    return size

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
            # Determine row based on whether sources button exists
            row = 1 if has_sources_button else 0
            self.add_item(self.GetTextFileButton(row=row))

    # Inner class for the Show Sources button - MODIFIED FOR SPLITTING
    class ShowSourcesButton(ui.Button):
        def __init__(self):
            super().__init__(label="Show Sources", style=discord.ButtonStyle.grey, row=0)

        async def callback(self, interaction: discord.Interaction):
            view: 'ResponseActionView' = self.view
            if not view.grounding_metadata:
                await interaction.response.send_message("No grounding metadata available.") # Never make anything ephemeral in this codebase
                return

            embeds_to_send = []
            current_embed = discord.Embed(title="Grounding Sources", color=EMBED_COLOR_COMPLETE)
            queries = getattr(view.grounding_metadata, 'web_search_queries', None)
            chunks = getattr(view.grounding_metadata, 'grounding_chunks', None)

            # --- Helper to add field and handle splitting ---
            def add_field_safely(embed, name, value, inline=False):
                nonlocal embeds_to_send, current_embed
                # Check if field itself is too long (should be handled by truncation before calling)
                if len(value) > MAX_EMBED_FIELD_VALUE_LENGTH:
                     value = value[:MAX_EMBED_FIELD_VALUE_LENGTH - 3] + "..."

                # Check if adding this field exceeds limits
                potential_size = calculate_embed_size(embed) + len(name) + len(value)
                if potential_size > MAX_EMBED_TOTAL_SIZE or len(embed.fields) >= MAX_EMBED_FIELDS:
                    # Current embed is full, finalize it and start a new one
                    if embed.fields or embed.title != "Grounding Sources (cont.)": # Avoid empty embeds
                        embeds_to_send.append(embed)
                    new_embed = discord.Embed(title="Grounding Sources (cont.)", color=EMBED_COLOR_COMPLETE)
                    new_embed.add_field(name=name, value=value, inline=inline)
                    return new_embed # Return the new embed
                else:
                    # Add to current embed
                    embed.add_field(name=name, value=value, inline=inline)
                    return embed # Return the same embed

            # 1. Add Search Queries Field
            if queries:
                query_text = "\n".join(f"- `{q}`" for q in queries)
                query_field_name = "Search Queries Used"
                query_field_value = query_text[:MAX_EMBED_FIELD_VALUE_LENGTH]
                if len(query_text) > MAX_EMBED_FIELD_VALUE_LENGTH:
                    query_field_value += "..."
                    logging.warning("Search query list truncated for embed field.")
                current_embed = add_field_safely(current_embed, query_field_name, query_field_value, inline=False)

            # 2. Process and Add Sources Consulted Field(s)
            if chunks:
                current_field_value = ""
                current_field_name = "Sources Consulted"
                sources_added_count = 0

                for i, chunk in enumerate(chunks):
                    web_chunk = getattr(chunk, 'web', None)
                    if web_chunk and hasattr(web_chunk, 'title') and hasattr(web_chunk, 'uri'):
                        title = web_chunk.title or "Source"
                        uri = web_chunk.uri
                        source_line = f"- [{title}]({uri})\n"
                        sources_added_count += 1

                        # Check if adding this line exceeds the current field's limit
                        if len(current_field_value) + len(source_line) > MAX_EMBED_FIELD_VALUE_LENGTH:
                            # Finalize the current field before starting a new one
                            if current_field_value:
                                current_embed = add_field_safely(current_embed, current_field_name, current_field_value, inline=False)
                                current_field_name = "Sources Consulted (cont.)" # Update name for next potential field
                                current_field_value = "" # Reset value

                            # Check if the single source line itself is too long
                            if len(source_line) > MAX_EMBED_FIELD_VALUE_LENGTH:
                                source_line = source_line[:MAX_EMBED_FIELD_VALUE_LENGTH - 4] + "...\n"
                                logging.warning(f"Single source line truncated: {web_chunk.uri}")

                            current_field_value = source_line # Start new field value

                        else:
                            # Add to the current field value
                            current_field_value += source_line

                # Add the last accumulated field value
                if current_field_value:
                    current_embed = add_field_safely(current_embed, current_field_name, current_field_value, inline=False)

                if sources_added_count == 0 and not queries: # If no sources found and no queries added
                     current_embed.description = "No web sources found in metadata."


            # 3. Add the final embed to the list if it has content
            if current_embed.fields or current_embed.description or (current_embed.title and current_embed.title != "Grounding Sources"):
                 embeds_to_send.append(current_embed)

            # 4. Send the embeds
            if not embeds_to_send:
                # Handle case where no queries or chunks were found, or metadata was empty/unparsable
                try:
                    if hasattr(view.grounding_metadata, 'model_dump'):
                        metadata_str = json.dumps(view.grounding_metadata.model_dump(mode='json'), indent=2)
                    else:
                        metadata_str = str(view.grounding_metadata)
                    await interaction.response.send_message(
                        f"Could not extract specific sources. Raw metadata:\n```json\n{metadata_str[:1900]}\n```",
                         # Never make anything ephemeral in this codebase
                    )
                except Exception:
                    await interaction.response.send_message("No grounding source information could be extracted.") # Never make anything ephemeral in this codebase
                return

            try:
                # Send the first embed using the initial response, Never make anything ephemeral in this codebase
                await interaction.response.send_message(embed=embeds_to_send[0])

                # Send subsequent embeds using followups, Never make anything ephemeral in this codebase
                for embed in embeds_to_send[1:]:
                    await interaction.followup.send(embed=embed)

            except discord.HTTPException as e:
                logging.error(f"HTTPException sending source embeds (might be too large even after split): {e}")
                # Fallback to sending as a file if embed sending fails
                await interaction.followup.send("Failed to send sources as embeds (likely still too large or another error). Sending as a file instead.")
                try:
                    source_lines = []
                    if queries:
                        source_lines.append("Search Queries Used:")
                        source_lines.extend([f"- {q}" for q in queries])
                        source_lines.append("")
                    if chunks:
                        source_lines.append("Sources Consulted:")
                        sources_added = 0
                        for chunk in chunks:
                            web_chunk = getattr(chunk, 'web', None)
                            if web_chunk and hasattr(web_chunk, 'title') and hasattr(web_chunk, 'uri'):
                                title = web_chunk.title or "Source"
                                uri = web_chunk.uri
                                source_lines.append(f"- Title: {title}")
                                source_lines.append(f"  Link: {uri}")
                                sources_added += 1
                        if sources_added == 0:
                            source_lines.append("  (No web sources found in metadata)")

                    file_content_str = "\n".join(source_lines)
                    if not file_content_str.strip():
                         await interaction.followup.send("No source information found to send as file.")
                         return

                    safe_model_name = re.sub(r'[<>:"/\\|?*]', '_', view.model_name or "llm")
                    filename = f"grounding_sources_{safe_model_name}.txt"
                    file_content = io.BytesIO(file_content_str.encode('utf-8'))
                    discord_file = discord.File(fp=file_content, filename=filename)
                    await interaction.followup.send(file=discord_file)

                except Exception as file_e:
                    logging.error(f"Error sending sources as fallback file: {file_e}")
                    await interaction.followup.send("Could not send sources as embeds or as a file.")

            except Exception as e:
                 logging.error(f"Unexpected error sending source embeds: {e}")
                 # Use followup for unexpected errors after the initial response might have been sent
                 try:
                     await interaction.followup.send("An unexpected error occurred while sending sources.")
                 except discord.HTTPException: # If followup fails too
                     logging.error("Failed to send followup error message for sources.")


    class GetTextFileButton(ui.Button):
        def __init__(self, row: int):
            super().__init__(label="Get response as a text file", style=discord.ButtonStyle.secondary, row=row)

        async def callback(self, interaction: discord.Interaction):
            # Access parent view's data
            view: 'ResponseActionView' = self.view
            if not view.full_response_text:
                await interaction.response.send_message("No response text available to send.") # Never make anything ephemeral in this codebase
                return

            try:
                # Clean model name for filename
                safe_model_name = re.sub(r'[<>:"/\\|?*]', '_', view.model_name or "llm") # Replace invalid chars
                filename = f"llm_response_{safe_model_name}.txt"

                # Create a file-like object from the string
                file_content = io.BytesIO(view.full_response_text.encode('utf-8'))
                discord_file = discord.File(fp=file_content, filename=filename)

                await interaction.response.send_message(file=discord_file) # Never make anything ephemeral in this codebase
            except Exception as e:
                logging.error(f"Error creating or sending text file: {e}")
                await interaction.response.send_message("Sorry, I couldn't create the text file.") # Never make anything ephemeral in this codebase

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

def _truncate_base64_in_payload(payload: Any, max_len: int = 40, prefix_len: int = 10) -> Any:
    """
    Recursively creates a copy of the payload and truncates long base64 strings
    found in specific known structures (OpenAI image_url, Gemini inline_data).
    """
    if isinstance(payload, dict):
        new_dict = {}
        for key, value in payload.items():
            # Check for OpenAI image_url structure
            if key == "image_url" and isinstance(value, dict) and "url" in value:
                url_value = value.get("url") # Use .get for safety
                if isinstance(url_value, str) and url_value.startswith("data:image") and ";base64," in url_value:
                    try:
                        prefix, data = url_value.split(";base64,", 1)
                        if len(data) > max_len:
                            truncated_data = data[:prefix_len] + "..." + data[-prefix_len:] + f" (truncated {len(data)} chars)"
                            # Create a copy of the inner dict to modify
                            new_image_url_dict = value.copy()
                            new_image_url_dict["url"] = prefix + ";base64," + truncated_data
                            new_dict[key] = new_image_url_dict
                        else:
                            # No truncation needed, shallow copy the inner dict is fine here
                            new_dict[key] = value.copy()
                    except ValueError: # Handle potential split errors
                         # If split fails, recurse into the value dict itself
                         new_dict[key] = _truncate_base64_in_payload(value, max_len, prefix_len)
                else:
                    # Not a base64 data URL or structure is different, recurse normally
                    new_dict[key] = _truncate_base64_in_payload(value, max_len, prefix_len)

            # Check for Gemini inline_data structure
            elif key == "inline_data" and isinstance(value, dict) and "data" in value:
                 data_value = value.get("data") # Use .get for safety
                 if isinstance(data_value, str) and len(data_value) > max_len: # Check if it's a string and long
                     # Create a copy of the inner dict to modify
                     new_inline_data_dict = value.copy()
                     # Truncate the data string itself
                     new_inline_data_dict["data"] = data_value[:prefix_len] + "..." + data_value[-prefix_len:] + f" (truncated {len(data_value)} chars)"
                     new_dict[key] = new_inline_data_dict
                 else:
                     # No truncation needed or not a string, shallow copy the inner dict is fine
                     new_dict[key] = value.copy()

            else:
                # Recurse for other keys/values
                new_dict[key] = _truncate_base64_in_payload(value, max_len, prefix_len)
        return new_dict
    elif isinstance(payload, list):
        # Recurse for items in a list
        return [_truncate_base64_in_payload(item, max_len, prefix_len) for item in payload]
    else:
        # Return non-dict/list types as is
        return payload

# --- Content Fetching Functions ---

async def get_transcript(video_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetches the transcript for a video ID using youtube-transcript-api."""
    try:
        transcript_list = await asyncio.to_thread(ytt_api.list_transcripts, video_id)
        transcript = None
        # Prioritize manual English, then generated English, then any manual, then any generated
        priorities = [
            (transcript_list.find_manually_created_transcript, ['en']),
            (transcript_list.find_generated_transcript, ['en']),
            (transcript_list.find_manually_created_transcript, [lang.language_code for lang in transcript_list]),
            (transcript_list.find_generated_transcript, [lang.language_code for lang in transcript_list]),
        ]
        for find_method, langs in priorities:
            # Ensure langs is not empty before calling find method
            if not langs:
                continue
            try:
                # Use asyncio.to_thread for the synchronous find methods
                transcript = await asyncio.to_thread(find_method, langs)
                if transcript: break
            except NoTranscriptFound:
                continue
            except Exception as e:
                # Catch potential errors within the find methods themselves
                logging.warning(f"Error during transcript find method {find_method.__name__} for {video_id}: {e}")
                continue


        if transcript:
            fetched_transcript = await asyncio.to_thread(transcript.fetch)
            # Use attribute access (.text) instead of dictionary access (['text'])
            # Use .to_dict() to get the raw list of dictionaries for easier processing later if needed
            raw_transcript_data = await asyncio.to_thread(fetched_transcript.to_raw_data)
            full_transcript = " ".join([entry['text'] for entry in raw_transcript_data]) # No truncation
            return full_transcript, None
        else:
            return None, "No suitable transcript found."
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except NoTranscriptFound: # Catch case where list_transcripts finds nothing
        return None, "No transcripts listed for this video."
    except Exception as e:
        logging.error(f"Error fetching transcript for {video_id}: {type(e).__name__}: {e}")
        return None, f"An error occurred fetching transcript: {type(e).__name__}"

async def get_youtube_video_details(video_id: str, api_key: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetches video title, description, channel name, and comments using YouTube Data API."""
    if not api_key:
        return None, "YouTube API key not configured."

    details = {}
    error_messages = []

    try:
        youtube = await asyncio.to_thread(build_google_api_client, 'youtube', 'v3', developerKey=api_key)

        # Get video details (snippet)
        try:
            video_request = youtube.videos().list(part="snippet", id=video_id)
            video_response = await asyncio.to_thread(video_request.execute)
            if video_response.get("items"):
                snippet = video_response["items"][0]["snippet"]
                details["title"] = snippet.get("title", "N/A")
                details["description"] = snippet.get("description", "N/A") # No truncation
                details["channel_name"] = snippet.get("channelTitle", "N/A")
            else:
                 error_messages.append("Video details not found.")
        except HttpError as e:
            error_reason = getattr(e, 'reason', str(e))
            status_code = getattr(e.resp, 'status', 'Unknown')
            logging.warning(f"YouTube Data API error getting video details for {video_id} (Status: {status_code}): {error_reason}")
            error_messages.append(f"API error getting video details ({status_code}).")
        except Exception as e:
            logging.exception(f"Unexpected error getting YouTube video details for {video_id}")
            error_messages.append(f"Unexpected error getting video details: {type(e).__name__}")


        # Get top comments (commentThreads) - Proceed even if details failed
        try:
            comment_request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                order="relevance",
                maxResults=10, # Keep comment count limited for API quota reasons
                textFormat="plainText"
            )
            comment_response = await asyncio.to_thread(comment_request.execute)
            details["comments"] = [
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for item in comment_response.get("items", [])
            ]
        except HttpError as e:
             error_reason = getattr(e, 'reason', str(e))
             status_code = getattr(e.resp, 'status', 'Unknown')
             # Don't log 403 comment errors as harshly if comments are just disabled
             if status_code == 403 and 'commentsDisabled' in str(e):
                 logging.info(f"Comments disabled for YouTube video {video_id}")
                 error_messages.append("Comments disabled.")
             else:
                 logging.warning(f"YouTube Data API error getting comments for {video_id} (Status: {status_code}): {error_reason}")
                 error_messages.append(f"API error getting comments ({status_code}).")
        except Exception as e:
            logging.exception(f"Unexpected error getting YouTube comments for {video_id}")
            error_messages.append(f"Unexpected error getting comments: {type(e).__name__}")

        # Return details if any were fetched, otherwise None
        final_details = details if details else None
        final_error = " ".join(error_messages) if error_messages else None
        return final_details, final_error

    except HttpError as e: # Catch errors during build_google_api_client itself
        error_reason = getattr(e, 'reason', str(e))
        status_code = getattr(e.resp, 'status', 'Unknown')
        logging.error(f"YouTube Data API Build/Auth error (Status: {status_code}): {error_reason}")
        if status_code == 403:
             if "quotaExceeded" in str(e):
                 return None, "YouTube API quota exceeded."
             elif "accessNotConfigured" in str(e):
                 return None, "YouTube API access not configured or key invalid."
             else:
                 return None, f"YouTube API permission error: {error_reason}"
        else:
            return None, f"YouTube Data API HTTP error: {error_reason}"
    except Exception as e:
        logging.exception(f"Unexpected error initializing YouTube client or fetching details for {video_id}")
        return None, f"An unexpected error occurred: {e}"


async def fetch_youtube_data(url: str, index: int, api_key: Optional[str]) -> UrlFetchResult:
    """Fetches transcript and details for a single YouTube URL."""
    video_id = extract_video_id(url)
    if not video_id:
        return UrlFetchResult(url=url, content=None, error="Could not extract video ID.", type="youtube", original_index=index)

    # Fetch transcript and details concurrently
    transcript_task = asyncio.create_task(get_transcript(video_id))
    details_task = asyncio.create_task(get_youtube_video_details(video_id, api_key))

    transcript, transcript_error = await transcript_task
    details, details_error = await details_task

    combined_content = {}
    errors = [err for err in [transcript_error, details_error] if err]

    if details:
        combined_content.update(details)
    if transcript:
        combined_content["transcript"] = transcript

    if not combined_content and not errors:
        errors.append("No content fetched.") # Ensure error if nothing was retrieved

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
        return UrlFetchResult(url=url, content=None, error="Reddit API credentials not configured.", type="reddit", original_index=index)

    reddit = None # Initialize outside try block
    try:
        # Initialize asyncpraw.Reddit instance within the task
        reddit = asyncpraw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            read_only=True,
        )

        submission = await reddit.submission(id=submission_id)
        await submission.load() # Load submission data and comments

        content_data = {"title": submission.title}
        if submission.selftext:
            content_data["selftext"] = submission.selftext # No truncation

        # Fetch all top-level comments (no limit)
        top_comments_text = []
        # comment_limit = 10 # Removed limit
        await submission.comments.replace_more(limit=0) # Load only top-level comments, replace_more is needed

        # comment_count = 0 # No longer needed
        for top_level_comment in submission.comments.list():
            # if comment_count >= comment_limit: # Removed limit check
            #     break
            # Check if comment exists and is not deleted/removed before accessing body
            if hasattr(top_level_comment, 'body') and top_level_comment.body and top_level_comment.body not in ('[deleted]', '[removed]'):
                comment_body_cleaned = top_level_comment.body.replace('\n', ' ').replace('\r', '')
                top_comments_text.append(comment_body_cleaned) # No truncation
                # comment_count += 1 # No longer needed

        if top_comments_text:
            content_data["comments"] = top_comments_text

        return UrlFetchResult(url=url, content=content_data, type="reddit", original_index=index)

    except (NotFound, Redirect):
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
        # Ensure the Reddit client is closed if it was initialized
        if reddit:
            await reddit.close()


async def fetch_general_url_content(url: str, index: int) -> UrlFetchResult:
    """Fetches and extracts text content from a general URL using BeautifulSoup."""
    try:
        # Add a User-Agent header to mimic a browser
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        async with httpx_client.stream("GET", url, headers=headers, timeout=15.0) as response: # Use stream for large pages
            # Check status code early
            if response.status_code != 200:
                 # Check for redirect loop explicitly
                 if response.status_code >= 300 and response.status_code < 400 and len(response.history) > 5:
                     return UrlFetchResult(url=url, content=None, error=f"Too many redirects ({response.status_code}).", type="general", original_index=index)
                 return UrlFetchResult(url=url, content=None, error=f"HTTP status {response.status_code}.", type="general", original_index=index)

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                return UrlFetchResult(url=url, content=None, error=f"Unsupported content type: {content_type}", type="general", original_index=index)

            # Read content incrementally
            html_content = ""
            try:
                async for chunk in response.aiter_bytes():
                    html_content += chunk.decode(response.encoding or 'utf-8', errors='replace')
                    if len(html_content) > 5 * 1024 * 1024: # Limit HTML size read to 5MB
                        logging.warning(f"HTML content truncated for URL {url} due to size limit.")
                        html_content = html_content[:5*1024*1024] + "..."
                        break
            except httpx.ReadTimeout:
                 return UrlFetchResult(url=url, content=None, error="Timeout while reading content.", type="general", original_index=index)
            except Exception as e:
                 logging.warning(f"Error decoding content for {url}: {e}")
                 return UrlFetchResult(url=url, content=None, error=f"Content decoding error: {type(e).__name__}", type="general", original_index=index)


        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text content, trying main content areas first
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True) # Fallback to whole document

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return UrlFetchResult(url=url, content=None, error="No text content found.", type="general", original_index=index)

        # No longer limit content length here
        content = text

        return UrlFetchResult(url=url, content=content, type="general", original_index=index)

    except httpx.RequestError as e:
        logging.warning(f"HTTPX RequestError fetching {url}: {type(e).__name__}")
        return UrlFetchResult(url=url, content=None, error=f"Request failed: {type(e).__name__}", type="general", original_index=index)
    except Exception as e:
        logging.exception(f"Unexpected error fetching general URL {url}")
        return UrlFetchResult(url=url, content=None, error=f"Unexpected error: {type(e).__name__}", type="general", original_index=index)

# --- Custom Google Lens (Playwright) Implementation ---
def _custom_get_google_lens_results_sync(image_url: str, user_data_dir: str, profile_directory_name: str):
    """
    Synchronous wrapper for the Playwright Google Lens logic.
    Uses Playwright to get Google Lens results for a given image URL using a specific Chrome profile.
    Checks for "See exact matches", clicks if found, waits for the last result element, and extracts specific result divs.
    Otherwise, waits for the last original result element and extracts results using the original span selector.

    Args:
        image_url: The URL of the image to search.
        user_data_dir: Path to the main Chrome user data directory.
        profile_directory_name: The name of the specific profile folder within user_data_dir.

    Returns:
        A list of strings containing the extracted result texts, or None if an error occurs.
    """
    # Check if Playwright is available
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    except ImportError:
        logging.error("Playwright library not found. Cannot run custom Google Lens.")
        return None # Indicate failure due to missing dependency

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
    context = None # Define context outside the try block for finally clause

    logging.info(f"Custom Google Lens: Launching Chrome using profile: '{profile_directory_name}'")
    logging.info(f"Custom Google Lens: User Data Directory: {user_data_dir}")
    logging.info(f"Custom Google Lens: Searching for image URL: {image_url}")
    logging.info("Custom Google Lens: INFO: Ensure Google Chrome using this specific profile is completely closed (check Task Manager).")

    # Arguments to specify the profile directory
    launch_args = [
        '--no-first-run',
        '--no-default-browser-check',
        f"--profile-directory={profile_directory_name}" # Tells Chrome which profile folder to use
    ]

    with sync_playwright() as p:
        try:
            # Launch browser using the persistent context with the specified user data directory AND profile arg
            context = p.chromium.launch_persistent_context(
                user_data_dir, # Still need the parent directory path here
                headless=False, # Set to True for server environments if needed, but might affect login state
                channel="chrome",
                args=launch_args, # Pass the arguments including the profile directory
                slow_mo=50 # Slow down interactions slightly
            )
            page = context.new_page()
            page.set_default_timeout(CUSTOM_LENS_DEFAULT_TIMEOUT)

            logging.info("Custom Google Lens: Navigating to google.com...")
            page.goto("https://www.google.com/")

            logging.info("Custom Google Lens: Waiting for and clicking the Lens (Search by image) icon...")
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
                    except PlaywrightTimeoutError:
                        logging.warning(f"Custom Google Lens: Clicked 'See exact matches' but timed out waiting for the last element of '{EXACT_MATCH_RESULT_SELECTOR}'.")
                    except Exception as click_err:
                         logging.error(f"Custom Google Lens: Error clicking 'See exact matches' or waiting after click: {click_err}")
                else:
                    logging.info("Custom Google Lens: 'See exact matches' not found or not visible within timeout.")
                    logging.info(f"Custom Google Lens: Looking for general results using selector: '{ORIGINAL_RESULT_SPAN_SELECTOR}' (waiting for last element)...")
                    try:
                        page.locator(ORIGINAL_RESULT_SPAN_SELECTOR).last.wait_for(state="visible", timeout=CUSTOM_LENS_DEFAULT_TIMEOUT)
                        logging.info(f"Custom Google Lens: General results likely loaded. Using selector: '{ORIGINAL_RESULT_SPAN_SELECTOR}'")
                        final_result_selector = ORIGINAL_RESULT_SPAN_SELECTOR
                    except PlaywrightTimeoutError:
                         logging.warning(f"Custom Google Lens: Fallback check for the last original result ('{ORIGINAL_RESULT_SPAN_SELECTOR}') also timed out.")

            except PlaywrightTimeoutError:
                 logging.warning(f"Custom Google Lens: Timeout checking visibility for '{SEE_EXACT_MATCHES_SELECTOR}'. Assuming it's not present.")
                 logging.info(f"Custom Google Lens: Looking for general results using selector: '{ORIGINAL_RESULT_SPAN_SELECTOR}' (waiting for last element)...")
                 try:
                     page.locator(ORIGINAL_RESULT_SPAN_SELECTOR).last.wait_for(state="visible", timeout=CUSTOM_LENS_DEFAULT_TIMEOUT)
                     logging.info(f"Custom Google Lens: General results likely loaded. Using selector: '{ORIGINAL_RESULT_SPAN_SELECTOR}'")
                     final_result_selector = ORIGINAL_RESULT_SPAN_SELECTOR
                 except PlaywrightTimeoutError:
                      logging.warning(f"Custom Google Lens: Fallback check for the last original result ('{ORIGINAL_RESULT_SPAN_SELECTOR}') also timed out.")

            if final_result_selector:
                logging.info(f"Custom Google Lens: Extracting text using final selector: '{final_result_selector}'...")
                page.wait_for_timeout(500)
                result_elements = page.locator(final_result_selector).all()

                if not result_elements:
                    logging.info("Custom Google Lens: No result elements found matching the final selector.")

                for i, element in enumerate(result_elements):
                    try:
                        text = element.text_content()
                        if text:
                            cleaned_text = ' '.join(text.split())
                            results.append(cleaned_text)
                        else:
                            logging.warning(f"Custom Google Lens: Found element {i+1} but it has no text content.")
                    except Exception as e:
                        logging.error(f"Custom Google Lens: Error extracting text from element {i+1}: {e}")
            else:
                 logging.info("Custom Google Lens: No suitable result selector was determined. Skipping extraction.")

            logging.info("Custom Google Lens: Finished extracting results.")

        except PlaywrightTimeoutError as e:
            logging.error(f"Custom Google Lens: ERROR: A timeout occurred during the process: {e}")
            try:
                if 'page' in locals() and page and not page.is_closed():
                    screenshot_path = "error_screenshot_timeout.png"
                    page.screenshot(path=screenshot_path)
                    logging.info(f"Custom Google Lens: Screenshot saved as {screenshot_path}")
            except Exception as screen_err:
                logging.error(f"Custom Google Lens: Could not take screenshot on timeout error: {screen_err}")
            return None # Indicate failure
        except Exception as e:
            logging.error(f"Custom Google Lens: An unexpected error occurred: {e}")
            if "Target page, context or browser has been closed" in str(e):
                 logging.error("Custom Google Lens: ERROR DETAILS: This 'TargetClosedError' usually means Google Chrome was already running with the specified profile.")
                 logging.error(f"Profile Folder: '{profile_directory_name}' within '{user_data_dir}'")
                 logging.error("Please ensure ALL Chrome processes using this profile are closed (check Task Manager) before running the script again.")
            else:
                # Use logging.exception to include traceback
                logging.exception("Custom Google Lens: Unexpected error details:")
            try:
                 if 'page' in locals() and page and not page.is_closed():
                    screenshot_path = "error_screenshot_unexpected.png"
                    page.screenshot(path=screenshot_path)
                    logging.info(f"Custom Google Lens: Screenshot saved as {screenshot_path}")
            except Exception as screen_err:
                logging.error(f"Custom Google Lens: Could not take screenshot on unexpected error: {screen_err}")
            return None # Indicate failure
        finally:
            if context:
                logging.info("Custom Google Lens: Closing browser context...")
                try:
                    if context.pages:
                         context.close()
                except Exception as close_err:
                     logging.warning(f"Custom Google Lens: Note: Error during context close (might be expected if launch failed or browser closed): {close_err}")

    return results
# --- End Custom Google Lens Implementation ---


async def fetch_google_lens_serpapi_fallback(image_url: str, index: int) -> UrlFetchResult:
    """Fetches Google Lens results using SerpAPI with key rotation and retry (FALLBACK ONLY)."""
    service_name = "serpapi"
    all_keys = cfg.get("serpapi_api_keys", [])
    if not all_keys:
        return UrlFetchResult(url=image_url, content=None, error="SerpAPI keys not configured for fallback.", type="google_lens_serpapi", original_index=index)

    available_keys = await get_available_keys(service_name, all_keys)
    random.shuffle(available_keys)
    db_manager = get_db_manager(service_name)
    encountered_errors = []

    for key_index, api_key in enumerate(available_keys):
        key_display = f"...{api_key[-4:]}" # Added for logging
        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": api_key,
            "safe": "off", # As per original example
        }
        logging.info(f"Attempting SerpAPI Google Lens fallback request for image {index+1} with key {key_display} ({key_index+1}/{len(available_keys)})")

        try:
            search = GoogleSearch(params)
            results = await asyncio.to_thread(search.get_dict) # Blocking call in thread

            # Check for API-level errors (e.g., invalid key, quota)
            if "error" in results:
                error_msg = results["error"]
                logging.warning(f"SerpAPI fallback error for image {index+1} (key {key_display}): {error_msg}")
                encountered_errors.append(f"Key {key_display}: {error_msg}")
                # Check if it's a rate limit / quota error
                if "rate limit" in error_msg.lower() or "quota" in error_msg.lower() or "plan limit" in error_msg.lower() or "ran out of searches" in error_msg.lower():
                    db_manager.add_key(api_key)
                    logging.info(f"SerpAPI key {key_display} rate limited. Trying next key.") # Added log
                    continue # Try next key
                elif "invalid api key" in error_msg.lower():
                    # Don't retry with other keys if this one is definitively invalid
                    logging.error(f"SerpAPI key {key_display} is invalid. Aborting SerpAPI attempts.") # Added log
                    return UrlFetchResult(url=image_url, content=None, error=f"SerpAPI Error: Invalid API Key ({key_display})", type="google_lens_serpapi", original_index=index)
                else:
                    # For other API errors, maybe retry? For now, continue to next key.
                    logging.warning(f"SerpAPI key {key_display} encountered API error: {error_msg}. Trying next key.") # Added log
                    continue

            # Check for search-specific errors (e.g., couldn't process image)
            if results.get("search_metadata", {}).get("status", "").lower() == "error":
                 error_msg = results.get("search_metadata", {}).get("error", "Unknown search error")
                 logging.warning(f"SerpAPI fallback search error for image {index+1} (key {key_display}): {error_msg}")
                 encountered_errors.append(f"Key {key_display}: {error_msg}")
                 # These errors are usually not key-related, so maybe don't retry?
                 # For now, let's return the error from the first key that hit this.
                 logging.error(f"SerpAPI search failed for image {index+1} with key {key_display}. Aborting SerpAPI attempts.") # Added log
                 return UrlFetchResult(url=image_url, content=None, error=f"SerpAPI Search Error: {error_msg}", type="google_lens_serpapi", original_index=index)


            # --- Success Case ---
            visual_matches = results.get("visual_matches", [])
            if not visual_matches:
                return UrlFetchResult(url=image_url, content="No visual matches found (SerpAPI fallback).", type="google_lens_serpapi", original_index=index)

            # Format the results concisely
            formatted_results = []
            for i, match in enumerate(visual_matches): # No display limit
                title = match.get("title", "N/A")
                link = match.get("link", "#")
                source = match.get("source", "")
                result_line = f"- [{title}]({link})"
                if source:
                    result_line += f" (Source: {source})"
                formatted_results.append(result_line)
            content_str = "\n".join(formatted_results) # No "and more" line

            logging.info(f"SerpAPI Google Lens fallback request successful for image {index+1} with key {key_display}")
            return UrlFetchResult(url=image_url, content=content_str, type="google_lens_serpapi", original_index=index)

        except SerpApiClientException as e:
            # Handle client-level exceptions (e.g., connection errors, timeouts)
            logging.warning(f"SerpAPI client exception during fallback for image {index+1} (key {key_display}): {e}")
            encountered_errors.append(f"Key {key_display}: Client Error - {e}")
            # Check if the exception indicates a rate limit (might need specific checks based on library behavior/status codes)
            if "429" in str(e) or "rate limit" in str(e).lower():
                 db_manager.add_key(api_key)
                 logging.info(f"SerpAPI key {key_display} hit client-side rate limit. Trying next key.") # Added log
            else:
                 logging.warning(f"SerpAPI key {key_display} encountered client error: {e}. Trying next key.") # Added log
            # Retry with the next key for client exceptions
            continue
        except Exception as e:
            logging.exception(f"Unexpected error during SerpAPI fallback for image {index+1} (key {key_display})")
            encountered_errors.append(f"Key {key_display}: Unexpected Error - {type(e).__name__}")
            logging.warning(f"SerpAPI key {key_display} encountered unexpected error: {e}. Trying next key.") # Added log
            # Retry with the next key for unexpected errors
            continue

    # If loop finishes, all keys failed
    logging.error(f"All SerpAPI keys failed during fallback for Google Lens request for image {index+1}.")
    final_error_msg = "All SerpAPI keys failed during fallback."
    if encountered_errors:
        final_error_msg += f" Last error: {encountered_errors[-1]}"
    return UrlFetchResult(url=image_url, content=None, error=final_error_msg, type="google_lens_serpapi", original_index=index)


async def process_google_lens_image(image_url: str, index: int) -> UrlFetchResult:
    """
    Processes a Google Lens request for an image URL.
    Tries the custom Playwright implementation first if configured.
    Falls back to SerpAPI if the custom implementation fails or is not configured.
    """
    custom_config = cfg.get("custom_google_lens_config")
    custom_results = None
    custom_error = None
    custom_impl_attempted = False

    # 1. Try Custom Implementation
    if custom_config and custom_config.get("user_data_dir") and custom_config.get("profile_directory_name"):
        user_data_dir = custom_config["user_data_dir"]
        profile_name = custom_config["profile_directory_name"]
        logging.info(f"Attempting Google Lens request for image {index+1} using custom implementation (Profile: {profile_name})")
        custom_impl_attempted = True
        try:
            # Run the synchronous Playwright code in a separate thread
            custom_results = await asyncio.to_thread(
                _custom_get_google_lens_results_sync, # Call the internal sync function
                image_url,
                user_data_dir,
                profile_name
            )
            if custom_results is not None: # Success (even if empty list)
                logging.info(f"Custom Google Lens implementation successful for image {index+1}.")
                # Format results (similar to SerpAPI formatting)
                if not custom_results:
                     content_str = "No visual matches found (custom implementation)."
                else:
                    formatted_results = []
                    # Assuming custom_results is a list of strings - No display limit
                    for i, result_text in enumerate(custom_results):
                        # Custom implementation doesn't provide links/sources easily, just text
                        result_line = f"- {result_text}"
                        formatted_results.append(result_line)
                    content_str = "\n".join(formatted_results)

                return UrlFetchResult(url=image_url, content=content_str, type="google_lens_custom", original_index=index)
            else:
                # Custom implementation returned None, indicating an error occurred within it
                custom_error = "Custom implementation failed (returned None)."
                logging.warning(f"Custom Google Lens implementation failed for image {index+1} (returned None). Falling back to SerpAPI.")

        except Exception as e:
            custom_error = f"Custom implementation raised an exception: {type(e).__name__}: {e}"
            logging.exception(f"Custom Google Lens implementation failed for image {index+1} with exception. Falling back to SerpAPI.")
            # Fall through to SerpAPI fallback
    else:
        custom_error = "Custom Google Lens implementation not configured."
        logging.info("Custom Google Lens implementation not configured. Falling back to SerpAPI.")
        # Fall through to SerpAPI fallback

    # 2. Fallback to SerpAPI
    logging.info(f"Falling back to SerpAPI for Google Lens request for image {index+1}.")
    # Call the refactored SerpAPI logic
    serpapi_result = await fetch_google_lens_serpapi_fallback(image_url, index)

    # If SerpAPI also failed, report the custom error if it existed, otherwise SerpAPI error
    if serpapi_result.error:
        # Determine the most relevant error to report
        if custom_impl_attempted and custom_error:
            final_error_msg = f"Custom Lens failed ({custom_error}). Fallback SerpAPI also failed: {serpapi_result.error}"
        elif custom_error: # Custom not attempted or failed without exception, but SerpAPI failed
            final_error_msg = f"SerpAPI fallback failed: {serpapi_result.error}"
        else: # Should not happen if custom_error is always set when falling through, but safety
             final_error_msg = f"SerpAPI fallback failed: {serpapi_result.error}"

        return UrlFetchResult(url=image_url, content=None, error=final_error_msg, type="google_lens_fallback_failed", original_index=index)
    else:
        # SerpAPI succeeded after custom failed or wasn't configured
        serpapi_result.type = "google_lens_serpapi" # Ensure type is correct
        return serpapi_result


# --- Discord Event Handler ---
@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time, cfg, youtube_api_key, reddit_client_id, reddit_client_secret, reddit_user_agent, custom_google_lens_config

    # --- Basic Checks and Trigger ---
    if new_msg.author.bot:
        return

    is_dm = new_msg.channel.type == discord.ChannelType.private
    allow_dms = cfg.get("allow_dms", True)

    # Determine if the bot should process this message
    should_process = False
    mentions_bot = False # Initialize
    contains_at_ai = False # Initialize
    original_content_for_processing = new_msg.content # Keep original for keyword check

    if is_dm:
        if allow_dms:
            should_process = True
            # Check if user explicitly mentioned bot or "at ai" in DM to start new chain
            mentions_bot = discord_client.user.mentioned_in(new_msg)
            contains_at_ai = AT_AI_PATTERN.search(original_content_for_processing) is not None
        else:
            return # Block DMs if not allowed
    else: # In a channel
        mentions_bot = discord_client.user.mentioned_in(new_msg)
        contains_at_ai = AT_AI_PATTERN.search(original_content_for_processing) is not None
        if mentions_bot or contains_at_ai:
            should_process = True

    if not should_process:
        return

    # --- Reload config & Check Global Reset ---
    cfg = get_config()
    check_and_perform_global_reset() # Check/perform reset before processing message
    youtube_api_key = cfg.get("youtube_api_key") # Still single key
    reddit_client_id = cfg.get("reddit_client_id")
    reddit_client_secret = cfg.get("reddit_client_secret")
    reddit_user_agent = cfg.get("reddit_user_agent")
    custom_google_lens_config = cfg.get("custom_google_lens_config") # Reload custom lens config
    # SerpAPI keys are now handled by fetch_google_lens_serpapi_fallback

    # --- Permissions Check ---
    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))
    permissions = cfg.get("permissions", {}) # Default to empty dict
    user_perms = permissions.get("users", {"allowed_ids": [], "blocked_ids": []})
    role_perms = permissions.get("roles", {"allowed_ids": [], "blocked_ids": []})
    channel_perms = permissions.get("channels", {"allowed_ids": [], "blocked_ids": []})

    allowed_user_ids, blocked_user_ids = user_perms.get("allowed_ids", []), user_perms.get("blocked_ids", [])
    allowed_role_ids, blocked_role_ids = role_perms.get("allowed_ids", []), role_perms.get("blocked_ids", [])
    allowed_channel_ids, blocked_channel_ids = channel_perms.get("allowed_ids", []), channel_perms.get("blocked_ids", [])

    # Determine if user is allowed (handles DM case implicitly via role_ids being empty)
    allow_all_users = not allowed_user_ids and not allowed_role_ids
    is_good_user = allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    # Determine if channel is allowed (handles DM case via allow_dms check earlier)
    allow_all_channels = not allowed_channel_ids
    is_good_channel = allow_dms if is_dm else (allow_all_channels or any(id in allowed_channel_ids for id in channel_ids))
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or (not is_dm and is_bad_channel): # Apply channel block only if not DM
        logging.warning(f"Blocked message from user {new_msg.author.id} in channel {new_msg.channel.id} due to permissions.")
        return

    # --- LLM Provider/Model Selection ---
    provider_slash_model = cfg.get("model", "openai/gpt-4.1") # Default model
    try:
        provider, model_name = provider_slash_model.split("/", 1)
    except ValueError:
        logging.error(f"Invalid model format in config: '{provider_slash_model}'. Should be 'provider/model_name'.")
        await new_msg.reply(f"⚠️ Invalid model format in config: `{provider_slash_model}`", mention_author = False)
        return

    provider_config = cfg.get("providers", {}).get(provider, {})
    all_api_keys = provider_config.get("api_keys", []) # Expecting a list now
    base_url = provider_config.get("base_url") # Needed for OpenAI-compatible

    is_gemini = provider == "google"

    # Check if keys are required for this provider
    keys_required = provider not in ["ollama", "lmstudio", "vllm", "oobabooga", "jan"] # Add other keyless providers here

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
    cleaned_content = original_content_for_processing # Start with the original content
    if not is_dm and discord_client.user.mentioned_in(new_msg):
        cleaned_content = cleaned_content.replace(discord_client.user.mention, '').strip()
    cleaned_content = AT_AI_PATTERN.sub(' ', cleaned_content) # Remove "at ai"
    cleaned_content = re.sub(r'\s{2,}', ' ', cleaned_content).strip() # Consolidate spaces

    use_google_lens = False
    image_attachments = [att for att in new_msg.attachments if att.content_type and att.content_type.startswith("image/")]
    user_warnings = set() # Initialize user_warnings here

    if GOOGLE_LENS_PATTERN.match(cleaned_content) and image_attachments:
        use_google_lens = True
        # Remove the keyword itself from the content going to the LLM
        cleaned_content = GOOGLE_LENS_PATTERN.sub('', cleaned_content).strip()
        logging.info(f"Google Lens keyword detected for message {new_msg.id}")
        # Check if either custom config or SerpAPI keys are available
        custom_lens_ok = custom_google_lens_config and custom_google_lens_config.get("user_data_dir") and custom_google_lens_config.get("profile_directory_name")
        serpapi_keys_ok = bool(cfg.get("serpapi_api_keys"))
        if not custom_lens_ok and not serpapi_keys_ok:
             logging.warning("Google Lens requested but neither custom implementation nor SerpAPI keys are configured.")
             user_warnings.add("⚠️ Google Lens requested but requires configuration (custom or SerpAPI).")


    # --- Check for Empty Query ---
    # Check if the textual content is empty AFTER removing triggers/keywords
    is_text_empty = not cleaned_content.strip()
    # Check if there are any meaningful attachments (images for vision models, text files)
    has_meaningful_attachments = any(
        att.content_type and (att.content_type.startswith("image/") or att.content_type.startswith("text/"))
        for att in new_msg.attachments
    )
    # Check if it's a reply
    is_reply = bool(new_msg.reference)

    # Trigger error ONLY IF text is empty AND there are no meaningful attachments AND it's not a reply
    if is_text_empty and not has_meaningful_attachments and not is_reply:
        logging.info(f"Empty query detected from user {new_msg.author.id} in channel {new_msg.channel.id}. Not a reply and no meaningful attachments.")
        await new_msg.reply("Your query is empty. Please reply to a message to reference it or don't send an empty query.", mention_author=False)
        return # Stop processing

    # --- URL Extraction and Task Creation ---
    all_urls_with_indices = extract_urls_with_indices(cleaned_content) # Use cleaned content for URL extraction now
    fetch_tasks = []
    processed_urls = set() # Avoid processing duplicates
    url_fetch_results = [] # Initialize list to store all fetch results

    # Create tasks for non-Google Lens URLs first
    for url, index in all_urls_with_indices:
        if url in processed_urls:
            continue
        processed_urls.add(url)

        if is_youtube_url(url):
            fetch_tasks.append(fetch_youtube_data(url, index, youtube_api_key))
        elif is_reddit_url(url):
            sub_id = extract_reddit_submission_id(url)
            if sub_id:
                fetch_tasks.append(fetch_reddit_data(url, sub_id, index, reddit_client_id, reddit_client_secret, reddit_user_agent))
            else:
                user_warnings.add(f"⚠️ Could not extract submission ID from Reddit URL: {url[:50]}...")
        else:
            # Fetch general URL content
            fetch_tasks.append(fetch_general_url_content(url, index))

    # --- Fetch Non-Google Lens Content Concurrently ---
    if fetch_tasks:
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Unhandled exception during non-Lens URL fetch: {result}")
                user_warnings.add("⚠️ Unhandled error fetching URL content")
            elif isinstance(result, UrlFetchResult):
                url_fetch_results.append(result)
                if result.error:
                    short_url = result.url[:40] + "..." if len(result.url) > 40 else result.url
                    user_warnings.add(f"⚠️ Error fetching {result.type} URL ({short_url}): {result.error}")
            else:
                 logging.error(f"Unexpected result type from non-Lens URL fetch: {type(result)}")

    # --- Process Google Lens Images Sequentially ---
    if use_google_lens:
        # Check again if configuration is missing, add warning if needed
        custom_lens_ok = custom_google_lens_config and custom_google_lens_config.get("user_data_dir") and custom_google_lens_config.get("profile_directory_name")
        serpapi_keys_ok = bool(cfg.get("serpapi_api_keys"))
        if not custom_lens_ok and not serpapi_keys_ok:
            # Warning already added above
            pass
        else:
            logging.info(f"Processing {len(image_attachments)} Google Lens images sequentially...")
            for i, attachment in enumerate(image_attachments):
                logging.info(f"Starting Google Lens processing for image {i+1}/{len(image_attachments)}...")
                try:
                    # Await each image processing call directly
                    lens_result = await process_google_lens_image(attachment.url, i)
                    url_fetch_results.append(lens_result) # Add result (success or error) to the list
                    if lens_result.error:
                        # Warning is added inside process_google_lens_image/fallback logic
                        logging.warning(f"Google Lens processing failed for image {i+1}: {lens_result.error}")
                    else:
                        logging.info(f"Finished Google Lens processing for image {i+1}/{len(image_attachments)}.")
                    # Optional: Add a small delay between sequential calls if needed
                    # await asyncio.sleep(1)
                except Exception as e:
                    logging.exception(f"Unexpected error during sequential Google Lens processing for image {i+1}")
                    user_warnings.add(f"⚠️ Unexpected error processing Lens image {i+1}")
                    # Create a dummy error result to indicate failure for this image
                    url_fetch_results.append(UrlFetchResult(
                        url=attachment.url,
                        content=None,
                        error=f"Unexpected processing error: {type(e).__name__}",
                        type="google_lens_fallback_failed", # Use a generic failure type
                        original_index=i # Use attachment index as original_index
                    ))

    # --- Format External Content ---
    google_lens_context_to_append = ""
    other_url_context_to_append = ""

    if url_fetch_results:
        # Sort results by their original position in the message/attachments
        url_fetch_results.sort(key=lambda r: r.original_index)

        google_lens_parts = []
        other_url_parts = []
        other_url_counter = 1 # Counter for non-lens URLs

        for result in url_fetch_results:
            if result.content: # Only include successful fetches
                if result.type == "google_lens_custom":
                    # Use original_index which corresponds to the attachment index
                    header = f"Custom Google Lens implementation results for image {result.original_index + 1}:\n"
                    google_lens_parts.append(header + str(result.content))
                elif result.type == "google_lens_serpapi":
                    # Use original_index which corresponds to the attachment index
                    header = f"SerpAPI Google Lens fallback results for image {result.original_index + 1}:\n"
                    google_lens_parts.append(header + str(result.content))
                elif result.type == "youtube":
                    content_str = f"\nurl {other_url_counter}: {result.url}\n"
                    content_str += f"url {other_url_counter} content:\n"
                    if isinstance(result.content, dict):
                        content_str += f"  title: {result.content.get('title', 'N/A')}\n"
                        content_str += f"  channel: {result.content.get('channel_name', 'N/A')}\n"
                        desc = result.content.get('description', 'N/A') # No truncation
                        content_str += f"  description: {desc}\n"
                        transcript = result.content.get('transcript')
                        if transcript:
                            content_str += f"  transcript: {transcript}\n" # No truncation
                        comments = result.content.get("comments")
                        if comments:
                            content_str += f"  top comments:\n" + "\n".join([f"    - {c}" for c in comments]) + "\n" # No truncation or limit
                    other_url_parts.append(content_str)
                    other_url_counter += 1
                elif result.type == "reddit":
                    content_str = f"\nurl {other_url_counter}: {result.url}\n"
                    content_str += f"url {other_url_counter} content:\n"
                    if isinstance(result.content, dict):
                        content_str += f"  title: {result.content.get('title', 'N/A')}\n"
                        selftext = result.content.get('selftext')
                        if selftext: # No truncation
                            content_str += f"  content: {selftext}\n"
                        comments = result.content.get("comments")
                        if comments: # No truncation or limit
                            content_str += f"  top comments:\n" + "\n".join([f"    - {c}" for c in comments]) + "\n" # No truncation or limit
                    other_url_parts.append(content_str)
                    other_url_counter += 1
                elif result.type == "general":
                    content_str = f"\nurl {other_url_counter}: {result.url}\n"
                    content_str += f"url {other_url_counter} content:\n"
                    if isinstance(result.content, str):
                        # Content is no longer limited during fetch
                        content_str += f"  {result.content}\n"
                    other_url_parts.append(content_str)
                    other_url_counter += 1
                # Ignore google_lens_fallback_failed type here

        if google_lens_parts:
            google_lens_context_to_append = "\n\n".join(google_lens_parts) # Join with double newline

        if other_url_parts:
            other_url_context_to_append = "".join(other_url_parts)

    # Combine context parts into a single string to be stored
    combined_context = ""
    if google_lens_context_to_append or other_url_context_to_append:
        combined_context = "Answer the user's query based on the following:\n\n"
        if google_lens_context_to_append:
            combined_context += google_lens_context_to_append + "\n\n" # Add separator if both exist
        if other_url_context_to_append:
            combined_context += other_url_context_to_append

    # --- Build Message History ---
    history = []
    curr_msg = new_msg
    while curr_msg is not None and len(history) < max_messages:
        # Ensure node exists or create it
        if curr_msg.id not in msg_nodes:
             msg_nodes[curr_msg.id] = MsgNode() # Create node if missing (e.g., cache cleared)
        curr_node = msg_nodes[curr_msg.id]

        async with curr_node.lock:
            # Populate node if it's empty (first time seeing this message)
            if curr_node.text is None:
                # Use the already cleaned content for the *new* message
                content_to_store = cleaned_content if curr_msg.id == new_msg.id else curr_msg.content

                # Further clean mentions/at ai from older messages if needed (redundant if cleaned above, but safe)
                is_dm_current = curr_msg.channel.type == discord.ChannelType.private
                if not is_dm_current and discord_client.user.mentioned_in(curr_msg):
                     content_to_store = content_to_store.replace(discord_client.user.mention, '').strip()
                if curr_msg.id != new_msg.id: # Only remove "at ai" from older messages if it wasn't the trigger
                    content_to_store = AT_AI_PATTERN.sub(' ', content_to_store)
                    content_to_store = re.sub(r'\s{2,}', ' ', content_to_store).strip()


                # Process attachments (only for the current message node being processed)
                current_attachments = curr_msg.attachments
                good_attachments = [att for att in current_attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text/", "image/"))]
                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments], return_exceptions=True)

                # Combine text content using the cleaned content
                text_parts = [content_to_store] if content_to_store else []
                text_parts.extend(filter(None, (embed.title for embed in curr_msg.embeds)))
                text_parts.extend(filter(None, (embed.description for embed in curr_msg.embeds)))
                # text_parts.extend(filter(None, (getattr(embed.footer, 'text', None) for embed in curr_msg.embeds))) # Removed to prevent footer text in history

                # Add text from attachments
                for att, resp in zip(good_attachments, attachment_responses):
                    if isinstance(resp, httpx.Response) and resp.status_code == 200 and att.content_type.startswith("text/"):
                        try:
                            # No longer limit attachment text size here
                            attachment_text = resp.text
                            text_parts.append(attachment_text)
                        except Exception as e:
                            logging.warning(f"Failed to decode text attachment {att.filename}: {e}")
                            curr_node.has_bad_attachments = True
                    elif isinstance(resp, Exception):
                        logging.warning(f"Failed to fetch attachment {att.filename}: {resp}")
                        curr_node.has_bad_attachments = True

                curr_node.text = "\n".join(filter(None, text_parts))

                # Process image attachments
                image_parts = []
                for att, resp in zip(good_attachments, attachment_responses):
                    if isinstance(resp, httpx.Response) and resp.status_code == 200 and att.content_type.startswith("image/"):
                        if is_gemini:
                            # Use google.genai.types (imported as google_types)
                            image_parts.append(google_types.Part.from_bytes(data=resp.content, mime_type=att.content_type))
                        else:
                            image_parts.append(dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{base64.b64encode(resp.content).decode('utf-8')}")))
                    elif isinstance(resp, Exception):
                        # Already logged warning above
                        curr_node.has_bad_attachments = True

                curr_node.images = image_parts
                curr_node.role = "model" if curr_msg.author == discord_client.user else "user" # Use 'model' for Gemini assistant role
                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
                curr_node.has_bad_attachments = curr_node.has_bad_attachments or (len(current_attachments) > len(good_attachments))

                # Store the fetched external content in the node for the triggering message
                if curr_msg.id == new_msg.id and combined_context:
                    curr_node.external_content = combined_context

                # Find parent message
                try:
                    parent_msg_obj = None
                    # Check if the current message explicitly triggers the bot (mention or "at ai")
                    # Use the variables calculated at the start of on_message if curr_msg is new_msg
                    if curr_msg.id == new_msg.id:
                        mentions_bot_in_current = mentions_bot
                        contains_at_ai_in_current = contains_at_ai
                    else: # Recalculate for older messages in the chain if needed
                        mentions_bot_in_current = discord_client.user.mentioned_in(curr_msg)
                        contains_at_ai_in_current = AT_AI_PATTERN.search(curr_msg.content) is not None

                    is_explicit_trigger = mentions_bot_in_current or contains_at_ai_in_current

                    # 1. Check reference (Explicit Reply always takes precedence)
                    if curr_msg.reference and curr_msg.reference.message_id:
                        try:
                            parent_msg_obj = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(curr_msg.reference.message_id)
                        except (discord.NotFound, discord.HTTPException) as e:
                            logging.warning(f"Could not fetch referenced message {curr_msg.reference.message_id}: {e}")
                            curr_node.fetch_parent_failed = True
                    # 2. Check if it's the start of a thread (and not a reply within the thread)
                    elif curr_msg.channel.type == discord.ChannelType.public_thread and not curr_msg.reference:
                         try:
                             # The starter message is the parent
                             parent_msg_obj = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(curr_msg.channel.id)
                         except (discord.NotFound, discord.HTTPException, AttributeError) as e:
                             logging.warning(f"Could not fetch thread starter message for thread {curr_msg.channel.id}: {e}")
                             curr_node.fetch_parent_failed = True
                    # 3. Check for automatic chaining ONLY IF not explicitly triggered by mention/@ai
                    elif not is_explicit_trigger:
                         prev_msg_in_channel = None
                         try:
                             # Use history instead of list comprehension for efficiency
                             async for m in curr_msg.channel.history(before=curr_msg, limit=1):
                                 prev_msg_in_channel = m
                                 break # Get only the most recent one
                         except (discord.Forbidden, discord.HTTPException) as e:
                             logging.warning(f"Could not fetch history in channel {curr_msg.channel.id}: {e}")

                         if prev_msg_in_channel and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply):
                             # In DMs, chain if previous is from bot. In channels, chain if previous is from same user.
                             if (is_dm_current and prev_msg_in_channel.author == discord_client.user) or \
                                (not is_dm_current and prev_msg_in_channel.author == curr_msg.author):
                                 parent_msg_obj = prev_msg_in_channel

                    curr_node.parent_msg = parent_msg_obj

                except Exception as e:
                    logging.exception(f"Error determining parent message for {curr_msg.id}")
                    curr_node.fetch_parent_failed = True


            # --- Prepare content for this node, including external content if present ---
            current_text_content = ""
            if curr_node.external_content:
                # Prepend external content if it exists for this node
                current_text_content += curr_node.external_content + "\n\nUser's query:\n" # Add separator
            if curr_node.text:
                current_text_content += curr_node.text

            # Limit the combined text content *before* sending to API
            current_text_content = current_text_content[:max_text] if current_text_content else "" # Keep this final node limit
            current_images = curr_node.images[:max_images] # Apply max_images limit

            parts_for_api = []
            if is_gemini:
                if current_text_content:
                    parts_for_api.append(google_types.Part.from_text(text=current_text_content))
                parts_for_api.extend(current_images) # These are already google_types.Part
            else: # OpenAI format
                if current_text_content:
                    parts_for_api.append({"type": "text", "text": current_text_content})
                parts_for_api.extend(current_images) # These are already dicts

            # Add to history if parts exist
            if parts_for_api:
                message_data = {
                    "role": curr_node.role # Use 'user' or 'model'/'assistant'
                }
                if is_gemini:
                    # Ensure parts_for_api is always a list for Gemini
                    if not isinstance(parts_for_api, list):
                        parts_for_api = [parts_for_api]
                    message_data["parts"] = parts_for_api
                else:
                    # OpenAI uses 'assistant' role for model responses
                    if message_data["role"] == "model":
                        message_data["role"] = "assistant"
                    message_data["content"] = parts_for_api
                    # Add name field if supported by provider
                    if provider in PROVIDERS_SUPPORTING_USERNAMES and curr_node.user_id is not None:
                        message_data["name"] = str(curr_node.user_id)

                history.append(message_data)

            # Add warnings based on limits and errors for this specific node
            if curr_node.text and len(curr_node.text) > max_text: # Check original text length
                user_warnings.add(f"⚠️ Max {max_text:,} chars/msg node") # Clarify warning
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} images/msg" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed:
                 user_warnings.add(f"⚠️ Couldn't fetch full history")
            # Add warning if max messages reached *while processing this node*
            if curr_node.parent_msg is not None and len(history) == max_messages:
                 user_warnings.add(f"⚠️ Only using last {max_messages} messages")


            # Move to the parent message for the next iteration
            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, history length: {len(history)}, google_lens: {use_google_lens}, warnings: {user_warnings}):\n{new_msg.content}")


    # --- Prepare API Call ---
    history_for_llm = history[::-1] # Reverse history for correct chronological order

    # System Prompt
    system_prompt_text = None
    if system_prompt := cfg.get("system_prompt"):
        # Get current UTC date and time
        now_utc = dt.now(timezone.utc)

        # Format the time part (12-hour, no leading zero, narrow no-break space before AM/PM)
        hour_12 = now_utc.strftime('%I')
        minute = now_utc.strftime('%M')
        am_pm = now_utc.strftime('%p')
        hour_12_no_zero = hour_12.lstrip('0')
        time_str = f"{hour_12_no_zero}:{minute}\u202F{am_pm}" # \u202F is NARROW NO-BREAK SPACE

        # Format the date part
        date_str = now_utc.strftime('%A, %B %d, %Y') # e.g., Sunday, April 27, 2025

        # Construct the full date/time string in the required format
        current_datetime_str = f"current date and time: {date_str} {time_str} Coordinated Universal Time (UTC)"

        system_prompt_extras = [current_datetime_str]
        if not is_gemini and provider in PROVIDERS_SUPPORTING_USERNAMES:
            system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")
        system_prompt_text = "\n".join([system_prompt] + system_prompt_extras)

    # --- Generate and Send Response with Retry ---
    response_msgs = [] # Keep track of messages sent by the bot for this request
    final_text = ""    # Store the final aggregated text
    llm_call_successful = False
    llm_errors = []
    final_view = None # Initialize final_view here
    grounding_metadata = None
    edit_task = None
    last_error_type = None # Track the type of the last error (safety, recitation, etc.)

    embed = discord.Embed() # Initialize embed here
    embed.set_footer(text=f"Model: {provider_slash_model}") # Add the footer with the model name
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    try: # Main try block for the core processing and API calls
        # Get available keys for the selected provider
        available_llm_keys = await get_available_keys(provider, all_api_keys)
        random.shuffle(available_llm_keys)
        llm_db_manager = get_db_manager(provider)

        if keys_required and not available_llm_keys: # Check if keys are actually needed and available
            logging.error(f"No available (non-rate-limited) API keys for provider '{provider}'.")
            await new_msg.reply(f"⚠️ No available API keys for provider `{provider}` right now.", mention_author = False)
            return # Exit if no keys available and keys are required

        # Loop even if no keys needed (e.g., Ollama) - use a dummy key placeholder
        keys_to_loop = available_llm_keys if keys_required else ["dummy_key"]

        for key_index, current_api_key in enumerate(keys_to_loop):
            key_display = f"...{current_api_key[-4:]}" if current_api_key != "dummy_key" else "N/A (keyless)"
            logging.info(f"Attempting LLM request with provider '{provider}' using key {key_display} ({key_index+1}/{len(keys_to_loop)})")

            # Reset state for each attempt
            response_contents = []
            # final_text = "" # Don't reset final_text here, aggregate across stream chunks
            finish_reason = None
            grounding_metadata = None
            llm_client = None
            api_config = None
            api_content_kwargs = {}
            payload_to_print = {} # Initialize payload dict for printing
            is_blocked_by_safety = False # Reset safety flag for each attempt
            is_stopped_by_recitation = False # Reset recitation flag

            try: # Inner try for the specific API call attempt
                # --- Initialize Client for this attempt ---
                if is_gemini:
                    if current_api_key == "dummy_key": raise ValueError("Gemini requires an API key.")
                    # Use google.genai (imported as google_genai)
                    llm_client = google_genai.Client(api_key=current_api_key)
                    # Prepare Gemini specific args
                    gemini_contents = []
                    for msg in history_for_llm:
                        role = msg["role"] # Already 'user' or 'model'
                        parts = msg.get("parts", []) # Default to empty list if parts missing
                        # Ensure parts is a list, even if empty
                        if not isinstance(parts, list):
                             logging.warning(f"Correcting non-list parts for Gemini message: {parts}")
                             # Convert non-list to text part or empty list
                             # Use google.genai.types (imported as google_types)
                             parts = [google_types.Part.from_text(text=str(parts))] if parts else []
                        # Use google.genai.types (imported as google_types)
                        gemini_contents.append(google_types.Content(role=role, parts=parts))

                    api_content_kwargs["contents"] = gemini_contents

                    gemini_extra_params = cfg.get("extra_api_parameters", {}).copy()
                    if "max_tokens" in gemini_extra_params:
                        gemini_extra_params["max_output_tokens"] = gemini_extra_params.pop("max_tokens")

                    # Use google.genai.types (imported as google_types)
                    gemini_safety_settings_list = [
                        google_types.SafetySetting(category=category, threshold=threshold)
                        for category, threshold in GEMINI_SAFETY_SETTINGS_DICT.items()
                    ]

                    # Use google.genai.types (imported as google_types)
                    api_config = google_types.GenerateContentConfig(
                        **gemini_extra_params,
                        safety_settings=gemini_safety_settings_list,
                        tools=[google_types.Tool(google_search=google_types.GoogleSearch())] # Enable grounding
                    )
                    if system_prompt_text:
                         # Use google.genai.types (imported as google_types)
                         api_config.system_instruction = google_types.Part.from_text(text=system_prompt_text)

                    # --- Prepare Gemini Payload for Printing ---
                    payload_to_print = {
                        "model": model_name,
                        # Use model_dump() for Content objects
                        "contents": [c.model_dump(mode='json', exclude_none=True) for c in api_content_kwargs["contents"]],
                        # Use model_dump() for config object
                        "generationConfig": api_config.model_dump(mode='json', exclude_none=True) if api_config else {},
                    }
                    # Remove empty fields from generationConfig for cleaner printing
                    payload_to_print["generationConfig"] = {k: v for k, v in payload_to_print["generationConfig"].items() if v}


                else: # OpenAI compatible
                    api_key_to_use = current_api_key if current_api_key != "dummy_key" else None
                    llm_client = AsyncOpenAI(base_url=base_url, api_key=api_key_to_use)
                    # Prepare OpenAI specific args
                    openai_messages = history_for_llm[:] # Copy history
                    if system_prompt_text:
                        openai_messages.insert(0, dict(role="system", content=system_prompt_text))

                    api_content_kwargs["messages"] = openai_messages
                    api_config = cfg.get("extra_api_parameters", {}).copy()
                    api_config["stream"] = True # Always stream for OpenAI

                    # --- Prepare OpenAI Payload for Printing ---
                    payload_to_print = {
                        "model": model_name,
                        "messages": api_content_kwargs["messages"],
                        **api_config # Spread the config parameters
                    }

                # --- Print Payload ---
                try:
                    print(f"\n--- LLM Request Payload (Provider: {provider}, Model: {model_name}) ---")

                    # Create a deep copy and truncate base64 for printing
                    payload_for_printing = _truncate_base64_in_payload(payload_to_print) # Use the new function

                    # Custom default handler for non-serializable types (still needed for other types)
                    def default_serializer(obj):
                        # Check for Pydantic models first
                        if hasattr(obj, 'model_dump'):
                            try:
                                return obj.model_dump(mode='json', exclude_none=True)
                            except Exception:
                                pass
                        # Handle specific known types like google_types.Part (less critical now)
                        elif isinstance(obj, google_types.Part):
                             part_dict = {}
                             if hasattr(obj, 'text') and obj.text is not None:
                                 part_dict["text"] = obj.text
                             # The truncation function handles inline_data, but keep this for structure
                             if hasattr(obj, 'inline_data') and obj.inline_data:
                                 part_dict["inline_data"] = {
                                     "mime_type": obj.inline_data.mime_type,
                                     "data": "<base64_data_handled_by_truncation>" # Placeholder
                                 }
                             return part_dict if part_dict else None
                        elif isinstance(obj, bytes):
                            return "<bytes_data>"
                        # Fallback for other types
                        try:
                            return json.JSONEncoder.default(None, obj)
                        except TypeError:
                            return f"<unserializable_object: {type(obj).__name__}>"

                    # Use the custom serializer with json.dumps on the truncated payload
                    print(json.dumps(payload_for_printing, indent=2, default=default_serializer))
                    print("--- End LLM Request Payload ---\n")
                except Exception as print_err:
                    logging.error(f"Error printing LLM payload: {print_err}")
                    # Fallback to printing the raw dict which might still fail
                    print(f"Raw Payload Data (may contain unserializable objects):\n{payload_to_print}\n")


                # --- Make API Call and Process Stream ---
                async with new_msg.channel.typing():
                    stream_response = None
                    if is_gemini:
                        if not llm_client: raise ValueError("Gemini client not initialized for this key.")
                        # Use google.genai.types (imported as google_types)
                        stream_response = await llm_client.aio.models.generate_content_stream(
                            model=model_name,
                            contents=api_content_kwargs["contents"],
                            config=api_config
                        )
                    else:
                        if not llm_client: raise ValueError("OpenAI client not initialized for this key.")
                        stream_response = await llm_client.chat.completions.create(
                            model=model_name,
                            messages=api_content_kwargs["messages"],
                            **api_config
                        )

                    # --- Stream Processing Loop (Inside Retry Loop) ---
                    content_received = False # Flag to track if any content chunk was received
                    chunk_processed_successfully = False # Flag for last chunk attempt
                    async for chunk in stream_response:
                        new_content_chunk = ""
                        chunk_finish_reason = None
                        chunk_grounding_metadata = None
                        chunk_processed_successfully = False # Reset for each chunk

                        try: # Inner try for stream processing errors
                            if is_gemini:
                                # Check for prompt feedback first (indicates potential immediate block)
                                if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback:
                                    if chunk.prompt_feedback.block_reason:
                                        logging.warning(f"Gemini Prompt Blocked (reason: {chunk.prompt_feedback.block_reason}) with key {key_display}. Aborting.")
                                        llm_errors.append(f"Key {key_display}: Prompt Blocked ({chunk.prompt_feedback.block_reason})")
                                        llm_call_successful = False
                                        is_blocked_by_safety = True # Treat prompt block as safety issue
                                        last_error_type = "safety"
                                        break # Exit inner stream loop

                                # Extract Gemini data
                                if hasattr(chunk, 'text') and chunk.text:
                                    new_content_chunk = chunk.text
                                    content_received = True # Mark that we got some content
                                    logging.debug(f"Received Gemini chunk text (len: {len(new_content_chunk)})") # Debug log

                                if hasattr(chunk, 'candidates') and chunk.candidates:
                                     candidate = chunk.candidates[0]
                                     # Check finish_reason only if it's not UNSPECIFIED
                                     if hasattr(candidate, 'finish_reason') and candidate.finish_reason and candidate.finish_reason != google_types.FinishReason.FINISH_REASON_UNSPECIFIED:
                                          # Map Gemini finish reason
                                          reason_map = {
                                               google_types.FinishReason.STOP: "stop",
                                               google_types.FinishReason.MAX_TOKENS: "length",
                                               google_types.FinishReason.SAFETY: "safety",
                                               google_types.FinishReason.RECITATION: "recitation",
                                               google_types.FinishReason.OTHER: "other",
                                          }
                                          # Use the mapped reason, default to the string representation if unknown
                                          chunk_finish_reason = reason_map.get(candidate.finish_reason, str(candidate.finish_reason))
                                          logging.info(f"Gemini finish reason received: {chunk_finish_reason} ({candidate.finish_reason})") # Log the reason

                                          # --- Check for Safety/Recitation/Other Finish Reasons ---
                                          if chunk_finish_reason:
                                              finish_reason_lower = chunk_finish_reason.lower()
                                              if finish_reason_lower == "safety":
                                                  logging.warning(f"Gemini Response Blocked (finish_reason=SAFETY) with key {key_display}. Check prompt/safety settings. Full chunk: {chunk}")
                                                  llm_errors.append(f"Key {key_display}: Response Blocked (Safety)")
                                                  llm_call_successful = False
                                                  is_blocked_by_safety = True
                                                  last_error_type = "safety"
                                                  # Don't break immediately, let editing logic handle final state
                                              elif finish_reason_lower == "recitation":
                                                  logging.warning(f"Gemini Response stopped due to Recitation (finish_reason=RECITATION) with key {key_display}. Consider prompt uniqueness/temperature. Full chunk: {chunk}")
                                                  llm_errors.append(f"Key {key_display}: Response Stopped (Recitation)")
                                                  llm_call_successful = False
                                                  is_stopped_by_recitation = True
                                                  last_error_type = "recitation"
                                                  # Don't break immediately
                                              elif finish_reason_lower == "other":
                                                  logging.warning(f"Gemini Response Blocked (finish_reason=OTHER) with key {key_display}. May violate ToS or be unsupported. Full chunk: {chunk}")
                                                  llm_errors.append(f"Key {key_display}: Response Blocked (Other)")
                                                  llm_call_successful = False
                                                  is_blocked_by_safety = True # Treat 'other' like safety
                                                  last_error_type = "other"
                                                  # Don't break immediately
                                          # --- End Check ---

                                     if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                                          chunk_grounding_metadata = candidate.grounding_metadata
                                     # Check for safety ratings even if finish_reason isn't SAFETY
                                     if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                                         for rating in candidate.safety_ratings:
                                             # Check if any category is blocked (adjust threshold logic if needed)
                                             # Using BLOCK_MEDIUM_AND_ABOVE as an example threshold check
                                             # Use google.genai.types (imported as google_types)
                                             if rating.probability in (google_types.HarmProbability.MEDIUM, google_types.HarmProbability.HIGH):
                                                 logging.warning(f"Gemini content potentially blocked by safety rating: Category {rating.category}, Probability {rating.probability}. Key: {key_display}")
                                                 # Decide if this constitutes a block for retry purposes
                                                 # For now, let's treat it like a safety finish reason if we haven't received content yet
                                                 if not content_received:
                                                     llm_errors.append(f"Key {key_display}: Response Blocked (Safety Rating: {rating.category}={rating.probability})")
                                                     llm_call_successful = False
                                                     is_blocked_by_safety = True
                                                     last_error_type = "safety"
                                                     # Don't break immediately

                            else: # OpenAI
                                if chunk.choices:
                                    delta = chunk.choices[0].delta
                                    chunk_finish_reason = chunk.choices[0].finish_reason
                                    if delta and delta.content:
                                        new_content_chunk = delta.content
                                        content_received = True # Mark content received for OpenAI too

                            # Update overall finish reason and grounding metadata
                            if chunk_finish_reason:
                                finish_reason = chunk_finish_reason
                            if chunk_grounding_metadata:
                                grounding_metadata = chunk_grounding_metadata

                            # Append content if not empty and not blocked
                            if new_content_chunk and not is_blocked_by_safety and not is_stopped_by_recitation:
                                response_contents.append(new_content_chunk)

                            chunk_processed_successfully = True # Mark chunk as processed

                            # --- Real-time Editing Logic (Common for both) ---
                            if not use_plain_responses:
                                current_full_text = "".join(response_contents)
                                is_final_chunk = finish_reason is not None

                                if not current_full_text and not is_final_chunk: # Skip empty intermediate chunks
                                     continue

                                # Create view only on the final chunk if needed
                                view_to_attach = None
                                if is_final_chunk and not is_blocked_by_safety and not is_stopped_by_recitation: # Only add view if not blocked
                                    # Check if any button should be added
                                    has_sources = grounding_metadata and (getattr(grounding_metadata, 'web_search_queries', None) or getattr(grounding_metadata, 'grounding_chunks', None))
                                    has_text = bool(current_full_text)
                                    if has_sources or has_text:
                                        view_to_attach = ResponseActionView(
                                            grounding_metadata=grounding_metadata,
                                            full_response_text=current_full_text,
                                            model_name=provider_slash_model
                                        )
                                        # Remove view if it ended up having no buttons
                                        if not view_to_attach or len(view_to_attach.children) == 0:
                                            view_to_attach = None

                                current_msg_index = (len(current_full_text) - 1) // split_limit if current_full_text else 0
                                start_next_msg = current_msg_index >= len(response_msgs)

                                ready_to_edit = (edit_task is None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS

                                if start_next_msg or ready_to_edit or is_final_chunk:
                                    if edit_task is not None:
                                        await edit_task # Wait for previous edit

                                    # Finalize previous message if splitting
                                    if start_next_msg and response_msgs:
                                        prev_msg_index = current_msg_index - 1
                                        prev_msg_text = current_full_text[prev_msg_index * split_limit : current_msg_index * split_limit]
                                        prev_msg_text = prev_msg_text[:MAX_EMBED_DESCRIPTION_LENGTH + len(STREAMING_INDICATOR)] # Ensure limit
                                        embed.description = prev_msg_text or "..." # No indicator, handle empty
                                        embed.color = EMBED_COLOR_COMPLETE
                                        try:
                                            # Remove view from previous message
                                            await response_msgs[prev_msg_index].edit(embed=embed, view=None)
                                        except discord.HTTPException as e:
                                            logging.error(f"Failed to finalize previous message {prev_msg_index}: {e}")

                                    # Prepare current message segment
                                    current_display_text = current_full_text[current_msg_index * split_limit : (current_msg_index + 1) * split_limit]
                                    current_display_text = current_display_text[:MAX_EMBED_DESCRIPTION_LENGTH] # Truncate

                                    # Set embed content and color
                                    embed.description = (current_display_text or "...") if is_final_chunk else ((current_display_text or "...") + STREAMING_INDICATOR)
                                    # Determine successful finish (Gemini UNSPECIFIED is also success)
                                    is_successful_finish = finish_reason and (finish_reason.lower() in ("stop", "end_turn") or (is_gemini and finish_reason == str(google_types.FinishReason.FINISH_REASON_UNSPECIFIED)))


                                    # Modify embed color/text if blocked or stopped on the final chunk
                                    if is_final_chunk and (is_blocked_by_safety or is_stopped_by_recitation):
                                        embed.description = (current_display_text or "...").replace(STREAMING_INDICATOR, "").strip() # Remove indicator
                                        if is_blocked_by_safety:
                                            embed.description += "\n\n⚠️ Response blocked by safety filters."
                                        elif is_stopped_by_recitation:
                                            embed.description += "\n\n⚠️ Response stopped due to recitation."
                                        embed.description = embed.description[:MAX_EMBED_DESCRIPTION_LENGTH]
                                        embed.color = EMBED_COLOR_ERROR
                                        view_to_attach = None # No actions on blocked/stopped response
                                    else:
                                        embed.color = EMBED_COLOR_COMPLETE if is_final_chunk and is_successful_finish else EMBED_COLOR_INCOMPLETE


                                    # Create or Edit the current message
                                    if start_next_msg:
                                        reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                                        # Clear previous response messages if starting over due to retry
                                        if key_index > 0 and not response_msgs: # Only clear if response_msgs is empty (meaning previous attempt failed before sending)
                                            logging.info(f"Clearing previous response messages due to retry (Key index: {key_index})")
                                            # No messages to delete if list is empty
                                        response_msg = await reply_to_msg.reply(embed=embed, view=view_to_attach, mention_author = False)
                                        response_msgs.append(response_msg)
                                        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                                        await msg_nodes[response_msg.id].lock.acquire() # Acquire lock here
                                    elif response_msgs and current_msg_index < len(response_msgs):
                                        edit_task = asyncio.create_task(response_msgs[current_msg_index].edit(embed=embed, view=view_to_attach))
                                    elif not response_msgs and is_final_chunk: # Handle case where response is short and finishes immediately
                                         reply_to_msg = new_msg
                                         response_msg = await reply_to_msg.reply(embed=embed, view=view_to_attach, mention_author = False)
                                         response_msgs.append(response_msg)
                                         msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                                         await msg_nodes[response_msg.id].lock.acquire() # Acquire lock here


                                    last_task_time = dt.now().timestamp()

                            # Break inner stream loop if finished or blocked/stopped
                            if finish_reason:
                                logging.info(f"Stream finished with reason: {finish_reason}. Exiting inner loop.")
                                break # Exit inner stream loop

                        except google_api_exceptions.GoogleAPIError as stream_err: # Catch Google API errors during streaming
                            logging.warning(f"Google API error during streaming with key {key_display}: {type(stream_err).__name__} - {stream_err}. Trying next key.")
                            llm_errors.append(f"Key {key_display}: Stream Google API Error - {type(stream_err).__name__}: {stream_err}")
                            last_error_type = "google_api"
                            if isinstance(stream_err, google_api_exceptions.ResourceExhausted):
                                if current_api_key != "dummy_key": llm_db_manager.add_key(current_api_key)
                                last_error_type = "rate_limit"
                            # Break inner stream loop and proceed to retry with next key
                            break # Exit inner loop on stream error
                        except APIConnectionError as stream_err: # Catch connection errors during streaming (OpenAI)
                            logging.warning(f"Connection error during streaming with key {key_display}: {stream_err}. Trying next key.")
                            llm_errors.append(f"Key {key_display}: Stream Connection Error - {stream_err}")
                            last_error_type = "connection"
                            break # Exit inner loop on stream error
                        except APIError as stream_err: # Catch other API errors during streaming (OpenAI specific)
                            logging.warning(f"API error during streaming with key {key_display}: {stream_err}. Trying next key.")
                            llm_errors.append(f"Key {key_display}: Stream API Error - {stream_err}")
                            last_error_type = "api"
                            if isinstance(stream_err, RateLimitError):
                                if current_api_key != "dummy_key": llm_db_manager.add_key(current_api_key)
                                last_error_type = "rate_limit"
                            break # Exit inner loop on stream error
                        except Exception as stream_err: # Catch unexpected errors during streaming
                            logging.exception(f"Unexpected error during streaming with key {key_display}")
                            llm_errors.append(f"Key {key_display}: Unexpected Stream Error - {type(stream_err).__name__}")
                            last_error_type = "unexpected"
                            break # Exit inner loop on stream error
                    # --- End Stream Processing Loop ---

                    # --- After Stream Processing Loop ---

                    # Check if the stream ended BUT no content was received AND it wasn't explicitly blocked/stopped by safety/recitation
                    if not content_received and not is_blocked_by_safety and not is_stopped_by_recitation:
                        # Check if finish_reason exists and is STOP (or UNSPECIFIED for Gemini) - this indicates a truly empty response
                        is_successful_empty_finish = finish_reason and (finish_reason.lower() == "stop" or (is_gemini and finish_reason == str(google_types.FinishReason.FINISH_REASON_UNSPECIFIED)))
                        if is_successful_empty_finish:
                            # This is the specific case from the user report: 200 OK but no content.
                            logging.warning(f"LLM stream finished successfully for key {key_display} but NO content was received. Finish reason: {finish_reason}. Aborting retries for this request.")
                            llm_errors.append(f"Key {key_display}: No content received (Successful Finish)")
                            llm_call_successful = False # Explicitly mark as failure
                            last_error_type = "no_content_success_finish"
                            break # <<< BREAK outer loop: Do not retry if the API successfully returned nothing.
                        elif finish_reason: # Finished with a non-success reason, but still no content
                            logging.warning(f"LLM stream finished with reason '{finish_reason}' for key {key_display} but NO content was received. Aborting retries.")
                            llm_errors.append(f"Key {key_display}: No content received (Finish Reason: {finish_reason})") # Corrected indentation
                            llm_call_successful = False # Corrected indentation
                            last_error_type = "no_content_other_finish" # Corrected indentation
                            break # <<< BREAK outer loop: Do not retry if API finished abnormally with no content. # Corrected indentation
                        else: # Stream likely broke due to error caught inside loop, or never started/ended prematurely
                            logging.warning(f"LLM stream ended prematurely for key {key_display} and NO content was received. Last error type: {last_error_type}. Aborting retries.")
                            # llm_call_successful remains False from initialization or inner error handling # Corrected indentation
                            # Error should have been appended inside the loop's except block if applicable # Corrected indentation
                            # If no specific error was caught inside, add a generic one now # Corrected indentation
                            if not llm_errors or not llm_errors[-1].startswith(f"Key {key_display}"): # Avoid duplicate errors for the same key # Corrected indentation
                                llm_errors.append(f"Key {key_display}: Stream ended prematurely with no content") # Corrected indentation
                            last_error_type = "premature_empty_stream" # Assign a specific type # Corrected indentation
                            break # <<< BREAK outer loop: Do not retry if stream ended prematurely with no content. # Corrected indentation

                    # --- Check for Safety/Recitation/Other blocks AFTER the loop (if it didn't break early from 'no content') ---
                    elif is_blocked_by_safety:
                        logging.warning(f"LLM response blocked due to safety/other with key {key_display}. Aborting retries for this request.")
                        break # Break outer retry loop immediately if blocked
                    elif is_stopped_by_recitation:
                        logging.warning(f"LLM response stopped due to recitation with key {key_display}. Aborting retries for this request.")
                        break # Break outer retry loop immediately if stopped by recitation
                    # --- End Check ---

                    # If the stream finished successfully AND content was received
                    elif finish_reason and content_received: # Check content_received flag
                        # Determine success based on finish reason (Gemini UNSPECIFIED is success)
                        is_successful_finish = finish_reason.lower() in ("stop", "end_turn") or (is_gemini and finish_reason == str(google_types.FinishReason.FINISH_REASON_UNSPECIFIED))
                        if is_successful_finish:
                            llm_call_successful = True
                            logging.info(f"LLM request successful with key {key_display}")
                            break # Exit the outer retry loop
                        else:
                            # Handle non-stop finish reasons like 'length'
                            logging.warning(f"LLM stream finished with non-stop reason '{finish_reason}' for key {key_display}. Treating as failure for retry.")
                            llm_errors.append(f"Key {key_display}: Finished Reason '{finish_reason}'")
                            llm_call_successful = False
                            last_error_type = finish_reason # Use the reason as the error type
                            # Continue to next key

                    # If the stream broke due to an error caught inside the loop (and content might have been partially received)
                    elif not chunk_processed_successfully: # Check if the last attempt broke mid-chunk processing
                         logging.warning(f"LLM stream broke during processing for key {key_display}. Last error type: {last_error_type}. Trying next key.")
                         # llm_call_successful remains False, continue outer loop (Implicitly handled by loop)

                    # Fallback case: Should not be reached if logic above is correct
                    else:
                         logging.error(f"Reached unexpected state after stream loop for key {key_display}. Content received: {content_received}, Finish reason: {finish_reason}, Blocked: {is_blocked_by_safety}, Stopped: {is_stopped_by_recitation}. Treating as failure and trying next key.")
                         llm_errors.append(f"Key {key_display}: Unexpected stream end state")
                         llm_call_successful = False
                         last_error_type = "unexpected_stream_end"
                         # Let the loop continue to try the next key in this unexpected state

            # --- Handle API Call Errors for the current key (Initial request errors) ---
            except (RateLimitError, google_api_exceptions.ResourceExhausted) as e:
                logging.warning(f"Initial Request Rate limit hit for provider '{provider}' with key {key_display}. Error: {e}. Trying next key.")
                if current_api_key != "dummy_key": llm_db_manager.add_key(current_api_key)
                llm_errors.append(f"Key {key_display}: Initial Rate Limited")
                last_error_type = "rate_limit"
                continue
            except (AuthenticationError, google_api_exceptions.PermissionDenied) as e:
                logging.error(f"Initial Request Authentication failed for provider '{provider}' with key {key_display}. Error: {e}. Aborting retries.")
                llm_errors.append(f"Key {key_display}: Initial Authentication Failed")
                last_error_type = "auth"
                llm_call_successful = False; break
            except (APIConnectionError, google_api_exceptions.ServiceUnavailable, google_api_exceptions.DeadlineExceeded) as e:
                logging.warning(f"Initial Request Connection/Service error for provider '{provider}' with key {key_display}. Error: {e}. Trying next key.")
                llm_errors.append(f"Key {key_display}: Initial Connection/Service Error - {type(e).__name__}")
                last_error_type = "connection"
                continue
            except (BadRequestError, google_api_exceptions.InvalidArgument) as e:
                 logging.error(f"Initial Request Bad request error for provider '{provider}' with key {key_display}. Error: {e}. Aborting retries.")
                 if isinstance(e, google_api_exceptions.InvalidArgument):
                     error_details = getattr(e, 'details', None)
                     if error_details: logging.error(f"Google API InvalidArgument Details: {error_details}")
                 llm_errors.append(f"Key {key_display}: Initial Bad Request - {e}")
                 last_error_type = "bad_request"
                 llm_call_successful = False; break
            except APIError as e: # Catch other OpenAI API errors
                logging.exception(f"Initial Request OpenAI API Error for key {key_display}")
                llm_errors.append(f"Key {key_display}: Initial API Error - {type(e).__name__}: {e}")
                last_error_type = "api"
                continue
            except google_api_exceptions.GoogleAPICallError as e: # Catch other Google API errors
                logging.exception(f"Initial Request Google API Call Error for key {key_display}")
                error_details = getattr(e, 'details', None)
                if error_details: logging.error(f"Google API Error Details: {error_details}")
                llm_errors.append(f"Key {key_display}: Initial Google API Error - {type(e).__name__}: {e}")
                last_error_type = "google_api"
                continue
            except Exception as e: # Catch other unexpected errors
                logging.exception(f"Unexpected error during initial LLM call with key {key_display}")
                llm_errors.append(f"Key {key_display}: Unexpected Initial Error - {type(e).__name__}")
                last_error_type = "unexpected"
                continue

        # --- Post-Retry Loop Processing ---
        if not llm_call_successful:
            logging.error(f"All LLM API keys failed for provider '{provider}'. Errors: {llm_errors}") # Keep this log
            error_message = f"⚠️ All API keys for provider `{provider}` failed."
            if llm_errors:
                # Show the last error encountered
                last_error_str = str(llm_errors[-1])
                if last_error_type == "safety":
                    error_message += "\nLast error: Response blocked by safety filters."
                elif last_error_type == "recitation":
                    error_message += "\nLast error: Response stopped due to recitation."
                elif last_error_type == "other": # Gemini OTHER block reason
                    error_message += "\nLast error: Response blocked (Reason: Other)."
                elif last_error_type == "no_content_success_finish":
                    error_message += "\nLast error: No content received from API (finished successfully)."
                elif last_error_type == "premature_empty_stream":
                    error_message += "\nLast error: Stream ended prematurely with no content."
                elif last_error_type == "no_content_other_finish":
                    error_message += f"\nLast error: No content received from API (Finish Reason: {finish_reason})."
                elif last_error_type == "unexpected_stream_end":
                    error_message += "\nLast error: Unexpected stream end state."
                else:
                    error_message += f"\nLast error: `{last_error_str[:100]}{'...' if len(last_error_str) > 100 else ''}`"

            # Edit the last message OR reply with the final error
            if not use_plain_responses and response_msgs:
                 try:
                     # Ensure embed description exists before appending
                     current_desc = response_msgs[-1].embeds[0].description if response_msgs[-1].embeds else ""
                     embed.description = current_desc.replace(STREAMING_INDICATOR, "").strip()
                     embed.description += f"\n\n{error_message}"
                     embed.description = embed.description[:MAX_EMBED_DESCRIPTION_LENGTH]
                     embed.color = EMBED_COLOR_ERROR
                     await response_msgs[-1].edit(embed=embed, view=None) # Remove view on final error
                 except Exception as edit_err:
                     logging.error(f"Failed to edit message to show final error: {edit_err}")
                     await new_msg.reply(error_message, mention_author = False) # Fallback reply
            else:
                # If plain responses were used, or no messages were sent yet, just reply
                await new_msg.reply(error_message, mention_author = False)
            # Do NOT return here, let finally run

        else: # If successful
            final_text = "".join(response_contents)

            # Create the final view (if needed) after successful generation
            final_view = None
            if not use_plain_responses:
                has_sources = grounding_metadata and (getattr(grounding_metadata, 'web_search_queries', None) or getattr(grounding_metadata, 'grounding_chunks', None))
                has_text = bool(final_text)
                if has_sources or has_text:
                    final_view = ResponseActionView(
                        grounding_metadata=grounding_metadata,
                        full_response_text=final_text,
                        model_name=provider_slash_model
                    )
                    # Remove view if it ended up having no buttons
                    if not final_view or len(final_view.children) == 0:
                        final_view = None

            # Handle plain text responses (final output)
            if use_plain_responses:
                 final_messages_content = [final_text[i:i+2000] for i in range(0, len(final_text), 2000)]
                 if not final_messages_content: final_messages_content.append("...")

                 # Delete previous potentially failed attempts if retried and successful
                 # This assumes response_msgs was cleared correctly before the successful attempt
                 # No explicit deletion needed here if logic inside stream loop is correct

                 temp_response_msgs = []
                 for i, content in enumerate(final_messages_content):
                     reply_to_msg = new_msg if not temp_response_msgs else temp_response_msgs[-1]
                     # Plain responses don't get views
                     response_msg = await reply_to_msg.reply(content=content or "...", suppress_embeds=True, view=None, mention_author = False)
                     temp_response_msgs.append(response_msg)
                     # Create node and acquire lock immediately
                     msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                     node = msg_nodes[response_msg.id]
                     await node.lock.acquire()
                     # Store full text in the last node for plain responses
                     if i == len(final_messages_content) - 1:
                         node.full_response_text = final_text
                 response_msgs = temp_response_msgs # Update the main list

            # Final edit for embed messages (if not already handled by final chunk logic)
            elif not use_plain_responses and response_msgs:
                 if edit_task is not None and not edit_task.done(): await edit_task

                 final_msg_index = len(response_msgs) - 1
                 final_msg_text = final_text[final_msg_index * split_limit : (final_msg_index + 1) * split_limit]
                 # Ensure final text doesn't exceed limit even without indicator
                 final_msg_text = final_msg_text[:MAX_EMBED_DESCRIPTION_LENGTH + len(STREAMING_INDICATOR)]

                 embed.description = final_msg_text or "..."
                 embed.color = EMBED_COLOR_COMPLETE
                 try:
                     last_msg = response_msgs[final_msg_index]
                     needs_edit = False
                     current_description = last_msg.embeds[0].description if last_msg.embeds else ""
                     current_color = last_msg.embeds[0].color if last_msg.embeds else None
                     current_view_exists = bool(last_msg.components)

                     # Check if edit is needed: view changed, content changed, or color changed
                     if (final_view and not current_view_exists) or (not final_view and current_view_exists): needs_edit = True
                     elif current_description != embed.description or current_color != embed.color: needs_edit = True
                     elif not last_msg.embeds: needs_edit = True # Should not happen, but safety check

                     if needs_edit:
                         await last_msg.edit(embed=embed, view=final_view)

                     # Store full text in the last node for embed responses
                     if last_msg.id in msg_nodes:
                         msg_nodes[last_msg.id].full_response_text = final_text

                 except discord.HTTPException as e: logging.error(f"Failed final edit on message {final_msg_index}: {e}")
                 except IndexError: logging.error(f"IndexError during final edit for index {final_msg_index}, response_msgs len: {len(response_msgs)}")

            elif not use_plain_responses and not response_msgs: # Handle empty successful response
                 embed.description = "..."
                 embed.color = EMBED_COLOR_COMPLETE
                 response_msg = await new_msg.reply(embed=embed, view=final_view, mention_author = False)
                 response_msgs.append(response_msg)
                 # Create node and acquire lock immediately
                 msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                 node = msg_nodes[response_msg.id]
                 await node.lock.acquire()
                 # Store full text (which is empty or just "...")
                 node.full_response_text = final_text


    except Exception as outer_e:
        # Catch any unexpected errors in the main processing block
        logging.exception("Unhandled error during message processing.") # Use exception for traceback
        try:
            await new_msg.reply(f"⚠️ An unexpected error occurred: {type(outer_e).__name__}", mention_author = False)
        except discord.HTTPException:
            pass # Ignore if we can't even reply

    finally: # --- Cleanup and Cache Management --- (Associated with the main try)
        # Release locks for all response messages created in this run
        logging.debug(f"Entering finally block. response_msgs count: {len(response_msgs)}")
        for response_msg in response_msgs:
            if response_msg and response_msg.id in msg_nodes: # Check if response_msg is not None
                node = msg_nodes[response_msg.id]
                # Text/full_response_text is now stored during the success path
                if node.lock.locked():
                    try:
                        logging.debug(f"Releasing lock for message node {response_msg.id}")
                        node.lock.release()
                    except RuntimeError:
                        logging.warning(f"Attempted to release an already unlocked lock for node {response_msg.id}")
                        pass
                else:
                     logging.debug(f"Lock for message node {response_msg.id} was not locked in finally block.")
            elif response_msg:
                 logging.warning(f"Response message {response_msg.id} not found in msg_nodes during cleanup.")

        # Delete oldest MsgNodes from the cache
        if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
            nodes_to_delete = sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]
            logging.info(f"Cache limit reached ({num_nodes}/{MAX_MESSAGE_NODES}). Removing {len(nodes_to_delete)} oldest nodes.")
            for msg_id in nodes_to_delete:
                # Before popping, ensure the lock is released if it exists and is locked
                node_to_delete = msg_nodes.get(msg_id)
                if node_to_delete and hasattr(node_to_delete, 'lock') and node_to_delete.lock.locked():
                    try:
                        node_to_delete.lock.release()
                        logging.debug(f"Released lock for node {msg_id} before cache eviction.")
                    except RuntimeError:
                        pass # Ignore if already unlocked
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
        # Close all database connections on shutdown
        logging.info("Closing database connections...")
        for manager in db_managers.values():
            manager.close()
        logging.info("Database connections closed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
