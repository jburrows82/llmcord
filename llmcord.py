import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
from typing import Literal, Optional, List, Dict, Any, Tuple, Union
import io
import re # Import re module
import urllib.parse

import asyncpraw
import discord
from discord import ui
import httpx
from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError, APIConnectionError # Import specific OpenAI errors
from google import genai as google_genai
from google.genai import types as google_types
from googleapiclient.discovery import build as build_google_api_client
from googleapiclient.errors import HttpError
from asyncprawcore.exceptions import NotFound, Redirect, Forbidden, RequestException as AsyncPrawRequestException
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yaml
from bs4 import BeautifulSoup # Added for general URL scraping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# --- Constants and Configuration ---
VISION_MODEL_TAGS = ("gpt-4", "o3", "o4", "claude-3", "gemini", "gemma", "llama", "pixtral", "mistral-small", "vision", "vl", "flash")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

# Gemini safety settings (BLOCK_NONE for all categories)
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
MAX_URL_CONTENT_LENGTH = 100000 # Limit for scraped web content per URL

# --- URL Regex Patterns ---
# General URL pattern (simplified, might need refinement for edge cases)
GENERAL_URL_PATTERN = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
# YouTube URL regex (captures full URL and video ID)
YOUTUBE_URL_PATTERN = re.compile(r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11}))')
# Reddit URL regex (captures full URL and submission ID)
REDDIT_URL_PATTERN = re.compile(r'(https?://(?:www\.)?reddit\.com/r/[a-zA-Z0-9_]+/comments/([a-zA-Z0-9]+))')
# Pattern to detect "at ai" as whole words, case-insensitive
AT_AI_PATTERN = re.compile(r'\bat ai\b', re.IGNORECASE)

# --- Initialization ---
ytt_api = YouTubeTranscriptApi() # Initialize youtube-transcript-api client

def get_config(filename="config.yaml"):
    try:
        with open(filename, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"CRITICAL: {filename} not found. Please copy config-example.yaml to {filename} and configure it.")
        exit()
    except yaml.YAMLError as e:
        logging.error(f"CRITICAL: Error parsing {filename}: {e}")
        exit()

cfg = get_config()
youtube_api_key = cfg.get("youtube_api_key")
reddit_client_id = cfg.get("reddit_client_id")
reddit_client_secret = cfg.get("reddit_client_secret")
reddit_user_agent = cfg.get("reddit_user_agent")

if not cfg.get("bot_token"):
    logging.error("CRITICAL: bot_token is not set in config.yaml")
    exit()

if client_id := cfg.get("client_id"): # Use .get() for safety
    logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")
else:
    logging.warning("client_id not found in config.yaml. Cannot generate invite URL.")

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(cfg.get("status_message") or "github.com/jakobdylanc/llmcord")[:128]) # Use .get()
discord_client = discord.Client(intents=intents, activity=activity)

httpx_client = httpx.AsyncClient(timeout=20.0, follow_redirects=True) # Increased timeout, enable redirects

msg_nodes = {}
last_task_time = 0

# --- Data Classes ---
@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)
    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None
    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False
    parent_msg: Optional[discord.Message] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

@dataclass
class UrlFetchResult:
    url: str
    content: Optional[Union[str, Dict[str, Any]]] # str for general, dict for YT/Reddit
    error: Optional[str] = None
    type: Literal["youtube", "reddit", "general"] = "general"
    original_index: int = -1 # To preserve order

# --- Discord UI ---
class SourcesView(ui.View):
    def __init__(self, grounding_metadata, timeout=300):
        super().__init__(timeout=timeout)
        self.grounding_metadata = grounding_metadata
        self.message = None # Will be set after sending the message

    @ui.button(label="Show Sources", style=discord.ButtonStyle.grey)
    async def show_sources_button(self, interaction: discord.Interaction, button: ui.Button):
        if not self.grounding_metadata:
            await interaction.response.send_message("No grounding metadata available.", ephemeral=True)
            return

        embed = discord.Embed(title="Grounding Sources", color=EMBED_COLOR_COMPLETE)
        field_count = 0

        # Add Search Queries field (usually short)
        if queries := getattr(self.grounding_metadata, 'web_search_queries', None):
            query_text = "\n".join(f"- `{q}`" for q in queries)
            if len(query_text) <= MAX_EMBED_FIELD_VALUE_LENGTH and field_count < MAX_EMBED_FIELDS:
                embed.add_field(name="Search Queries Used", value=query_text, inline=False)
                field_count += 1
            else:
                logging.warning("Search query list too long for embed field.")
                # Optionally handle very long query lists here (e.g., truncate or split)

        # Add Sources Consulted field(s), splitting if necessary
        if chunks := getattr(self.grounding_metadata, 'grounding_chunks', None):
            current_field_value = ""
            field_title = "Sources Consulted"
            sources_added = 0

            for chunk in chunks:
                # Check for web chunk structure
                web_chunk = getattr(chunk, 'web', None)
                if web_chunk and hasattr(web_chunk, 'title') and hasattr(web_chunk, 'uri'):
                    title = web_chunk.title or "Source" # Fallback title
                    uri = web_chunk.uri
                    source_line = f"- [{title}]({uri})\n"

                    # Check if adding this line exceeds the limit for the current field
                    if len(current_field_value) + len(source_line) > MAX_EMBED_FIELD_VALUE_LENGTH:
                        # Add the current field if it has content
                        if current_field_value and field_count < MAX_EMBED_FIELDS:
                            embed.add_field(name=field_title, value=current_field_value, inline=False)
                            field_count += 1
                            field_title = "Sources Consulted (cont.)" # Change title for subsequent fields
                        elif field_count >= MAX_EMBED_FIELDS:
                             logging.warning("Max embed fields reached while adding sources.")
                             break # Stop adding sources if max fields reached

                        # Start a new field, checking if the single line itself is too long
                        if len(source_line) <= MAX_EMBED_FIELD_VALUE_LENGTH:
                            current_field_value = source_line
                        else:
                            # Handle case where a single source line is too long (e.g., truncate)
                            truncated_line = source_line[:MAX_EMBED_FIELD_VALUE_LENGTH-4] + "...\n"
                            current_field_value = truncated_line
                            logging.warning(f"Single source line truncated: {source_line}")

                    else:
                        # Add the line to the current field
                        current_field_value += source_line
                    sources_added += 1

            # Add the last field if it has content and we haven't hit the limit
            if current_field_value and field_count < MAX_EMBED_FIELDS:
                embed.add_field(name=field_title, value=current_field_value, inline=False)
                field_count += 1

            if sources_added == 0 and not embed.fields: # If no web chunks were found and no queries
                 embed.description = "No web sources or search queries found in metadata."

        # If somehow still no fields (e.g., queries were too long and no sources)
        if not embed.fields and not embed.description:
             # Fallback: Show raw metadata if possible, else generic message
             try:
                 metadata_str = str(self.grounding_metadata) # Or use a specific formatting method if available
                 embed.description = f"```json\n{metadata_str[:MAX_EMBED_DESCRIPTION_LENGTH-10]}\n```" # Truncate raw data
             except Exception:
                 embed.description = "Could not display raw grounding metadata."

        # Check if embed is empty before sending
        if not embed.fields and not embed.description and not embed.title:
             await interaction.response.send_message("Could not extract source information.", ephemeral=True)
        else:
            await interaction.response.send_message(embed=embed, ephemeral=True)


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
            full_transcript = " ".join([entry['text'] for entry in fetched_transcript]) # Use dict access for fetched
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
                details["description"] = snippet.get("description", "N/A")
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
                maxResults=10, # Reduced comment count
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
        return UrlFetchResult(url=url, error="Could not extract video ID.", type="youtube", original_index=index)

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
        return UrlFetchResult(url=url, error="Reddit API credentials not configured.", type="reddit", original_index=index)

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
            content_data["selftext"] = submission.selftext

        # Fetch top comments
        top_comments_text = []
        comment_limit = 10
        await submission.comments.replace_more(limit=0) # Load only top-level comments

        comment_count = 0
        for top_level_comment in submission.comments.list():
            if comment_count >= comment_limit:
                break
            # Check if comment exists and is not deleted/removed before accessing body
            if hasattr(top_level_comment, 'body') and top_level_comment.body and top_level_comment.body not in ('[deleted]', '[removed]'):
                comment_body_cleaned = top_level_comment.body.replace('\n', ' ').replace('\r', '')
                top_comments_text.append(comment_body_cleaned)
                comment_count += 1

        if top_comments_text:
            content_data["comments"] = top_comments_text

        return UrlFetchResult(url=url, content=content_data, type="reddit", original_index=index)

    except (NotFound, Redirect):
        return UrlFetchResult(url=url, error="Submission not found or invalid URL.", type="reddit", original_index=index)
    except Forbidden as e:
         logging.warning(f"Reddit API Forbidden error for {url}: {e}")
         return UrlFetchResult(url=url, error="Reddit API access forbidden.", type="reddit", original_index=index)
    except AsyncPrawRequestException as e:
        logging.warning(f"Reddit API Request error for {url}: {e}")
        return UrlFetchResult(url=url, error=f"Reddit API request error: {type(e).__name__}", type="reddit", original_index=index)
    except Exception as e:
        logging.exception(f"Unexpected error fetching Reddit content for {url}")
        return UrlFetchResult(url=url, error=f"Unexpected error: {type(e).__name__}", type="reddit", original_index=index)
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
                     return UrlFetchResult(url=url, error=f"Too many redirects ({response.status_code}).", type="general", original_index=index)
                 return UrlFetchResult(url=url, error=f"HTTP status {response.status_code}.", type="general", original_index=index)

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                return UrlFetchResult(url=url, error=f"Unsupported content type: {content_type}", type="general", original_index=index)

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
                 return UrlFetchResult(url=url, error="Timeout while reading content.", type="general", original_index=index)
            except Exception as e:
                 logging.warning(f"Error decoding content for {url}: {e}")
                 return UrlFetchResult(url=url, error=f"Content decoding error: {type(e).__name__}", type="general", original_index=index)


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
            return UrlFetchResult(url=url, error="No text content found.", type="general", original_index=index)

        # Limit content length
        content = text[:MAX_URL_CONTENT_LENGTH] + ('...' if len(text) > MAX_URL_CONTENT_LENGTH else '')

        return UrlFetchResult(url=url, content=content, type="general", original_index=index)

    except httpx.RequestError as e:
        logging.warning(f"HTTPX RequestError fetching {url}: {type(e).__name__}")
        return UrlFetchResult(url=url, error=f"Request failed: {type(e).__name__}", type="general", original_index=index)
    except Exception as e:
        logging.exception(f"Unexpected error fetching general URL {url}")
        return UrlFetchResult(url=url, error=f"Unexpected error: {type(e).__name__}", type="general", original_index=index)


# --- Discord Event Handler ---
@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time, cfg, youtube_api_key, reddit_client_id, reddit_client_secret, reddit_user_agent

    # --- Basic Checks and Trigger ---
    if new_msg.author.bot:
        return

    is_dm = new_msg.channel.type == discord.ChannelType.private
    allow_dms = cfg.get("allow_dms", True)

    # Determine if the bot should process this message
    should_process = False
    mentions_bot = False # Initialize
    contains_at_ai = False # Initialize

    if is_dm:
        if allow_dms:
            should_process = True
            # Check if user explicitly mentioned bot or "at ai" in DM to start new chain
            mentions_bot = discord_client.user in new_msg.mentions
            contains_at_ai = AT_AI_PATTERN.search(new_msg.content) is not None
        else:
            return # Block DMs if not allowed
    else: # In a channel
        mentions_bot = discord_client.user in new_msg.mentions
        contains_at_ai = AT_AI_PATTERN.search(new_msg.content) is not None
        if mentions_bot or contains_at_ai:
            should_process = True

    if not should_process:
        return

    # --- Reload config ---
    cfg = get_config()
    youtube_api_key = cfg.get("youtube_api_key")
    reddit_client_id = cfg.get("reddit_client_id")
    reddit_client_secret = cfg.get("reddit_client_secret")
    reddit_user_agent = cfg.get("reddit_user_agent")

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

    # --- LLM Client Initialization ---
    provider_slash_model = cfg.get("model", "openai/gpt-4.1") # Default model
    provider, model = provider_slash_model.split("/", 1)
    provider_config = cfg.get("providers", {}).get(provider, {})
    base_url = provider_config.get("base_url")
    api_key = provider_config.get("api_key")

    is_gemini = provider == "google"
    gemini_client = None
    openai_client = None

    try:
        if is_gemini:
            if not api_key:
                raise ValueError("Google API key is missing in config.yaml for the 'google' provider.")
            gemini_client = google_genai.Client(api_key=api_key)
        else:
            openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key or "sk-no-key-required")
    except Exception as e:
        logging.exception(f"Failed to initialize LLM client for provider '{provider}'")
        await new_msg.reply(f"⚠️ Failed to configure LLM API: {e}", silent=True)
        return

    # --- Configuration Values ---
    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
    max_text = cfg.get("max_text", 100000)
    max_images = cfg.get("max_images", 5) if accept_images else 0
    max_messages = cfg.get("max_messages", 25)
    use_plain_responses = cfg.get("use_plain_responses", False)
    split_limit = MAX_EMBED_DESCRIPTION_LENGTH if not use_plain_responses else 2000

    # --- URL Extraction and Categorization ---
    user_warnings = set()
    all_urls_with_indices = extract_urls_with_indices(new_msg.content)
    fetch_tasks = []
    processed_urls = set() # Avoid processing duplicates

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
                # Get original content for cleaning
                original_content = curr_msg.content
                cleaned_content = original_content

                # Clean mentions from all messages (for consistency with original logic)
                is_dm_current = curr_msg.channel.type == discord.ChannelType.private
                if not is_dm_current and discord_client.user.mentioned_in(curr_msg):
                     cleaned_content = cleaned_content.replace(discord_client.user.mention, '').strip()

                # Remove "at ai" ONLY from the new message that triggered the bot
                if curr_msg.id == new_msg.id:
                     # Replace "at ai" (case-insensitive, whole word) with a single space
                     cleaned_content = AT_AI_PATTERN.sub(' ', cleaned_content)
                     # Replace multiple spaces (possibly resulting from removal) with a single space
                     cleaned_content = re.sub(r'\s{2,}', ' ', cleaned_content).strip()

                # Process attachments
                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text/", "image/"))]
                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments], return_exceptions=True)

                # Combine text content using the cleaned content
                text_parts = [cleaned_content] if cleaned_content else []
                text_parts.extend(filter(None, (embed.title for embed in curr_msg.embeds)))
                text_parts.extend(filter(None, (embed.description for embed in curr_msg.embeds)))
                text_parts.extend(filter(None, (getattr(embed.footer, 'text', None) for embed in curr_msg.embeds)))

                # Add text from attachments
                for att, resp in zip(good_attachments, attachment_responses):
                    if isinstance(resp, httpx.Response) and resp.status_code == 200 and att.content_type.startswith("text/"):
                        try:
                            # Limit attachment text size
                            attachment_text = resp.text[:max_text // 2] # Limit text attachment size
                            if len(resp.text) > max_text // 2:
                                attachment_text += "..."
                                user_warnings.add(f"⚠️ Truncated text attachment: {att.filename}")
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
                            image_parts.append(google_types.Part.from_bytes(data=resp.content, mime_type=att.content_type))
                        else:
                            image_parts.append(dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}")))
                    elif isinstance(resp, Exception):
                        # Already logged warning above
                        curr_node.has_bad_attachments = True

                curr_node.images = image_parts
                curr_node.role = "assistant" if curr_msg.author == discord_client.user else "user"
                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
                curr_node.has_bad_attachments = curr_node.has_bad_attachments or (len(curr_msg.attachments) > len(good_attachments))

                # Find parent message
                try:
                    parent_msg_obj = None
                    # Check if the current message explicitly triggers the bot (mention or "at ai")
                    # Use the variables calculated at the start of on_message if curr_msg is new_msg
                    if curr_msg.id == new_msg.id:
                        mentions_bot_in_current = mentions_bot
                        contains_at_ai_in_current = contains_at_ai
                    else: # Recalculate for older messages in the chain if needed (though unlikely needed here)
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


            # Prepare parts for the current message node for the LLM API
            current_text_content = curr_node.text[:max_text] if curr_node.text else ""
            current_images = curr_node.images[:max_images]

            parts_for_api = []
            if is_gemini:
                if current_text_content:
                    parts_for_api.append(google_types.Part.from_text(text=current_text_content))
                parts_for_api.extend(current_images)
            else: # OpenAI format
                if current_text_content:
                    parts_for_api.append({"type": "text", "text": current_text_content})
                parts_for_api.extend(current_images) # These are already dicts

            # Add to history if parts exist
            if parts_for_api:
                message_data = {
                    "role": "model" if curr_node.role == "assistant" else "user" # Gemini roles
                }
                if is_gemini:
                    # Ensure parts_for_api is always a list for Gemini
                    message_data["parts"] = parts_for_api if isinstance(parts_for_api, list) else [parts_for_api]
                else:
                    message_data["role"] = curr_node.role # OpenAI roles
                    message_data["content"] = parts_for_api
                    # Add name field if supported by provider
                    if provider in PROVIDERS_SUPPORTING_USERNAMES and curr_node.user_id is not None:
                        message_data["name"] = str(curr_node.user_id)

                history.append(message_data)

            # Add warnings based on limits and errors for this specific node
            if curr_node.text and len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} chars/msg")
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

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, history length: {len(history)}):\n{new_msg.content}")

    # --- Fetch External Content Concurrently ---
    url_fetch_results = []
    if fetch_tasks:
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                # This shouldn't happen if individual fetchers handle errors, but good fallback
                logging.error(f"Unhandled exception during URL fetch: {result}")
                user_warnings.add("⚠️ Unhandled error fetching URL content")
            elif isinstance(result, UrlFetchResult):
                url_fetch_results.append(result)
                if result.error:
                    # Shorten URL in warning
                    short_url = result.url[:40] + "..." if len(result.url) > 40 else result.url
                    user_warnings.add(f"⚠️ Error fetching {result.type} URL ({short_url}): {result.error}")
            else:
                 logging.error(f"Unexpected result type from URL fetch: {type(result)}")


    # --- Format External Content ---
    context_to_append = ""
    if url_fetch_results:
        # Sort results by their original position in the message
        url_fetch_results.sort(key=lambda r: r.original_index)

        formatted_parts = []
        url_counter = 1
        for result in url_fetch_results:
            if result.content: # Only include successful fetches
                content_str = f"\nurl {url_counter}: {result.url}\n"
                content_str += f"url {url_counter} content:\n"
                if result.type == "youtube":
                    content_str += f"  title: {result.content.get('title', 'N/A')}\n"
                    content_str += f"  channel: {result.content.get('channel_name', 'N/A')}\n"
                    desc = result.content.get('description', 'N/A')
                    content_str += f"  description: {desc[:500]}{'...' if len(desc) > 500 else ''}\n"
                    transcript = result.content.get('transcript')
                    if transcript:
                        content_str += f"  transcript: {transcript[:MAX_URL_CONTENT_LENGTH]}{'...' if len(transcript) > MAX_URL_CONTENT_LENGTH else ''}\n"
                    comments = result.content.get("comments")
                    if comments:
                        content_str += f"  top comments:\n" + "\n".join([f"    - {c[:150]}{'...' if len(c) > 150 else ''}" for c in comments[:5]]) + "\n" # Limit comments shown
                elif result.type == "reddit":
                    content_str += f"  title: {result.content.get('title', 'N/A')}\n"
                    selftext = result.content.get('selftext')
                    if selftext:
                        content_str += f"  content: {selftext[:MAX_URL_CONTENT_LENGTH]}{'...' if len(selftext) > MAX_URL_CONTENT_LENGTH else ''}\n"
                    comments = result.content.get("comments")
                    if comments:
                        content_str += f"  top comments:\n" + "\n".join([f"    - {c[:150]}{'...' if len(c) > 150 else ''}" for c in comments[:5]]) + "\n" # Limit comments shown
                elif result.type == "general":
                    # Content is already limited string
                    content_str += f"  {result.content}\n"

                formatted_parts.append(content_str)
                url_counter += 1

        if formatted_parts:
            context_to_append = "Answer the user's query based on the following:\n" + "".join(formatted_parts)


    # --- Prepare API Call ---
    history_for_llm = history[::-1] # Reverse history for correct chronological order

    # Prepend external content context if available
    if context_to_append:
        context_header = context_to_append + "\n\nUser's query:\n" # Use the formatted context
        if history_for_llm and history_for_llm[0]['role'] == 'user':
            first_user_msg = history_for_llm[0]
            if is_gemini:
                # Find or add text part in Gemini message
                text_part_found = False
                # Ensure 'parts' exists and is a list
                if 'parts' not in first_user_msg or not isinstance(first_user_msg['parts'], list):
                    first_user_msg['parts'] = []

                for part in first_user_msg['parts']:
                    if hasattr(part, 'text') and part.text is not None:
                        part.text = context_header + part.text
                        text_part_found = True
                        break
                if not text_part_found:
                    first_user_msg['parts'].insert(0, google_types.Part.from_text(text=context_header))
            else: # OpenAI
                # Ensure 'content' exists before modifying
                if 'content' not in first_user_msg:
                    first_user_msg['content'] = []

                # Find or add text part in OpenAI message content list
                if isinstance(first_user_msg['content'], list):
                    text_part_found = False
                    for part in first_user_msg['content']:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            part['text'] = context_header + part.get('text', '')
                            text_part_found = True
                            break
                    if not text_part_found:
                        first_user_msg['content'].insert(0, {'type': 'text', 'text': context_header})
                elif isinstance(first_user_msg['content'], str): # Handle case where content is just a string
                     first_user_msg['content'] = context_header + first_user_msg['content']
                else: # Fallback: create content list if it's missing or wrong type
                     first_user_msg['content'] = [{'type': 'text', 'text': context_header}]

        else:
            # Insert context as a new user message if history is empty or starts with assistant/model
            new_user_message_role = 'user'
            if is_gemini:
                 history_for_llm.insert(0, {'role': new_user_message_role, 'parts': [google_types.Part.from_text(text=context_header)]})
            else:
                 history_for_llm.insert(0, {'role': new_user_message_role, 'content': context_header})


    # System Prompt
    system_prompt_text = None
    if system_prompt := cfg.get("system_prompt"):
        system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
        if not is_gemini and provider in PROVIDERS_SUPPORTING_USERNAMES:
            system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")
        system_prompt_text = "\n".join([system_prompt] + system_prompt_extras)

    # API Arguments
    api_config = None
    api_content_kwargs = {}

    if is_gemini:
        gemini_contents = []
        for msg in history_for_llm:
            role = msg["role"] # Already 'user' or 'model'
            parts = msg.get("parts", []) # Default to empty list if parts missing
            # Ensure parts is a list, even if empty
            if not isinstance(parts, list):
                 logging.warning(f"Correcting non-list parts for Gemini message: {parts}")
                 # Convert non-list to text part or empty list
                 parts = [google_types.Part.from_text(text=str(parts))] if parts else []
            gemini_contents.append(google_types.Content(role=role, parts=parts))

        api_content_kwargs["contents"] = gemini_contents

        gemini_extra_params = cfg.get("extra_api_parameters", {}).copy()
        if "max_tokens" in gemini_extra_params:
            gemini_extra_params["max_output_tokens"] = gemini_extra_params.pop("max_tokens")

        gemini_safety_settings_list = [
            google_types.SafetySetting(category=category, threshold=threshold)
            for category, threshold in GEMINI_SAFETY_SETTINGS_DICT.items()
        ]

        api_config = google_types.GenerateContentConfig(
            **gemini_extra_params,
            safety_settings=gemini_safety_settings_list,
            tools=[google_types.Tool(google_search=google_types.GoogleSearch())] # Enable grounding
        )
        if system_prompt_text:
             api_config.system_instruction = google_types.Part.from_text(text=system_prompt_text)

    else: # OpenAI
        openai_messages = history_for_llm
        if system_prompt_text:
            openai_messages.insert(0, dict(role="system", content=system_prompt_text))

        api_content_kwargs["messages"] = openai_messages
        api_config = cfg.get("extra_api_parameters", {}).copy()
        api_config["stream"] = True # Always stream for OpenAI

    # --- Generate and Send Response ---
    curr_content = finish_reason = edit_task = None
    grounding_metadata = None
    response_msgs = []
    response_contents = []
    final_text = "" # Initialize final_text
    view = None # Initialize view

    embed = discord.Embed()
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    try:
        async with new_msg.channel.typing():
            stream_response = None
            if is_gemini:
                if not gemini_client: raise ValueError("Gemini client not initialized")
                stream_response = await gemini_client.aio.models.generate_content_stream(
                    model=model,
                    contents=api_content_kwargs["contents"],
                    config=api_config
                )
            else:
                if not openai_client: raise ValueError("OpenAI client not initialized")
                stream_response = await openai_client.chat.completions.create(
                    model=model,
                    messages=api_content_kwargs["messages"],
                    **api_config
                )

            # --- Stream Processing Loop ---
            async for chunk in stream_response:
                new_content_chunk = ""
                chunk_finish_reason = None
                chunk_grounding_metadata = None

                if is_gemini:
                    # Extract Gemini data
                    if hasattr(chunk, 'text') and chunk.text:
                        new_content_chunk = chunk.text
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                         candidate = chunk.candidates[0]
                         if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                              # Map Gemini finish reason if needed, default to 'stop' if successful
                              reason_map = {
                                   google_types.FinishReason.STOP: "stop",
                                   google_types.FinishReason.MAX_TOKENS: "length",
                                   google_types.FinishReason.SAFETY: "safety",
                                   google_types.FinishReason.RECITATION: "recitation",
                                   # Add other mappings as needed
                              }
                              # Use FINISH_REASON_UNSPECIFIED as a successful stop condition too
                              chunk_finish_reason = reason_map.get(candidate.finish_reason, "stop" if candidate.finish_reason in (google_types.FinishReason.FINISH_REASON_UNSPECIFIED, google_types.FinishReason.STOP) else str(candidate.finish_reason))
                         if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                              chunk_grounding_metadata = candidate.grounding_metadata
                else: # OpenAI
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        chunk_finish_reason = chunk.choices[0].finish_reason
                        if delta and delta.content:
                            new_content_chunk = delta.content

                # Update overall finish reason and grounding metadata
                if chunk_finish_reason:
                    finish_reason = chunk_finish_reason
                if chunk_grounding_metadata:
                    grounding_metadata = chunk_grounding_metadata # Keep the latest

                # Append content if not empty
                if new_content_chunk:
                    response_contents.append(new_content_chunk)

                # --- Real-time Editing Logic (Common for both) ---
                if not use_plain_responses:
                    current_full_text = "".join(response_contents)
                    if not current_full_text and not finish_reason: # Skip empty intermediate chunks
                         continue

                    # Create view if grounding metadata exists *during* streaming
                    if view is None and grounding_metadata and (getattr(grounding_metadata, 'web_search_queries', None) or getattr(grounding_metadata, 'grounding_chunks', None)):
                        view = SourcesView(grounding_metadata)

                    current_msg_index = (len(current_full_text) - 1) // split_limit if current_full_text else 0
                    start_next_msg = current_msg_index >= len(response_msgs)

                    ready_to_edit = (edit_task is None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                    is_final_chunk = finish_reason is not None

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
                        is_successful_finish = finish_reason and finish_reason.lower() in ("stop", "end_turn") # Define success
                        embed.color = EMBED_COLOR_COMPLETE if is_final_chunk and is_successful_finish else EMBED_COLOR_INCOMPLETE

                        # Attach view only if it's the final chunk and grounding exists
                        view_to_attach = view if is_final_chunk else None

                        # Create or Edit the current message
                        if start_next_msg:
                            reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                            response_msg = await reply_to_msg.reply(embed=embed, view=view_to_attach, silent=True)
                            response_msgs.append(response_msg)
                            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()
                        elif response_msgs and current_msg_index < len(response_msgs):
                            edit_task = asyncio.create_task(response_msgs[current_msg_index].edit(embed=embed, view=view_to_attach))
                        elif not response_msgs and is_final_chunk: # Handle case where response is short and finishes immediately
                             reply_to_msg = new_msg
                             response_msg = await reply_to_msg.reply(embed=embed, view=view_to_attach, silent=True)
                             response_msgs.append(response_msg)
                             msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                             await msg_nodes[response_msg.id].lock.acquire()


                        last_task_time = dt.now().timestamp()

                # Break loop if finished
                if finish_reason:
                    break
            # --- End Stream Processing Loop ---

            final_text = "".join(response_contents)

            # Create view if grounding metadata exists (final check)
            if view is None and grounding_metadata and (getattr(grounding_metadata, 'web_search_queries', None) or getattr(grounding_metadata, 'grounding_chunks', None)):
                view = SourcesView(grounding_metadata)

            # Handle plain text responses
            if use_plain_responses:
                 final_messages_content = [final_text[i:i+2000] for i in range(0, len(final_text), 2000)]
                 if not final_messages_content: # Handle empty response
                     final_messages_content.append("...")

                 temp_response_msgs = []
                 for i, content in enumerate(final_messages_content):
                     reply_to_msg = new_msg if not temp_response_msgs else temp_response_msgs[-1]
                     current_view = view if (i == len(final_messages_content) - 1) else None
                     response_msg = await reply_to_msg.reply(content=content or "...", suppress_embeds=True, view=current_view, silent=True)
                     temp_response_msgs.append(response_msg)
                     msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                     await msg_nodes[response_msg.id].lock.acquire()
                 response_msgs = temp_response_msgs

            # Final edit for embed messages (if not already handled by final chunk logic)
            elif not use_plain_responses and response_msgs:
                 if edit_task is not None and not edit_task.done():
                     await edit_task # Ensure last edit task completes

                 final_msg_index = len(response_msgs) - 1
                 final_msg_text = final_text[final_msg_index * split_limit : (final_msg_index + 1) * split_limit]
                 # Ensure final text doesn't exceed limit even without indicator
                 final_msg_text = final_msg_text[:MAX_EMBED_DESCRIPTION_LENGTH + len(STREAMING_INDICATOR)]

                 # Ensure the last message has the final content and view
                 embed.description = final_msg_text or "..."
                 embed.color = EMBED_COLOR_COMPLETE
                 try:
                     # Check if view needs to be added/updated or content/color changed
                     last_msg = response_msgs[final_msg_index]
                     needs_edit = False
                     if view and not last_msg.components:
                         needs_edit = True
                     elif not view and last_msg.components:
                          needs_edit = True
                     elif last_msg.embeds and (last_msg.embeds[0].description != embed.description or last_msg.embeds[0].color != embed.color):
                          needs_edit = True
                     elif not last_msg.embeds: # Should not happen, but safety check
                          needs_edit = True

                     if needs_edit:
                          await last_msg.edit(embed=embed, view=view)
                 except discord.HTTPException as e:
                      logging.error(f"Failed final edit on message {final_msg_index}: {e}")
                 except IndexError:
                      logging.error(f"IndexError during final edit for index {final_msg_index}, response_msgs len: {len(response_msgs)}")

            elif not use_plain_responses and not response_msgs:
                 # Handle case where response was empty and no initial message was sent
                 embed.description = "..."
                 embed.color = EMBED_COLOR_COMPLETE
                 response_msg = await new_msg.reply(embed=embed, view=view, silent=True)
                 response_msgs.append(response_msg)
                 msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                 await msg_nodes[response_msg.id].lock.acquire()


    # --- Error Handling ---
    except google_types.BlockedPromptError as e:
         logging.warning(f"Gemini Prompt Blocked: {e}")
         await new_msg.reply("⚠️ My prompt was blocked by safety filters.", silent=True)
    except google_types.StopCandidateError as e:
         logging.warning(f"Gemini Response Blocked: {e}")
         # Edit the last message to show it was blocked if possible
         if not use_plain_responses and response_msgs:
             try:
                 embed.description = (embed.description or "") + "\n\n⚠️ Response blocked by safety filters."
                 embed.description = embed.description.replace(STREAMING_INDICATOR, "").strip()[:MAX_EMBED_DESCRIPTION_LENGTH]
                 embed.color = discord.Color.red()
                 await response_msgs[-1].edit(embed=embed, view=None) # Remove view on block
             except Exception as edit_err:
                 logging.error(f"Failed to edit message to show safety block: {edit_err}")
                 await new_msg.reply("⚠️ My response was blocked by safety filters.", silent=True)
         else:
            await new_msg.reply("⚠️ My response was blocked by safety filters.", silent=True)
    except google_types.GoogleAPICallError as e:
        logging.exception("Google API Call Error")
        await new_msg.reply(f"⚠️ Google API Error: {e}", silent=True)
    except Exception as e: # Catch OpenAI errors and others
        logging.exception("Error during LLM call")
        error_message = f"⚠️ An error occurred: {type(e).__name__}"

        # Add more specific error details if possible and safe
        if hasattr(e, 'message'): error_message += f": {e.message}"
        elif hasattr(e, 'body') and isinstance(e.body, dict) and 'message' in e.body: error_message += f": {e.body['message']}" # OpenAI specific
        elif hasattr(e, 'args') and e.args: error_message += f": {e.args[0]}"

        # Check for specific OpenAI errors if applicable
        if not is_gemini and isinstance(e, APIError):
             if isinstance(e, RateLimitError): error_message = "⚠️ Rate limit exceeded. Please try again later."
             elif isinstance(e, AuthenticationError): error_message = "⚠️ Authentication error. Check your API key."
             elif isinstance(e, APIConnectionError): error_message = "⚠️ Could not connect to the API."
             # Add more specific OpenAI error checks if needed

        await new_msg.reply(error_message, silent=True)

    # --- Cleanup and Cache Management ---
    finally:
        # Release locks and store final text for all response messages
        for response_msg in response_msgs:
            if response_msg.id in msg_nodes:
                # Ensure node exists before accessing attributes
                node = msg_nodes[response_msg.id]
                node.text = final_text # Store aggregated final text
                if node.lock.locked():
                    try:
                        node.lock.release()
                    except RuntimeError: # Lock might already be released
                        pass


        # Delete oldest MsgNodes (lowest message IDs) from the cache
        if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
            nodes_to_delete = sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]
            logging.info(f"Cache limit reached ({num_nodes}/{MAX_MESSAGE_NODES}). Removing {len(nodes_to_delete)} oldest nodes.")
            for msg_id in nodes_to_delete:
                msg_nodes.pop(msg_id, None)
                # No need to acquire/release lock here, just pop


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

if __name__ == "__main__":
    asyncio.run(main())