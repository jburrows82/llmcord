import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
from typing import Literal, Optional, List, Dict, Any, Tuple
import io
import re
import urllib.parse

import asyncpraw
import discord
from discord import ui
import httpx
from openai import AsyncOpenAI
from google import genai as google_genai
# Corrected import: removed generative_models
from google.genai import types as google_types
from googleapiclient.discovery import build as build_google_api_client
from googleapiclient.errors import HttpError
from asyncprawcore.exceptions import NotFound, Redirect
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("gpt-4", "o3", "o4", "claude-3", "gemini", "gemma", "llama", "pixtral", "mistral-small", "vision", "vl", "flash") # Added flash for gemini
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

# Gemini safety settings mapping (Dictionary format is fine for definition)
GEMINI_SAFETY_SETTINGS_DICT = {
    google_types.HarmCategory.HARM_CATEGORY_HARASSMENT: google_types.HarmBlockThreshold.BLOCK_NONE,
    google_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: google_types.HarmBlockThreshold.BLOCK_NONE,
    google_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: google_types.HarmBlockThreshold.BLOCK_NONE,
    google_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: google_types.HarmBlockThreshold.BLOCK_NONE,
    # google_types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: google_types.HarmBlockThreshold.BLOCK_NONE, # Not supported by all models yet
}
MAX_MESSAGE_NODES = 500
MAX_EMBED_FIELD_VALUE_LENGTH = 1024
MAX_EMBED_FIELDS = 25
# Define Discord's embed description limit (leaving space for indicator)
MAX_EMBED_DESCRIPTION_LENGTH = 4096 - len(STREAMING_INDICATOR)

# YouTube URL regex
YOUTUBE_URL_PATTERN = re.compile(r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11}))')

# Reddit URL regex (captures submission ID)
REDDIT_URL_PATTERN = re.compile(r'(https?://(?:www\.)?reddit\.com/r/[a-zA-Z0-9_]+/comments/([a-zA-Z0-9]+))')

ytt_api = YouTubeTranscriptApi() # Initialize youtube-transcript-api client

def get_config(filename="config.yaml"):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


cfg = get_config()
youtube_api_key = cfg.get("youtube_api_key")
reddit_client_id = cfg.get("reddit_client_id")
reddit_client_secret = cfg.get("reddit_client_secret")
reddit_user_agent = cfg.get("reddit_user_agent")

if client_id := cfg["client_id"]:
    logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(cfg["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_client = discord.Client(intents=intents, activity=activity)

httpx_client = httpx.AsyncClient()

msg_nodes = {}
last_task_time = 0


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
                if hasattr(chunk, 'web') and chunk.web and hasattr(chunk.web, 'title') and hasattr(chunk.web, 'uri'):
                    source_line = f"- [{chunk.web.title}]({chunk.web.uri})\n"

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
             embed.description = f"```json\n{self.grounding_metadata}\n```" # Fallback

        # Check if embed is empty before sending
        if not embed.fields and not embed.description and not embed.title:
             await interaction.response.send_message("Could not extract source information.", ephemeral=True)
        else:
            await interaction.response.send_message(embed=embed, ephemeral=True)


def extract_youtube_urls(text: str) -> List[str]:
    """Extracts YouTube video URLs from text."""
    return [match[0] for match in YOUTUBE_URL_PATTERN.findall(text)]

def extract_reddit_urls(text: str) -> List[Tuple[str, str]]:
    """Extracts Reddit submission URLs and their IDs from text."""
    # Returns list of tuples: (full_url, submission_id)
    return [(match[0], match[1]) for match in REDDIT_URL_PATTERN.findall(text)]

def extract_video_id(url: str) -> Optional[str]:
    """Extracts the video ID from a YouTube URL."""
    match = YOUTUBE_URL_PATTERN.search(url)
    return match.group(2) if match else None

async def get_transcript(video_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetches the transcript for a video ID using youtube-transcript-api."""
    try:
        # Run synchronous library call in a separate thread
        transcript_list = await asyncio.to_thread(ytt_api.list_transcripts, video_id)
        # Prioritize manual transcripts, fallback to generated, prefer English
        transcript = None
        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except NoTranscriptFound:
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
            except NoTranscriptFound:
                 # Try finding *any* transcript if English isn't available
                 transcript = next(iter(transcript_list), None)

        if transcript:
            fetched_transcript = await asyncio.to_thread(transcript.fetch)
            # --- FIX: Use attribute access (.text) instead of dictionary access (['text']) ---
            full_transcript = " ".join([entry.text for entry in fetched_transcript])
            return full_transcript, None
        else:
            return None, "No suitable transcript found."
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return None, "No transcript found for this video."
    except Exception as e:
        # Log the actual exception type and message for better debugging
        logging.error(f"Error fetching transcript for {video_id}: {type(e).__name__}: {e}")
        return None, f"An error occurred while fetching the transcript: {type(e).__name__}"


async def get_youtube_video_details(video_id: str, api_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetches video title, description, channel name, and comments using YouTube Data API."""
    if not api_key:
        return None, "YouTube API key not configured."

    try:
        # Use asyncio.to_thread for the synchronous googleapiclient calls
        youtube = await asyncio.to_thread(build_google_api_client, 'youtube', 'v3', developerKey=api_key)

        # Get video details (snippet)
        video_request = youtube.videos().list(part="snippet", id=video_id)
        video_response = await asyncio.to_thread(video_request.execute)

        if not video_response.get("items"):
            return None, "Video not found or unavailable."

        snippet = video_response["items"][0]["snippet"]
        details = {
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "channel_name": snippet.get("channelTitle"),
        }

        # Get top comments (commentThreads)
        comment_request = youtube.commentThreads().list(part="snippet", videoId=video_id, order="relevance", maxResults=50, textFormat="plainText")
        comment_response = await asyncio.to_thread(comment_request.execute)
        details["comments"] = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in comment_response.get("items", [])]

        return details, None
    except HttpError as e:
        error_reason = e.reason if hasattr(e, 'reason') else str(e)
        logging.error(f"YouTube Data API error for {video_id}: {error_reason}")
        if e.resp.status == 403:
             if "quotaExceeded" in str(e):
                 return None, "YouTube API quota exceeded."
             else:
                 return None, f"YouTube API permission error: {error_reason}"
        elif e.resp.status == 404:
            return None, "Video not found via YouTube API."
        else:
            return None, f"YouTube Data API HTTP error: {error_reason}"
    except Exception as e:
        logging.exception(f"Unexpected error fetching YouTube details for {video_id}")
        return None, f"An unexpected error occurred: {e}"

async def get_reddit_content(urls_with_ids: List[Tuple[str, str]], client_id: str, client_secret: str, user_agent: str) -> Tuple[str, List[str]]:
    """Fetches content from multiple Reddit submission URLs concurrently."""
    if not all([client_id, client_secret, user_agent]):
        return "", ["Reddit API credentials not configured."]

    content_parts = []
    errors = []

    # Initialize asyncpraw.Reddit instance
    # Using read_only=True as we only need to fetch data
    reddit = asyncpraw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        read_only=True, # Set to True for read-only operations
    )

    async def fetch_single_submission(url: str, submission_id: str, index: int):
        nonlocal content_parts, errors
        try:
            # Use await for asyncpraw methods
            submission = await reddit.submission(id=submission_id) # No need for fetch=True, load() handles it
            await submission.load() # Ensure submission data and comments are loaded

            title = submission.title
            selftext = submission.selftext

            # Fetch top comments (limit to 10 for brevity)
            top_comments_text = []
            comment_limit = 10
            # Ensure comments are loaded before iterating
            await submission.comments.replace_more(limit=0) # Load top-level comments only

            # Iterate through the loaded comments
            comment_count = 0
            for top_level_comment in submission.comments.list():
                 if comment_count >= comment_limit:
                     break
                 # Check if comment body exists and is not deleted/removed
                 if hasattr(top_level_comment, 'body') and top_level_comment.body and top_level_comment.body not in ('[deleted]', '[removed]'):
                     # Limit comment length and replace newlines for cleaner output
                     comment_body_cleaned = top_level_comment.body.replace('\n', ' ').replace('\r', '')
                     top_comments_text.append(f"    - {comment_body_cleaned[:150]}...")
                     comment_count += 1


            content_str = f"\nreddit url {index+1}: {url}\n"
            content_str += f"reddit url {index+1} content:\n"
            content_str += f"  title: {title}\n"
            if selftext:
                content_str += f"  content: {selftext[:1000]}...\n" # Limit selftext length
            if top_comments_text:
                content_str += f"  top comments:\n" + "\n".join(top_comments_text) + "\n"
            content_parts.append((index, content_str)) # Store index for sorting
        except (NotFound, Redirect):
            errors.append(f"Reddit submission not found or invalid URL {index+1}: {url}")
        except Exception as e:
            logging.exception(f"Error fetching Reddit content for {url}")
            errors.append(f"Error fetching Reddit URL {index+1}: {type(e).__name__}")

    tasks = [fetch_single_submission(url, sub_id, i) for i, (url, sub_id) in enumerate(urls_with_ids)]
    await asyncio.gather(*tasks)

    # Close the Reddit session when done
    await reddit.close()

    # Sort content parts by index to maintain order
    content_parts.sort(key=lambda x: x[0])

    return "".join([part[1] for part in content_parts]), errors


@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_client.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    cfg = get_config()

    allow_dms = cfg["allow_dms"]
    permissions = cfg["permissions"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = cfg["model"]
    provider, model = provider_slash_model.split("/", 1)
    global youtube_api_key # Use the global key loaded earlier
    global reddit_client_id, reddit_client_secret, reddit_user_agent # Use global Reddit creds
    base_url = cfg["providers"][provider].get("base_url") # May be None for google
    # Reload API key in case config changed
    youtube_api_key = cfg.get("youtube_api_key")
    api_key = cfg["providers"][provider].get("api_key")

    is_gemini = provider == "google"
    gemini_client = None # Initialize gemini_client to None

    if not is_gemini:
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key or "sk-no-key-required")
    else:
        if not api_key:
            logging.error("Google API key is missing in config.yaml for the 'google' provider.")
            await new_msg.reply("⚠️ Configuration error: Google API key is missing.", silent=True)
            return
        try:
            # Initialize the client here instead of using configure
            gemini_client = google_genai.Client(api_key=api_key)
        except Exception as e:
            logging.exception("Failed to configure Google GenAI client")
            await new_msg.reply(f"⚠️ Failed to configure Google API: {e}", silent=True)
            return


    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)

    max_text = cfg["max_text"]
    max_images = cfg["max_images"] if accept_images else 0
    max_messages = cfg["max_messages"]

    use_plain_responses = cfg["use_plain_responses"]
    # Use the Discord embed description limit for splitting logic if using embeds
    # Otherwise, use the standard message limit
    split_limit = MAX_EMBED_DESCRIPTION_LENGTH if not use_plain_responses else 2000

    # --- YouTube URL Processing ---
    youtube_urls = extract_youtube_urls(new_msg.content)
    youtube_content_to_append = ""
    youtube_data_tasks = []
    youtube_fetch_errors = []
    user_warnings = set() # Initialize user_warnings here

    if youtube_urls:
        youtube_data_tasks = []
        processed_urls = set()
        unique_urls = []
        for url in youtube_urls:
            if url not in processed_urls:
                unique_urls.append(url)
                processed_urls.add(url)

        async def fetch_and_format_youtube_data(url: str, index: int):
            video_id = extract_video_id(url)
            if not video_id:
                return None, f"Could not extract video ID from URL {index+1}: {url}"

            # Fetch transcript and details concurrently
            transcript_task = asyncio.create_task(get_transcript(video_id))
            details_task = asyncio.create_task(get_youtube_video_details(video_id, youtube_api_key))

            transcript, transcript_error = await transcript_task
            details, details_error = await details_task

            errors = [err for err in [transcript_error, details_error] if err]

            content_str = f"\nyoutube url {index+1}: {url}\n"
            content_str += f"youtube url {index+1} content:\n"
            if details:
                content_str += f"  title: {details.get('title', 'N/A')}\n"
                content_str += f"  channel: {details.get('channel_name', 'N/A')}\n"
                content_str += f"  description: {details.get('description', 'N/A')[:500]}...\n" # Limit description length
            if transcript:
                content_str += f"  transcription: {transcript[:1000]}...\n" # Limit transcript length
            if details and details.get("comments"):
                content_str += f"  top comments:\n" + "\n".join([f"    - {c[:150]}..." for c in details['comments'][:10]]) + "\n" # Limit comments
            return content_str, errors

        youtube_data_tasks = [fetch_and_format_youtube_data(url, i) for i, url in enumerate(unique_urls)]

    # --- Reddit URL Processing ---
    reddit_urls_with_ids = extract_reddit_urls(new_msg.content)
    reddit_content_to_append = ""
    reddit_data_task = None
    reddit_fetch_errors = []

    if reddit_urls_with_ids:
        # Reload Reddit creds in case config changed
        reddit_client_id = cfg.get("reddit_client_id")
        reddit_client_secret = cfg.get("reddit_client_secret")
        reddit_user_agent = cfg.get("reddit_user_agent")
        if not all([reddit_client_id, reddit_client_secret, reddit_user_agent]):
            user_warnings.add("⚠️ Reddit API credentials not configured")
        else:
            reddit_data_task = asyncio.create_task(get_reddit_content(reddit_urls_with_ids, reddit_client_id, reddit_client_secret, reddit_user_agent))

    # Build message chain and set user warnings
    history = [] # Renamed from messages to avoid confusion with discord.Message
    # user_warnings = set() # Moved initialization up
    curr_msg = new_msg

    while curr_msg != None and len(history) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_client.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, getattr(embed.footer, 'text', None)))) for embed in curr_msg.embeds] # Added getattr for footer
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                if is_gemini:
                     curr_node.images = [
                        google_types.Part.from_bytes(data=resp.content, mime_type=att.content_type)
                        for att, resp in zip(good_attachments, attachment_responses)
                        if att.content_type.startswith("image")]
                else:
                    curr_node.images = [
                        dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                        for att, resp in zip(good_attachments, attachment_responses)
                        if att.content_type.startswith("image")
                    ]
                curr_node.role = "assistant" if curr_msg.author == discord_client.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_client.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in new_msg.channel.history(before=curr_msg, limit=1)] or [None])[0]) # Use new_msg.channel
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_client.user if is_dm else curr_msg.author) # Check author based on DM or not
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and hasattr(curr_msg.channel, 'parent') and curr_msg.channel.parent and curr_msg.channel.parent.type == discord.ChannelType.text # Check parent exists

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            # Prepare parts for the current message node
            if curr_node.images[:max_images]:
                if is_gemini:
                    # Use text= keyword argument
                    parts = ([google_types.Part.from_text(text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
                else:
                    parts = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                if is_gemini:
                    # Use text= keyword argument, ensure list even if empty text
                    parts = [google_types.Part.from_text(text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []
                else:
                    parts = curr_node.text[:max_text]


            if parts != "" and parts != []:
                message_data = dict(role=curr_node.role if not is_gemini else ("model" if curr_node.role == "assistant" else "user")) # Map roles for Gemini
                if is_gemini:
                    message_data["parts"] = parts
                else:
                    message_data["content"] = parts
                    if any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES) and curr_node.user_id is not None:
                        message_data["name"] = str(curr_node.user_id)
                history.append(message_data)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(history) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(history)} message{'' if len(history) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(history)}):\n{new_msg.content}")

    # --- Await External Content Fetching and Format ---
    if youtube_urls:
        youtube_results = await asyncio.gather(*youtube_data_tasks)
        formatted_youtube_content = []
        for content_str, errors in youtube_results:
            if content_str:
                formatted_youtube_content.append(content_str)
            if errors:
                youtube_fetch_errors.extend(errors)

        if formatted_youtube_content:
            youtube_content_to_append = "".join(formatted_youtube_content) # Append later with header

        if youtube_fetch_errors:
            user_warnings.add(f"⚠️ Could not fetch all YouTube data ({len(youtube_fetch_errors)} errors)")

    if reddit_data_task:
        reddit_content_to_append, reddit_fetch_errors = await reddit_data_task
        if reddit_fetch_errors:
            user_warnings.add(f"⚠️ Could not fetch all Reddit data ({len(reddit_fetch_errors)} errors)")

    # Combine external content
    context_to_append = ""
    if youtube_content_to_append or reddit_content_to_append:
        context_to_append = "Answer the user's query based on the following:\n" + youtube_content_to_append + reddit_content_to_append

    # Prepare API call arguments
    system_prompt_text = None
    if system_prompt := cfg["system_prompt"]:
        system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
        # Gemini doesn't support the 'name' field, so don't add username instructions for it
        if not is_gemini and any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES):
            system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")
        system_prompt_text = "\n".join([system_prompt] + system_prompt_extras)

    # Prepare kwargs for the API call, separating config/generation_config
    api_config = None # Initialize config object
    api_content_kwargs = {}

    # --- Prepend YouTube/Reddit content to the history/query ---
    # We'll prepend it as a separate user message before the actual user query
    # This keeps the structure clean and aligns with the example format.
    history_for_llm = history[::-1] # Reverse history for correct order

    if context_to_append:
        # Modify the *first* user message in the reversed history to include the context
        # Or, if the history starts with assistant, insert a new user message
        if history_for_llm and history_for_llm[0].get('role') == ('user' if not is_gemini else 'user'):
             # Prepend to the text part of the first user message
             first_user_msg_content = history_for_llm[0].get('content', '') if not is_gemini else history_for_llm[0].get('parts', [])
             prepended_content = context_to_append + "\n\nUser's query:\n" # Use combined context
             if isinstance(first_user_msg_content, str): # OpenAI string content
                 history_for_llm[0]['content'] = prepended_content + first_user_msg_content
             elif isinstance(first_user_msg_content, list): # OpenAI/Gemini list content
                 # Find the first text part or create one
                 text_part_found = False
                 for part in first_user_msg_content:
                     if isinstance(part, dict) and part.get('type') == 'text': # OpenAI text part
                         part['text'] = prepended_content + part.get('text', '')
                         text_part_found = True
                         break
                     # Corrected check using google_types.Part
                     elif isinstance(part, google_types.Part) and hasattr(part, 'text') and part.text is not None: # Gemini text part
                         part.text = prepended_content + part.text
                         text_part_found = True
                         break
                 if not text_part_found: # If no text part exists (e.g., image-only message), insert one
                     if is_gemini:
                         history_for_llm[0]['parts'].insert(0, google_types.Part.from_text(text=prepended_content))
                     else:
                         # Ensure 'content' is a list before inserting
                         if not isinstance(history_for_llm[0]['content'], list):
                             history_for_llm[0]['content'] = [] # Initialize as list if it wasn't
                         history_for_llm[0]['content'].insert(0, {'type': 'text', 'text': prepended_content})
        else:
            # If history is empty or starts with assistant, insert a new user message
            new_user_message_role = 'user' if not is_gemini else 'user'
            if is_gemini:
                 history_for_llm.insert(0, {'role': new_user_message_role, 'parts': [google_types.Part.from_text(text=context_to_append)]}) # Use combined context
            else:
                 history_for_llm.insert(0, {'role': new_user_message_role, 'content': context_to_append}) # Use combined context

    if is_gemini:
        # Convert potentially modified history to Gemini format
        gemini_contents = []
        for msg in history_for_llm: # Use the potentially modified history
            role = msg["role"] # Already mapped user/model
            parts = msg["parts"]
            gemini_contents.append(google_types.Content(role=role, parts=parts))

        api_content_kwargs["contents"] = gemini_contents

        # Prepare Gemini generation config, renaming max_tokens if necessary
        gemini_extra_params = cfg["extra_api_parameters"].copy()
        if "max_tokens" in gemini_extra_params:
            gemini_extra_params["max_output_tokens"] = gemini_extra_params.pop("max_tokens")

        # Convert safety settings dict to list of SafetySetting objects
        gemini_safety_settings_list = [
            google_types.SafetySetting(category=category, threshold=threshold)
            for category, threshold in GEMINI_SAFETY_SETTINGS_DICT.items()
        ]

        # Create the config object, passing generation params directly
        api_config = google_types.GenerateContentConfig(
            # Pass generation parameters directly using **
            **gemini_extra_params,
            # Pass the list of SafetySetting objects
            safety_settings=gemini_safety_settings_list,
            tools=[google_types.Tool(google_search=google_types.GoogleSearch())] # Enable grounding
        )
        if system_prompt_text:
             # Use text= keyword argument
             api_config.system_instruction = google_types.Part.from_text(text=system_prompt_text)

    else:
        # Use the potentially modified history for OpenAI
        openai_messages = history_for_llm
        if system_prompt_text:
            openai_messages.insert(0, dict(role="system", content=system_prompt_text)) # Insert system prompt at the beginning

        api_content_kwargs["messages"] = openai_messages
        # OpenAI config is passed directly as kwargs or extra_body
        api_config = cfg["extra_api_parameters"].copy()
        api_config["stream"] = True

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = edit_task = None
    grounding_metadata = None # Initialize grounding_metadata here
    response_msgs = []
    response_contents = [] # Store chunks here

    embed = discord.Embed()
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    try:
        async with new_msg.channel.typing():
            if is_gemini:
                # Use google-genai client initialized earlier
                if not gemini_client: # Should not happen if check at start passed
                     raise ValueError("Gemini client not initialized")

                # Use aio client for async streaming
                stream_response = await gemini_client.aio.models.generate_content_stream(
                    model=model,
                    contents=api_content_kwargs["contents"],
                    config=api_config # Pass the combined config object
                )

                async for chunk in stream_response:
                    # Check for grounding metadata in the current chunk
                    if hasattr(chunk, 'candidates') and chunk.candidates and hasattr(chunk.candidates[0], 'grounding_metadata'):
                        latest_metadata = chunk.candidates[0].grounding_metadata
                        if latest_metadata: # Store if it exists
                            grounding_metadata = latest_metadata

                    # Check if chunk has text before appending
                    if hasattr(chunk, 'text') and chunk.text:
                        response_contents.append(chunk.text)
                        # --- Real-time editing logic for Gemini ---
                        current_full_text = "".join(response_contents)
                        if not use_plain_responses:
                            # Calculate which message index the *end* of the text belongs to
                            current_msg_index = (len(current_full_text) - 1) // split_limit
                            # Check if we need to start a new message
                            start_next_msg = current_msg_index >= len(response_msgs)

                            ready_to_edit = (edit_task is None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                            is_final_chunk = False # Assume not final unless loop finishes

                            # --- Trigger message creation or edit ---
                            if start_next_msg or ready_to_edit:
                                if edit_task is not None:
                                    await edit_task # Wait for any previous edit task

                                # --- Handle splitting: Update the PREVIOUS message first ---
                                if start_next_msg and response_msgs:
                                    prev_msg_index = current_msg_index - 1
                                    # Get the complete text for the previous message segment
                                    prev_msg_text = current_full_text[prev_msg_index * split_limit : current_msg_index * split_limit]
                                    prev_msg_text = prev_msg_text[:MAX_EMBED_DESCRIPTION_LENGTH + len(STREAMING_INDICATOR)] # Ensure limit
                                    # Final update for the previous message
                                    embed.description = prev_msg_text # No indicator
                                    embed.color = EMBED_COLOR_COMPLETE
                                    try:
                                        await response_msgs[prev_msg_index].edit(embed=embed)
                                    except discord.HTTPException as e:
                                        logging.error(f"Failed to finalize previous message {prev_msg_index}: {e}")
                                    # Optional small delay if needed: await asyncio.sleep(0.1)

                                # --- Determine text for the CURRENT message segment ---
                                current_display_text = current_full_text[current_msg_index * split_limit : (current_msg_index + 1) * split_limit]
                                current_display_text = current_display_text[:MAX_EMBED_DESCRIPTION_LENGTH] # Truncate for safety

                                # Set description and color for the current/new message
                                embed.description = current_display_text + STREAMING_INDICATOR
                                embed.color = EMBED_COLOR_INCOMPLETE

                                # --- Create or Edit the CURRENT message ---
                                if start_next_msg:
                                    reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                                    response_msg = await reply_to_msg.reply(embed=embed, silent=True) # Send new message
                                    response_msgs.append(response_msg)
                                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                                    await msg_nodes[response_msg.id].lock.acquire()
                                elif response_msgs: # If not starting a new one, edit the existing one for this index
                                    if current_msg_index < len(response_msgs):
                                        edit_task = asyncio.create_task(response_msgs[current_msg_index].edit(embed=embed))
                                    else:
                                        logging.error(f"Edit index {current_msg_index} out of bounds for response_msgs (len {len(response_msgs)})")

                                last_task_time = dt.now().timestamp()
                        # --- End real-time editing logic ---

                finish_reason = "stop" # Assume stop if stream finishes without error for Gemini

            else:
                # Use openai client
                # Pass messages directly, other config in kwargs
                stream_response = await openai_client.chat.completions.create(
                    model=model,
                    messages=api_content_kwargs["messages"],
                    **api_config # Pass stream=True and extra_body here
                )
                async for curr_chunk in stream_response:
                    if finish_reason != None: # Check if already finished
                        break

                    finish_reason = curr_chunk.choices[0].finish_reason
                    new_content_chunk = curr_chunk.choices[0].delta.content or ""

                    if not response_contents and not new_content_chunk: # Skip empty initial chunks
                        continue

                    response_contents.append(new_content_chunk) # Append the chunk

                    # --- Real-time editing logic for OpenAI ---
                    if not use_plain_responses:
                        current_full_text = "".join(response_contents)
                        # Determine which message this chunk belongs to
                        current_msg_index = (len(current_full_text) -1) // split_limit
                        start_next_msg = current_msg_index >= len(response_msgs)

                        ready_to_edit = (edit_task is None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                        is_final_chunk = finish_reason is not None

                        # --- Trigger message creation or edit ---
                        if start_next_msg or ready_to_edit or is_final_chunk:
                            if edit_task is not None:
                                await edit_task # Wait for any previous edit task

                            # --- Handle splitting: Update the PREVIOUS message first ---
                            if start_next_msg and response_msgs:
                                prev_msg_index = current_msg_index - 1
                                prev_msg_text = current_full_text[prev_msg_index * split_limit : current_msg_index * split_limit]
                                prev_msg_text = prev_msg_text[:MAX_EMBED_DESCRIPTION_LENGTH + len(STREAMING_INDICATOR)] # Ensure limit
                                embed.description = prev_msg_text # No indicator
                                embed.color = EMBED_COLOR_COMPLETE
                                try:
                                    await response_msgs[prev_msg_index].edit(embed=embed)
                                except discord.HTTPException as e:
                                     logging.error(f"Failed to finalize previous message {prev_msg_index}: {e}")
                                # Optional small delay if needed: await asyncio.sleep(0.1)

                            # --- Determine text for the CURRENT message segment ---
                            current_display_text = current_full_text[current_msg_index * split_limit : (current_msg_index + 1) * split_limit]
                            current_display_text = current_display_text[:MAX_EMBED_DESCRIPTION_LENGTH] # Truncate for safety

                            # Set description and color for the current/new message
                            embed.description = current_display_text if is_final_chunk else (current_display_text + STREAMING_INDICATOR)
                            embed.color = EMBED_COLOR_COMPLETE if is_final_chunk and finish_reason and finish_reason.lower() in ("stop", "end_turn") else EMBED_COLOR_INCOMPLETE

                            # --- Create or Edit the CURRENT message ---
                            if start_next_msg:
                                reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                                response_msg = await reply_to_msg.reply(embed=embed, silent=True) # Send new message
                                response_msgs.append(response_msg)
                                msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                                await msg_nodes[response_msg.id].lock.acquire()
                            elif response_msgs: # If not starting a new one, edit the existing one for this index
                                if current_msg_index < len(response_msgs):
                                     edit_task = asyncio.create_task(response_msgs[current_msg_index].edit(embed=embed))
                                else:
                                     logging.error(f"Edit index {current_msg_index} out of bounds for response_msgs (len {len(response_msgs)})")


                            last_task_time = dt.now().timestamp()
                    # --- End real-time editing logic ---

            # --- Post-Streaming Processing ---
            final_text = "".join(response_contents)
            # grounding_metadata is now potentially populated from the loop for Gemini
            view = None

            # Create the view only if grounding_metadata was found and has relevant info
            if grounding_metadata and (getattr(grounding_metadata, 'web_search_queries', None) or getattr(grounding_metadata, 'grounding_chunks', None)):
                view = SourcesView(grounding_metadata)

            # Send final message(s) using plain text if configured
            if use_plain_responses:
                 # Split final_text into messages based on standard 2000 char limit
                 final_messages_content = [final_text[i:i+2000] for i in range(0, len(final_text), 2000)]
                 if not final_messages_content and final_text == "": # Ensure at least one empty message if response was empty
                     final_messages_content.append("...") # Placeholder for empty response
                 elif not final_messages_content and final_text != "": # Handle case where final_text is not empty but splitting resulted in empty list
                     final_messages_content = [final_text[:2000]]


                 # Ensure response_msgs list is populated correctly if it was empty
                 temp_response_msgs = []
                 for i, content in enumerate(final_messages_content):
                     reply_to_msg = new_msg if not temp_response_msgs else temp_response_msgs[-1]
                     # Attach view only to the last message if applicable
                     current_view = view if (i == len(final_messages_content) - 1) else None
                     response_msg = await reply_to_msg.reply(content=content or "...", suppress_embeds=True, view=current_view, silent=True) # Ensure content is not empty
                     temp_response_msgs.append(response_msg)
                     msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                     await msg_nodes[response_msg.id].lock.acquire() # Lock immediately
                 response_msgs = temp_response_msgs # Assign the populated list back

            # Final edit for embed messages (if not using plain responses)
            elif not use_plain_responses and response_msgs:
                # Final update for the VERY LAST message segment
                final_msg_index = len(response_msgs) - 1
                # Calculate final text for the last segment, respecting embed limits
                final_msg_text = final_text[final_msg_index * split_limit : (final_msg_index + 1) * split_limit]
                final_msg_text = final_msg_text[:MAX_EMBED_DESCRIPTION_LENGTH + len(STREAMING_INDICATOR)] # Use full 4096 limit for final

                # Wait for any pending edit task on the last message
                if edit_task is not None and not edit_task.done():
                    await edit_task

                embed.description = final_msg_text or "..." # Final text without indicator, handle empty
                embed.color = EMBED_COLOR_COMPLETE
                # Edit only the last message with the final text and view
                await response_msgs[final_msg_index].edit(embed=embed, view=view)
            elif not use_plain_responses and not response_msgs:
                 # Handle case where response was empty and no initial message was sent
                 embed.description = "..." # Placeholder for empty response
                 embed.color = EMBED_COLOR_COMPLETE
                 response_msg = await new_msg.reply(embed=embed, view=view, silent=True)
                 response_msgs.append(response_msg)
                 msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                 await msg_nodes[response_msg.id].lock.acquire() # Lock immediately

    # Correct the exception handling paths using specific Error types
    except google_types.BlockedPromptError as e:
         logging.warning(f"Gemini Prompt Blocked: {e}")
         await new_msg.reply("⚠️ My prompt was blocked by safety filters.", silent=True)
    except google_types.StopCandidateError as e:
         logging.warning(f"Gemini Response Blocked: {e}")
         await new_msg.reply("⚠️ My response was blocked by safety filters.", silent=True)
    # Catch potential API errors more broadly
    except google_types.GoogleAPICallError as e:
        logging.exception("Google API Call Error")
        await new_msg.reply(f"⚠️ Google API Error: {e}", silent=True)
    except Exception as e:
        logging.exception("Error during LLM call")
        error_message = f"⚠️ An error occurred: {type(e).__name__}"
        # Add more specific error details if possible and safe
        if hasattr(e, 'message'):
            error_message += f": {e.message}"
        elif hasattr(e, 'args'):
             error_message += f": {e.args[0]}" if e.args else ""
        await new_msg.reply(error_message, silent=True)


    # Release locks and store final text for all response messages
    for response_msg in response_msgs:
        if response_msg.id in msg_nodes:
            msg_nodes[response_msg.id].text = final_text # Use aggregated final text
            if msg_nodes[response_msg.id].lock.locked():
                msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            # Ensure lock is acquired before potentially deleting
            node_to_delete = msg_nodes.get(msg_id)
            if node_to_delete:
                async with node_to_delete.lock:
                    msg_nodes.pop(msg_id, None)


async def main():
    await discord_client.start(cfg["bot_token"])


asyncio.run(main())