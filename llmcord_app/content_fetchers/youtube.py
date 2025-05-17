import asyncio
import logging
import xml.etree.ElementTree  # Import for specific error handling
from typing import Tuple, Optional, Dict, Any

from googleapiclient.discovery import build as build_google_api_client
from googleapiclient.errors import HttpError
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

# Import proxy types conditionally or handle import error
try:
    from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig
except ImportError:
    WebshareProxyConfig = None
    GenericProxyConfig = None
    logging.info("youtube_transcript_api.proxies not found. Proxy support disabled.")

from ..models import UrlFetchResult
from ..utils import extract_video_id

# Module-level globals
_initial_proxy_config_data: Optional[Dict] = None
ytt_api: Optional[YouTubeTranscriptApi] = None  # Default, no-proxy instance
youtube_service_client = None  # Global client for YouTube Data API


def _build_proxy_object(proxy_config_dict: Optional[Dict]) -> Optional[Any]:
    """
    Builds a proxy configuration object (WebshareProxyConfig or GenericProxyConfig)
    from a dictionary. Returns None if config is invalid or types are not available.
    """
    if not proxy_config_dict or not isinstance(proxy_config_dict, dict):
        return None

    proxy_type = proxy_config_dict.get("type")

    if WebshareProxyConfig is None and GenericProxyConfig is None and proxy_type:
        logging.warning(
            "Proxy types (WebshareProxyConfig, GenericProxyConfig) not imported. Cannot build proxy object."
        )
        return None

    if (
        proxy_type == "webshare"
        and WebshareProxyConfig
        and proxy_config_dict.get("username")
        and proxy_config_dict.get("password")
    ):
        try:
            return WebshareProxyConfig(
                proxy_username=proxy_config_dict["username"],
                proxy_password=proxy_config_dict["password"],
            )
        except KeyError:
            logging.error("Webshare proxy config missing 'username' or 'password'.")
            return None
        except Exception as e:
            logging.error(f"Error initializing WebshareProxyConfig: {e}")
            return None
    elif (
        proxy_type == "generic"
        and GenericProxyConfig
        and (proxy_config_dict.get("http_url") or proxy_config_dict.get("https_url"))
    ):
        try:
            return GenericProxyConfig(
                http_url=proxy_config_dict.get("http_url"),
                https_url=proxy_config_dict.get("https_url"),
            )
        except KeyError:
            logging.error("Generic proxy config missing 'http_url' or 'https_url'.")
            return None
        except Exception as e:
            logging.error(f"Error initializing GenericProxyConfig: {e}")
            return None
    elif proxy_type:
        logging.warning(
            f"Invalid or incomplete proxy type '{proxy_type}' in config. Not using proxy for this attempt."
        )
    return None


def initialize_youtube_data_api(api_key: Optional[str]):
    """Initializes the YouTube Data API service client."""
    global youtube_service_client
    if not api_key:
        logging.warning("YouTube Data API key not provided. Cannot initialize client.")
        youtube_service_client = None
        return
    try:
        youtube_service_client = build_google_api_client(
            "youtube", "v3", developerKey=api_key
        )
        logging.info("YouTube Data API service client initialized successfully.")
    except HttpError as e_build:
        error_reason = getattr(e_build, "reason", str(e_build))
        status_code = getattr(e_build.resp, "status", "Unknown")
        logging.error(
            f"Failed to initialize YouTube Data API client (Status: {status_code}): {error_reason}. "
            "Video detail fetching will likely fail."
        )
        youtube_service_client = None
    except Exception:
        logging.exception("Unexpected error initializing YouTube Data API client.")
        youtube_service_client = None


def initialize_ytt_api(proxy_config_data_from_main: Optional[Dict] = None):
    """
    Initializes the global ytt_api instance (without proxy) and stores
    the proxy configuration for potential later use.
    """
    global ytt_api, _initial_proxy_config_data
    _initial_proxy_config_data = proxy_config_data_from_main
    try:
        ytt_api = YouTubeTranscriptApi(proxy_config=None)  # Initialize without proxy
        logging.info("Global YouTubeTranscriptApi (ytt_api) initialized without proxy.")
        if _initial_proxy_config_data:
            logging.info("Proxy configuration has been stored for potential retry.")
    except Exception as e:
        logging.error(
            f"Failed to initialize YouTubeTranscriptApi: {e}. Transcript fetching may fail."
        )
        ytt_api = None  # Ensure ytt_api is None if initialization fails


async def _fetch_transcript_core(
    video_id: str, api_client: YouTubeTranscriptApi
) -> Tuple[Optional[str], Optional[str]]:
    """
    Core logic to fetch transcript using a given YouTubeTranscriptApi client.
    Raises exceptions like ParseError, TranscriptsDisabled, NoTranscriptFound.
    """
    # Use asyncio.to_thread for the synchronous calls
    transcript_list = await asyncio.to_thread(api_client.list_transcripts, video_id)
    transcript_obj = (
        None  # Renamed from transcript to avoid conflict with outer scope in some IDEs
    )
    # Prioritize manual English, then generated English, then any manual, then any generated
    priorities = [
        (transcript_list.find_manually_created_transcript, ["en"]),
        (transcript_list.find_generated_transcript, ["en"]),
        (
            transcript_list.find_manually_created_transcript,
            [lang.language_code for lang in transcript_list],
        ),
        (
            transcript_list.find_generated_transcript,
            [lang.language_code for lang in transcript_list],
        ),
    ]
    for find_method, langs in priorities:
        if not langs:
            continue
        try:
            transcript_obj = await asyncio.to_thread(find_method, langs)
            if transcript_obj:
                break
        except NoTranscriptFound:
            continue
        except Exception as e_find:  # Catch errors during find_method
            logging.warning(
                f"Error during transcript find method {find_method.__name__} for {video_id}: {e_find}"
            )
            continue  # Try next priority

    if transcript_obj:
        fetched_transcript_data = await asyncio.to_thread(transcript_obj.fetch)
        # The .to_raw_data() method was removed in youtube-transcript-api 0.6.0
        # The fetched_transcript_data is already the list of dicts.
        full_transcript = " ".join([entry.text for entry in fetched_transcript_data])
        return full_transcript, None
    else:
        # This case should be caught by NoTranscriptFound from list_transcripts or find_transcript
        # but as a fallback:
        raise NoTranscriptFound(video_id)


async def get_transcript(video_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetches the transcript for a video ID.
    Retries with proxy if initial attempt fails with a specific ParseError.
    """
    global ytt_api, _initial_proxy_config_data
    if ytt_api is None:
        logging.error(
            "YouTubeTranscriptApi (ytt_api) not initialized. Cannot fetch transcript."
        )
        return None, "Transcript API not initialized."

    try:
        # First attempt (no proxy)
        return await _fetch_transcript_core(video_id, ytt_api)
    except xml.etree.ElementTree.ParseError as e_parse:
        logging.warning(
            f"Initial transcript fetch for {video_id} failed with ParseError: {e_parse}. "
            "This often indicates a block. Attempting proxy retry if configured."
        )
        if _initial_proxy_config_data:
            proxy_obj_for_retry = _build_proxy_object(_initial_proxy_config_data)
            if proxy_obj_for_retry:
                logging.info(f"Retrying transcript fetch for {video_id} with proxy.")
                proxy_api_client = YouTubeTranscriptApi(
                    proxy_config=proxy_obj_for_retry
                )
                try:
                    return await _fetch_transcript_core(video_id, proxy_api_client)
                except xml.etree.ElementTree.ParseError as e_parse_proxy:
                    logging.error(
                        f"Proxy transcript fetch for {video_id} also failed with ParseError: {e_parse_proxy}"
                    )
                    return None, "Transcript parsing failed even with proxy."
                except TranscriptsDisabled:
                    return None, "Transcripts are disabled for this video."
                except NoTranscriptFound:
                    return None, "No transcripts listed for this video (proxy attempt)."
                except Exception as e_proxy_other:
                    logging.error(
                        f"Proxy transcript fetch for {video_id} failed with other error: {e_proxy_other}",
                        exc_info=True,
                    )
                    return (
                        None,
                        f"Proxy transcript fetch failed: {type(e_proxy_other).__name__}",
                    )
            else:
                logging.info(
                    f"Proxy is configured but failed to build proxy object for {video_id}. No retry with proxy."
                )
                return (
                    None,
                    "Transcript parsing failed (likely blocked; proxy config error).",
                )
        else:
            logging.info(f"No proxy configured for {video_id}. No retry with proxy.")
            return None, "Transcript parsing failed (likely blocked; consider proxy)."
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return None, "No transcripts listed for this video."
    except Exception as e_other:
        e_str = str(e_other).lower()
        if (
            "requestblocked" in e_str
            or "ipblocked" in e_str
            or "too many requests" in e_str
        ):
            logging.error(
                f"YouTube blocked transcript request for {video_id} (direct): {type(e_other).__name__}: {e_other}"
            )
            # This error type might also benefit from a proxy retry, but current request is specific to ParseError.
            return (
                None,
                "YouTube blocked the transcript request (IP ban or rate limit).",
            )

        logging.error(
            f"Unexpected error fetching transcript for {video_id} (direct): {type(e_other).__name__}: {e_other}",
            exc_info=True,
        )
        return None, f"An error occurred fetching transcript: {type(e_other).__name__}"


async def get_youtube_video_details(
    video_id: str, api_key: Optional[str]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetches video title, description, channel name, and comments using YouTube Data API."""
    if not api_key:  # Check if api_key is None or empty string
        logging.info(
            "YouTube API key not configured or empty. Skipping video detail fetch."
        )
        return None, "YouTube API key not configured."
    if not youtube_service_client:
        logging.warning(
            "YouTube Data API client not initialized. Cannot fetch video details."
        )
        # Attempt a one-off initialization if key is present but client is not
        if api_key:
            initialize_youtube_data_api(api_key)
            if not youtube_service_client:  # Check again after attempt
                return None, "YouTube Data API client failed to initialize."
        else:  # Should have been caught by the first check, but for safety
            return None, "YouTube Data API client not available (no key)."

    details = {}
    error_messages = []

    try:
        # Run blocking Google API client calls in a separate thread
        def sync_get_details():
            # No need for global youtube_service_client here, it's accessed from module scope
            _details = {}
            _errors = []

            # Get video details (snippet)
            try:
                video_request = youtube_service_client.videos().list(
                    part="snippet", id=video_id
                )
                video_response = video_request.execute()
                if video_response.get("items"):
                    snippet = video_response["items"][0]["snippet"]
                    _details["title"] = snippet.get("title", "N/A")
                    _details["description"] = snippet.get("description", "N/A")
                    _details["channel_name"] = snippet.get("channelTitle", "N/A")
                else:
                    _errors.append("Video details not found.")
            except HttpError as e_vid:
                error_reason = getattr(e_vid, "reason", str(e_vid))
                status_code = getattr(e_vid.resp, "status", "Unknown")
                logging.warning(
                    f"YouTube Data API error getting video details for {video_id} (Status: {status_code}): {error_reason}"
                )
                _errors.append(f"API error getting video details ({status_code}).")
            except Exception as e_vid_other:
                logging.exception(
                    f"Unexpected error getting YouTube video details for {video_id}"
                )
                _errors.append(
                    f"Unexpected error getting video details: {type(e_vid_other).__name__}"
                )

            # Get top comments (commentThreads)
            try:
                comment_request = youtube_service_client.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    order="relevance",
                    maxResults=10,  # Fetch up to 10 top-level comments
                    textFormat="plainText",
                )
                comment_response = comment_request.execute()
                _details["comments"] = [
                    item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    for item in comment_response.get("items", [])
                ]
            except HttpError as e_com:
                error_reason = getattr(e_com, "reason", str(e_com))
                status_code = getattr(e_com.resp, "status", "Unknown")
                if (
                    status_code == 403 and "commentsDisabled" in str(e_com).lower()
                ):  # More robust check
                    logging.info(f"Comments disabled for YouTube video {video_id}")
                    _errors.append("Comments disabled.")
                else:
                    logging.warning(
                        f"YouTube Data API error getting comments for {video_id} (Status: {status_code}): {error_reason}"
                    )
                    _errors.append(f"API error getting comments ({status_code}).")
            except Exception as e_com_other:
                logging.exception(
                    f"Unexpected error getting YouTube comments for {video_id}"
                )
                _errors.append(
                    f"Unexpected error getting comments: {type(e_com_other).__name__}"
                )
            return _details, _errors

        details, error_messages = await asyncio.to_thread(sync_get_details)

        final_details = details if details else None
        final_error = " ".join(error_messages) if error_messages else None
        return final_details, final_error

    except Exception as e:
        logging.exception(
            f"Error running YouTube detail fetch in thread for {video_id}"
        )
        return None, f"An unexpected error occurred: {e}"


async def fetch_youtube_data(
    url: str, index: int, api_key: Optional[str]
) -> UrlFetchResult:
    """Fetches transcript and details for a single YouTube URL."""
    video_id = extract_video_id(url)
    if not video_id:
        return UrlFetchResult(
            url=url,
            content=None,
            error="Could not extract video ID.",
            type="youtube",
            original_index=index,
        )

    # Fetch transcript and details concurrently
    transcript_task = asyncio.create_task(get_transcript(video_id))
    # Pass api_key to get_youtube_video_details
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
        errors.append("No content fetched.")

    return UrlFetchResult(
        url=url,
        content=combined_content if combined_content else None,
        error=" ".join(errors) if errors else None,
        type="youtube",
        original_index=index,
    )
