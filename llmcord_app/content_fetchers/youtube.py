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

# Initialize ytt_api globally within this module, potentially configured with proxy
ytt_api = None


def initialize_ytt_api(proxy_config_data: Optional[Dict] = None):
    """Initializes the YouTubeTranscriptApi instance, potentially with proxy."""
    global ytt_api
    ytt_proxy_config = None
    if proxy_config_data and isinstance(proxy_config_data, dict):
        proxy_type = proxy_config_data.get("type")
        if (
            proxy_type == "webshare"
            and WebshareProxyConfig
            and proxy_config_data.get("username")
            and proxy_config_data.get("password")
        ):
            try:
                ytt_proxy_config = WebshareProxyConfig(
                    proxy_username=proxy_config_data["username"],
                    proxy_password=proxy_config_data["password"],
                )
                logging.info("Using Webshare proxy for YouTube transcripts.")
            except KeyError:
                logging.error("Webshare proxy config missing 'username' or 'password'.")
            except Exception as e:
                logging.error(f"Error initializing WebshareProxyConfig: {e}")

        elif (
            proxy_type == "generic"
            and GenericProxyConfig
            and (
                proxy_config_data.get("http_url") or proxy_config_data.get("https_url")
            )
        ):
            try:
                ytt_proxy_config = GenericProxyConfig(
                    http_url=proxy_config_data.get("http_url"),
                    https_url=proxy_config_data.get("https_url"),
                )
                logging.info("Using generic proxy for YouTube transcripts.")
            except KeyError:
                logging.error("Generic proxy config missing 'http_url' or 'https_url'.")
            except Exception as e:
                logging.error(f"Error initializing GenericProxyConfig: {e}")

        elif proxy_type:
            logging.warning(
                f"Invalid or incomplete proxy type '{proxy_type}' in config.yaml. Not using proxy."
            )
        # else: No proxy type specified or config is empty/invalid

    try:
        ytt_api = YouTubeTranscriptApi(proxy_config=ytt_proxy_config)
        logging.info(
            f"YouTubeTranscriptApi initialized {'with proxy' if ytt_proxy_config else 'without proxy'}."
        )
    except Exception as e:
        logging.error(
            f"Failed to initialize YouTubeTranscriptApi: {e}. Transcript fetching may fail."
        )
        # Fallback to initializing without proxy if config caused an error
        ytt_api = YouTubeTranscriptApi()


async def get_transcript(video_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetches the transcript for a video ID using youtube-transcript-api."""
    global ytt_api
    if ytt_api is None:
        # This should ideally not happen if initialize_ytt_api is called early
        logging.error("YouTubeTranscriptApi not initialized. Cannot fetch transcript.")
        return None, "Transcript API not initialized."

    try:
        # Use asyncio.to_thread for the synchronous calls
        transcript_list = await asyncio.to_thread(ytt_api.list_transcripts, video_id)
        transcript = None
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
            # Ensure langs is not empty before calling find method
            if not langs:
                continue
            try:
                transcript = await asyncio.to_thread(find_method, langs)
                if transcript:
                    break
            except NoTranscriptFound:
                continue
            except Exception as e:
                logging.warning(
                    f"Error during transcript find method {find_method.__name__} for {video_id}: {e}"
                )
                continue

        if transcript:
            fetched_transcript = await asyncio.to_thread(transcript.fetch)
            raw_transcript_data = await asyncio.to_thread(
                fetched_transcript.to_raw_data
            )
            full_transcript = " ".join([entry["text"] for entry in raw_transcript_data])
            return full_transcript, None
        else:
            return None, "No suitable transcript found."
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return None, "No transcripts listed for this video."
    except xml.etree.ElementTree.ParseError as e:
        logging.error(
            f"XML ParseError fetching transcript for {video_id}: {e}. "
            f"This often means the request was blocked by YouTube or returned an empty/invalid (non-XML) response. "
            f"Consider configuring a proxy in config.yaml if this issue persists."
        )
        return None, "Transcript parsing failed (likely blocked; consider proxy)."
    except Exception as e:
        e_str = str(e).lower()
        if (
            "requestblocked" in e_str
            or "ipblocked" in e_str
            or "too many requests" in e_str
        ):
            logging.error(
                f"YouTube blocked transcript request for {video_id}: {type(e).__name__}: {e}"
            )
            return (
                None,
                "YouTube blocked the transcript request (IP ban or rate limit).",
            )
        logging.error(
            f"Error fetching transcript for {video_id}: {type(e).__name__}: {e}",
            exc_info=True,
        )
        return None, f"An error occurred fetching transcript: {type(e).__name__}"


async def get_youtube_video_details(
    video_id: str, api_key: Optional[str]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetches video title, description, channel name, and comments using YouTube Data API."""
    if not api_key:
        return None, "YouTube API key not configured."

    details = {}
    error_messages = []

    try:
        # Run blocking Google API client calls in a separate thread
        def sync_get_details():
            _details = {}
            _errors = []
            try:
                youtube = build_google_api_client("youtube", "v3", developerKey=api_key)

                # Get video details (snippet)
                try:
                    video_request = youtube.videos().list(part="snippet", id=video_id)
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

                # Get top comments (commentThreads) - Proceed even if details failed
                try:
                    comment_request = youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        order="relevance",
                        maxResults=10,
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
                    if status_code == 403 and "commentsDisabled" in str(e_com):
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

            except (
                HttpError
            ) as e_build:  # Catch errors during build_google_api_client itself
                error_reason = getattr(e_build, "reason", str(e_build))
                status_code = getattr(e_build.resp, "status", "Unknown")
                logging.error(
                    f"YouTube Data API Build/Auth error (Status: {status_code}): {error_reason}"
                )
                if status_code == 403:
                    if "quotaExceeded" in str(e_build):
                        _errors.append("YouTube API quota exceeded.")
                    elif "accessNotConfigured" in str(e_build):
                        _errors.append(
                            "YouTube API access not configured or key invalid."
                        )
                    else:
                        _errors.append(f"YouTube API permission error: {error_reason}")
                else:
                    _errors.append(f"YouTube Data API HTTP error: {error_reason}")
                return None, _errors
            except Exception as e_build_other:
                logging.exception(
                    f"Unexpected error initializing YouTube client for {video_id}"
                )
                _errors.append(
                    f"An unexpected error occurred during client init: {e_build_other}"
                )
                return None, _errors

        details, error_messages = await asyncio.to_thread(sync_get_details)

        # Return details if any were fetched, otherwise None
        final_details = details if details else None
        final_error = " ".join(error_messages) if error_messages else None
        return final_details, final_error

    except Exception as e:  # Catch potential errors in asyncio.to_thread itself
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
        errors.append("No content fetched.")  # Ensure error if nothing was retrieved

    return UrlFetchResult(
        url=url,
        content=combined_content if combined_content else None,
        error=" ".join(errors) if errors else None,
        type="youtube",
        original_index=index,
    )
