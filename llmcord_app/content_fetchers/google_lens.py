import asyncio
import logging
import random
from typing import List, Dict, Any
import discord

from serpapi import GoogleSearch
from serpapi.serp_api_client_exception import SerpApiClientException

from ..models import UrlFetchResult
from ..rate_limiter import get_db_manager, get_available_keys


async def fetch_google_lens_serpapi(
    image_url: str, index: int, all_keys: List[str]
) -> UrlFetchResult:
    """Fetches Google Lens results using SerpAPI with key rotation and retry (PRIMARY)."""
    service_name = "serpapi"
    if not all_keys:
        return UrlFetchResult(
            url=image_url,
            content=None,
            error="SerpAPI keys not configured.",
            type="google_lens_serpapi",
            original_index=index,
        )

    available_keys = await get_available_keys(service_name, all_keys)
    if not available_keys:  # Check if keys are available *after* filtering
        logging.warning(
            f"All SerpAPI keys are currently rate-limited for Google Lens request for image {index + 1}."
        )
        return UrlFetchResult(
            url=image_url,
            content=None,
            error="All SerpAPI keys are rate-limited.",
            type="google_lens_serpapi",
            original_index=index,
        )

    random.shuffle(available_keys)
    db_manager = get_db_manager(service_name)
    encountered_errors = []

    for key_index, api_key in enumerate(available_keys):
        key_display = f"...{api_key[-4:]}"
        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": api_key,
            "safe": "off",  # Keep safe search off
            "no_cache": True,  # Force fresh results
            "hl": "en",  # Set language to English
            "country": "us",  # Set country to United States
        }
        logging.info(
            f"Attempting SerpAPI Google Lens request for image {index + 1} with key {key_display} ({key_index + 1}/{len(available_keys)})"
        )

        try:
            # Run the synchronous SerpAPI call in a separate thread
            def sync_serpapi_call():
                search = GoogleSearch(params)
                return search.get_dict()

            results = await asyncio.to_thread(sync_serpapi_call)

            # Check for API-level errors (e.g., invalid key, quota)
            if "error" in results:
                error_msg = results["error"]
                logging.warning(
                    f"SerpAPI error for image {index + 1} (key {key_display}): {error_msg}"
                )
                encountered_errors.append(f"Key {key_display}: {error_msg}")
                # Check if it's a rate limit / quota error
                if (
                    "rate limit" in error_msg.lower()
                    or "quota" in error_msg.lower()
                    or "plan limit" in error_msg.lower()
                    or "ran out of searches" in error_msg.lower()
                ):
                    db_manager.add_key(api_key)
                    logging.info(
                        f"SerpAPI key {key_display} rate limited. Trying next key."
                    )
                    continue  # Try next key
                elif "invalid api key" in error_msg.lower():
                    logging.error(
                        f"SerpAPI key {key_display} is invalid. Aborting SerpAPI attempts."
                    )
                    # Don't mark invalid keys as rate-limited, just stop trying
                    return UrlFetchResult(
                        url=image_url,
                        content=None,
                        error=f"SerpAPI Error: Invalid API Key ({key_display})",
                        type="google_lens_serpapi",
                        original_index=index,
                    )
                else:
                    # For other API errors, log and try next key
                    logging.warning(
                        f"SerpAPI key {key_display} encountered API error: {error_msg}. Trying next key."
                    )
                    continue

            # Check for search-specific errors (e.g., couldn't process image)
            search_metadata = results.get("search_metadata", {})
            if search_metadata.get("status", "").lower() == "error":
                error_msg = search_metadata.get("error", "Unknown search error")
                logging.warning(
                    f"SerpAPI search error for image {index + 1} (key {key_display}): {error_msg}"
                )
                encountered_errors.append(
                    f"Key {key_display}: Search Error - {error_msg}"
                )
                # These errors are usually not key-related, so don't mark key as limited.
                # Return the error from the first key that hit this.
                logging.error(
                    f"SerpAPI search failed for image {index + 1} with key {key_display}. Aborting SerpAPI attempts."
                )
                return UrlFetchResult(
                    url=image_url,
                    content=None,
                    error=f"SerpAPI Search Error: {error_msg}",
                    type="google_lens_serpapi",
                    original_index=index,
                )

            visual_matches = results.get("visual_matches", [])
            if not visual_matches:
                logging.info(
                    f"SerpAPI Google Lens request successful for image {index + 1} with key {key_display}, but no visual matches found."
                )
                return UrlFetchResult(
                    url=image_url,
                    content="No visual matches found (SerpAPI).",
                    type="google_lens_serpapi",
                    original_index=index,
                )

            # Format the results concisely
            formatted_results = []
            for i, match in enumerate(visual_matches):  # No display limit
                title = match.get("title", "N/A")
                link = match.get("link", "#")
                source = match.get("source", "")
                escaped_title = discord.utils.escape_markdown(
                    title
                )
                result_line = f"- [{escaped_title}]({link})"
                if source:
                    result_line += f" (Source: {discord.utils.escape_markdown(source)})"
                formatted_results.append(result_line)
            content_str = "\n".join(formatted_results)

            logging.info(
                f"SerpAPI Google Lens request successful for image {index + 1} with key {key_display}"
            )
            return UrlFetchResult(
                url=image_url,
                content=content_str,
                type="google_lens_serpapi",
                original_index=index,
            )

        except SerpApiClientException as e:
            # Handle client-level exceptions (e.g., connection errors, timeouts)
            logging.warning(
                f"SerpAPI client exception for image {index + 1} (key {key_display}): {e}"
            )
            encountered_errors.append(f"Key {key_display}: Client Error - {e}")
            # Check if the exception indicates a rate limit (might need specific checks based on library behavior/status codes)
            # SerpApiClientException often wraps HTTP errors, check status code if possible
            status_code = None
            if hasattr(e, "response") and hasattr(e.response, "status_code"):
                status_code = e.response.status_code

            if status_code == 429 or "rate limit" in str(e).lower():
                db_manager.add_key(api_key)
                logging.info(
                    f"SerpAPI key {key_display} hit client-side rate limit (Status: {status_code}). Trying next key."
                )
            else:
                logging.warning(
                    f"SerpAPI key {key_display} encountered client error (Status: {status_code}): {e}. Trying next key."
                )
            # Retry with the next key for client exceptions
            continue
        except Exception as e:
            logging.exception(
                f"Unexpected error during SerpAPI request for image {index + 1} (key {key_display})"
            )
            encountered_errors.append(
                f"Key {key_display}: Unexpected Error - {type(e).__name__}"
            )
            logging.warning(
                f"SerpAPI key {key_display} encountered unexpected error: {e}. Trying next key."
            )
            # Retry with the next key for unexpected errors
            continue

    # If loop finishes, all keys failed
    logging.error(
        f"All available SerpAPI keys failed for Google Lens request for image {index + 1}."
    )
    final_error_msg = "All available SerpAPI keys failed."
    if encountered_errors:
        final_error_msg += f" Last error: {encountered_errors[-1]}"
    return UrlFetchResult(
        url=image_url,
        content=None,
        error=final_error_msg,
        type="google_lens_serpapi",
        original_index=index,
    )


async def process_google_lens_image(
    image_url: str, index: int, cfg: Dict[str, Any]
) -> UrlFetchResult:
    """
    Processes a Google Lens request for an image URL using SerpAPI.

    Args:
        image_url: The URL of the image to process.
        index: The index of the image in the message attachments (for logging/context).
        cfg: The loaded application configuration dictionary.

    Returns:
        A UrlFetchResult object.
    """
    serpapi_keys = cfg.get("serpapi_api_keys", [])
    serpapi_keys_ok = bool(serpapi_keys)
    serpapi_error_message = "SerpAPI not attempted (no keys configured)."

    if serpapi_keys_ok:
        logging.info(
            f"Attempting Google Lens request for image {index + 1} using SerpAPI."
        )
        serpapi_result = await fetch_google_lens_serpapi(image_url, index, serpapi_keys)

        if not serpapi_result.error:
            logging.info(
                f"SerpAPI Google Lens request successful for image {index + 1}."
            )
            return serpapi_result
        else:
            serpapi_error_message = serpapi_result.error
            logging.warning(
                f"SerpAPI Google Lens request failed for image {index + 1}: {serpapi_error_message}."
            )
    else:
        logging.info("SerpAPI keys not configured. Skipping Google Lens request.")

    # If SerpAPI was not configured or failed
    final_error_msg = (
        f"Google Lens processing failed. SerpAPI Error: {serpapi_error_message}"
    )
    return UrlFetchResult(
        url=image_url,
        content=None,
        error=final_error_msg,
        type="google_lens_serpapi",
        original_index=index,
    )
