import asyncio
import logging
from typing import List, Set, Dict, Any, Tuple

import discord
import httpx

from . import models  # Use relative import within the package
from .utils import (
    extract_urls_with_indices,
    is_youtube_url,
    is_reddit_url,
    is_image_url,
    extract_reddit_submission_id,
)
from .content_fetchers import (
    fetch_youtube_data,
    fetch_reddit_data,
    # fetch_general_url_content, # No longer directly used here for batching
    process_google_lens_image,
)
from .content_fetchers.web import (
    fetch_general_url_content as fetch_general_url_content_dynamic,
)  # Renamed for clarity
from .constants import (
    MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY,
    FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY,
    DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR,
    DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR,
    JINA_ENGINE_MODE_CONFIG_KEY,
    DEFAULT_JINA_ENGINE_MODE,
    JINA_WAIT_FOR_SELECTOR_CONFIG_KEY,  # Added
    JINA_TIMEOUT_CONFIG_KEY,
    DEFAULT_JINA_WAIT_FOR_SELECTOR,
    DEFAULT_JINA_TIMEOUT,
    # Crawl4AI
    CRAWL4AI_CACHE_MODE_CONFIG_KEY,  # Added
    DEFAULT_CRAWL4AI_CACHE_MODE,  # Added
)


async def fetch_external_content(
    cleaned_content: str,
    image_attachments: List[discord.Attachment],
    use_google_lens: bool,
    max_files_per_message: int,
    user_warnings: Set[str],
    config: Dict[str, Any],
    httpx_client: httpx.AsyncClient,
) -> List[models.UrlFetchResult]:
    """Fetches content from URLs and Google Lens. General URLs are batched with Crawl4AI."""
    all_urls_with_indices = extract_urls_with_indices(cleaned_content)
    other_fetch_tasks = []  # For YouTube, Reddit, image URLs
    general_urls_to_batch: List[Tuple[str, int]] = []  # For Crawl4AI batching
    processed_urls_for_batching = set()  # To avoid duplicate general URLs in the batch
    url_fetch_results = []
    max_text_length = config.get(
        "searxng_url_content_max_length"
    )  # Get max_text_length from config

    # Separate general URLs for batching, other types for individual tasks
    for url, index in all_urls_with_indices:
        # Check if the URL is wrapped in backticks
        is_wrapped_in_backticks = False
        # Check character before the URL
        char_before_is_backtick = index > 0 and cleaned_content[index - 1] == "`"
        # Check character after the URL
        char_after_is_backtick = (index + len(url)) < len(
            cleaned_content
        ) and cleaned_content[index + len(url)] == "`"

        if char_before_is_backtick and char_after_is_backtick:
            is_wrapped_in_backticks = True

        if is_wrapped_in_backticks:
            logging.info(
                f"Skipping content extraction for URL wrapped in backticks: {url}"
            )
            # DO NOT add to processed_urls_for_batching here.
            # A non-backticked version of the same URL elsewhere should still be processed.
            continue  # Skip fetching and further processing for this URL

        if (
            url in processed_urls_for_batching
        ):  # Check if already added to general batch or other tasks
            continue

        if is_youtube_url(url):
            other_fetch_tasks.append(
                fetch_youtube_data(url, index, config.get("youtube_api_key"))
            )
            processed_urls_for_batching.add(url)
        elif is_reddit_url(url):
            sub_id = extract_reddit_submission_id(url)
            if sub_id:
                other_fetch_tasks.append(
                    fetch_reddit_data(
                        url,
                        sub_id,
                        index,
                        config.get("reddit_client_id"),
                        config.get("reddit_client_secret"),
                        config.get("reddit_user_agent"),
                    )
                )
            else:
                user_warnings.add(
                    f"⚠️ Could not extract submission ID from Reddit URL: {url[:50]}..."
                )
            processed_urls_for_batching.add(url)
        elif is_image_url(url):

            async def fetch_image_url_content_wrapper(
                img_url: str, img_idx: int
            ) -> models.UrlFetchResult:
                # This is the same inner function as before, just defined here for clarity
                try:
                    logging.info(f"Attempting to download image URL: {img_url}")
                    async with httpx_client.stream(
                        "GET", img_url, timeout=15.0
                    ) as response:
                        if response.status_code == 200:
                            content_type = response.headers.get(
                                "content-type", ""
                            ).lower()
                            if content_type.startswith("image/"):
                                img_bytes_list = [
                                    chunk async for chunk in response.aiter_bytes()
                                ]
                                img_bytes = b"".join(img_bytes_list)
                                logging.info(
                                    f"Successfully downloaded image from URL: {img_url} ({len(img_bytes)} bytes)"
                                )
                                return models.UrlFetchResult(
                                    url=img_url,
                                    content=img_bytes,
                                    type="image_url_content",
                                    original_index=img_idx,
                                )
                            else:
                                return models.UrlFetchResult(
                                    url=img_url,
                                    content=None,
                                    error=f"Not an image content type: {content_type}",
                                    type="image_url_content",
                                    original_index=img_idx,
                                )
                        else:
                            return models.UrlFetchResult(
                                url=img_url,
                                content=None,
                                error=f"HTTP status {response.status_code}",
                                type="image_url_content",
                                original_index=img_idx,
                            )
                except httpx.RequestError as e:
                    return models.UrlFetchResult(
                        url=img_url,
                        content=None,
                        error=f"Request error: {type(e).__name__}",
                        type="image_url_content",
                        original_index=img_idx,
                    )
                except Exception as e:
                    return models.UrlFetchResult(
                        url=img_url,
                        content=None,
                        error=f"Unexpected error: {type(e).__name__}",
                        type="image_url_content",
                        original_index=img_idx,
                    )

            other_fetch_tasks.append(fetch_image_url_content_wrapper(url, index))
            processed_urls_for_batching.add(url)
        else:  # General web page, add to batch list
            if url not in processed_urls_for_batching:
                general_urls_to_batch.append((url, index))
                processed_urls_for_batching.add(url)

    # Batch process general URLs with Crawl4AI
    if general_urls_to_batch:
        main_extractor_method = config.get(
            MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY,
            DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR,
        )
        fallback_extractor_method = config.get(
            FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY,
            DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR,
        )
        jina_mode = config.get(JINA_ENGINE_MODE_CONFIG_KEY, DEFAULT_JINA_ENGINE_MODE)
        jina_selector = config.get(  # Added
            JINA_WAIT_FOR_SELECTOR_CONFIG_KEY, DEFAULT_JINA_WAIT_FOR_SELECTOR
        )
        jina_timeout_val = config.get(JINA_TIMEOUT_CONFIG_KEY, DEFAULT_JINA_TIMEOUT)
        crawl4ai_cache = config.get(  # Added
            CRAWL4AI_CACHE_MODE_CONFIG_KEY, DEFAULT_CRAWL4AI_CACHE_MODE
        )
        logging.info(
            f"Processing {len(general_urls_to_batch)} general URLs with main: '{main_extractor_method}', fallback: '{fallback_extractor_method}', Jina mode: '{jina_mode}', Selector: '{jina_selector}', Timeout: {jina_timeout_val}, Crawl4AI Cache: '{crawl4ai_cache}'."
        )

        general_url_processing_tasks = []
        for url_str, original_idx_val in general_urls_to_batch:
            general_url_processing_tasks.append(
                fetch_general_url_content_dynamic(
                    url=url_str,
                    index=original_idx_val,
                    client=httpx_client,
                    main_extractor=main_extractor_method,
                    fallback_extractor=fallback_extractor_method,
                    max_text_length=max_text_length,
                    jina_engine_mode=jina_mode,
                    jina_wait_for_selector=jina_selector,
                    jina_timeout=jina_timeout_val,
                    crawl4ai_cache_mode=crawl4ai_cache,  # Pass Crawl4AI cache mode
                )
            )

        try:
            batch_general_results = await asyncio.gather(
                *general_url_processing_tasks, return_exceptions=True
            )
            for result_item in batch_general_results:
                if isinstance(result_item, models.UrlFetchResult):
                    url_fetch_results.append(result_item)
                    if result_item.error:
                        user_warnings.add(
                            f"⚠️ Fetch failed for general URL {result_item.url[:40]}...: {result_item.error}"
                        )
                elif isinstance(result_item, Exception):
                    logging.error(
                        f"Unhandled exception during general URL batch processing: {result_item}",
                        exc_info=True,
                    )
                    # Find which URL this exception corresponds to if possible, or add a generic warning
                    # This part is tricky as gather doesn't directly map exceptions back to input tasks easily without more complex tracking
                    user_warnings.add("⚠️ Error during batch web page processing.")
        except Exception as e_general_batch:
            logging.error(
                f"General URL batch processing gather failed: {e_general_batch}",
                exc_info=True,
            )
            user_warnings.add("⚠️ Critical error during batch web page processing.")

    # Fetch other non-general URLs concurrently
    if other_fetch_tasks:
        results_others = await asyncio.gather(
            *other_fetch_tasks, return_exceptions=True
        )
        for result_other in results_others:
            if isinstance(result_other, Exception):
                logging.error(
                    f"Unhandled exception during other URL fetch: {result_other}",
                    exc_info=True,
                )
                user_warnings.add("⚠️ Unhandled error fetching some URL content")
            elif isinstance(result_other, models.UrlFetchResult):
                url_fetch_results.append(result_other)
                if result_other.error:
                    short_url_other = (
                        result_other.url[:40] + "..."
                        if len(result_other.url) > 40
                        else result_other.url
                    )
                    user_warnings.add(
                        f"⚠️ Error fetching {result_other.type} URL ({short_url_other}): {result_other.error}"
                    )
            else:
                logging.error(
                    f"Unexpected result type from other URL fetch: {type(result_other)}"
                )

    # Process Google Lens Images Concurrently
    if use_google_lens:
        lens_images_to_process = image_attachments[
            :max_files_per_message
        ]  # Use max_files_per_message
        if len(image_attachments) > max_files_per_message:
            user_warnings.add(
                f"⚠️ Only processing first {max_files_per_message} images for Google Lens."
            )

        if lens_images_to_process:
            logging.info(
                f"Processing {len(lens_images_to_process)} Google Lens images concurrently..."
            )
            lens_processing_tasks = []
            for i, attachment in enumerate(lens_images_to_process):
                lens_processing_tasks.append(
                    process_google_lens_image(attachment.url, i, config)
                )

            try:
                lens_results_gathered = await asyncio.gather(
                    *lens_processing_tasks, return_exceptions=True
                )
                for i, lens_result_or_exc in enumerate(lens_results_gathered):
                    attachment_url_for_error = lens_images_to_process[i].url
                    if isinstance(lens_result_or_exc, models.UrlFetchResult):
                        url_fetch_results.append(lens_result_or_exc)
                        if lens_result_or_exc.error:
                            logging.warning(
                                f"Google Lens processing failed for image {lens_result_or_exc.original_index + 1} ({attachment_url_for_error}): {lens_result_or_exc.error}"
                            )
                            user_warnings.add(
                                f"⚠️ Google Lens failed for image {lens_result_or_exc.original_index + 1}: {lens_result_or_exc.error[:100]}..."
                            )
                        else:
                            logging.info(
                                f"Finished Google Lens processing for image {lens_result_or_exc.original_index + 1} ({attachment_url_for_error})."
                            )
                    elif isinstance(lens_result_or_exc, Exception):
                        logging.exception(
                            f"Unexpected exception during concurrent Google Lens processing for image {i + 1} ({attachment_url_for_error})"
                        )
                        user_warnings.add(
                            f"⚠️ Unexpected error processing Lens image {i + 1}"
                        )
                        url_fetch_results.append(
                            models.UrlFetchResult(
                                url=attachment_url_for_error,
                                content=None,
                                error=f"Unexpected processing error: {type(lens_result_or_exc).__name__}",
                                type="google_lens_fallback_failed",
                                original_index=i,  # original_index here refers to the index in lens_images_to_process
                            )
                        )
            except Exception as e_lens_gather:
                logging.error(
                    f"Google Lens batch processing gather failed: {e_lens_gather}",
                    exc_info=True,
                )
                user_warnings.add(
                    "⚠️ Critical error during batch Google Lens processing."
                )

    return url_fetch_results


def format_external_content(
    url_fetch_results: List[models.UrlFetchResult],
) -> Dict[str, str]:
    """
    Formats fetched URL and Google Lens content into separate strings.
    Returns a dictionary with keys 'user_urls' and 'lens'.
    Search results (SearxNG) are handled separately and not formatted by this function.
    """
    formatted_content_dict = {"user_urls": "", "lens": ""}
    if not url_fetch_results:
        return formatted_content_dict

    google_lens_parts_list = []
    user_provided_url_parts_list = []
    url_counter = 1  # Counter for user-provided URLs

    # Sort results by original position to maintain order
    url_fetch_results.sort(key=lambda r: r.original_index)

    for result in url_fetch_results:
        if result.content:  # Only include successful fetches
            if result.type == "google_lens_serpapi":
                header = (
                    f"Google Lens results for image {result.original_index + 1}:\n"
                )
                google_lens_parts_list.append(header + str(result.content))
            elif result.type in [
                "youtube",
                "reddit",
                "general",
                "general_crawl4ai",
                "general_jina",
            ]:
                # These are considered user-provided URLs for the purpose of this formatting
                content_str = f"URL {url_counter} ({result.type} - {result.url}):\n"
                if isinstance(result.content, dict):  # YouTube or Reddit
                    if result.type == "youtube":
                        content_str += (
                            f"  Title: {result.content.get('title', 'N/A')}\n"
                        )
                        content_str += f"  Channel: {result.content.get('channel_name', 'N/A')}\n"
                        desc = result.content.get("description", "N/A")
                        content_str += f"  Description: {desc}\n"
                        transcript = result.content.get("transcript")
                        if transcript:
                            content_str += f"  Transcript: {transcript}\n"
                        comments = result.content.get("comments")
                        if comments:
                            content_str += "  Top Comments:\n" + "\n".join(
                                [f"    - {c}" for c in comments]
                            )
                    elif result.type == "reddit":
                        content_str += (
                            f"  Title: {result.content.get('title', 'N/A')}\n"
                        )
                        selftext = result.content.get("selftext")
                        if selftext:
                            content_str += f"  Content: {selftext}\n"
                        comments = result.content.get("comments")
                        if comments:
                            content_str += "  Top Comments:\n" + "\n".join(
                                [f"    - {c}" for c in comments]
                            )
                elif isinstance(result.content, str):  # General web page
                    content_str += f"  Content: {result.content}\n"
                
                user_provided_url_parts_list.append(content_str)
                url_counter += 1
            # image_url_content types are handled as binary data for API parts,
            # not formatted into text here for history persistence.

    if google_lens_parts_list:
        formatted_content_dict["lens"] = "\n\n".join(google_lens_parts_list)

    if user_provided_url_parts_list:
        formatted_content_dict["user_urls"] = "\n\n".join(
            user_provided_url_parts_list
        )

    return formatted_content_dict
