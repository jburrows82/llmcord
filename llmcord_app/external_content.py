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
from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CacheMode,
)  # Added Crawl4AI imports
from .content_fetchers import (
    fetch_youtube_data,
    fetch_reddit_data,
    # fetch_general_url_content, # No longer directly used here for batching
    process_google_lens_image,
)
from .content_fetchers.web import fetch_with_beautifulsoup  # Import fallback


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
        logging.info(
            f"Batch processing {len(general_urls_to_batch)} general URLs with Crawl4AI."
        )
        url_strings_for_crawl4ai = [item[0] for item in general_urls_to_batch]
        url_to_original_index_map = {url: idx for url, idx in general_urls_to_batch}

        try:
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,  # Consider making this configurable
                # Add other relevant Crawl4AI configs here if needed
            )
            async with AsyncWebCrawler() as crawler:  # Single crawler instance
                crawl4ai_batch_results = await crawler.arun_many(
                    url_strings_for_crawl4ai, config=run_config
                )

            for crawl_result in crawl4ai_batch_results:
                original_idx = url_to_original_index_map.get(crawl_result.url, -1)
                if (
                    crawl_result.success
                    and crawl_result.markdown
                    and crawl_result.markdown.raw_markdown
                ):
                    content = crawl_result.markdown.raw_markdown
                    if (
                        max_text_length is not None
                        and content
                        and len(content) > max_text_length
                    ):
                        content = content[: max_text_length - 3] + "..."
                    url_fetch_results.append(
                        models.UrlFetchResult(
                            url=crawl_result.url,
                            content=content,
                            type="general_crawl4ai",
                            original_index=original_idx,
                        )
                    )
                else:
                    error_msg_crawl4ai = (
                        crawl_result.error_message
                        or "Crawl4AI: Unknown error or no markdown."
                    )
                    logging.warning(
                        f"Crawl4AI failed for {crawl_result.url} (Error: {error_msg_crawl4ai}). Falling back to BeautifulSoup."
                    )
                    bs_fallback_result = await fetch_with_beautifulsoup(
                        crawl_result.url, original_idx, httpx_client, max_text_length
                    )
                    url_fetch_results.append(bs_fallback_result)
                    if bs_fallback_result.error:
                        user_warnings.add(
                            f"⚠️ Fallback fetch failed for {crawl_result.url[:40]}...: {bs_fallback_result.error}"
                        )

        except Exception as e_crawl4ai_batch:
            logging.error(
                f"Crawl4AI arun_many batch processing failed: {e_crawl4ai_batch}",
                exc_info=True,
            )
            user_warnings.add("⚠️ Error during batch web page processing with Crawl4AI.")
            # Fallback for all URLs in this batch if arun_many itself fails
            for url_str, original_idx_val in general_urls_to_batch:
                logging.info(
                    f"Falling back to BeautifulSoup for {url_str} due to arun_many failure."
                )
                bs_fallback_result = await fetch_with_beautifulsoup(
                    url_str, original_idx_val, httpx_client, max_text_length
                )
                url_fetch_results.append(bs_fallback_result)
                if bs_fallback_result.error:
                    user_warnings.add(
                        f"⚠️ Fallback fetch failed for {url_str[:40]}...: {bs_fallback_result.error}"
                    )

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

    # Process Google Lens Images Sequentially (remains the same)
    if use_google_lens:
        lens_images_to_process = image_attachments[
            :max_files_per_message
        ]  # Use max_files_per_message
        if len(image_attachments) > max_files_per_message:
            user_warnings.add(
                f"⚠️ Only processing first {max_files_per_message} images for Google Lens."
            )

        logging.info(
            f"Processing {len(lens_images_to_process)} Google Lens images sequentially..."
        )
        for i, attachment in enumerate(lens_images_to_process):
            logging.info(
                f"Starting Google Lens processing for image {i + 1}/{len(lens_images_to_process)}..."
            )
            try:
                # Pass config to process_google_lens_image
                lens_result = await process_google_lens_image(
                    attachment.url, i, config
                )  # Pass config here
                url_fetch_results.append(lens_result)
                if lens_result.error:
                    logging.warning(
                        f"Google Lens processing failed for image {i + 1}: {lens_result.error}"
                    )
                    user_warnings.add(
                        f"⚠️ Google Lens failed for image {i + 1}: {lens_result.error[:100]}..."
                    )
                else:
                    logging.info(f"Finished Google Lens processing for image {i + 1}.")
            except Exception as e:
                logging.exception(
                    f"Unexpected error during sequential Google Lens processing for image {i + 1}"
                )
                user_warnings.add(f"⚠️ Unexpected error processing Lens image {i + 1}")
                # Use the correct UrlFetchResult class
                url_fetch_results.append(
                    models.UrlFetchResult(
                        url=attachment.url,
                        content=None,
                        error=f"Unexpected processing error: {type(e).__name__}",
                        type="google_lens_fallback_failed",
                        original_index=i,
                    )
                )

    return url_fetch_results


def format_external_content(url_fetch_results: List[models.UrlFetchResult]) -> str:
    """Formats fetched URL/Lens content into a string for the LLM."""
    if not url_fetch_results:
        return ""

    google_lens_parts = []
    other_url_parts = []
    other_url_counter = 1

    # Sort results by original position
    url_fetch_results.sort(key=lambda r: r.original_index)

    for result in url_fetch_results:
        if result.content:  # Only include successful fetches
            if result.type == "google_lens_serpapi":
                header = f"SerpAPI Google Lens results for image {result.original_index + 1}:\n"
                google_lens_parts.append(header + str(result.content))
            elif result.type == "youtube":
                content_str = f"\nurl {other_url_counter}: {result.url}\nurl {other_url_counter} content:\n"
                if isinstance(result.content, dict):
                    content_str += f"  title: {result.content.get('title', 'N/A')}\n"
                    content_str += (
                        f"  channel: {result.content.get('channel_name', 'N/A')}\n"
                    )
                    desc = result.content.get("description", "N/A")
                    content_str += f"  description: {desc}\n"
                    transcript = result.content.get("transcript")
                    if transcript:
                        content_str += f"  transcript: {transcript}\n"
                    comments = result.content.get("comments")
                    if comments:
                        content_str += (
                            "  top comments:\n"
                            + "\n".join([f"    - {c}" for c in comments])
                            + "\n"
                        )
                other_url_parts.append(content_str)
                other_url_counter += 1
            elif result.type == "reddit":
                content_str = f"\nurl {other_url_counter}: {result.url}\nurl {other_url_counter} content:\n"
                if isinstance(result.content, dict):
                    content_str += f"  title: {result.content.get('title', 'N/A')}\n"
                    selftext = result.content.get("selftext")
                    if selftext:
                        content_str += f"  content: {selftext}\n"
                    comments = result.content.get("comments")
                    if comments:
                        content_str += (
                            "  top comments:\n"
                            + "\n".join([f"    - {c}" for c in comments])
                            + "\n"
                        )
                other_url_parts.append(content_str)
                other_url_counter += 1
            elif result.type == "general" or result.type == "general_crawl4ai":
                content_str = f"\nurl {other_url_counter}: {result.url} (source type: {result.type})\nurl {other_url_counter} content:\n"
                if isinstance(result.content, str):
                    content_str += f"  {result.content}\n"
                other_url_parts.append(content_str)
                other_url_counter += 1
            # image_url_content types are handled as binary data and not formatted into combined_context text

    combined_context = ""
    if google_lens_parts or other_url_parts:
        combined_context = "Answer the user's query based on the following:\n\n"
        if google_lens_parts:
            combined_context += "\n\n".join(google_lens_parts) + "\n\n"
        if other_url_parts:
            combined_context += "".join(other_url_parts)

    return combined_context.strip()
