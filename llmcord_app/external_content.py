import asyncio
import logging
from typing import List, Set, Dict, Any

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
    fetch_general_url_content,
    process_google_lens_image,
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
    """Fetches content from URLs and Google Lens."""
    all_urls_with_indices = extract_urls_with_indices(cleaned_content)
    fetch_tasks = []
    processed_urls = set()
    url_fetch_results = []

    # Create tasks for non-Google Lens URLs
    for url, index in all_urls_with_indices:
        if url in processed_urls:
            continue
        processed_urls.add(url)

        if is_youtube_url(url):
            fetch_tasks.append(
                fetch_youtube_data(url, index, config.get("youtube_api_key"))
            )
        elif is_reddit_url(url):
            sub_id = extract_reddit_submission_id(url)
            if sub_id:
                fetch_tasks.append(
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
        elif is_image_url(url):  # Check for image URLs
            # Define an async function to download the image and return a UrlFetchResult
            async def fetch_image_url_content(
                img_url: str, img_idx: int
            ) -> models.UrlFetchResult:
                try:
                    logging.info(f"Attempting to download image URL: {img_url}")
                    # Use the passed httpx_client
                    async with httpx_client.stream(
                        "GET", img_url, timeout=15.0
                    ) as response:
                        if response.status_code == 200:
                            content_type = response.headers.get(
                                "content-type", ""
                            ).lower()
                            if content_type.startswith("image/"):
                                img_bytes_list = []
                                async for chunk in response.aiter_bytes():
                                    img_bytes_list.append(chunk)
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
                                logging.warning(
                                    f"URL {img_url} is an image URL but content type is '{content_type}'. Skipping."
                                )
                                return models.UrlFetchResult(
                                    url=img_url,
                                    content=None,
                                    error=f"Not an image content type: {content_type}",
                                    type="image_url_content",
                                    original_index=img_idx,
                                )
                        else:
                            logging.warning(
                                f"Failed to download image URL {img_url}. Status: {response.status_code}"
                            )
                            return models.UrlFetchResult(
                                url=img_url,
                                content=None,
                                error=f"HTTP status {response.status_code}",
                                type="image_url_content",
                                original_index=img_idx,
                            )
                except httpx.RequestError as e:
                    logging.warning(
                        f"RequestError downloading image URL {img_url}: {e}"
                    )
                    return models.UrlFetchResult(
                        url=img_url,
                        content=None,
                        error=f"Request error: {type(e).__name__}",
                        type="image_url_content",
                        original_index=img_idx,
                    )
                except Exception as e:
                    logging.exception(
                        f"Unexpected error downloading image URL {img_url}"
                    )
                    return models.UrlFetchResult(
                        url=img_url,
                        content=None,
                        error=f"Unexpected error: {type(e).__name__}",
                        type="image_url_content",
                        original_index=img_idx,
                    )

            fetch_tasks.append(fetch_image_url_content(url, index))
        else:  # General web page
            fetch_tasks.append(
                fetch_general_url_content(url, index, httpx_client)  # Pass the client
            )

    # Fetch non-Lens content concurrently
    if fetch_tasks:
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logging.error(
                    f"Unhandled exception during non-Lens URL fetch: {result}",
                    exc_info=True,
                )
                user_warnings.add("⚠️ Unhandled error fetching URL content")
            # Use the correct UrlFetchResult class
            elif isinstance(result, models.UrlFetchResult):
                url_fetch_results.append(result)
                if result.error:
                    short_url = (
                        result.url[:40] + "..." if len(result.url) > 40 else result.url
                    )
                    user_warnings.add(
                        f"⚠️ Error fetching {result.type} URL ({short_url}): {result.error}"
                    )
            else:
                logging.error(
                    f"Unexpected result type from non-Lens URL fetch: {type(result)}"
                )

    # Process Google Lens Images Sequentially
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
            elif result.type == "general":
                content_str = f"\nurl {other_url_counter}: {result.url}\nurl {other_url_counter} content:\n"
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
