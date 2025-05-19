import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator, Tuple

import discord  # For discord.utils.escape_markdown
import httpx

from . import models
from .constants import (
    GROUNDING_MODEL_PROVIDER,
    GROUNDING_MODEL_NAME,
    SEARXNG_BASE_URL_CONFIG_KEY,
    SEARXNG_NUM_RESULTS,
    SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY,
    AllKeysFailedError,
)
from .content_fetchers import (
    fetch_searxng_results,
    fetch_youtube_data,
    fetch_reddit_data,
    fetch_general_url_content,
)
from .utils import is_youtube_url, is_reddit_url, extract_reddit_submission_id


async def get_web_search_queries_from_gemini(
    history_for_gemini_grounding: List[Dict[str, Any]],
    system_prompt_text_for_grounding: Optional[str],
    config: Dict[str, Any],
    generate_response_stream_func: Callable[
        [str, str, List[Dict[str, Any]], Optional[str], Dict[str, Any], Dict[str, Any]],
        AsyncGenerator[
            Tuple[Optional[str], Optional[str], Optional[Any], Optional[str]], None
        ],
    ],
) -> Optional[List[str]]:
    """
    Calls Gemini to get web search queries from its grounding metadata.
    """
    logging.info("Attempting to get web search queries from Gemini for grounding...")
    gemini_provider_config = config.get("providers", {}).get(
        GROUNDING_MODEL_PROVIDER, {}
    )
    if not gemini_provider_config or not gemini_provider_config.get("api_keys"):
        logging.warning(
            f"Cannot perform Gemini grounding step: Provider '{GROUNDING_MODEL_PROVIDER}' not configured with API keys."
        )
        return None

    all_web_search_queries = set()  # Use a set to store unique queries

    try:
        # Use a minimal set of extra_params for the grounding call
        grounding_extra_params = {
            "temperature": 1,
            "thinking_budget": 0,
        }

        stream_generator = generate_response_stream_func(
            provider=GROUNDING_MODEL_PROVIDER,
            model_name=GROUNDING_MODEL_NAME,
            history_for_llm=history_for_gemini_grounding,
            system_prompt_text=system_prompt_text_for_grounding,
            provider_config=gemini_provider_config,
            extra_params=grounding_extra_params,
        )

        async for _, _, chunk_grounding_metadata, error_message in stream_generator:
            if error_message:
                logging.error(f"Error during Gemini grounding call: {error_message}")
                return None  # Abort on first error

            if chunk_grounding_metadata:
                if (
                    hasattr(chunk_grounding_metadata, "web_search_queries")
                    and chunk_grounding_metadata.web_search_queries
                ):
                    for query in chunk_grounding_metadata.web_search_queries:
                        if isinstance(query, str) and query.strip():
                            all_web_search_queries.add(query.strip())
                            logging.debug(
                                f"Gemini grounding produced search query: '{query.strip()}'"
                            )

        if all_web_search_queries:
            logging.info(
                f"Gemini grounding produced {len(all_web_search_queries)} unique search queries."
            )
            return list(all_web_search_queries)
        else:
            logging.info(
                "Gemini grounding call completed but found no web_search_queries in metadata."
            )
            return None

    except (
        AllKeysFailedError
    ) as e:  # Make sure AllKeysFailedError is accessible or handled
        logging.error(
            f"All API keys failed for Gemini grounding model ({GROUNDING_MODEL_PROVIDER}/{GROUNDING_MODEL_NAME}): {e}"
        )
        return None
    except Exception:
        logging.exception(
            "Unexpected error during Gemini grounding call to get search queries:"
        )
        return None


async def fetch_and_format_searxng_results(
    queries: List[str],
    user_query_for_log: str,  # For logging context
    config: Dict[str, Any],
    httpx_client: httpx.AsyncClient,
) -> Optional[str]:
    """
    Fetches search results from SearxNG for given queries,
    then fetches content of those URLs (uniquely) and formats them.
    Uses specific fetchers for YouTube and Reddit URLs.
    """
    if not queries:
        return None

    searxng_base_url = config.get(SEARXNG_BASE_URL_CONFIG_KEY)
    if not searxng_base_url:
        logging.warning(
            "SearxNG base URL not configured. Skipping web search enhancement."
        )
        return None

    logging.info(
        f"Fetching SearxNG results for {len(queries)} queries related to user query: '{user_query_for_log[:100]}...'"
    )
    searxng_content_limit = config.get(SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY)

    # API keys for specific fetchers
    youtube_api_key = config.get("youtube_api_key")
    reddit_client_id = config.get("reddit_client_id")
    reddit_client_secret = config.get("reddit_client_secret")
    reddit_user_agent = config.get("reddit_user_agent")

    # Fetch SearxNG URLs for all queries concurrently
    searxng_tasks = []
    for query in queries:
        searxng_tasks.append(
            fetch_searxng_results(
                query, httpx_client, searxng_base_url, SEARXNG_NUM_RESULTS
            )
        )

    try:
        list_of_url_lists_per_query = await asyncio.gather(
            *searxng_tasks, return_exceptions=True
        )
    except Exception:
        logging.exception("Error gathering SearxNG results.")
        return None

    # Collect all unique URLs from all queries
    unique_urls_to_fetch_content = set()
    for i, query_urls_or_exc in enumerate(list_of_url_lists_per_query):
        query_str = queries[i]
        if isinstance(query_urls_or_exc, Exception):
            logging.error(
                f"Failed to get SearxNG results for query '{query_str}': {query_urls_or_exc}"
            )
            continue
        if not query_urls_or_exc:
            logging.info(f"No URLs returned by SearxNG for query: '{query_str}'")
            continue
        for url_str in query_urls_or_exc:
            if isinstance(url_str, str):  # Ensure it's a string
                unique_urls_to_fetch_content.add(url_str)

    if not unique_urls_to_fetch_content:
        logging.info(
            "No unique URLs found from SearxNG results to process content for."
        )
        return None

    # Fetch content for all unique URLs concurrently using appropriate fetchers
    url_content_processing_tasks = []
    for idx, url_str in enumerate(list(unique_urls_to_fetch_content)):
        if is_youtube_url(url_str):
            url_content_processing_tasks.append(
                fetch_youtube_data(url_str, idx, youtube_api_key)
            )
        elif is_reddit_url(url_str):
            submission_id = extract_reddit_submission_id(url_str)
            if submission_id:
                url_content_processing_tasks.append(
                    fetch_reddit_data(
                        url_str,
                        submission_id,
                        idx,
                        reddit_client_id,
                        reddit_client_secret,
                        reddit_user_agent,
                    )
                )
            else:
                logging.warning(
                    f"Could not extract submission ID from SearxNG Reddit URL: {url_str}. Skipping."
                )

                async def dummy_failed_result():
                    return models.UrlFetchResult(
                        url=url_str,
                        content=None,
                        error="Failed to extract Reddit submission ID.",
                        type="reddit",
                        original_index=idx,
                    )

                url_content_processing_tasks.append(dummy_failed_result())
        else:
            url_content_processing_tasks.append(
                fetch_general_url_content(
                    url_str,
                    idx,
                    httpx_client,
                    max_text_length=searxng_content_limit,  # Pass httpx_client
                )
            )

    try:
        fetched_unique_content_results = await asyncio.gather(
            *url_content_processing_tasks, return_exceptions=True
        )
    except Exception:
        logging.exception("Error gathering content from unique SearxNG URLs.")
        return None

    url_to_content_map: Dict[str, models.UrlFetchResult] = {}
    for result_or_exc in fetched_unique_content_results:
        if isinstance(result_or_exc, models.UrlFetchResult):
            url_to_content_map[result_or_exc.url] = result_or_exc
        elif isinstance(result_or_exc, Exception):
            logging.error(
                f"Exception while processing a unique URL for content: {result_or_exc}"
            )

    formatted_query_blocks = []
    query_counter = 1
    limit = searxng_content_limit

    for i, query_str in enumerate(queries):
        urls_for_this_query_or_exc = list_of_url_lists_per_query[i]

        if (
            isinstance(urls_for_this_query_or_exc, Exception)
            or not urls_for_this_query_or_exc
        ):
            continue

        current_query_url_content_parts = []
        url_counter_for_query = 1

        for url_from_searxng_for_query in urls_for_this_query_or_exc:
            if not isinstance(url_from_searxng_for_query, str):
                continue

            content_result = url_to_content_map.get(url_from_searxng_for_query)

            if content_result and content_result.content and not content_result.error:
                content_str_part = ""
                if content_result.type == "youtube" and isinstance(
                    content_result.content, dict
                ):
                    data = content_result.content
                    title = data.get("title", "N/A")
                    content_str_part += (
                        f"  title: {discord.utils.escape_markdown(title)}\n"
                    )
                    channel = data.get("channel_name", "N/A")
                    content_str_part += (
                        f"  channel: {discord.utils.escape_markdown(channel)}\n"
                    )
                    desc = data.get("description", "N/A")
                    if limit and len(desc) > limit:
                        desc = desc[: limit - 3] + "..."
                    content_str_part += (
                        f"  description: {discord.utils.escape_markdown(desc)}\n"
                    )
                    transcript = data.get("transcript")
                    if transcript:
                        if limit and len(transcript) > limit:
                            transcript = transcript[: limit - 3] + "..."
                        content_str_part += f"  transcript: {discord.utils.escape_markdown(transcript)}\n"
                    comments_list = data.get("comments")
                    if comments_list:
                        comments_str = "\n".join(
                            [
                                f"    - {discord.utils.escape_markdown(c)}"
                                for c in comments_list
                            ]
                        )
                        if limit and len(comments_str) > limit:
                            comments_str = comments_str[: limit - 3] + "..."
                        content_str_part += f"  top comments:\n{comments_str}\n"
                elif content_result.type == "reddit" and isinstance(
                    content_result.content, dict
                ):
                    data = content_result.content
                    title = data.get("title", "N/A")
                    content_str_part += (
                        f"  title: {discord.utils.escape_markdown(title)}\n"
                    )
                    selftext = data.get("selftext")
                    if selftext:
                        if limit and len(selftext) > limit:
                            selftext = selftext[: limit - 3] + "..."
                        content_str_part += (
                            f"  content: {discord.utils.escape_markdown(selftext)}\n"
                        )
                    comments_list = data.get("comments")
                    if comments_list:
                        comments_str = "\n".join(
                            [
                                f"    - {discord.utils.escape_markdown(c)}"
                                for c in comments_list
                            ]
                        )
                        if limit and len(comments_str) > limit:
                            comments_str = comments_str[: limit - 3] + "..."
                        content_str_part += f"  top comments:\n{comments_str}\n"
                elif content_result.type == "general" and isinstance(
                    content_result.content, str
                ):
                    content_str_part = discord.utils.escape_markdown(
                        content_result.content
                    )
                else:
                    raw_content_str = str(content_result.content)
                    if limit and len(raw_content_str) > limit:
                        raw_content_str = raw_content_str[: limit - 3] + "..."
                    content_str_part = discord.utils.escape_markdown(raw_content_str)

                current_query_url_content_parts.append(
                    f"URL {url_counter_for_query}: {content_result.url}\n"
                    f"URL {url_counter_for_query} content:\n{content_str_part.strip()}\n"
                )
                url_counter_for_query += 1
            elif content_result and content_result.error:
                logging.warning(
                    f"Skipping SearxNG URL {content_result.url} for query '{query_str}' due to fetch error: {content_result.error}"
                )
            elif not content_result:
                logging.warning(
                    f"Content for SearxNG URL {url_from_searxng_for_query} (query: '{query_str}') not found in pre-fetched map."
                )

        if current_query_url_content_parts:
            query_block_header = f'Query {query_counter} ("{discord.utils.escape_markdown(query_str)}") search results:\n\n'
            formatted_query_blocks.append(
                query_block_header + "\n".join(current_query_url_content_parts)
            )
            query_counter += 1

    if formatted_query_blocks:
        final_context = (
            "Answer the user's query based on the following:\n\n"
            + "\n\n".join(formatted_query_blocks)
        )
        return final_context.strip()

    logging.info(
        "No content successfully fetched and formatted from SearxNG result URLs."
    )
    return None
