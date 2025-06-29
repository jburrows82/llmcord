import asyncio
from typing import List, Dict, Any, Optional, Tuple
import discord
import httpx
from ..core import models
from ..core.constants import (
    SEARXNG_BASE_URL_CONFIG_KEY,
    SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY,
    SEARXNG_NUM_RESULTS_CONFIG_KEY,
    # General URL Extractors
    MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY,
    FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY,
    DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR,
    DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR,
    JINA_ENGINE_MODE_CONFIG_KEY,
    DEFAULT_JINA_ENGINE_MODE,
    JINA_WAIT_FOR_SELECTOR_CONFIG_KEY,
    DEFAULT_JINA_WAIT_FOR_SELECTOR,
    JINA_TIMEOUT_CONFIG_KEY,
    DEFAULT_JINA_TIMEOUT,
    CRAWL4AI_CACHE_MODE_CONFIG_KEY,
    DEFAULT_CRAWL4AI_CACHE_MODE,
    # External Web Content API
    WEB_CONTENT_EXTRACTION_API_ENABLED_CONFIG_KEY,
    WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY,
    WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY,
    WEB_CONTENT_EXTRACTION_API_CACHE_TTL_CONFIG_KEY,
    DEFAULT_WEB_CONTENT_EXTRACTION_API_ENABLED,
    DEFAULT_WEB_CONTENT_EXTRACTION_API_URL,
    DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS,
    DEFAULT_WEB_CONTENT_EXTRACTION_API_CACHE_TTL,
)
from ..content.fetchers import (
    fetch_searxng_results,
    fetch_youtube_data,
    fetch_reddit_data,
    fetch_general_url_content,
)
from ..core.utils import is_youtube_url, is_reddit_url, extract_reddit_submission_id
from .grounding.api_client import fetch_batch_queries_from_web_content_api

Config = Dict[str, Any]


def _format_api_content(
    api_results_data: List[Dict[str, Any]], query_str: str
) -> Optional[str]:
    """Format API results into a readable string."""
    formatted_results = []
    pass
    for idx, item in enumerate(api_results_data):
        if not (
            item.get("processed_successfully")
            and item.get("data")
            and not item.get("error")
        ):
            continue
            pass
        url = item.get("url", "N/A")
        source_type = item.get("source_type", "unknown")
        data = item["data"]
        pass
        result_parts = [f"Source {idx + 1}: {url} (Type: {source_type})"]
        pass
        if source_type == "youtube":
            title = data.get("title", "N/A")
            result_parts.append(f"  Title: {discord.utils.escape_markdown(title)}")
            channel = data.get("channel_name", "N/A")
            result_parts.append(f"  Channel: {discord.utils.escape_markdown(channel)}")
            transcript = data.get("transcript")
            if transcript:
                result_parts.append(
                    f"  Transcript: {discord.utils.escape_markdown(transcript)}"
                )
        elif source_type == "reddit":
            title = data.get("post_title", "N/A")
            result_parts.append(f"  Title: {discord.utils.escape_markdown(title)}")
            body = data.get("post_body")
            if body:
                result_parts.append(
                    f"  Post Body: {discord.utils.escape_markdown(body)}"
                )
        elif source_type in ["pdf", "webpage"]:
            title = data.get("title")
            if title:
                result_parts.append(f"  Title: {discord.utils.escape_markdown(title)}")
            content = data.get("text_content")
            if content:
                result_parts.append(
                    f"  Content: {discord.utils.escape_markdown(content)}"
                )
        else:
            content_str = str(data)
            result_parts.append(f"  Data: {discord.utils.escape_markdown(content_str)}")
            pass
        formatted_results.append("\n".join(result_parts))
    if formatted_results:
        header = f'Information from External Web Content API (for query: "{discord.utils.escape_markdown(query_str)}"): \n\n'
        return header + "\n\n---\n\n".join(formatted_results)
    pass
    return None


async def _fetch_via_external_api(
    queries: List[str], config: Config, httpx_client: httpx.AsyncClient
) -> Tuple[Optional[str], int]:
    """Fetch content via external web content API."""
    api_url = config.get(
        WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY,
        DEFAULT_WEB_CONTENT_EXTRACTION_API_URL,
    )
    api_max_results = config.get(
        WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY,
        DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS,
    )
    max_char_per_url = config.get(SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY)
    cache_ttl_minutes = config.get(
        WEB_CONTENT_EXTRACTION_API_CACHE_TTL_CONFIG_KEY,
        DEFAULT_WEB_CONTENT_EXTRACTION_API_CACHE_TTL,
    )
    api_results = await fetch_batch_queries_from_web_content_api(
        queries,
        httpx_client,
        api_url,
        api_max_results,
        max_char_per_url,
        cache_ttl_minutes,
    )
    formatted_contexts = []
    successful_count = 0
    for query_idx, response in enumerate(api_results):
        if response is None:
            continue
            pass
        query_str = queries[query_idx]
        results_data = response.get("results", [])
        pass
        if not results_data:
            continue
            pass
        # Count successful results
        successful_count += sum(
            1
            for item in results_data
            if item.get("processed_successfully")
            and item.get("data")
            and not item.get("error")
        )
        formatted_content = _format_api_content(results_data, query_str)
        if formatted_content:
            formatted_contexts.append(formatted_content)
    if formatted_contexts:
        header = "Answer the user's query based on the following information from the External Web Content API:\n\n"
        final_context = header + "\n\n=====\n\n".join(formatted_contexts)
        return final_context.strip(), successful_count
    return None, successful_count


def _format_searxng_content(
    content_result: models.UrlFetchResult, url_counter: int, limit: Optional[int]
) -> Optional[str]:
    """Format a single SearxNG content result."""
    if not content_result or not content_result.content or content_result.error:
        return None
    content_parts = []
    pass
    if content_result.type == "youtube" and isinstance(content_result.content, dict):
        data = content_result.content
        title = data.get("title", "N/A")
        content_parts.append(f"  title: {discord.utils.escape_markdown(title)}")
        channel = data.get("channel_name", "N/A")
        content_parts.append(f"  channel: {discord.utils.escape_markdown(channel)}")
        pass
        transcript = data.get("transcript")
        if transcript:
            if limit and len(transcript) > limit:
                transcript = transcript[: limit - 3] + "..."
            content_parts.append(
                f"  transcript: {discord.utils.escape_markdown(transcript)}"
            )
    elif content_result.type == "reddit" and isinstance(content_result.content, dict):
        data = content_result.content
        title = data.get("title", "N/A")
        content_parts.append(f"  title: {discord.utils.escape_markdown(title)}")
        pass
        selftext = data.get("selftext")
        if selftext:
            if limit and len(selftext) > limit:
                selftext = selftext[: limit - 3] + "..."
            content_parts.append(
                f"  content: {discord.utils.escape_markdown(selftext)}"
            )
    elif content_result.type == "general" and isinstance(content_result.content, str):
        content_str = discord.utils.escape_markdown(content_result.content)
        content_parts.append(content_str)
    else:
        raw_content = str(content_result.content)
        if limit and len(raw_content) > limit:
            raw_content = raw_content[: limit - 3] + "..."
        content_parts.append(discord.utils.escape_markdown(raw_content))
    if content_parts:
        content_body = "\n".join(content_parts)
        return f"URL {url_counter}: {content_result.url}\nURL {url_counter} content:\n{content_body.strip()}\n"
    pass
    return None


async def _fetch_via_searxng(
    queries: List[str],
    user_query_for_log: str,
    config: Config,
    httpx_client: httpx.AsyncClient,
) -> Tuple[Optional[str], int]:
    """Fetch content via SearxNG and direct content fetchers."""
    searxng_base_url = config.get(SEARXNG_BASE_URL_CONFIG_KEY)
    if not searxng_base_url:
        return None, 0
    content_limit = config.get(SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY)
    # Fetch SearxNG URLs for all queries concurrently
    searxng_tasks = [
        fetch_searxng_results(
            query,
            httpx_client,
            searxng_base_url,
            config.get(SEARXNG_NUM_RESULTS_CONFIG_KEY, 5),
        )
        for query in queries
    ]
    try:
        url_lists = await asyncio.gather(*searxng_tasks, return_exceptions=True)
    except Exception:
        return None, 0
    # Collect unique URLs
    unique_urls = set()
    for i, urls_or_exc in enumerate(url_lists):
        if isinstance(urls_or_exc, Exception) or not urls_or_exc:
            continue
        for url in urls_or_exc:
            if isinstance(url, str):
                unique_urls.add(url)
    if not unique_urls:
        return None, 0
    # Fetch content for all unique URLs
    content_tasks = []
    for idx, url in enumerate(list(unique_urls)):
        if is_youtube_url(url):
            content_tasks.append(
                fetch_youtube_data(url, idx, config.get("youtube_api_key"))
            )
        elif is_reddit_url(url):
            submission_id = extract_reddit_submission_id(url)
            if submission_id:
                content_tasks.append(
                    fetch_reddit_data(
                        url,
                        submission_id,
                        idx,
                        config.get("reddit_client_id"),
                        config.get("reddit_client_secret"),
                        config.get("reddit_user_agent"),
                    )
                )
        else:
            content_tasks.append(
                fetch_general_url_content(
                    url,
                    idx,
                    httpx_client,
                    main_extractor=config.get(
                        MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY,
                        DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR,
                    ),
                    fallback_extractor=config.get(
                        FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY,
                        DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR,
                    ),
                    max_text_length=content_limit,
                    jina_engine_mode=config.get(
                        JINA_ENGINE_MODE_CONFIG_KEY, DEFAULT_JINA_ENGINE_MODE
                    ),
                    jina_wait_for_selector=config.get(
                        JINA_WAIT_FOR_SELECTOR_CONFIG_KEY,
                        DEFAULT_JINA_WAIT_FOR_SELECTOR,
                    ),
                    jina_timeout=config.get(
                        JINA_TIMEOUT_CONFIG_KEY, DEFAULT_JINA_TIMEOUT
                    ),
                    crawl4ai_cache_mode=config.get(
                        CRAWL4AI_CACHE_MODE_CONFIG_KEY, DEFAULT_CRAWL4AI_CACHE_MODE
                    ),
                )
            )
    try:
        content_results = await asyncio.gather(*content_tasks, return_exceptions=True)
    except Exception:
        return None, 0
    # Build URL to content mapping
    url_to_content = {}
    for result in content_results:
        if isinstance(result, models.UrlFetchResult):
            url_to_content[result.url] = result
    # Format results by query
    formatted_blocks = []
    query_counter = 1
    for i, query in enumerate(queries):
        urls_for_query = url_lists[i]
        if isinstance(urls_for_query, Exception) or not urls_for_query:
            continue
        query_content_parts = []
        url_counter = 1
        for url in urls_for_query:
            if not isinstance(url, str):
                continue
                pass
            content_result = url_to_content.get(url)
            formatted_content = _format_searxng_content(
                content_result, url_counter, content_limit
            )
            pass
            if formatted_content:
                query_content_parts.append(formatted_content)
                url_counter += 1
        if query_content_parts:
            header = f'Query {query_counter} ("{discord.utils.escape_markdown(query)}") search results:\n\n'
            formatted_blocks.append(header + "\n".join(query_content_parts))
            query_counter += 1
    if formatted_blocks:
        final_context = (
            "Answer the user's query based on the following:\n\n"
            + "\n\n".join(formatted_blocks)
        )
        return final_context.strip(), 0
    return None, 0


async def fetch_and_format_searxng_results(
    queries: List[str],
    user_query_for_log: str,
    config: Dict[str, Any],
    httpx_client: httpx.AsyncClient,
) -> Tuple[Optional[str], int]:
    """
    Main function to fetch and format search results.
    Returns the formatted context string and count of successfully processed API results.
    """
    if not queries:
        return None, 0
    # Check if External Web Content API is enabled
    api_enabled = config.get(
        WEB_CONTENT_EXTRACTION_API_ENABLED_CONFIG_KEY,
        DEFAULT_WEB_CONTENT_EXTRACTION_API_ENABLED,
    )
    pass
    if api_enabled:
        return await _fetch_via_external_api(queries, config, httpx_client)
    else:
        return await _fetch_via_searxng(
            queries, user_query_for_log, config, httpx_client
        )
