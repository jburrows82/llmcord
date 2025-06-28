import asyncio
import logging
import json  # For parsing LLM response
import base64  # Added for image encoding
import hashlib  # Added for caching
from datetime import datetime, timedelta  # Added for date/time and cache expiry
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator, Tuple

import discord  # For discord.utils.escape_markdown
import httpx

from ..core import models  # Use relative import
from ..core.constants import (
    SEARXNG_BASE_URL_CONFIG_KEY,
    SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY,
    GROUNDING_MODEL_CONFIG_KEY,
    GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY,  # Added
    GROUNDING_MODEL_TOP_K_CONFIG_KEY,  # Added
    GROUNDING_MODEL_TOP_P_CONFIG_KEY,  # Added
    DEFAULT_GROUNDING_MODEL_TEMPERATURE,  # Added
    DEFAULT_GROUNDING_MODEL_TOP_K,  # Added
    DEFAULT_GROUNDING_MODEL_TOP_P,  # Added
    # Grounding Model Thinking Budget
    GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY,
    GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY,
    GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET,
    GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE,
    SEARXNG_NUM_RESULTS_CONFIG_KEY,
    AllKeysFailedError,
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
    # Crawl4AI
    CRAWL4AI_CACHE_MODE_CONFIG_KEY,  # Added
    DEFAULT_CRAWL4AI_CACHE_MODE,  # Added
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
    fetch_general_url_content,  # This is the dynamic one from content_fetchers.web
)
from ..core.utils import is_youtube_url, is_reddit_url, extract_reddit_submission_id

Config = Dict[str, Any]  # For type hinting and accessing config
# The generate_response_stream function will be passed as an argument
# Note: LLMHandler import removed, will use Callable for the stream function

# Simple in-memory cache for API responses
_api_response_cache: Dict[str, Tuple[Dict[str, Any], datetime, int]] = {}

MAX_CONCURRENT_WEB_CONTENT_API_REQUESTS = (
    10  # Hard-limit to avoid saturating the local API server
)


def _get_cache_key(
    query: str,
    api_url: str,
    api_max_results: int,
    max_char_per_url: Optional[int] = None,
) -> str:
    """Generate a cache key for API requests."""
    key_data = f"{query}|{api_url}|{api_max_results}|{max_char_per_url}"
    return hashlib.md5(key_data.encode()).hexdigest()


def _get_cached_response(
    cache_key: str, cache_ttl_minutes: int
) -> Optional[Dict[str, Any]]:
    """Get cached response if it exists and hasn't expired."""
    if cache_key in _api_response_cache:
        cached_response, cache_time, stored_ttl = _api_response_cache[cache_key]
        # Use the stored TTL from when the item was cached, not the current call's TTL
        if datetime.now() - cache_time < timedelta(minutes=stored_ttl):
            return cached_response
        else:
            # Remove expired cache entry
            del _api_response_cache[cache_key]
    return None


def _cache_response(
    cache_key: str, response: Dict[str, Any], cache_ttl_minutes: int
) -> None:
    """Cache an API response."""
    _api_response_cache[cache_key] = (response, datetime.now(), cache_ttl_minutes)

    # Periodically clean up expired entries (every 100 cache operations)
    if len(_api_response_cache) > 0 and len(_api_response_cache) % 100 == 0:
        _cleanup_expired_cache()


def _cleanup_expired_cache() -> None:
    """Remove expired entries from the cache."""
    current_time = datetime.now()
    expired_keys = [
        key
        for key, (_, cache_time, stored_ttl) in _api_response_cache.items()
        if current_time - cache_time >= timedelta(minutes=stored_ttl)
    ]
    for key in expired_keys:
        del _api_response_cache[key]
    if expired_keys:
        logging.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


async def _fetch_batch_queries_from_web_content_api(
    queries: List[str],
    client: httpx.AsyncClient,
    api_url: str,
    api_max_results: int,
    max_char_per_url: Optional[int] = None,
    cache_ttl_minutes: int = DEFAULT_WEB_CONTENT_EXTRACTION_API_CACHE_TTL,
) -> List[Optional[Dict[str, Any]]]:
    """
    Optimized batch fetching with caching and reduced timeout.
    Returns results in the same order as input queries.
    This implementation additionally:
    1. Eliminates duplicate uncached queries so we never hit the API twice for the same query in a single batch.
    2. Uses a semaphore to constrain the level of concurrency to avoid overwhelming the local API while still achieving high throughput.
    """
    # Pre-allocate result list so we can easily fill by index later on.
    results: List[Optional[Dict[str, Any]]] = [None] * len(queries)

    # ---------------------------------------------------------------------
    # Pass 1 – satisfy as many queries as possible from the in-memory cache.
    # ---------------------------------------------------------------------
    uncached_query_to_indices: Dict[str, List[int]] = {}
    for idx, query in enumerate(queries):
        cache_key = _get_cache_key(query, api_url, api_max_results, max_char_per_url)
        cached_result = _get_cached_response(cache_key, cache_ttl_minutes)
        if cached_result is not None:
            logging.debug(f"Using cached result for query: '{query}'")
            results[idx] = cached_result
        else:
            # Keep track of every position in *queries* that needs this uncached query.
            uncached_query_to_indices.setdefault(query, []).append(idx)

    # Short-circuit if everything was already cached.
    if not uncached_query_to_indices:
        return results

    logging.info(
        f"Fetching {len(uncached_query_to_indices)} unique uncached queries (mapped from {sum(len(v) for v in uncached_query_to_indices.values())} total) from External Web Content API"
    )

    # ---------------------------------------------------------------------
    # Pass 2 – perform the actual network requests, constrained by a semaphore.
    # ---------------------------------------------------------------------
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WEB_CONTENT_API_REQUESTS)

    async def _sem_fetch(q: str) -> Optional[Dict[str, Any]]:
        """Wrapper that enforces the concurrency limit via *semaphore*."""
        async with semaphore:
            return await _fetch_single_query_from_web_content_api(
                q, client, api_url, api_max_results, max_char_per_url
            )

    # Kick off tasks for every unique uncached query.
    fetch_tasks = {
        query: asyncio.create_task(_sem_fetch(query))
        for query in uncached_query_to_indices
    }

    # Wait for all tasks to finish (preserving errors for logging below).
    await asyncio.gather(*fetch_tasks.values(), return_exceptions=True)

    # ---------------------------------------------------------------------
    # Pass 3 – populate *results* and cache successful responses.
    # ---------------------------------------------------------------------
    for query, task in fetch_tasks.items():
        if task.cancelled():
            continue  # Skip cancelled tasks (shouldn't normally happen)

        exc = task.exception()
        if exc is not None:
            logging.error(f"Exception during API call for query '{query}': {exc}")
            continue  # Leave corresponding indices as None

        result_payload: Optional[Dict[str, Any]] = task.result()

        if result_payload is not None:
            # Cache the successful response so future batches can reuse it.
            cache_key = _get_cache_key(
                query, api_url, api_max_results, max_char_per_url
            )
            _cache_response(cache_key, result_payload, cache_ttl_minutes)

        # Assign the fetched result (whatever it is) to every original index
        for idx in uncached_query_to_indices[query]:
            results[idx] = result_payload

    return results


async def get_web_search_queries_from_gemini(
    history_for_gemini_grounding: List[Dict[str, Any]],
    system_prompt_text_for_grounding: Optional[str],
    config: Dict[str, Any],
    # Pass the generate_response_stream function as a callable
    generate_response_stream_func: Callable[
        [
            str,
            str,
            List[Dict[str, Any]],
            Optional[str],
            Dict[str, Any],
            Dict[str, Any],
            Dict[str, Any],
        ],
        AsyncGenerator[
            Tuple[
                Optional[str],
                Optional[str],
                Optional[Any],
                Optional[str],
                Optional[bytes],
                Optional[str],
            ],
            None,
        ],
    ],
) -> Optional[List[str]]:
    """
    Calls Gemini to get web search queries from its grounding metadata.
    Returns search queries immediately when found, without waiting for full response.
    """
    logging.info("Attempting to get web search queries from Gemini for grounding...")
    grounding_model_str = config.get(
        GROUNDING_MODEL_CONFIG_KEY, "google/gemini-2.5-flash"
    )
    try:
        grounding_provider, grounding_model_name = grounding_model_str.split("/", 1)
    except ValueError:
        logging.error(
            f"Invalid format for '{GROUNDING_MODEL_CONFIG_KEY}': {grounding_model_str}. Expected 'provider/model_name'."
        )
        return None

    gemini_provider_config = config.get("providers", {}).get(grounding_provider, {})
    if not gemini_provider_config or not gemini_provider_config.get("api_keys"):
        logging.warning(
            f"Cannot perform Gemini grounding step: Provider '{grounding_provider}' (from '{GROUNDING_MODEL_CONFIG_KEY}') not configured with API keys."
        )
        return None

    all_web_search_queries = set()  # Use a set to store unique queries

    try:
        # Parameters for the grounding model call.
        # These are fetched from the application config, with defaults.
        grounding_temperature = config.get(
            GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY, DEFAULT_GROUNDING_MODEL_TEMPERATURE
        )
        grounding_top_k = config.get(
            GROUNDING_MODEL_TOP_K_CONFIG_KEY, DEFAULT_GROUNDING_MODEL_TOP_K
        )
        grounding_top_p = config.get(
            GROUNDING_MODEL_TOP_P_CONFIG_KEY, DEFAULT_GROUNDING_MODEL_TOP_P
        )

        grounding_extra_params = {
            "temperature": grounding_temperature,
            "top_k": grounding_top_k,
            "top_p": grounding_top_p,
        }
        # Add thinking budget if configured and model is Gemini
        grounding_use_thinking_budget = config.get(
            GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY,
            GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET,
        )
        if grounding_use_thinking_budget and grounding_provider == "google":
            grounding_thinking_budget_value = config.get(
                GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY,
                GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE,
            )
            # Ensure the model is Flash, as thinkingBudget is only supported in Gemini 2.5 Flash
            if "flash" in grounding_model_name.lower():
                grounding_extra_params["thinking_budget"] = (
                    grounding_thinking_budget_value
                )
                logging.info(
                    f"Applying thinking_budget: {grounding_thinking_budget_value} to Gemini grounding model {grounding_model_name}"
                )
            else:
                logging.warning(
                    f"Thinking budget is configured for grounding model {grounding_model_name}, but it's not a Gemini Flash model. Ignoring thinking_budget."
                )

        stream_generator = generate_response_stream_func(
            provider=grounding_provider,
            model_name=grounding_model_name,
            history_for_llm=history_for_gemini_grounding,
            system_prompt_text=system_prompt_text_for_grounding,
            provider_config=gemini_provider_config,
            extra_params=grounding_extra_params,
            app_config=config,  # Pass app_config
        )

        async for (
            _,
            _,
            chunk_grounding_metadata,
            error_message,
            _,
            _,
        ) in stream_generator:
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
                    
                    # Return search queries immediately when found
                    if all_web_search_queries:
                        logging.info(
                            f"Gemini grounding produced {len(all_web_search_queries)} unique search queries. Returning immediately without waiting for full response."
                        )
                        return list(all_web_search_queries)

        # If we get here, no search queries were found
        logging.info(
            "Gemini grounding call completed but found no web_search_queries in metadata."
        )
        return None

    except (
        AllKeysFailedError
    ) as e:  # Make sure AllKeysFailedError is accessible or handled
        logging.error(
            f"All API keys failed for Gemini grounding model ({grounding_provider}/{grounding_model_name}): {e}"
        )
        return None
    except Exception:
        logging.exception(
            "Unexpected error during Gemini grounding call to get search queries:"
        )
        return None


async def get_web_search_queries_from_gemini_force_stop(
    history_for_gemini_grounding: List[Dict[str, Any]],
    system_prompt_text_for_grounding: Optional[str],
    config: Dict[str, Any],
    # Pass the generate_response_stream function as a callable
    generate_response_stream_func: Callable[
        [
            str,
            str,
            List[Dict[str, Any]],
            Optional[str],
            Dict[str, Any],
            Dict[str, Any],
            Dict[str, Any],
        ],
        AsyncGenerator[
            Tuple[
                Optional[str],
                Optional[str],
                Optional[Any],
                Optional[str],
                Optional[bytes],
                Optional[str],
            ],
            None,
        ],
    ],
) -> Optional[List[str]]:
    """
    Calls Gemini to get web search queries from its grounding metadata.
    Forcefully stops the stream immediately after getting search queries to minimize API usage.
    """
    logging.info("Attempting to get web search queries from Gemini for grounding (force-stop mode)...")
    grounding_model_str = config.get(
        GROUNDING_MODEL_CONFIG_KEY, "google/gemini-2.5-flash"
    )
    try:
        grounding_provider, grounding_model_name = grounding_model_str.split("/", 1)
    except ValueError:
        logging.error(
            f"Invalid format for '{GROUNDING_MODEL_CONFIG_KEY}': {grounding_model_str}. Expected 'provider/model_name'."
        )
        return None

    gemini_provider_config = config.get("providers", {}).get(grounding_provider, {})
    if not gemini_provider_config or not gemini_provider_config.get("api_keys"):
        logging.warning(
            f"Cannot perform Gemini grounding step: Provider '{grounding_provider}' (from '{GROUNDING_MODEL_CONFIG_KEY}') not configured with API keys."
        )
        return None

    all_web_search_queries = set()  # Use a set to store unique queries

    try:
        # Parameters for the grounding model call.
        # These are fetched from the application config, with defaults.
        grounding_temperature = config.get(
            GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY, DEFAULT_GROUNDING_MODEL_TEMPERATURE
        )
        grounding_top_k = config.get(
            GROUNDING_MODEL_TOP_K_CONFIG_KEY, DEFAULT_GROUNDING_MODEL_TOP_K
        )
        grounding_top_p = config.get(
            GROUNDING_MODEL_TOP_P_CONFIG_KEY, DEFAULT_GROUNDING_MODEL_TOP_P
        )

        grounding_extra_params = {
            "temperature": grounding_temperature,
            "top_k": grounding_top_k,
            "top_p": grounding_top_p,
        }
        # Add thinking budget if configured and model is Gemini
        grounding_use_thinking_budget = config.get(
            GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY,
            GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET,
        )
        if grounding_use_thinking_budget and grounding_provider == "google":
            grounding_thinking_budget_value = config.get(
                GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY,
                GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE,
            )
            # Ensure the model is Flash, as thinkingBudget is only supported in Gemini 2.5 Flash
            if "flash" in grounding_model_name.lower():
                grounding_extra_params["thinking_budget"] = (
                    grounding_thinking_budget_value
                )
                logging.info(
                    f"Applying thinking_budget: {grounding_thinking_budget_value} to Gemini grounding model {grounding_model_name}"
                )
            else:
                logging.warning(
                    f"Thinking budget is configured for grounding model {grounding_model_name}, but it's not a Gemini Flash model. Ignoring thinking_budget."
                )

        stream_generator = generate_response_stream_func(
            provider=grounding_provider,
            model_name=grounding_model_name,
            history_for_llm=history_for_gemini_grounding,
            system_prompt_text=system_prompt_text_for_grounding,
            provider_config=gemini_provider_config,
            extra_params=grounding_extra_params,
            app_config=config,  # Pass app_config
        )

        # Use asyncio.wait_for with a timeout to force-stop if queries are found
        try:
            async for (
                _,
                _,
                chunk_grounding_metadata,
                error_message,
                _,
                _,
            ) in stream_generator:
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
                        
                        # Force stop by breaking and closing the generator
                        if all_web_search_queries:
                            logging.info(
                                f"Gemini grounding produced {len(all_web_search_queries)} unique search queries. Force-stopping stream immediately."
                            )
                            # Try to close the generator to stop the stream
                            try:
                                await stream_generator.aclose()
                            except Exception as e:
                                logging.debug(f"Error closing stream generator: {e}")
                            return list(all_web_search_queries)

        except asyncio.CancelledError:
            logging.info("Gemini grounding stream was cancelled after getting search queries.")
            if all_web_search_queries:
                return list(all_web_search_queries)
            return None

        # If we get here, no search queries were found
        logging.info(
            "Gemini grounding call completed but found no web_search_queries in metadata."
        )
        return None

    except (
        AllKeysFailedError
    ) as e:  # Make sure AllKeysFailedError is accessible or handled
        logging.error(
            f"All API keys failed for Gemini grounding model ({grounding_provider}/{grounding_model_name}): {e}"
        )
        return None
    except Exception:
        logging.exception(
            "Unexpected error during Gemini grounding call to get search queries:"
        )
        return None


async def _fetch_single_query_from_web_content_api(
    query_item_str: str,
    client: httpx.AsyncClient,
    api_url: str,
    api_max_results: int,
    max_char_per_url: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Helper to fetch and do initial processing for a single query to the external web content API."""
    logging.info(
        f"External Web Content API: Initiating fetch for query: '{query_item_str}'"
    )
    payload = {"query": query_item_str, "max_results": api_max_results}
    if max_char_per_url is not None:
        payload["max_char_per_url"] = max_char_per_url
    try:
        # Optimized timeout with granular settings for better performance
        response = await client.post(
            api_url,
            json=payload,
            timeout=httpx.Timeout(connect=8.0, read=15.0, write=8.0, pool=5.0),
        )
        response.raise_for_status()
        api_response_json = response.json()

        # Check for API's own error field
        if api_response_json.get("error"):
            logging.error(
                f"External Web Content API returned a global error for query '{query_item_str}': {api_response_json['error']}"
            )
            return None  # API indicated an error for this specific query
        return api_response_json

    except httpx.RequestError as e:
        logging.error(
            f"HTTP RequestError calling External Web Content API for query '{query_item_str}' at {api_url}: {e}"
        )
        return None
    except httpx.HTTPStatusError as e:
        logging.error(
            f"HTTP StatusError from External Web Content API for query '{query_item_str}' (status {e.response.status_code}): {e.response.text}"
        )
        return None
    except json.JSONDecodeError as e:  # If response.json() fails
        logging.error(
            f"JSONDecodeError parsing External Web Content API response for query '{query_item_str}': {e}"
        )
        return None
    except Exception as e_generic:  # Catch other unexpected errors during the fetch
        logging.exception(
            f"Unexpected error while fetching/parsing External Web Content API response for query '{query_item_str}': {e_generic}"
        )
        return None


async def fetch_and_format_searxng_results(
    queries: List[str],
    user_query_for_log: str,  # For logging context
    config: Dict[str, Any],
    httpx_client: httpx.AsyncClient,
) -> Tuple[Optional[str], int]:
    """
    Fetches search results from SearxNG for given queries,
    then fetches content of those URLs (uniquely) and formats them.
    Uses specific fetchers for YouTube and Reddit URLs.
    Returns the formatted context string and the count of successfully processed API results.
    """
    if not queries:
        return None, 0

    successful_api_results_count = 0
    # --- External Web Content API Integration ---
    api_enabled = config.get(
        WEB_CONTENT_EXTRACTION_API_ENABLED_CONFIG_KEY,
        DEFAULT_WEB_CONTENT_EXTRACTION_API_ENABLED,
    )
    api_url = config.get(
        WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY,
        DEFAULT_WEB_CONTENT_EXTRACTION_API_URL,
    )
    api_max_results = config.get(
        WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY,
        DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS,
    )
    if api_enabled:
        max_char_per_url = config.get(
            SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY
        )  # Send to API as max_char_per_url parameter
        cache_ttl_minutes = config.get(
            WEB_CONTENT_EXTRACTION_API_CACHE_TTL_CONFIG_KEY,
            DEFAULT_WEB_CONTENT_EXTRACTION_API_CACHE_TTL,
        )
        all_formatted_contexts_from_api = []
        logging.info(
            f"External Web Content API is enabled. Processing {len(queries)} queries with caching (TTL: {cache_ttl_minutes}min) and batch optimization."
        )

        # Use optimized batch fetching with caching
        api_call_results = await _fetch_batch_queries_from_web_content_api(
            queries,
            httpx_client,
            api_url,
            api_max_results,
            max_char_per_url,
            cache_ttl_minutes,
        )

        for query_idx, api_response_json in enumerate(api_call_results):
            query_item_str = queries[query_idx]  # Get the original query for context

            if api_response_json is None:
                # This means the batch fetch returned None for this query,
                # indicating an error that was already logged (e.g., API error field, HTTP error, JSON decode).
                logging.debug(
                    f"Skipping query '{query_item_str}' due to previously logged API call failure or empty/error response from API."
                )
                continue

            # --- Start of existing processing logic for a single successful API response ---
            logging.info(
                f"External Web Content API: Processing successful response for query {query_idx + 1}/{len(queries)}: '{query_item_str}'"
            )

            api_results_data = api_response_json.get("results", [])
            if not api_results_data:
                logging.info(
                    f"External Web Content API returned no 'results' data for query '{query_item_str}'."
                )
                continue  # Skip to the next query's result

            current_query_formatted_api_results_content = []
            for item_idx, item in enumerate(api_results_data):
                item_str_parts = []
                item_url = item.get("url", "N/A")
                item_source_type = item.get("source_type", "unknown")

                item_str_parts.append(
                    f"Source {item_idx + 1}: {item_url} (Type: {item_source_type})"
                )

                if (
                    item.get("processed_successfully")
                    and item.get("data")
                    and not item.get("error")
                ):
                    successful_api_results_count += (
                        1  # Increment count for successful items
                    )
                    data = item["data"]

                    if item_source_type == "youtube":
                        title = data.get("title", "N/A")
                        item_str_parts.append(
                            f"  Title: {discord.utils.escape_markdown(title)}"
                        )
                        channel = data.get("channel_name", "N/A")
                        item_str_parts.append(
                            f"  Channel: {discord.utils.escape_markdown(channel)}"
                        )
                        transcript = data.get("transcript")
                        if transcript:
                            # No client-side truncation needed when using web content extraction API
                            # as the API handles max_char_per_url server-side
                            item_str_parts.append(
                                f"  Transcript: {discord.utils.escape_markdown(transcript)}"
                            )
                        comments = data.get("comments")
                        if comments and isinstance(comments, list):
                            comment_texts = [
                                f"    - {discord.utils.escape_markdown(c.get('text', ''))}"
                                for c in comments
                                if isinstance(c, dict) and c.get("text")
                            ]
                            if comment_texts:
                                item_str_parts.append(
                                    "  Comments:\n" + "\n".join(comment_texts)
                                )

                    elif item_source_type == "reddit":
                        title = data.get("post_title", "N/A")
                        item_str_parts.append(
                            f"  Title: {discord.utils.escape_markdown(title)}"
                        )
                        body = data.get("post_body")
                        if body:
                            # No client-side truncation needed when using web content extraction API
                            # as the API handles max_char_per_url server-side
                            item_str_parts.append(
                                f"  Post Body: {discord.utils.escape_markdown(body)}"
                            )
                        comments = data.get("comments")
                        if comments and isinstance(comments, list):
                            comment_texts = [
                                f"    - {discord.utils.escape_markdown(c.get('text', ''))} (Score: {c.get('score', 'N/A')})"
                                for c in comments
                                if isinstance(c, dict) and c.get("text")
                            ]
                            if comment_texts:
                                item_str_parts.append(
                                    "  Comments:\n" + "\n".join(comment_texts)
                                )

                    elif item_source_type == "pdf":
                        text_content = data.get("text_content")
                        if text_content:
                            # No client-side truncation needed when using web content extraction API
                            # as the API handles max_char_per_url server-side
                            item_str_parts.append(
                                f"  Content: {discord.utils.escape_markdown(text_content)}"
                            )

                    elif item_source_type == "webpage":
                        title = data.get("title")
                        if title:
                            item_str_parts.append(
                                f"  Title: {discord.utils.escape_markdown(title)}"
                            )
                        text_content = data.get("text_content")
                        if text_content:
                            # No client-side truncation needed when using web content extraction API
                            # as the API handles max_char_per_url server-side
                            item_str_parts.append(
                                f"  Content: {discord.utils.escape_markdown(text_content)}"
                            )
                    else:  # unknown or other types
                        raw_content_str = str(data)
                        # No client-side truncation needed when using web content extraction API
                        # as the API handles max_char_per_url server-side
                        item_str_parts.append(
                            f"  Data: {discord.utils.escape_markdown(raw_content_str)}"
                        )
                    current_query_formatted_api_results_content.append(
                        "\n".join(item_str_parts)
                    )
                elif item.get("error"):
                    logging.warning(
                        f"External API processing error for URL {item_url} (query: '{query_item_str}'): {item.get('error')}"
                    )
                    item_str_parts.append(f"  Error: {item.get('error')}")
                    current_query_formatted_api_results_content.append(
                        "\n".join(item_str_parts)
                    )
            # End of loop for 'item in api_results_data'

            if current_query_formatted_api_results_content:
                header = (
                    f"Information from External Web Content API "
                    f'(for query: "{discord.utils.escape_markdown(query_item_str)}"): \n\n'
                )
                single_query_context = header + "\n\n---\n\n".join(
                    current_query_formatted_api_results_content
                )
                all_formatted_contexts_from_api.append(single_query_context)
            else:
                logging.info(
                    f"External Web Content API did not return any processable content for query '{query_item_str}' after processing results."
                )
            # --- End of existing processing logic for a single successful API response ---
        # End of loop for 'result_or_exc in api_call_results_or_exceptions'

        if all_formatted_contexts_from_api:
            # Add a general header if combining results from multiple API queries
            combined_header = "Answer the user's query based on the following information from the External Web Content API (results from one or more queries):\n\n"
            final_api_context = combined_header + "\n\n=====\n\n".join(
                all_formatted_contexts_from_api
            )
            logging.info(
                f"Aggregated formatted context from External Web Content API for {len(all_formatted_contexts_from_api)} queries: {final_api_context[:500]}..."
            )
            return final_api_context.strip(), successful_api_results_count
        else:
            logging.info(
                "External Web Content API processing completed for all queries, but no processable content was gathered."
            )
            return None, successful_api_results_count

    # --- Original SearXNG and content fetching logic (if external API is not enabled or fails to return content) ---
    # successful_api_results_count will remain 0 if this path is taken.
    searxng_base_url = config.get(SEARXNG_BASE_URL_CONFIG_KEY)
    if not searxng_base_url:
        logging.warning(
            "SearxNG base URL not configured. Skipping web search enhancement."
        )
        return None, 0

    logging.info(
        f"Fetching SearxNG results for {len(queries)} queries related to user query: '{user_query_for_log[:100]}...'"
    )
    searxng_content_limit = config.get(
        SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY
    )  # Already defined as api_content_limit if API was enabled

    # API keys for specific fetchers (LLMCord's internal fetchers)
    youtube_api_key = config.get("youtube_api_key")
    reddit_client_id = config.get("reddit_client_id")
    reddit_client_secret = config.get("reddit_client_secret")
    reddit_user_agent = config.get("reddit_user_agent")

    # Fetch SearxNG URLs for all queries concurrently
    searxng_tasks = []
    for query in queries:
        num_results_to_fetch = config.get(SEARXNG_NUM_RESULTS_CONFIG_KEY, 5)
        searxng_tasks.append(
            fetch_searxng_results(
                query, httpx_client, searxng_base_url, num_results_to_fetch
            )
        )

    try:
        list_of_url_lists_per_query = await asyncio.gather(
            *searxng_tasks, return_exceptions=True
        )
    except Exception:
        logging.exception("Error gathering SearxNG results.")
        return None, 0

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
        return None, 0

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
                    main_extractor=config.get(
                        MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY,
                        DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR,
                    ),
                    fallback_extractor=config.get(
                        FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY,
                        DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR,
                    ),
                    max_text_length=searxng_content_limit,
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
                    crawl4ai_cache_mode=config.get(  # Added
                        CRAWL4AI_CACHE_MODE_CONFIG_KEY, DEFAULT_CRAWL4AI_CACHE_MODE
                    ),
                )
            )

    try:
        fetched_unique_content_results = await asyncio.gather(
            *url_content_processing_tasks, return_exceptions=True
        )
    except Exception:
        logging.exception("Error gathering content from unique SearxNG URLs.")
        return None, 0

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
                    content_str_part = discord.utils.escape_markdown(raw_content_str)  # type: ignore

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
        return final_context.strip(), 0  # 0 successful API results for SearxNG path

    logging.info(
        "No content successfully fetched and formatted from SearxNG result URLs."
    )
    return None, 0


async def generate_search_queries_with_custom_prompt(
    latest_query: str,
    chat_history: List[Dict[str, Any]],
    config: Config,
    # Use Callable for the generate_response_stream function, similar to get_web_search_queries_from_gemini
    generate_response_stream_func: Callable[
        [
            str,
            str,
            List[Dict[str, Any]],
            Optional[str],
            Dict[str, Any],
            Dict[str, Any],
            Config,
        ],  # Added Config to signature based on llm_handler call
        AsyncGenerator[
            Tuple[
                Optional[str],
                Optional[str],
                Optional[Any],
                Optional[str],
                Optional[bytes],
                Optional[str],
            ],
            None,
        ],
    ],
    current_model_id: str,
    httpx_client: httpx.AsyncClient,  # Added httpx_client
    image_urls: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """
    Generates search queries using a custom prompt with a non-Gemini model, potentially with images.
    """
    logging.info(
        f"Attempting to generate search queries with custom prompt for model: {current_model_id}"
    )

    alt_search_config_dict = config.get("alternative_search_query_generation", {})
    prompt_template = alt_search_config_dict.get(
        "search_query_generation_prompt_template", ""
    )
    system_prompt_template = (
        alt_search_config_dict.get(  # New: Get system prompt template
            "search_query_generation_system_prompt", ""
        )
    )

    if not prompt_template:
        # This condition covers both missing 'alternative_search_query_generation' key
        # and missing 'search_query_generation_prompt_template' key within it,
        # or if the template itself is an empty string.
        logging.warning(
            "Search query generation prompt template is not configured, not found, or is empty. Skipping."
        )
        return None

    now = datetime.now()
    current_date_str = now.strftime("%Y-%m-%d")
    current_day_of_week_str = now.strftime("%A")
    current_time_str = now.strftime("%I:%M %p")  # e.g., "02:30 PM"

    # Prepare system prompt
    final_system_prompt_text = None
    if system_prompt_template:
        final_system_prompt_text = system_prompt_template.replace(
            "{current_date}", current_date_str
        )
        # Add other placeholders if needed for system prompt, e.g., {current_day_of_week}, {current_time}

        logging.info(
            f"Using system prompt for search query generation: {final_system_prompt_text}"
        )

    # 1. Format Chat History (from chat_history[:-1])
    formatted_chat_history_parts = []
    history_to_format = chat_history[:-1]  # History leading up to the latest query

    for message_dict in history_to_format:
        role = message_dict.get("role")
        text_content = ""

        if role == "model" or role == "assistant":
            role_for_display = "assistant"
        elif role == "user":
            role_for_display = "user"
        else:
            continue  # Skip unknown roles

        # Extract text content from message_dict
        # message_dict structure can be like:

        # OpenAI: {"role": "user/assistant", "content": "text_string" or [{"type":"text", "text":"..."}, {"type":"image_url", ...}]}
        if "parts" in message_dict:  # Likely Gemini-style from build_message_history
            for part in message_dict["parts"]:
                if hasattr(part, "text") and part.text:
                    text_content = part.text
                    break
        elif "content" in message_dict:  # Likely OpenAI-style
            if isinstance(message_dict["content"], str):
                text_content = message_dict["content"]
            elif isinstance(message_dict["content"], list):
                for item_part in message_dict["content"]:
                    if isinstance(item_part, dict) and item_part.get("type") == "text":
                        text_content = item_part.get("text", "")
                        break

        if text_content.strip():
            formatted_chat_history_parts.append(
                f"{role_for_display}: {text_content.strip()}"
            )

    formatted_chat_history_string = "\n\n".join(formatted_chat_history_parts)

    # 2. Prepare the Single Prompt Text
    # Replace basic placeholders first
    current_prompt_text = prompt_template.replace("{latest_query}", latest_query)
    current_prompt_text = current_prompt_text.replace(
        "{current_date}", current_date_str
    )
    current_prompt_text = current_prompt_text.replace(
        "{current_day_of_week}", current_day_of_week_str
    )
    current_prompt_text = current_prompt_text.replace(
        "{current_time}", current_time_str
    )

    # Inject formatted chat history
    if "{chat_history}" in current_prompt_text:
        final_prompt_text = current_prompt_text.replace(
            "{chat_history}", formatted_chat_history_string
        )
    else:
        logging.warning(
            "'{chat_history}' placeholder not found in search_query_generation_prompt_template. Chat history will not be textually injected as a block."
        )
        # Optionally, append if placeholder is missing and history exists
        # if formatted_chat_history_string:

        final_prompt_text = (
            current_prompt_text  # Default to not appending if placeholder missing
        )

    # 3. Process Images (same as before)
    processed_image_data_urls = []
    if image_urls and httpx_client:
        for img_url in image_urls:
            try:
                response = await httpx_client.get(img_url, timeout=10)
                response.raise_for_status()
                image_bytes = await response.aread()
                mime_type = response.headers.get("Content-Type", "image/jpeg")
                if not mime_type.startswith("image/"):
                    logging.warning(
                        f"Content-Type '{mime_type}' from {img_url} is not an image type. Skipping image."
                    )
                    continue
                base64_encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                data_url = f"data:{mime_type};base64,{base64_encoded_image}"
                processed_image_data_urls.append(data_url)
            except Exception as e:
                logging.error(
                    f"Error processing image {img_url} for search query gen: {e}"
                )

    # 4. Construct user_prompt_content_parts for the API call
    # This will be a list of parts, e.g., [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
    # This structure is generally for OpenAI-like models; llm_handler will adapt it for Gemini if needed.
    user_prompt_content_parts_for_api = []
    user_prompt_content_parts_for_api.append(
        {"type": "text", "text": final_prompt_text}
    )

    if processed_image_data_urls:
        for data_url in processed_image_data_urls:
            user_prompt_content_parts_for_api.append(
                {"type": "image_url", "image_url": {"url": data_url}}
            )

    # 5. Construct messages_for_llm as a single user turn
    messages_for_llm = [{"role": "user", "content": user_prompt_content_parts_for_api}]

    try:
        provider_name, model_name = current_model_id.split("/", 1)
    except ValueError:
        logging.error(
            f"Invalid format for current_model_id: {current_model_id}. Expected 'provider/model_name'."
        )
        return None

    provider_config = config.get("providers", {}).get(provider_name, {})
    if not provider_config or not provider_config.get("api_keys"):
        logging.warning(
            f"Cannot generate search queries: Provider '{provider_name}' (from current_model_id '{current_model_id}') not configured with API keys."
        )
        return None

    final_response_text = ""
    try:
        # Parameters for the LLM call.
        # Using default temperature, top_k, top_p from the model's configuration or llm_handler defaults.
        # If specific overrides are needed, they could be fetched from config.alternative_search_query_generation
        extra_params = {}  # Let the stream function use its defaults or model-specific settings

        # Call the passed stream generation function
        stream_generator = generate_response_stream_func(
            provider=provider_name,
            model_name=model_name,
            history_for_llm=messages_for_llm,
            system_prompt_text=final_system_prompt_text,  # Use the prepared system prompt
            provider_config=provider_config,
            extra_params=extra_params,
            app_config=config,  # Pass app_config
        )

        async for text_chunk, _, _, error_message, _, _ in stream_generator:  # type: ignore
            if error_message:
                logging.error(
                    f"LLM call error during custom search query generation: {error_message}"
                )
                return None  # Abort on first error
            if text_chunk:
                final_response_text += text_chunk

        logging.debug(
            f"LLM response for custom search query generation: {final_response_text}"
        )

        if not final_response_text:
            logging.warning(
                "LLM returned an empty response for search query generation."
            )
            return None

        # llm_response_content is the raw string from LLM (final_response_text in this context)
        content_to_process = final_response_text.strip()

        # Markdown stripping logic
        if content_to_process.startswith("```json"):
            content_to_process = content_to_process[len("```json") :].strip()
        elif content_to_process.startswith("```"):
            content_to_process = content_to_process[len("```") :].strip()

        if content_to_process.endswith("```"):
            content_to_process = content_to_process[: -len("```")].strip()

        # Parse as JSON and expect the new structure
        try:
            parsed_response = json.loads(content_to_process)
            if (
                isinstance(parsed_response, dict)
                and "web_search_required" in parsed_response
                and isinstance(parsed_response["web_search_required"], bool)
            ):
                if parsed_response["web_search_required"]:
                    search_queries = parsed_response.get("search_queries")
                    if isinstance(search_queries, list) and all(
                        isinstance(q, str) for q in search_queries
                    ):
                        logging.info(
                            f"Successfully generated {len(search_queries)} search queries via custom prompt."
                        )
                        return {
                            "web_search_required": True,
                            "search_queries": search_queries,
                        }
                    else:
                        logging.warning(
                            f"web_search_required is true but 'search_queries' is missing or invalid. Response: {content_to_process}"
                        )
                        return {
                            "web_search_required": True,
                            "search_queries": [],
                        }
                else:
                    logging.info(
                        "LLM indicated no web search is needed via 'web_search_required': false."
                    )
                    return {"web_search_required": False}
            else:
                logging.warning(
                    f"Parsed JSON does not match expected structure. Response: {content_to_process}"
                )
                return {"web_search_required": False}
        except json.JSONDecodeError:
            logging.warning(
                f"Failed to parse LLM response as JSON for search query generation. Response: {content_to_process}"
            )
            return {"web_search_required": False}

    except AllKeysFailedError as e:
        logging.error(
            f"All API keys failed for custom search query generation model ({provider_name}/{model_name}): {e}"
        )
        return None
    except Exception:
        logging.exception(
            "Unexpected error during custom search query generation LLM call:"
        )
        return None
