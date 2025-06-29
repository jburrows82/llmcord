import asyncio
import json
from typing import List, Dict, Any, Optional
import httpx
from ...core.constants import DEFAULT_WEB_CONTENT_EXTRACTION_API_CACHE_TTL
from .cache import get_cache_key, get_cached_response, cache_response
MAX_CONCURRENT_WEB_CONTENT_API_REQUESTS = 10
async def fetch_single_query_from_web_content_api(
    query_item_str: str,
    client: httpx.AsyncClient,
    api_url: str,
    api_max_results: int,
    max_char_per_url: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Helper to fetch and do initial processing for a single query to the external web content API."""
    payload = {"query": query_item_str, "max_results": api_max_results}
    if max_char_per_url is not None:
        payload["max_char_per_url"] = max_char_per_url
    try:
        # Optimized timeout with granular settings for better performance
        response = await client.post(
            api_url,
            json=payload,
            timeout=httpx.Timeout(connect=8.0, read=15.0, write=8.0, pool=5.0),
        response.raise_for_status()
        api_response_json = response.json()
        # Check for API's own error field
        if api_response_json.get("error"):
            return None  # API indicated an error for this specific query
        return api_response_json
    except httpx.RequestError as e:
        return None
    except httpx.HTTPStatusError as e:
        return None
    except json.JSONDecodeError as e:  # If response.json() fails
        return None
    except Exception as e_generic:  # Catch other unexpected errors during the fetch
        return None
async def fetch_batch_queries_from_web_content_api(
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
    """
    # Pre-allocate result list so we can easily fill by index later on.
    results: List[Optional[Dict[str, Any]]] = [None] * len(queries)
    # Pass 1 – satisfy as many queries as possible from the in-memory cache.
    uncached_query_to_indices: Dict[str, List[int]] = {}
    for idx, query in enumerate(queries):
        cache_key = get_cache_key(query, api_url, api_max_results, max_char_per_url)
        cached_result = get_cached_response(cache_key, cache_ttl_minutes)
        if cached_result is not None:
            results[idx] = cached_result
        else:
            # Keep track of every position in *queries* that needs this uncached query.
            uncached_query_to_indices.setdefault(query, []).append(idx)
    # Short-circuit if everything was already cached.
    if not uncached_query_to_indices:
        return results
    # Pass 2 – perform the actual network requests, constrained by a semaphore.
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WEB_CONTENT_API_REQUESTS)
    async def _sem_fetch(q: str) -> Optional[Dict[str, Any]]:
        """Wrapper that enforces the concurrency limit via *semaphore*."""
        async with semaphore:
            return await fetch_single_query_from_web_content_api(
                q, client, api_url, api_max_results, max_char_per_url
    # Kick off tasks for every unique uncached query.
    fetch_tasks = {
        query: asyncio.create_task(_sem_fetch(query))
        for query in uncached_query_to_indices
    }
    # Wait for all tasks to finish (preserving errors for logging below).
    await asyncio.gather(*fetch_tasks.values(), return_exceptions=True)
    # Pass 3 – populate *results* and cache successful responses.
    for query, task in fetch_tasks.items():
        if task.cancelled():
            continue  # Skip cancelled tasks (shouldn't normally happen)
        exc = task.exception()
        if exc is not None:
            continue  # Leave corresponding indices as None
        result_payload: Optional[Dict[str, Any]] = task.result()
        if result_payload is not None:
            # Cache the successful response so future batches can reuse it.
            cache_key = get_cache_key(
                query, api_url, api_max_results, max_char_per_url
            cache_response(cache_key, result_payload, cache_ttl_minutes)
        # Assign the fetched result (whatever it is) to every original index
        for idx in uncached_query_to_indices[query]:
            results[idx] = result_payload
    return results 