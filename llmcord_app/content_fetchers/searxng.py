import logging
from typing import List
import httpx
import urllib.parse

# httpx_client will be passed in from the bot instance


async def fetch_searxng_results(
    query: str,
    httpx_client: httpx.AsyncClient,
    base_url: str,
    num_results: int = 5,
    category: str = "general",  # Default to general, can be configurable
) -> List[str]:  # Returns a list of URLs
    """Fetches search results from a SearxNG instance."""
    urls = []
    if not base_url:
        logging.error("SearxNG base URL not configured.")
        return urls

    # Ensure base_url is clean and join with 'search'
    # Example: http://localhost:18088/search
    parsed_base_url = urllib.parse.urlsplit(base_url)
    search_url = urllib.parse.urlunsplit(
        (
            parsed_base_url.scheme,
            parsed_base_url.netloc,
            urllib.parse.urljoin(parsed_base_url.path, "search"),  # Join path carefully
            "",
            "",
        )  # Query and fragment are empty here, will be added by params
    )

    params = {
        "q": query,
        "format": "json",
        "categories": category,
        # As per SearxNG docs, there's no direct 'count' or 'limit' for results.
        # We fetch one page and take the top `num_results`.
    }

    try:
        logging.info(
            f"Querying SearxNG ('{query}') at {search_url} with params {params}"
        )
        response = await httpx_client.get(search_url, params=params, timeout=15.0)
        response.raise_for_status()

        data = response.json()
        results_data = data.get("results", [])

        for result in results_data:
            if len(urls) >= num_results:
                break
            if "url" in result and isinstance(result["url"], str):
                urls.append(result["url"])
            # Optional: could extract title/snippet here if needed for future enhancements
            # title = result.get("title")
            # content_snippet = result.get("content")

        logging.info(
            f"SearxNG returned {len(urls)} URLs for query '{query}'. Requested up to {num_results}."
        )

    except httpx.HTTPStatusError as e:
        logging.error(
            f"SearxNG HTTP error for query '{query}': {e.response.status_code} - {e.response.text}"
        )
    except httpx.RequestError as e:
        logging.error(
            f"SearxNG request error for query '{query}': {type(e).__name__} - {e}"
        )
    except ValueError as e:  # Includes JSONDecodeError
        logging.error(
            f"SearxNG JSON decode error for query '{query}': {e} (Response: {response.text[:200]}...)"
        )
    except Exception:
        logging.exception(
            f"Unexpected error fetching SearxNG results for query '{query}'"
        )

    return urls
