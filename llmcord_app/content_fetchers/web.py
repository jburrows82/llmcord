import logging
import re
from typing import Optional
import urllib.request  # Added for Jina
import asyncio  # Added for Jina

import httpx
from bs4 import BeautifulSoup
from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CacheMode,
)  # Added Crawl4AI imports

from ..models import UrlFetchResult


from ..constants import DEFAULT_JINA_ENGINE_MODE  # Added for Jina engine mode


async def fetch_with_jina(
    url: str,
    index: int,
    max_text_length: Optional[int] = None,
    jina_engine_mode: str = DEFAULT_JINA_ENGINE_MODE,
) -> UrlFetchResult:
    """Fetches content using Jina Reader."""
    logging.info(
        f"Attempting to fetch URL with Jina Reader: {url} (Engine: {jina_engine_mode})"
    )
    jina_reader_url = f"https://r.jina.ai/{url}"
    headers = {"X-Engine": jina_engine_mode}
    try:
        # urllib.request is synchronous, so run it in a thread
        def sync_jina_fetch():
            req = urllib.request.Request(jina_reader_url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as response:
                content_bytes = response.read()
                # Jina reader usually returns markdown, try decoding as UTF-8
                return content_bytes.decode("utf-8", errors="replace")

        content = await asyncio.to_thread(sync_jina_fetch)

        if content:
            if max_text_length is not None and len(content) > max_text_length:
                logging.info(
                    f"Jina: Truncating content for {url} from {len(content)} to {max_text_length} chars."
                )
                content = content[: max_text_length - 3] + "..."
            logging.info(f"Jina: Successfully fetched and processed {url}")
            return UrlFetchResult(
                url=url, content=content, type="general_jina", original_index=index
            )
        else:
            logging.warning(f"Jina: No content extracted from {url}")
            return UrlFetchResult(
                url=url,
                content=None,
                error="Jina: No content extracted.",
                type="general_jina",
                original_index=index,
            )
    except urllib.error.HTTPError as e:
        logging.warning(
            f"Jina: HTTPError for {url} ({jina_reader_url}): {e.code} {e.reason}"
        )
        return UrlFetchResult(
            url=url,
            content=None,
            error=f"Jina HTTPError: {e.code} {e.reason}",
            type="general_jina",
            original_index=index,
        )
    except urllib.error.URLError as e:
        logging.warning(f"Jina: URLError for {url} ({jina_reader_url}): {e.reason}")
        return UrlFetchResult(
            url=url,
            content=None,
            error=f"Jina URLError: {e.reason}",
            type="general_jina",
            original_index=index,
        )
    except Exception as e:
        logging.error(
            f"Jina: Exception during fetch for {url} ({jina_reader_url}): {e}",
            exc_info=True,
        )
        return UrlFetchResult(
            url=url,
            content=None,
            error=f"Jina Exception: {type(e).__name__}",
            type="general_jina",
            original_index=index,
        )


async def fetch_with_crawl4ai(
    url: str, index: int, max_text_length: Optional[int] = None
) -> UrlFetchResult:
    """Fetches content using Crawl4AI."""
    logging.info(f"Attempting to fetch URL with Crawl4AI: {url}")
    try:
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  # Or CacheMode.ENABLED if caching is desired
            # Add other relevant Crawl4AI configs here if needed, e.g., user_agent
            # For now, using default markdown generator
        )
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=run_config)

        if result.success and result.markdown:
            content = (
                result.markdown.raw_markdown
            )  # Using raw_markdown for less processing
            if (
                max_text_length is not None
                and content
                and len(content) > max_text_length
            ):
                logging.info(
                    f"Crawl4AI: Truncating content for {url} from {len(content)} to {max_text_length} chars."
                )
                content = content[: max_text_length - 3] + "..."

            if not content:
                logging.warning(f"Crawl4AI: No markdown content extracted from {url}")
                return UrlFetchResult(
                    url=url,
                    content=None,
                    error="Crawl4AI: No markdown content extracted.",
                    type="general_crawl4ai",
                    original_index=index,
                )

            logging.info(f"Crawl4AI: Successfully fetched and processed {url}")
            return UrlFetchResult(
                url=url, content=content, type="general_crawl4ai", original_index=index
            )
        elif result.success and not result.markdown:
            logging.warning(
                f"Crawl4AI: Crawl successful but no markdown content for {url}"
            )
            return UrlFetchResult(
                url=url,
                content=None,
                error="Crawl4AI: Crawl successful but no markdown content.",
                type="general_crawl4ai",
                original_index=index,
            )
        else:
            error_msg = result.error_message or "Crawl4AI: Unknown error"
            logging.warning(f"Crawl4AI: Failed to fetch {url}. Error: {error_msg}")
            return UrlFetchResult(
                url=url,
                content=None,
                error=f"Crawl4AI: {error_msg}",
                type="general_crawl4ai",
                original_index=index,
            )
    except Exception as e:
        logging.error(f"Crawl4AI: Exception during fetch for {url}: {e}", exc_info=True)
        return UrlFetchResult(
            url=url,
            content=None,
            error=f"Crawl4AI Exception: {type(e).__name__}",
            type="general_crawl4ai",
            original_index=index,
        )


async def fetch_with_beautifulsoup(
    url: str,
    index: int,
    client: httpx.AsyncClient,
    max_text_length: Optional[int] = None,
) -> UrlFetchResult:
    """Fetches and extracts text content from a general URL using BeautifulSoup (fallback)."""
    logging.info(f"Attempting to fetch URL with BeautifulSoup (fallback): {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with client.stream("GET", url, headers=headers, timeout=15.0) as response:
            if response.status_code != 200:
                if (
                    response.status_code >= 300
                    and response.status_code < 400
                    and len(response.history) > 5
                ):
                    logging.warning(f"BS4: Too many redirects for URL: {url}")
                    return UrlFetchResult(
                        url=url,
                        content=None,
                        error=f"BS4: Too many redirects ({response.status_code}).",
                        type="general",
                        original_index=index,
                    )
                logging.warning(
                    f"BS4: HTTP status {response.status_code} for URL: {url}"
                )
                return UrlFetchResult(
                    url=url,
                    content=None,
                    error=f"BS4: HTTP status {response.status_code}.",
                    type="general",
                    original_index=index,
                )

            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                logging.warning(
                    f"BS4: Unsupported content type '{content_type}' for URL: {url}"
                )
                return UrlFetchResult(
                    url=url,
                    content=None,
                    error=f"BS4: Unsupported content type: {content_type}",
                    type="general",
                    original_index=index,
                )

            html_content = ""
            html_limit = 5 * 1024 * 1024
            try:
                async for chunk in response.aiter_bytes():
                    try:
                        html_content += chunk.decode(
                            response.encoding or "utf-8", errors="replace"
                        )
                    except UnicodeDecodeError as decode_err:
                        logging.warning(
                            f"BS4: Unicode decode error for {url}: {decode_err}. Skipping chunk."
                        )
                        continue
                    if len(html_content) > html_limit:
                        logging.warning(
                            f"BS4: HTML content truncated for {url} ({html_limit} bytes)."
                        )
                        html_content = html_content[:html_limit] + "..."
                        break
            except httpx.ReadTimeout:
                logging.warning(f"BS4: Timeout reading content from {url}")
                return UrlFetchResult(
                    url=url,
                    content=None,
                    error="BS4: Timeout reading content.",
                    type="general",
                    original_index=index,
                )
            except Exception as e:
                logging.warning(f"BS4: Error reading content stream for {url}: {e}")
                return UrlFetchResult(
                    url=url,
                    content=None,
                    error=f"BS4: Content reading error: {type(e).__name__}",
                    type="general",
                    original_index=index,
                )

        try:
            soup = BeautifulSoup(html_content, "lxml")
        except ImportError:
            soup = BeautifulSoup(html_content, "html.parser")

        for script_or_style in soup(
            ["script", "style", "header", "footer", "nav", "aside"]
        ):
            script_or_style.decompose()

        main_content_tag = (
            soup.find("main") or soup.find("article") or soup.find("body")
        )
        text = (
            main_content_tag.get_text(separator=" ", strip=True)
            if main_content_tag
            else soup.get_text(separator=" ", strip=True)
        )
        text = re.sub(r"\s+", " ", text).strip()

        if max_text_length is not None and text and len(text) > max_text_length:
            logging.info(
                f"BS4: Truncating content for {url} from {len(text)} to {max_text_length} chars."
            )
            text = text[: max_text_length - 3] + "..."

        if not text:
            logging.warning(f"BS4: No text content found after parsing {url}")
            return UrlFetchResult(
                url=url,
                content=None,
                error="BS4: No text content found.",
                type="general",
                original_index=index,
            )
        logging.info(f"BS4: Successfully fetched and processed {url}")
        return UrlFetchResult(
            url=url, content=text, type="general", original_index=index
        )

    except httpx.RequestError as e:
        error_type = type(e).__name__
        logging.warning(f"BS4: HTTPX RequestError for {url}: {error_type} - {e}")
        error_msg_map = {
            "ConnectTimeout": "Connection timed out",
            "ReadTimeout": "Read timed out",
            "ConnectError": f"Connection error: {e}",
            "TooManyRedirects": "Too many redirects",
        }
        error_msg = error_msg_map.get(error_type, f"Request failed: {error_type}")
        return UrlFetchResult(
            url=url,
            content=None,
            error=f"BS4: {error_msg}",
            type="general",
            original_index=index,
        )
    except Exception as e:
        logging.error(f"BS4: Unexpected error for {url}: {e}", exc_info=True)
        return UrlFetchResult(
            url=url,
            content=None,
            error=f"BS4 Exception: {type(e).__name__}",
            type="general",
            original_index=index,
        )


async def fetch_general_url_content(
    url: str,
    index: int,
    client: httpx.AsyncClient,
    main_extractor: str,
    fallback_extractor: str,
    max_text_length: Optional[int] = None,
    jina_engine_mode: str = DEFAULT_JINA_ENGINE_MODE,  # Added jina_engine_mode
) -> UrlFetchResult:
    """
    Fetches content from a general URL using the specified main and fallback extractors.
    """
    chosen_main_fetcher = None
    chosen_fallback_fetcher = None
    main_fetcher_name = ""
    fallback_fetcher_name = ""

    # Define helper functions to capture the 'client' variable from the outer scope
    async def _bs_fetcher_wrapper(u_param: str, i_param: int, mtl_param: Optional[int]):
        return await fetch_with_beautifulsoup(u_param, i_param, client, mtl_param)

    async def _jina_fetcher_wrapper(
        u_param: str, i_param: int, mtl_param: Optional[int]
    ):
        return await fetch_with_jina(u_param, i_param, mtl_param, jina_engine_mode)

    if main_extractor == "crawl4ai":
        chosen_main_fetcher = fetch_with_crawl4ai
        main_fetcher_name = "Crawl4AI"
    elif main_extractor == "beautifulsoup":
        chosen_main_fetcher = _bs_fetcher_wrapper
        main_fetcher_name = "BeautifulSoup"
    elif main_extractor == "jina":
        chosen_main_fetcher = _jina_fetcher_wrapper  # Use wrapper
        main_fetcher_name = "Jina"
    else:  # Should not happen due to config validation
        logging.error(
            f"Invalid main_extractor specified: {main_extractor}. Defaulting to crawl4ai."
        )
        chosen_main_fetcher = fetch_with_crawl4ai
        main_fetcher_name = "Crawl4AI (defaulted)"

    if fallback_extractor == "crawl4ai":
        chosen_fallback_fetcher = fetch_with_crawl4ai
        fallback_fetcher_name = "Crawl4AI"
    elif fallback_extractor == "beautifulsoup":
        chosen_fallback_fetcher = _bs_fetcher_wrapper
        fallback_fetcher_name = "BeautifulSoup"
    elif fallback_extractor == "jina":
        chosen_fallback_fetcher = _jina_fetcher_wrapper  # Use wrapper
        fallback_fetcher_name = "Jina"
    else:  # Should not happen
        logging.error(
            f"Invalid fallback_extractor specified: {fallback_extractor}. Defaulting to beautifulsoup."
        )
        chosen_fallback_fetcher = _bs_fetcher_wrapper
        fallback_fetcher_name = "BeautifulSoup (defaulted)"

    logging.info(
        f"Attempting general URL fetch for {url} with main extractor: {main_fetcher_name}"
    )
    main_result = await chosen_main_fetcher(url, index, max_text_length)

    if main_result.content and not main_result.error:
        return main_result
    else:
        logging.warning(
            f"{main_fetcher_name} failed for {url} (Error: {main_result.error}). Falling back to {fallback_fetcher_name}."
        )

        # Ensure main and fallback are not the same to avoid re-running the same failed fetcher
        if main_extractor == fallback_extractor:
            logging.warning(
                f"Main and fallback extractors are the same ({main_extractor}). Not re-running failed fetcher."
            )
            # Return the error from the first attempt
            return UrlFetchResult(
                url=url,
                content=None,
                error=f"{main_fetcher_name} Error: {main_result.error} (Fallback was same method).",
                type="general",
                original_index=index,
            )

        fallback_result = await chosen_fallback_fetcher(url, index, max_text_length)
        if fallback_result.content and not fallback_result.error:
            return fallback_result
        else:
            # Both failed
            final_error_message = f"Failed with {main_fetcher_name} (Error: {main_result.error}) and {fallback_fetcher_name} (Error: {fallback_result.error})."
            logging.error(f"Both extractors failed for {url}: {final_error_message}")
            return UrlFetchResult(
                url=url,
                content=None,
                error=final_error_message,
                type="general",
                original_index=index,
            )
