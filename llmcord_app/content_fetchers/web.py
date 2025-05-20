import logging
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CacheMode,
)  # Added Crawl4AI imports

from ..models import UrlFetchResult


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
    client: httpx.AsyncClient,  # httpx client for BeautifulSoup fallback
    max_text_length: Optional[int] = None,
) -> UrlFetchResult:
    """
    Fetches content from a general URL.
    Tries Crawl4AI first, then falls back to BeautifulSoup.
    """
    crawl4ai_result = await fetch_with_crawl4ai(url, index, max_text_length)

    if crawl4ai_result.content and not crawl4ai_result.error:
        return crawl4ai_result
    else:
        logging.warning(
            f"Crawl4AI failed for {url} (Error: {crawl4ai_result.error}). Falling back to BeautifulSoup."
        )
        # Pass the httpx client to the BeautifulSoup fetcher
        bs_result = await fetch_with_beautifulsoup(url, index, client, max_text_length)
        if bs_result.content and not bs_result.error:
            return bs_result
        else:
            # Both failed, return Crawl4AI's error if it exists, else BS's, or a generic one
            final_error = (
                "Failed to fetch content with both Crawl4AI and BeautifulSoup."
            )
            if crawl4ai_result.error and bs_result.error:
                final_error = f"Crawl4AI Error: {crawl4ai_result.error}. BeautifulSoup Error: {bs_result.error}"
            elif crawl4ai_result.error:
                final_error = f"Crawl4AI Error: {crawl4ai_result.error} (BS fallback also failed)."
            elif bs_result.error:
                final_error = (
                    f"BeautifulSoup Error: {bs_result.error} (Crawl4AI also failed)."
                )

            return UrlFetchResult(
                url=url,
                content=None,
                error=final_error,
                type="general",  # Fallback type if both fail
                original_index=index,
            )
