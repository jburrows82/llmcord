import logging
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from ..models import UrlFetchResult

# Define httpx_client here or pass it in
# For simplicity, let's define it here for this module's use
# In a larger app, dependency injection might be better
# httpx_client = httpx.AsyncClient(timeout=20.0, follow_redirects=True) # Removed module-level client


async def fetch_general_url_content(
    url: str,
    index: int,
    client: httpx.AsyncClient,
    max_text_length: Optional[int] = None,
) -> UrlFetchResult:
    """Fetches and extracts text content from a general URL using BeautifulSoup."""
    try:
        # Add a User-Agent header to mimic a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        logging.debug(f"Fetching general URL: {url}")
        async with client.stream(  # Use passed-in client
            "GET", url, headers=headers, timeout=15.0
        ) as response:
            # Check status code early
            if response.status_code != 200:
                # Check for redirect loop explicitly
                if (
                    response.status_code >= 300
                    and response.status_code < 400
                    and len(response.history) > 5
                ):
                    logging.warning(f"Too many redirects for URL: {url}")
                    return UrlFetchResult(
                        url=url,
                        content=None,
                        error=f"Too many redirects ({response.status_code}).",
                        type="general",
                        original_index=index,
                    )
                logging.warning(f"HTTP status {response.status_code} for URL: {url}")
                return UrlFetchResult(
                    url=url,
                    content=None,
                    error=f"HTTP status {response.status_code}.",
                    type="general",
                    original_index=index,
                )

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                logging.warning(
                    f"Unsupported content type '{content_type}' for URL: {url}"
                )
                return UrlFetchResult(
                    url=url,
                    content=None,
                    error=f"Unsupported content type: {content_type}",
                    type="general",
                    original_index=index,
                )

            # Read content incrementally
            html_content = ""
            html_limit = 5 * 1024 * 1024  # 5MB limit for HTML source
            try:
                async for chunk in response.aiter_bytes():
                    # Decode chunk by chunk to handle large files better
                    try:
                        html_content += chunk.decode(
                            response.encoding or "utf-8", errors="replace"
                        )
                    except UnicodeDecodeError as decode_err:
                        logging.warning(
                            f"Unicode decode error reading chunk from {url}: {decode_err}. Skipping chunk."
                        )
                        continue

                    if len(html_content) > html_limit:
                        logging.warning(
                            f"HTML content truncated for URL {url} due to size limit ({html_limit} bytes)."
                        )
                        html_content = html_content[:html_limit] + "..."
                        break
            except httpx.ReadTimeout:
                logging.warning(f"Timeout while reading content from URL: {url}")
                return UrlFetchResult(
                    url=url,
                    content=None,
                    error="Timeout while reading content.",
                    type="general",
                    original_index=index,
                )
            except Exception as e:
                logging.warning(f"Error reading content stream for {url}: {e}")
                return UrlFetchResult(
                    url=url,
                    content=None,
                    error=f"Content reading error: {type(e).__name__}",
                    type="general",
                    original_index=index,
                )

        logging.debug(
            f"Successfully read HTML content (length: {len(html_content)}) for URL: {url}"
        )

        # Parse with BeautifulSoup
        try:
            # Use 'lxml' for potentially faster parsing if available, fallback to 'html.parser'
            try:
                soup = BeautifulSoup(html_content, "lxml")
            except ImportError:
                soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script_or_style in soup(
                ["script", "style", "header", "footer", "nav", "aside"]
            ):  # Remove more non-content tags
                script_or_style.decompose()

            # Get text content, trying main content areas first
            main_content = (
                soup.find("main") or soup.find("article") or soup.find("body")
            )
            if main_content:
                text = main_content.get_text(separator=" ", strip=True)
            else:
                text = soup.get_text(
                    separator=" ", strip=True
                )  # Fallback to whole document

            # Clean up whitespace: replace multiple spaces/newlines with a single space
            text = re.sub(r"\s+", " ", text).strip()

            # --- NEW: Truncate extracted text if limit is provided ---
            if max_text_length is not None and text and len(text) > max_text_length:
                logging.info(
                    f"Truncating extracted text content for URL {url} from {len(text)} to {max_text_length} characters."
                )
                text = (
                    text[: max_text_length - 3] + "..."
                )  # Ensure space for ellipsis if text is long enough

            if not text:
                logging.warning(f"No text content found after parsing URL: {url}")
                return UrlFetchResult(
                    url=url,
                    content=None,
                    error="No text content found.",
                    type="general",
                    original_index=index,
                )

            # Content is no longer limited here, the LLM handler will limit the history
            content = text
            logging.debug(
                f"Extracted text content (length: {len(content)}) for URL: {url}"
            )

            return UrlFetchResult(
                url=url, content=content, type="general", original_index=index
            )

        except Exception as parse_err:
            logging.error(
                f"Error parsing HTML content for {url} with BeautifulSoup: {parse_err}",
                exc_info=True,
            )
            return UrlFetchResult(
                url=url,
                content=None,
                error=f"HTML parsing error: {type(parse_err).__name__}",
                type="general",
                original_index=index,
            )

    except httpx.RequestError as e:
        # Log specific httpx errors if possible
        error_type = type(e).__name__
        logging.warning(f"HTTPX RequestError fetching {url}: {error_type} - {e}")
        # Provide more specific error messages
        if isinstance(e, httpx.ConnectTimeout):
            error_msg = "Connection timed out"
        elif isinstance(e, httpx.ReadTimeout):
            error_msg = "Read timed out"
        elif isinstance(e, httpx.ConnectError):
            error_msg = "Connection error"
        elif isinstance(e, httpx.TooManyRedirects):
            error_msg = "Too many redirects"
        else:
            error_msg = f"Request failed: {error_type}"
        return UrlFetchResult(
            url=url, content=None, error=error_msg, type="general", original_index=index
        )
    except Exception as e:
        logging.exception(f"Unexpected error fetching general URL {url}")
        return UrlFetchResult(
            url=url,
            content=None,
            error=f"Unexpected error: {type(e).__name__}",
            type="general",
            original_index=index,
        )
