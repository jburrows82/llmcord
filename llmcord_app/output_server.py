import logging
from typing import Optional, Dict, Any
import httpx
from bs4 import BeautifulSoup

from .constants import (
    OUTPUT_SHARING_CONFIG_KEY,
    TEXTIS_ENABLED_CONFIG_KEY,  # Renamed from NGROK_ENABLED_CONFIG_KEY
    URL_SHORTENER_ENABLED_CONFIG_KEY,
    URL_SHORTENER_SERVICE_CONFIG_KEY,
)

async def _async_shorten_url_tinyurl(long_url: str) -> Optional[str]:
    """Asynchronously shortens a URL using TinyURL's API."""
    if not long_url:
        return None
    try:
        api_url = f"http://tinyurl.com/api-create.php?url={long_url}"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(api_url)
            response.raise_for_status()
        short_url = response.text.strip()
        # Basic validation: check if it looks like a URL
        if short_url.startswith("http://") or short_url.startswith("https://"):
            logging.info(
                f"Successfully shortened {long_url} to {short_url} using TinyURL."
            )
            return short_url
        else:
            logging.warning(
                f"TinyURL returned an unexpected response: '{short_url}' for URL: {long_url}"
            )
            return None
    except httpx.RequestError as e:
        logging.error(f"Error requesting TinyURL for {long_url}: {e}")
        return None
    except httpx.HTTPStatusError as e:
        logging.error(
            f"TinyURL API returned an error for {long_url}: {e.response.status_code} - {e.response.text}"
        )
        return None
    except Exception as e:
        logging.error(
            f"Unexpected error shortening URL {long_url} with TinyURL: {e}",
            exc_info=True,
        )
        return None

async def share_to_textis(text_content: str) -> Optional[str]:
    """
    Shares text_content to text.is and returns the generated URL.
    """
    base_url = "https://text.is/"
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            # 1. GET the page to obtain CSRF token
            logging.debug(f"Fetching initial page from {base_url} to get CSRF token.")
            response_get = await client.get(base_url)
            response_get.raise_for_status()
            
            soup_get = BeautifulSoup(response_get.text, "html.parser")
            csrf_token_input = soup_get.find("input", {"name": "csrfmiddlewaretoken"})
            
            if not csrf_token_input or not csrf_token_input.get("value"):
                logging.error("Could not find CSRF token on text.is page.")
                return None
            csrf_token = csrf_token_input["value"]
            logging.debug(f"Found CSRF token: {csrf_token}")

            # 2. POST the content
            payload = {
                "csrfmiddlewaretoken": csrf_token,
                "text": text_content,
            }
            
            logging.debug(f"Posting content to {base_url}")
            response_post = await client.post(base_url, data=payload)
            response_post.raise_for_status()

            # 3. Extract the canonical URL from the response page
            # httpx.AsyncClient with follow_redirects=True should mean response_post.url is the final URL.
            final_url = str(response_post.url)
            
            # Check if the final_url is the base_url. If so, the redirect might not have happened as expected,
            # or the POST didn't result in a new paste. Try to parse from HTML as a fallback.
            if final_url == base_url:
                logging.warning(f"Response URL {final_url} is the same as the base URL. Attempting to parse canonical link from HTML.")
                soup_post = BeautifulSoup(response_post.text, "html.parser")
                canonical_link_tag = soup_post.find("link", {"rel": "canonical"})
                if canonical_link_tag and canonical_link_tag.get("href"):
                    final_url = canonical_link_tag["href"]
                    logging.debug(f"Extracted canonical URL from meta tag: {final_url}")
                else:
                    logging.error(f"Could not determine the final URL from text.is response. Response URL was base URL, and no canonical link found. Content: {response_post.text[:500]}")
                    return None
            else:
                logging.debug(f"Determined final URL from response_post.url (after potential redirects): {final_url}")

            # Final validation of the extracted URL:
            # It must start with the base_url and be longer than the base_url itself (i.e., have a path component).
            if not final_url.startswith(base_url) or len(final_url) <= len(base_url) or final_url == base_url:
                 logging.error(f"Generated URL '{final_url}' does not look like a valid text.is paste URL (must start with '{base_url}' and have a path).")
                 return None

            logging.info(f"Successfully obtained text.is URL: {final_url}")
            return final_url

    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error occurred while interacting with text.is: {e.response.status_code} - {e.response.text[:200]}")
        return None
    except httpx.RequestError as e:
        logging.error(f"Request error occurred while interacting with text.is: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred in share_to_textis: {e}", exc_info=True)
        return None

async def start_output_server(
    text_content: str, config: Dict[str, Any]
) -> Optional[str]:
    """
    Shares markdown text_content to text.is.
    Returns the public URL to the paste if successful, otherwise None.
    """
    output_sharing_cfg = config.get(OUTPUT_SHARING_CONFIG_KEY, {})
    textis_enabled = output_sharing_cfg.get(TEXTIS_ENABLED_CONFIG_KEY, False)

    if not textis_enabled:
        logging.info("Output sharing via text.is is disabled in config.")
        return None

    if not text_content:
        logging.warning("No text content provided to share.")
        return None

    logging.info("Attempting to share content to text.is...")
    public_url = await share_to_textis(text_content)

    if not public_url:
        logging.error("Failed to share content to text.is.")
        return None
    
    logging.info(f"Successfully shared to text.is: {public_url}")

    # URL Shortening
    shortener_enabled = output_sharing_cfg.get(URL_SHORTENER_ENABLED_CONFIG_KEY, False)
    shortener_service = output_sharing_cfg.get(URL_SHORTENER_SERVICE_CONFIG_KEY, "tinyurl")
    
    final_url_to_share = public_url

    if shortener_enabled:
        logging.info(f"URL shortener enabled, service: {shortener_service}")
        if shortener_service.lower() == "tinyurl":
            shortened_url = await _async_shorten_url_tinyurl(public_url)
            if shortened_url:
                final_url_to_share = shortened_url
            else:
                logging.warning(
                    f"Failed to shorten URL with TinyURL, using original: {public_url}"
                )
        else:
            logging.warning(
                f"Unsupported URL shortener service: '{shortener_service}'. Using original URL."
            )
            
    logging.info(f"Final URL to share: {final_url_to_share}")
    return final_url_to_share

def stop_output_server():
    """
    Placeholder function. No server to stop with text.is implementation.
    """
    logging.info("stop_output_server called. No active server to stop for text.is.")
    pass

async def cleanup_shared_html_dir():
    """
    Placeholder function. No local HTML directory to clean up with text.is implementation.
    """
    logging.info("cleanup_shared_html_dir called. No local directory to clean for text.is.")
    pass
