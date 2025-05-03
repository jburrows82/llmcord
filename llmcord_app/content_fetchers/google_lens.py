import asyncio
import logging
import os
import random
import re
from typing import Optional, List, Tuple, Set, Dict, Any
import discord # <-- Added import

from serpapi import GoogleSearch
from serpapi.serp_api_client_exception import SerpApiClientException

from ..models import UrlFetchResult
from ..rate_limiter import get_db_manager, get_available_keys
from ..constants import (
    LENS_ICON_SELECTOR, PASTE_LINK_INPUT_SELECTOR, SEE_EXACT_MATCHES_SELECTOR,
    EXACT_MATCH_RESULT_SELECTOR, INITIAL_RESULTS_WAIT_SELECTOR, ORIGINAL_RESULT_SPAN_SELECTOR,
    CUSTOM_LENS_DEFAULT_TIMEOUT, CUSTOM_LENS_SHORT_TIMEOUT
)

# Import Playwright conditionally
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    logging.warning("Playwright not found. Custom Google Lens fallback implementation will not work.")
    logging.warning("Install it using: pip install playwright")
    logging.warning("Then install browsers: python -m playwright install chrome")
    PLAYWRIGHT_AVAILABLE = False
    # Define dummy class if playwright is missing to avoid NameErrors later
    class PlaywrightTimeoutError(Exception): pass
    def sync_playwright():
        class DummyContextManager:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        return DummyContextManager()

# --- Custom Google Lens (Playwright) Implementation ---
def _custom_get_google_lens_results_sync_fallback(image_url: str, user_data_dir: str, profile_directory_name: str) -> Optional[List[str]]:
    """
    Synchronous wrapper for the Playwright Google Lens logic (FALLBACK ONLY).
    Uses Playwright to get Google Lens results for a given image URL using a specific Chrome profile.

    Args:
        image_url: The URL of the image to search.
        user_data_dir: Path to the main Chrome user data directory.
        profile_directory_name: The name of the specific profile folder within user_data_dir.

    Returns:
        A list of strings containing the extracted result texts, or None if an error occurs.
    """
    if not PLAYWRIGHT_AVAILABLE:
        logging.error("Playwright library not found. Cannot run custom Google Lens fallback.")
        return None # Indicate failure due to missing dependency

    if not user_data_dir or not profile_directory_name:
        logging.error("Custom Google Lens Fallback: Chrome user_data_dir or profile_directory_name not provided.")
        return None

    if not os.path.exists(user_data_dir):
        logging.error(f"Custom Google Lens Fallback: Chrome user data directory not found at: {user_data_dir}")
        return None

    profile_path = os.path.join(user_data_dir, profile_directory_name)
    if not os.path.exists(profile_path):
        logging.error(f"Custom Google Lens Fallback: Specific profile directory not found at: {profile_path}")
        logging.error(f"Custom Google Lens Fallback: Please ensure '{profile_directory_name}' is the correct folder name inside '{user_data_dir}'.")
        return None

    results = []
    context = None # Define context outside the try block for finally clause

    logging.info(f"Custom Google Lens Fallback: Launching Chrome using profile: '{profile_directory_name}'")
    logging.info(f"Custom Google Lens Fallback: User Data Directory: {user_data_dir}")
    logging.info(f"Custom Google Lens Fallback: Searching for image URL: {image_url}")
    logging.info("Custom Google Lens Fallback: INFO: Ensure Google Chrome using this specific profile is completely closed (check Task Manager).")

    # Arguments to specify the profile directory
    launch_args = [
        '--no-first-run',
        '--no-default-browser-check',
        f"--profile-directory={profile_directory_name}" # Tells Chrome which profile folder to use
    ]

    with sync_playwright() as p:
        browser = None
        page = None
        try:
            # Launch browser using the persistent context with the specified user data directory AND profile arg
            # Headless=False is often required for persistent context/logins to work reliably
            context = p.chromium.launch_persistent_context(
                user_data_dir, # Still need the parent directory path here
                headless=False, # Set to True ONLY if you are sure login state isn't needed and it works on your server
                channel="chrome",
                args=launch_args, # Pass the arguments including the profile directory
                slow_mo=50 # Slow down interactions slightly
            )
            page = context.new_page()
            page.set_default_timeout(CUSTOM_LENS_DEFAULT_TIMEOUT)

            logging.info("Custom Google Lens Fallback: Navigating to google.com...")
            page.goto("https://www.google.com/", wait_until='domcontentloaded') # Wait for DOM load

            logging.info("Custom Google Lens Fallback: Waiting for and clicking the Lens (Search by image) icon...")
            lens_icon = page.locator(LENS_ICON_SELECTOR)
            lens_icon.wait_for(state="visible", timeout=CUSTOM_LENS_SHORT_TIMEOUT) # Shorter timeout for initial icon
            lens_icon.click()

            logging.info("Custom Google Lens Fallback: Waiting for the image link input field...")
            paste_link_input = page.locator(PASTE_LINK_INPUT_SELECTOR)
            paste_link_input.wait_for(state="visible")

            logging.info("Custom Google Lens Fallback: Pasting the image URL...")
            paste_link_input.fill(image_url)

            logging.info("Custom Google Lens Fallback: Submitting the search (pressing Enter)...")
            # Add a small delay before pressing Enter, sometimes helps
            page.wait_for_timeout(500)
            paste_link_input.press("Enter")

            logging.info("Custom Google Lens Fallback: Waiting for initial Lens results page to load...")
            try:
                 # Wait for a more specific element that indicates results are loading/loaded
                 page.wait_for_selector(INITIAL_RESULTS_WAIT_SELECTOR, state="visible", timeout=CUSTOM_LENS_DEFAULT_TIMEOUT)
                 logging.info("Custom Google Lens Fallback: Initial results container found.")
            except PlaywrightTimeoutError:
                 logging.warning(f"Custom Google Lens Fallback: Timed out waiting for initial results container ('{INITIAL_RESULTS_WAIT_SELECTOR}'). Proceeding cautiously.")
                 current_url = page.url
                 if "google.com/search?" not in current_url or ("lens" not in current_url and "source=lns" not in current_url):
                     logging.warning(f"Custom Google Lens Fallback: Current URL doesn't look like a Lens results page: {current_url}")
                 # Don't return yet, try checking for results anyway

            logging.info(f"Custom Google Lens Fallback: Checking for '{SEE_EXACT_MATCHES_SELECTOR}'...")
            see_exact_matches_button = page.locator(SEE_EXACT_MATCHES_SELECTOR)
            final_result_selector = None
            result_elements = []

            try:
                # Use is_visible with a timeout to check if the button appears
                if see_exact_matches_button.is_visible(timeout=CUSTOM_LENS_SHORT_TIMEOUT):
                    logging.info("Custom Google Lens Fallback: Found 'See exact matches', clicking it...")
                    try: # Nested try for click and wait
                        see_exact_matches_button.click()
                        logging.info("Custom Google Lens Fallback: Waiting for exact match results to load (waiting for last element)...")
                        # Wait for the *last* element matching the selector to ensure page is populated
                        page.locator(EXACT_MATCH_RESULT_SELECTOR).last.wait_for(state="visible", timeout=CUSTOM_LENS_DEFAULT_TIMEOUT)
                        logging.info(f"Custom Google Lens Fallback: Exact match results likely loaded. Using selector: '{EXACT_MATCH_RESULT_SELECTOR}'")
                        final_result_selector = EXACT_MATCH_RESULT_SELECTOR
                    except PlaywrightTimeoutError:
                        logging.warning(f"Custom Google Lens Fallback: Clicked 'See exact matches' but timed out waiting for the last element of '{EXACT_MATCH_RESULT_SELECTOR}'. May proceed with partial results if any.")
                        # Still set the selector, maybe some loaded
                        final_result_selector = EXACT_MATCH_RESULT_SELECTOR
                    except Exception as click_err:
                         logging.error(f"Custom Google Lens Fallback: Error clicking 'See exact matches' or waiting after click: {click_err}")
                         # Fallback to original results if click fails badly
                         final_result_selector = ORIGINAL_RESULT_SPAN_SELECTOR
                else:
                    logging.info("Custom Google Lens Fallback: 'See exact matches' not found or not visible within timeout.")
                    final_result_selector = ORIGINAL_RESULT_SPAN_SELECTOR # Use original selector

            except PlaywrightTimeoutError:
                 logging.warning(f"Custom Google Lens Fallback: Timeout checking visibility for '{SEE_EXACT_MATCHES_SELECTOR}'. Assuming it's not present.")
                 final_result_selector = ORIGINAL_RESULT_SPAN_SELECTOR # Use original selector

            # Wait for general results if using the original selector
            if final_result_selector == ORIGINAL_RESULT_SPAN_SELECTOR:
                logging.info(f"Custom Google Lens Fallback: Looking for general results using selector: '{ORIGINAL_RESULT_SPAN_SELECTOR}' (waiting for last element)...")
                try:
                    page.locator(ORIGINAL_RESULT_SPAN_SELECTOR).last.wait_for(state="visible", timeout=CUSTOM_LENS_DEFAULT_TIMEOUT)
                    logging.info(f"Custom Google Lens Fallback: General results likely loaded.")
                except PlaywrightTimeoutError:
                     logging.warning(f"Custom Google Lens Fallback: Fallback check for the last original result ('{ORIGINAL_RESULT_SPAN_SELECTOR}') also timed out. May get no results.")
                     final_result_selector = None # Indicate no results likely

            # Extract results using the determined selector
            if final_result_selector:
                logging.info(f"Custom Google Lens Fallback: Extracting text using final selector: '{final_result_selector}'...")
                # Add a small delay before extracting, sometimes helps ensure rendering
                page.wait_for_timeout(1000)
                result_elements = page.locator(final_result_selector).all()

                if not result_elements:
                    logging.info("Custom Google Lens Fallback: No result elements found matching the final selector.")

                for i, element in enumerate(result_elements):
                    try:
                        text = element.text_content()
                        if text:
                            cleaned_text = ' '.join(text.split()) # Clean whitespace
                            results.append(cleaned_text)
                        else:
                            logging.warning(f"Custom Google Lens Fallback: Found element {i+1} but it has no text content.")
                    except Exception as e:
                        logging.error(f"Custom Google Lens Fallback: Error extracting text from element {i+1}: {e}")
            else:
                 logging.info("Custom Google Lens Fallback: No suitable result selector was determined. Skipping extraction.")

            logging.info(f"Custom Google Lens Fallback: Finished extracting results. Found {len(results)} items.")

        except PlaywrightTimeoutError as e:
            logging.error(f"Custom Google Lens Fallback: ERROR: A timeout occurred during the process: {e}")
            try:
                if page and not page.is_closed():
                    screenshot_path = f"error_screenshot_timeout_{random.randint(1000,9999)}.png"
                    page.screenshot(path=screenshot_path)
                    logging.info(f"Custom Google Lens Fallback: Screenshot saved as {screenshot_path}")
            except Exception as screen_err:
                logging.error(f"Custom Google Lens Fallback: Could not take screenshot on timeout error: {screen_err}")
            return None # Indicate failure
        except Exception as e:
            logging.error(f"Custom Google Lens Fallback: An unexpected error occurred: {e}")
            if "Target page, context or browser has been closed" in str(e) or "Target closed" in str(e):
                 logging.error("Custom Google Lens Fallback: ERROR DETAILS: This 'TargetClosedError' usually means Google Chrome was already running with the specified profile, or the browser crashed.")
                 logging.error(f"Custom Google Lens Fallback: Profile Folder: '{profile_directory_name}' within '{user_data_dir}'")
                 logging.error("Custom Google Lens Fallback: Please ensure ALL Chrome processes using this profile are closed (check Task Manager/Activity Monitor) before running the script again.")
            else:
                # Use logging.exception to include traceback for other errors
                logging.exception("Custom Google Lens Fallback: Unexpected error details:")
            try:
                 if page and not page.is_closed():
                    screenshot_path = f"error_screenshot_unexpected_{random.randint(1000,9999)}.png"
                    page.screenshot(path=screenshot_path)
                    logging.info(f"Custom Google Lens Fallback: Screenshot saved as {screenshot_path}")
            except Exception as screen_err:
                logging.error(f"Custom Google Lens Fallback: Could not take screenshot on unexpected error: {screen_err}")
            return None # Indicate failure
        finally:
            if page and not page.is_closed():
                logging.debug("Custom Google Lens Fallback: Closing page...")
                try:
                    page.close()
                except Exception as page_close_err:
                    logging.warning(f"Custom Google Lens Fallback: Error closing page: {page_close_err}")
            if context:
                logging.info("Custom Google Lens Fallback: Closing browser context...")
                try:
                    context.close()
                except Exception as close_err:
                     logging.warning(f"Custom Google Lens Fallback: Note: Error during context close (might be expected if launch failed or browser closed): {close_err}")

    return results
# --- End Custom Google Lens Implementation ---


async def fetch_google_lens_serpapi(image_url: str, index: int, all_keys: List[str]) -> UrlFetchResult:
    """Fetches Google Lens results using SerpAPI with key rotation and retry (PRIMARY)."""
    service_name = "serpapi"
    if not all_keys:
        return UrlFetchResult(url=image_url, content=None, error="SerpAPI keys not configured.", type="google_lens_serpapi", original_index=index)

    available_keys = await get_available_keys(service_name, all_keys)
    if not available_keys: # Check if keys are available *after* filtering
        logging.warning(f"All SerpAPI keys are currently rate-limited for Google Lens request for image {index+1}.")
        return UrlFetchResult(url=image_url, content=None, error="All SerpAPI keys are rate-limited.", type="google_lens_serpapi", original_index=index)

    random.shuffle(available_keys)
    db_manager = get_db_manager(service_name)
    encountered_errors = []

    for key_index, api_key in enumerate(available_keys):
        key_display = f"...{api_key[-4:]}"
        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": api_key,
            "safe": "off", # Keep safe search off
        }
        logging.info(f"Attempting SerpAPI Google Lens request for image {index+1} with key {key_display} ({key_index+1}/{len(available_keys)})")

        try:
            # Run the synchronous SerpAPI call in a separate thread
            def sync_serpapi_call():
                search = GoogleSearch(params)
                return search.get_dict()

            results = await asyncio.to_thread(sync_serpapi_call)

            # Check for API-level errors (e.g., invalid key, quota)
            if "error" in results:
                error_msg = results["error"]
                logging.warning(f"SerpAPI error for image {index+1} (key {key_display}): {error_msg}")
                encountered_errors.append(f"Key {key_display}: {error_msg}")
                # Check if it's a rate limit / quota error
                if "rate limit" in error_msg.lower() or "quota" in error_msg.lower() or "plan limit" in error_msg.lower() or "ran out of searches" in error_msg.lower():
                    db_manager.add_key(api_key)
                    logging.info(f"SerpAPI key {key_display} rate limited. Trying next key.")
                    continue # Try next key
                elif "invalid api key" in error_msg.lower():
                    logging.error(f"SerpAPI key {key_display} is invalid. Aborting SerpAPI attempts.")
                    # Don't mark invalid keys as rate-limited, just stop trying
                    return UrlFetchResult(url=image_url, content=None, error=f"SerpAPI Error: Invalid API Key ({key_display})", type="google_lens_serpapi", original_index=index)
                else:
                    # For other API errors, log and try next key
                    logging.warning(f"SerpAPI key {key_display} encountered API error: {error_msg}. Trying next key.")
                    continue

            # Check for search-specific errors (e.g., couldn't process image)
            search_metadata = results.get("search_metadata", {})
            if search_metadata.get("status", "").lower() == "error":
                 error_msg = search_metadata.get("error", "Unknown search error")
                 logging.warning(f"SerpAPI search error for image {index+1} (key {key_display}): {error_msg}")
                 encountered_errors.append(f"Key {key_display}: Search Error - {error_msg}")
                 # These errors are usually not key-related, so don't mark key as limited.
                 # Return the error from the first key that hit this.
                 logging.error(f"SerpAPI search failed for image {index+1} with key {key_display}. Aborting SerpAPI attempts.")
                 return UrlFetchResult(url=image_url, content=None, error=f"SerpAPI Search Error: {error_msg}", type="google_lens_serpapi", original_index=index)

            # --- Success Case ---
            visual_matches = results.get("visual_matches", [])
            if not visual_matches:
                logging.info(f"SerpAPI Google Lens request successful for image {index+1} with key {key_display}, but no visual matches found.")
                return UrlFetchResult(url=image_url, content="No visual matches found (SerpAPI).", type="google_lens_serpapi", original_index=index)

            # Format the results concisely
            formatted_results = []
            for i, match in enumerate(visual_matches): # No display limit
                title = match.get("title", "N/A")
                link = match.get("link", "#")
                source = match.get("source", "")
                # Escape markdown in title
                escaped_title = discord.utils.escape_markdown(title) # Use discord.utils here
                result_line = f"- [{escaped_title}]({link})"
                if source:
                    result_line += f" (Source: {discord.utils.escape_markdown(source)})" # Escape source too
                formatted_results.append(result_line)
            content_str = "\n".join(formatted_results)

            logging.info(f"SerpAPI Google Lens request successful for image {index+1} with key {key_display}")
            return UrlFetchResult(url=image_url, content=content_str, type="google_lens_serpapi", original_index=index)

        except SerpApiClientException as e:
            # Handle client-level exceptions (e.g., connection errors, timeouts)
            logging.warning(f"SerpAPI client exception for image {index+1} (key {key_display}): {e}")
            encountered_errors.append(f"Key {key_display}: Client Error - {e}")
            # Check if the exception indicates a rate limit (might need specific checks based on library behavior/status codes)
            # SerpApiClientException often wraps HTTP errors, check status code if possible
            status_code = None
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code

            if status_code == 429 or "rate limit" in str(e).lower():
                 db_manager.add_key(api_key)
                 logging.info(f"SerpAPI key {key_display} hit client-side rate limit (Status: {status_code}). Trying next key.")
            else:
                 logging.warning(f"SerpAPI key {key_display} encountered client error (Status: {status_code}): {e}. Trying next key.")
            # Retry with the next key for client exceptions
            continue
        except Exception as e:
            logging.exception(f"Unexpected error during SerpAPI request for image {index+1} (key {key_display})")
            encountered_errors.append(f"Key {key_display}: Unexpected Error - {type(e).__name__}")
            logging.warning(f"SerpAPI key {key_display} encountered unexpected error: {e}. Trying next key.")
            # Retry with the next key for unexpected errors
            continue

    # If loop finishes, all keys failed
    logging.error(f"All available SerpAPI keys failed for Google Lens request for image {index+1}.")
    final_error_msg = "All available SerpAPI keys failed."
    if encountered_errors:
        final_error_msg += f" Last error: {encountered_errors[-1]}"
    return UrlFetchResult(url=image_url, content=None, error=final_error_msg, type="google_lens_serpapi", original_index=index)


async def process_google_lens_image(image_url: str, index: int, cfg: Dict[str, Any]) -> UrlFetchResult:
    """
    Processes a Google Lens request for an image URL using configured methods.
    Tries SerpAPI first. Falls back to custom Playwright if SerpAPI fails and custom is configured.

    Args:
        image_url: The URL of the image to process.
        index: The index of the image in the message attachments (for logging/context).
        cfg: The loaded application configuration dictionary.

    Returns:
        A UrlFetchResult object.
    """
    custom_config = cfg.get("custom_google_lens_config")
    serpapi_keys = cfg.get("serpapi_api_keys", [])
    serpapi_keys_ok = bool(serpapi_keys)
    custom_impl_configured = custom_config and custom_config.get("user_data_dir") and custom_config.get("profile_directory_name")
    serpapi_error = "SerpAPI not attempted (no keys configured)." # Default error if keys missing
    custom_error = "Custom fallback not attempted (not configured)." # Default error if not configured

    # 1. Try SerpAPI (Primary)
    if serpapi_keys_ok:
        logging.info(f"Attempting Google Lens request for image {index+1} using primary implementation (SerpAPI)")
        serpapi_result = await fetch_google_lens_serpapi(image_url, index, serpapi_keys)

        # Check if SerpAPI succeeded
        if not serpapi_result.error:
            logging.info(f"SerpAPI Google Lens primary implementation successful for image {index+1}.")
            return serpapi_result # Return successful SerpAPI result
        else:
            serpapi_error = serpapi_result.error # Store the error message
            logging.warning(f"SerpAPI Google Lens primary implementation failed for image {index+1}: {serpapi_error}. Checking fallback.")
    else:
         logging.info("SerpAPI keys not configured. Skipping primary Google Lens implementation.")


    # 2. Fallback to Custom Implementation (if SerpAPI failed AND custom is configured)
    if custom_impl_configured:
        user_data_dir = custom_config["user_data_dir"]
        profile_name = custom_config["profile_directory_name"]
        logging.info(f"Falling back to custom Google Lens implementation for image {index+1} (Profile: {profile_name})")
        try:
            # Run the synchronous Playwright code in a separate thread
            custom_results = await asyncio.to_thread(
                _custom_get_google_lens_results_sync_fallback, # Call the fallback sync function
                image_url,
                user_data_dir,
                profile_name
            )
            if custom_results is not None: # Success (even if empty list)
                logging.info(f"Custom Google Lens fallback implementation successful for image {index+1}.")
                content_str = "\n".join([f"- {r}" for r in custom_results]) if custom_results else "No visual matches found (custom fallback)."

                return UrlFetchResult(url=image_url, content=content_str, type="google_lens_custom", original_index=index)
            else:
                # Custom implementation returned None, indicating an error occurred within it
                custom_error = "Custom fallback failed (returned None)."
                logging.warning(f"Custom Google Lens fallback implementation failed for image {index+1} (returned None).")
                # Fall through to final error reporting

        except Exception as e:
            custom_error = f"Custom fallback implementation raised an exception: {type(e).__name__}: {e}"
            logging.exception(f"Custom Google Lens fallback implementation failed for image {index+1} with exception.")
            # Fall through to final error reporting
    else:
        custom_error = "Custom Google Lens fallback implementation not configured."
        logging.info("Custom Google Lens fallback implementation not configured.")
        # Fall through to final error reporting

    # 3. Both Primary (SerpAPI) and Fallback (Custom) failed or weren't possible
    logging.error(f"Both SerpAPI and Custom Google Lens failed for image {index+1}.")
    final_error_msg = f"SerpAPI failed ({serpapi_error}). "
    final_error_msg += f"Custom fallback also failed or not configured ({custom_error})."

    return UrlFetchResult(url=image_url, content=None, error=final_error_msg, type="google_lens_fallback_failed", original_index=index)