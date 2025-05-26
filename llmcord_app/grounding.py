import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator, Tuple

import discord  # For discord.utils.escape_markdown
import httpx

from . import models  # Use relative import
from .constants import (
    SEARXNG_BASE_URL_CONFIG_KEY,
    SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY,
    GROUNDING_MODEL_CONFIG_KEY,
    GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY,  # Added
    GROUNDING_MODEL_TOP_K_CONFIG_KEY,  # Added
    GROUNDING_MODEL_TOP_P_CONFIG_KEY,  # Added
    DEFAULT_GROUNDING_MODEL_TEMPERATURE, # Added
    DEFAULT_GROUNDING_MODEL_TOP_K, # Added
    DEFAULT_GROUNDING_MODEL_TOP_P, # Added
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
    DEFAULT_WEB_CONTENT_EXTRACTION_API_ENABLED,
    DEFAULT_WEB_CONTENT_EXTRACTION_API_URL,
    DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS,
)
from .content_fetchers import (
    fetch_searxng_results,
    fetch_youtube_data,
    fetch_reddit_data,
    fetch_general_url_content,  # This is the dynamic one from content_fetchers.web
)
from .utils import is_youtube_url, is_reddit_url, extract_reddit_submission_id
# The generate_response_stream function will be passed as an argument


async def get_web_search_queries_from_gemini(
    history_for_gemini_grounding: List[Dict[str, Any]],
    system_prompt_text_for_grounding: Optional[str],
    config: Dict[str, Any],
    # Pass the generate_response_stream function as a callable
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
    grounding_model_str = config.get(
        GROUNDING_MODEL_CONFIG_KEY, "google/gemini-2.5-flash-preview-05-20"
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
            "thinking_budget": 0,  # Keep thinking budget minimal for grounding
        }

        stream_generator = generate_response_stream_func(
            provider=grounding_provider,
            model_name=grounding_model_name,
            history_for_llm=history_for_gemini_grounding,
            system_prompt_text=system_prompt_text_for_grounding,
            provider_config=gemini_provider_config,
            extra_params=grounding_extra_params,
            app_config=config,  # Pass app_config
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
            f"All API keys failed for Gemini grounding model ({grounding_provider}/{grounding_model_name}): {e}"
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
    api_content_limit = config.get(SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY) # Reuse existing limit

    if api_enabled:
        logging.info(
            f"External Web Content API is enabled. Using it for query: '{queries[0]}'"
        )
        payload = {"query": queries[0], "max_results": api_max_results}
        try:
            response = await httpx_client.post(api_url, json=payload, timeout=30.0) # Consider making timeout configurable
            response.raise_for_status()
            api_response_json = response.json()

            if api_response_json.get("error"):
                logging.error(
                    f"External Web Content API returned a global error: {api_response_json['error']}"
                )
                return None

            api_results_data = api_response_json.get("results", [])
            if not api_results_data:
                logging.info("External Web Content API returned no results.")
                return None

            formatted_api_results_content = []
            for item_idx, item in enumerate(api_results_data):
                item_str_parts = []
                item_url = item.get("url", "N/A")
                item_source_type = item.get("source_type", "unknown")
                
                item_str_parts.append(f"Source {item_idx + 1}: {item_url} (Type: {item_source_type})")

                if item.get("processed_successfully") and item.get("data"):
                    data = item["data"]
                    
                    if item_source_type == "youtube":
                        title = data.get("title", "N/A")
                        item_str_parts.append(f"  Title: {discord.utils.escape_markdown(title)}")
                        channel = data.get("channel_name", "N/A")
                        item_str_parts.append(f"  Channel: {discord.utils.escape_markdown(channel)}")
                        transcript = data.get("transcript")
                        if transcript:
                            if api_content_limit and len(transcript) > api_content_limit:
                                transcript = transcript[: api_content_limit - 3] + "..."
                            item_str_parts.append(f"  Transcript: {discord.utils.escape_markdown(transcript)}")
                        comments = data.get("comments")
                        if comments and isinstance(comments, list):
                            comment_texts = [
                                f"    - {discord.utils.escape_markdown(c.get('text', ''))}"
                                for c in comments if isinstance(c, dict) and c.get("text")
                            ]
                            if comment_texts:
                                item_str_parts.append("  Comments:\n" + "\n".join(comment_texts))
                    
                    elif item_source_type == "reddit":
                        title = data.get("post_title", "N/A")
                        item_str_parts.append(f"  Title: {discord.utils.escape_markdown(title)}")
                        body = data.get("post_body")
                        if body:
                            if api_content_limit and len(body) > api_content_limit:
                                body = body[: api_content_limit - 3] + "..."
                            item_str_parts.append(f"  Post Body: {discord.utils.escape_markdown(body)}")
                        comments = data.get("comments")
                        if comments and isinstance(comments, list):
                            comment_texts = [
                                f"    - {discord.utils.escape_markdown(c.get('text', ''))} (Score: {c.get('score', 'N/A')})"
                                for c in comments if isinstance(c, dict) and c.get("text")
                            ]
                            if comment_texts:
                                item_str_parts.append("  Comments:\n" + "\n".join(comment_texts))
                                
                    elif item_source_type == "pdf":
                        text_content = data.get("text_content")
                        if text_content:
                            if api_content_limit and len(text_content) > api_content_limit:
                                text_content = text_content[: api_content_limit - 3] + "..."
                            item_str_parts.append(f"  Content: {discord.utils.escape_markdown(text_content)}")
                            
                    elif item_source_type == "webpage":
                        title = data.get("title")
                        if title:
                            item_str_parts.append(f"  Title: {discord.utils.escape_markdown(title)}")
                        text_content = data.get("text_content")
                        if text_content:
                            if api_content_limit and len(text_content) > api_content_limit:
                                text_content = text_content[: api_content_limit - 3] + "..."
                            item_str_parts.append(f"  Content: {discord.utils.escape_markdown(text_content)}")
                    else: # unknown or other types
                        raw_content_str = str(data)
                        if api_content_limit and len(raw_content_str) > api_content_limit:
                             raw_content_str = raw_content_str[:api_content_limit-3]+"..."
                        item_str_parts.append(f"  Data: {discord.utils.escape_markdown(raw_content_str)}")
                    
                    formatted_api_results_content.append("\n".join(item_str_parts))
                elif item.get("error"):
                    logging.warning(
                        f"External API processing error for URL {item_url}: {item.get('error')}"
                    )
                    item_str_parts.append(f"  Error: {item.get('error')}")
                    formatted_api_results_content.append("\n".join(item_str_parts))


            if formatted_api_results_content:
                query_for_header = queries[0] # Since we used queries[0] for the API call
                header = (
                    f"Answer the user's query based on the following information from the External Web Content API "
                    f'(for query: "{discord.utils.escape_markdown(query_for_header)}"): \n\n'
                )
                final_context = header + "\n\n---\n\n".join(formatted_api_results_content)
                logging.info(f"Formatted context from External Web Content API: {final_context[:500]}...")
                return final_context.strip()
            else:
                logging.info("External Web Content API did not return any processable content.")
                return None

        except httpx.RequestError as e:
            logging.error(f"Error calling External Web Content API at {api_url}: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logging.error(
                f"External Web Content API request failed with status {e.response.status_code}: {e.response.text}"
            )
            return None
        except Exception as e:
            logging.exception("Unexpected error while processing External Web Content API response.")
            return None
    
    # --- Original SearXNG and content fetching logic (if external API is not enabled or fails to return content) ---
    searxng_base_url = config.get(SEARXNG_BASE_URL_CONFIG_KEY)
    if not searxng_base_url:
        logging.warning(
            "SearxNG base URL not configured. Skipping web search enhancement."
        )
        return None

    logging.info(
        f"Fetching SearxNG results for {len(queries)} queries related to user query: '{user_query_for_log[:100]}...'"
    )
    searxng_content_limit = config.get(SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY) # Already defined as api_content_limit if API was enabled

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
        return final_context.strip()

    logging.info(
        "No content successfully fetched and formatted from SearxNG result URLs."
    )
    return None
