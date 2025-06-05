import logging
import re
from typing import Tuple, Dict, Any, List, Set, Optional

import httpx  # Added import
import discord  # Required for discord.utils.escape_markdown if used here, or pass formatted text

from ..core.constants import (
    AT_AI_PATTERN,  # May not be needed here if content is already cleaned
    PROVIDERS_SUPPORTING_USERNAMES,  # For build_message_history call
    GROUNDING_SYSTEM_PROMPT_CONFIG_KEY,
    GROUNDING_MODEL_CONFIG_KEY,
)
from ..core import models
from ..core.utils import (
    extract_urls_with_indices,  # May not be needed if URLs are pre-extracted
    extract_text_from_pdf_bytes,  # For build_message_history
)
from ..llm.handler import (
    generate_response_stream,
)  # For grounding/search query generation
from .external_content import (
    fetch_external_content,
    format_external_content,
)
from ..services.grounding import (
    get_web_search_queries_from_gemini,
    fetch_and_format_searxng_results,
    generate_search_queries_with_custom_prompt,
)
from ..messaging.history_utils import (
    build_message_history,
)  # If history building is part of this scope
from ..services.prompt_utils import prepare_system_prompt


async def extract_urls_from_replied_message(new_msg: discord.Message) -> str:
    """
    Extracts URLs from the message that was replied to, if any.
    Returns a string containing any URLs found in the replied-to message.
    """
    if not new_msg.reference or not new_msg.reference.message_id:
        return ""
    
    try:
        # Try to get the referenced message from cache first
        referenced_msg = new_msg.reference.cached_message
        if not referenced_msg:
            # If not in cache, fetch it
            referenced_msg = await new_msg.channel.fetch_message(new_msg.reference.message_id)
        
        if referenced_msg and referenced_msg.content:
            # Extract URLs from the referenced message content
            urls_with_indices = extract_urls_with_indices(referenced_msg.content)
            if urls_with_indices:
                urls = [url for url, _ in urls_with_indices]
                urls_content = " ".join(urls)
                logging.info(f"Found {len(urls)} URL(s) in replied-to message {referenced_msg.id}: {urls}")
                return urls_content
                
    except (discord.NotFound, discord.HTTPException) as e:
        logging.warning(f"Could not fetch referenced message {new_msg.reference.message_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error extracting URLs from referenced message: {e}", exc_info=True)
    
    return ""


ProcessContentResult = Tuple[
    Optional[str],  # formatted_user_urls_content
    Optional[str],  # formatted_google_lens_content
    Optional[str],  # searxng_derived_context_str
    List[models.UrlFetchResult],  # url_fetch_results
    bool,  # custom_search_queries_generated_flag
    int,  # successful_api_results_count
    str,  # potentially modified cleaned_content (if image URLs removed)
]


async def process_content_and_grounding(
    new_msg: discord.Message,  # Pass new_msg for history building context
    cleaned_content: str,
    image_attachments: List[discord.Attachment],
    use_google_lens: bool,
    provider: str,  # Current provider (after model selection)
    model_name: str,  # Current model name
    is_gemini_provider: bool,  # Whether the current provider is Gemini
    is_grok_model: bool,  # Whether the current model is Grok
    config: Dict[str, Any],
    user_warnings: Set[str],
    httpx_client: httpx.AsyncClient,
    # For build_message_history if called from here
    max_messages: int,
    max_tokens_for_text_config: int,
    max_files_per_message_config: int,  # Renamed to avoid conflict
    current_model_accepts_files: bool,  # Renamed
    msg_nodes_cache: dict,
    bot_user_obj: discord.User,
    models_module: Any,
    google_types_module: Any,
) -> ProcessContentResult:
    """
    Handles fetching external content (user URLs, Google Lens) and performing
    web grounding (SearxNG via Gemini or custom prompt) if applicable.
    Returns formatted content strings and related metadata.
    """
    formatted_user_urls_content: Optional[str] = ""
    formatted_google_lens_content: Optional[str] = ""
    searxng_derived_context_str: Optional[str] = ""
    url_fetch_results: List[models.UrlFetchResult] = []

    custom_search_performed = False
    custom_search_queries_generated_flag = False
    successful_api_results_count = 0

    # This will be the content passed to build_message_history later
    content_after_processing = cleaned_content

    # Extract URLs from the current message content
    all_urls_in_cleaned_content = extract_urls_with_indices(cleaned_content)
    
    # Extract URLs from replied-to message if present
    replied_urls_content = await extract_urls_from_replied_message(new_msg)
    
    # Combine current message content with URLs from replied-to message for URL processing
    combined_content_for_url_extraction = cleaned_content
    if replied_urls_content:
        combined_content_for_url_extraction = f"{cleaned_content} {replied_urls_content}".strip()
        logging.info(f"Added URLs from replied-to message to content processing. Combined content: {len(combined_content_for_url_extraction)} chars")
    
    # Re-extract URLs from combined content
    all_urls_in_combined_content = extract_urls_with_indices(combined_content_for_url_extraction)
    user_has_provided_urls = bool(all_urls_in_combined_content)
    
    has_only_backticked_urls = False

    if user_has_provided_urls:
        all_backticked = True
        for url, index_pos in all_urls_in_combined_content:
            char_before_is_backtick = (
                index_pos > 0 and combined_content_for_url_extraction[index_pos - 1] == "`"
            )
            char_after_is_backtick = (index_pos + len(url)) < len(
                combined_content_for_url_extraction
            ) and combined_content_for_url_extraction[index_pos + len(url)] == "`"
            if not (char_before_is_backtick and char_after_is_backtick):
                all_backticked = False
                break
        if all_backticked:
            has_only_backticked_urls = True

    if use_google_lens:
        url_fetch_results = await fetch_external_content(
            combined_content_for_url_extraction,
            image_attachments,
            True,  # use_google_lens is true here
            max_files_per_message_config,
            user_warnings,
            config,
            httpx_client,
        )
        if url_fetch_results:
            formatted_parts = format_external_content(url_fetch_results)
            formatted_user_urls_content = formatted_parts.get("user_urls", "")
            formatted_google_lens_content = formatted_parts.get("lens", "")
            if formatted_user_urls_content:
                logging.info(
                    f"Formatted content from user-provided URLs (Lens path): {len(formatted_user_urls_content)} chars"
                )
            if formatted_google_lens_content:
                logging.info(
                    f"Formatted content from Google Lens: {len(formatted_google_lens_content)} chars"
                )
        logging.info(
            "Google Lens was active. User-provided URLs and Lens content (if any) have been processed."
        )
        # SearxNG/grounding is skipped if Lens is active.
    else:  # Not using Google Lens, proceed with other content fetching/grounding
        alt_search_config_dict = config.get("alternative_search_query_generation", {})
        is_alt_search_enabled = alt_search_config_dict.get("enabled", False)
        trigger_alternative_search = False

        if is_alt_search_enabled:
            if (
                user_has_provided_urls and not has_only_backticked_urls
            ):  # Don't trigger if user explicitly gave URLs unless they are all backticked
                logging.info(
                    "User query contains non-backticked URLs. Skipping alternative search query generation."
                )
            elif model_name == "gemini-2.0-flash-preview-image-generation":
                # Don't trigger alternative search for image generation model
                logging.info(
                    "Image generation model detected. Skipping alternative search query generation."
                )
            else:
                if (
                    provider != "google"
                ):  # Only trigger for non-Google (non-Gemini) models
                    trigger_alternative_search = True

        if trigger_alternative_search:
            logging.info(
                f"Attempting alternative search query generation for model {provider}/{model_name}"
            )
            # History for custom prompt needs to be built *without* the content that will be generated by this step
            history_for_custom_prompt = await build_message_history(
                new_msg=new_msg,
                initial_cleaned_content=cleaned_content,
                current_formatted_user_urls="",  # No user URLs for this specific call's context
                current_formatted_google_lens="",  # No Lens for this specific call's context
                current_formatted_search_results="",  # No search results for this specific call's context
                max_messages=max_messages,
                max_tokens_for_text=max_tokens_for_text_config,
                max_files_per_message=max_files_per_message_config,
                accept_files=current_model_accepts_files,  # Vision capability of the *current final model*
                use_google_lens_for_current=False,
                is_target_provider_gemini=(
                    provider == "google"
                ),  # Based on current final model
                target_provider_name=provider,
                target_model_name=model_name,
                user_warnings=user_warnings,
                current_message_url_fetch_results=None,
                msg_nodes_cache=msg_nodes_cache,
                bot_user_obj=bot_user_obj,
                httpx_async_client=httpx_client,
                models_module=models_module,
                google_types_module=google_types_module,
                extract_text_from_pdf_bytes_func=extract_text_from_pdf_bytes,
                at_ai_pattern_re=AT_AI_PATTERN,
                providers_supporting_usernames_const=PROVIDERS_SUPPORTING_USERNAMES,
                system_prompt_text_for_budgeting=None,  # System prompt for query gen is handled within generate_search_queries_with_custom_prompt
                config=config,
            )

            if history_for_custom_prompt:
                image_urls_for_query_gen = [
                    att.url for att in image_attachments
                ]  # Pass current message images
                custom_search_queries_result = (
                    await generate_search_queries_with_custom_prompt(
                        latest_query=cleaned_content,
                        chat_history=history_for_custom_prompt,
                        config=config,
                        generate_response_stream_func=generate_response_stream,
                        current_model_id=f"{provider}/{model_name}",
                        httpx_client=httpx_client,
                        image_urls=image_urls_for_query_gen,
                    )
                )
                if isinstance(
                    custom_search_queries_result, dict
                ) and custom_search_queries_result.get("web_search_required"):
                    queries = custom_search_queries_result.get("search_queries", [])
                    if queries:
                        custom_search_queries_generated_flag = True
                        logging.info(
                            f"Custom prompt generated {len(queries)} search queries: {queries}"
                        )
                        (
                            searxng_derived_context_str,
                            successful_api_results_count,
                        ) = await fetch_and_format_searxng_results(
                            queries, cleaned_content, config, httpx_client
                        )
                        if searxng_derived_context_str:
                            logging.info(
                                f"Fetched SearxNG results (custom prompt): {len(searxng_derived_context_str)} chars"
                            )
            custom_search_performed = True
        elif (
            (not user_has_provided_urls or has_only_backticked_urls)
            and not is_gemini_provider
            and not is_grok_model
            and not custom_search_performed
        ):
            logging.info(
                f"Target model '{provider}/{model_name}' is non-Gemini/non-Grok, no user URLs or only backticked. Attempting Gemini grounding pre-step."
            )
            # History for Gemini grounding
            grounding_model_for_history = config.get(
                GROUNDING_MODEL_CONFIG_KEY, "google/gemini-2.5-flash-preview-05-20"
            )
            grounding_provider_for_history, grounding_model_name_for_history = (
                grounding_model_for_history.split("/", 1)
            )

            history_for_gemini_grounding = await build_message_history(
                new_msg=new_msg,
                initial_cleaned_content=cleaned_content,
                current_formatted_user_urls="",
                current_formatted_google_lens="",
                current_formatted_search_results="",
                max_messages=max_messages,
                max_tokens_for_text=max_tokens_for_text_config,
                max_files_per_message=max_files_per_message_config,
                accept_files=True,  # Grounding model (Gemini Flash) accepts files
                use_google_lens_for_current=False,
                is_target_provider_gemini=True,  # Grounding is with Gemini
                target_provider_name=grounding_provider_for_history,
                target_model_name=grounding_model_name_for_history,
                user_warnings=user_warnings,
                current_message_url_fetch_results=None,
                msg_nodes_cache=msg_nodes_cache,
                bot_user_obj=bot_user_obj,
                httpx_async_client=httpx_client,
                models_module=models_module,
                google_types_module=google_types_module,
                extract_text_from_pdf_bytes_func=extract_text_from_pdf_bytes,
                at_ai_pattern_re=AT_AI_PATTERN,
                providers_supporting_usernames_const=PROVIDERS_SUPPORTING_USERNAMES,
                system_prompt_text_for_budgeting=prepare_system_prompt(
                    True,
                    grounding_provider_for_history,
                    config.get(GROUNDING_SYSTEM_PROMPT_CONFIG_KEY),
                ),
                config=config,
            )
            if history_for_gemini_grounding:
                system_prompt_for_grounding = prepare_system_prompt(
                    True,
                    grounding_provider_for_history,
                    config.get(GROUNDING_SYSTEM_PROMPT_CONFIG_KEY),
                )
                web_search_queries = await get_web_search_queries_from_gemini(
                    history_for_gemini_grounding,
                    system_prompt_for_grounding,
                    config,
                    generate_response_stream,
                )
                if web_search_queries:
                    custom_search_queries_generated_flag = (
                        True  # Indicates internet was used
                    )
                    (
                        searxng_derived_context_str,
                        successful_api_results_count,
                    ) = await fetch_and_format_searxng_results(
                        web_search_queries, cleaned_content, config, httpx_client
                    )
                    if searxng_derived_context_str:
                        logging.info(
                            f"Formatted SearxNG results (Gemini grounding): {len(searxng_derived_context_str)} chars"
                        )
        elif (
            user_has_provided_urls and not custom_search_performed
        ):  # Process user-provided URLs if no other search happened
            url_fetch_results = await fetch_external_content(
                combined_content_for_url_extraction,
                image_attachments,
                False,  # use_google_lens is False here
                max_files_per_message_config,
                user_warnings,
                config,
                httpx_client,
            )
            if url_fetch_results:
                formatted_parts = format_external_content(url_fetch_results)
                formatted_user_urls_content = formatted_parts.get("user_urls", "")
                if formatted_user_urls_content:
                    logging.info(
                        f"Formatted content from user-provided URLs (direct path): {len(formatted_user_urls_content)} chars"
                    )

    # Remove successfully fetched image URLs from cleaned_content to avoid re-processing by LLM as text
    if url_fetch_results:
        successfully_fetched_image_urls = {
            res.url
            for res in url_fetch_results
            if res.type == "image_url_content" and res.content and not res.error
        }
        if successfully_fetched_image_urls:
            temp_cleaned_content = (
                content_after_processing  # Start with current state of content
            )
            for img_url in successfully_fetched_image_urls:
                temp_cleaned_content = temp_cleaned_content.replace(img_url, "")
            content_after_processing = re.sub(
                r"\s{2,}", " ", temp_cleaned_content
            ).strip()
            logging.info(
                f"Removed {len(successfully_fetched_image_urls)} successfully fetched image URLs from content_after_processing."
            )

    return (
        formatted_user_urls_content,
        formatted_google_lens_content,
        searxng_derived_context_str,
        url_fetch_results,
        custom_search_queries_generated_flag,
        successful_api_results_count,
        content_after_processing,
    )
