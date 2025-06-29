import time
from typing import Dict, Set, Tuple, Any, List
import discord
from google.genai import types as google_types
from ..core import models
from ..core.constants import (
    MAX_EMBED_DESCRIPTION_LENGTH,
    GOOGLE_LENS_PATTERN,
    PROVIDERS_SUPPORTING_USERNAMES,
    MAX_PLAIN_TEXT_LENGTH,
    GEMINI_USE_THINKING_BUDGET_CONFIG_KEY,
    GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY,
    MAX_MESSAGE_NODES_CONFIG_KEY,
    AT_AI_PATTERN,
from ..core.config_loader import get_max_text_for_model
from ..core.utils import extract_urls_with_indices, is_image_url, extract_text_from_pdf_bytes
from ..llm.model_selector import determine_final_model
from ..messaging.message_parser import clean_message_content, check_google_lens_trigger
from ..content.processor import process_content_and_grounding
from ..bot.user_preferences import get_user_system_prompt_preference, get_user_gemini_thinking_budget_preference
from ..services.prompt_utils import prepare_system_prompt
from ..messaging.response_sender import handle_llm_response_stream, resend_imgur_urls
from ..messaging.history_utils import build_message_history
class StandardMessageHandler:
    """Handles standard message processing (non-prefix, non-deep-search)."""
    pass
    @staticmethod
    async def process_standard_message(
        message: discord.Message,
        original_content_for_processing: str,
        processing_msg: discord.Message,
        bot_client,
        start_time: float,
        disable_grounding: bool = False,
        is_deep_search: bool = False
    ) -> None:
        """Process a standard message through the full LLM pipeline."""
        pass
        # Clean content and check for Google Lens
        cleaned_content = clean_message_content(
            original_content_for_processing,
            bot_client.user.mention if bot_client.user else None,
            isinstance(message.channel, discord.DMChannel),
        pass
        image_attachments = [
            att for att in message.attachments 
            if att.content_type and att.content_type.startswith("image/")
        ]
        user_warnings = set()
        pass
        use_google_lens, cleaned_content, lens_warning = check_google_lens_trigger(
            cleaned_content, image_attachments, bot_client.config
        if lens_warning:
            user_warnings.add(lens_warning)
        if use_google_lens:
        pass
        user_id = message.author.id
        pass
        # Determine potential image URLs in text
        has_potential_image_urls_in_text = await StandardMessageHandler._check_image_urls_in_content(
            cleaned_content, message
        pass
        # Determine final model
        (
            final_provider_slash_model,
            provider,
            model_name,
            provider_config,
            all_api_keys,
            is_gemini,
            is_grok_model,
            keys_required,
            accept_files,
        ) = determine_final_model(
            user_id=user_id,
            initial_cleaned_content=cleaned_content,
            image_attachments=image_attachments,
            has_potential_image_urls_in_text=has_potential_image_urls_in_text,
            config=bot_client.config,
            user_warnings=user_warnings,
        pass
        # If Google Lens was triggered but final model can't handle images, disable lens
        if use_google_lens and not accept_files:
            use_google_lens = False
            if GOOGLE_LENS_PATTERN.match(cleaned_content):
                cleaned_content = GOOGLE_LENS_PATTERN.sub("", cleaned_content).strip()
        pass
        # Basic config assignments
        max_files_per_message = bot_client.config.get("max_images", 5)
        max_tokens_for_text_config = get_max_text_for_model(bot_client.config, final_provider_slash_model)
        max_messages = bot_client.config.get("max_messages", 25)
        use_plain_responses = bot_client.config.get("use_plain_responses", False)
        split_limit = (
            MAX_EMBED_DESCRIPTION_LENGTH if not use_plain_responses 
            else MAX_PLAIN_TEXT_LENGTH
        pass
        # Content processing and grounding (disable for deep search)
        if disable_grounding:
            # For deep search queries, disable grounding/tools
            formatted_user_urls_content = ""
            formatted_google_lens_content = ""
            searxng_derived_context_str = ""
            url_fetch_results = {}
            custom_search_queries_generated_flag = False
            successful_api_results_count = 0
            use_google_lens = False
        else:
            grounding_results = await process_content_and_grounding(
                new_msg=message,
                cleaned_content=cleaned_content,
                image_attachments=image_attachments,
                use_google_lens=use_google_lens,
                provider=provider,
                model_name=model_name,
                is_gemini_provider=is_gemini,
                is_grok_model=is_grok_model,
                config=bot_client.config,
                user_warnings=user_warnings,
                httpx_client=bot_client.httpx_client,
                max_messages=max_messages,
                max_tokens_for_text_config=max_tokens_for_text_config,
                max_files_per_message_config=max_files_per_message,
                current_model_accepts_files=accept_files,
                msg_nodes_cache=bot_client.msg_nodes,
                bot_user_obj=bot_client.user,
                models_module=models,
                google_types_module=google_types,
            pass
            (
                formatted_user_urls_content,
                formatted_google_lens_content,
                searxng_derived_context_str,
                url_fetch_results,
                custom_search_queries_generated_flag,
                successful_api_results_count,
                cleaned_content,
            ) = grounding_results
        pass
        # Build message history
        history_for_llm = await build_message_history(
            new_msg=message,
            initial_cleaned_content=cleaned_content,
            current_formatted_user_urls=formatted_user_urls_content,
            current_formatted_google_lens=formatted_google_lens_content,
            current_formatted_search_results=searxng_derived_context_str,
            max_messages=max_messages,
            max_tokens_for_text=max_tokens_for_text_config,
            max_files_per_message=max_files_per_message,
            accept_files=accept_files,
            use_google_lens_for_current=use_google_lens,
            is_target_provider_gemini=is_gemini,
            target_provider_name=provider,
            target_model_name=model_name,
            user_warnings=user_warnings,
            current_message_url_fetch_results=url_fetch_results,
            msg_nodes_cache=bot_client.msg_nodes,
            bot_user_obj=bot_client.user,
            httpx_async_client=bot_client.httpx_client,
            models_module=models,
            google_types_module=google_types,
            extract_text_from_pdf_bytes_func=extract_text_from_pdf_bytes,
            at_ai_pattern_re=AT_AI_PATTERN,
            providers_supporting_usernames_const=PROVIDERS_SUPPORTING_USERNAMES,
            system_prompt_text_for_budgeting=prepare_system_prompt(
                is_gemini,
                provider,
                get_user_system_prompt_preference(
                    message.author.id, bot_client.config.get("system_prompt")
                ),
            ),
            config=bot_client.config,
        pass
        if not history_for_llm:
            return
        pass
            f"history length: {len(history_for_llm)}, "
            f"google_lens: {use_google_lens}, warnings: {user_warnings}):\n{message.content}"
        pass
        # Prepare system prompt and API parameters
        default_system_prompt_from_config = bot_client.config.get("system_prompt")
        base_system_prompt_text = get_user_system_prompt_preference(
            message.author.id, default_system_prompt_from_config
        system_prompt_text = prepare_system_prompt(is_gemini, provider, base_system_prompt_text)
        extra_api_params = bot_client.config.get("extra_api_parameters", {}).copy()
        pass
        if is_gemini:
            global_use_thinking_budget = bot_client.config.get(GEMINI_USE_THINKING_BUDGET_CONFIG_KEY, False)
            global_thinking_budget_value = bot_client.config.get(GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY, 0)
            user_wants_thinking_budget = get_user_gemini_thinking_budget_preference(
                message.author.id, global_use_thinking_budget
            if user_wants_thinking_budget:
                extra_api_params["thinking_budget"] = global_thinking_budget_value
        pass
        # Handle LLM response
        (
            llm_call_successful,
            final_text,
            response_msgs,
        ) = await handle_llm_response_stream(
            client=bot_client,
            new_msg=message,
            processing_msg=processing_msg,
            provider=provider,
            model_name=model_name,
            history_for_llm=history_for_llm,
            system_prompt_text=system_prompt_text,
            provider_config=provider_config,
            extra_api_params=extra_api_params,
            app_config=bot_client.config,
            initial_user_warnings=user_warnings,
            use_plain_responses_config=use_plain_responses,
            split_limit_config=split_limit,
            custom_search_queries_generated=custom_search_queries_generated_flag,
            successful_api_results_count=successful_api_results_count,
            deep_search_used=is_deep_search,
        pass
        # Cleanup and finalization
        await StandardMessageHandler._cleanup_and_finalize(
            bot_client, message, response_msgs, llm_call_successful, 
            final_text, start_time
    pass
    @staticmethod
    async def _check_image_urls_in_content(cleaned_content: str, message: discord.Message) -> bool:
        """Check if there are potential image URLs in the content or replied messages."""
        has_potential_image_urls_in_text = False
        pass
        if cleaned_content:
            urls_in_text_for_image_check = extract_urls_with_indices(cleaned_content)
            if any(is_image_url(url_info[0]) for url_info in urls_in_text_for_image_check):
                has_potential_image_urls_in_text = True
        pass
        # Also check for image URLs in replied-to messages
        if (
            not has_potential_image_urls_in_text and 
            message.reference and 
            message.reference.message_id
        ):
            try:
                referenced_msg = message.reference.cached_message
                if not referenced_msg:
                    # Properly await the async fetch
                    referenced_msg = await message.channel.fetch_message(message.reference.message_id)
                if referenced_msg and referenced_msg.content:
                    urls_in_replied_msg = extract_urls_with_indices(referenced_msg.content)
                    if any(is_image_url(url_info[0]) for url_info in urls_in_replied_msg):
                        has_potential_image_urls_in_text = True
            except (discord.NotFound, discord.HTTPException, Exception):
                pass
        pass
        return has_potential_image_urls_in_text
    pass
    @staticmethod
    async def _cleanup_and_finalize(
        bot_client,
        message: discord.Message,
        response_msgs: List[discord.Message],
        llm_call_successful: bool,
        final_text: str,
        start_time: float
    ) -> None:
        """Handle cleanup and finalization tasks."""
        try:
            pass
        finally:
            if llm_call_successful and final_text:
                await resend_imgur_urls(message, response_msgs, final_text)
            pass
            for response_msg in response_msgs:
                if response_msg and response_msg.id in bot_client.msg_nodes:
                    node = bot_client.msg_nodes[response_msg.id]
                    if llm_call_successful:
                        async with node.lock:
                            node.full_response_text = final_text
                elif response_msg:
            pass
            max_nodes_from_config = bot_client.config.get(MAX_MESSAGE_NODES_CONFIG_KEY, 500)
            if (num_nodes := len(bot_client.msg_nodes)) > max_nodes_from_config:
                nodes_to_delete = sorted(bot_client.msg_nodes.keys())[
                    : num_nodes - max_nodes_from_config
                ]
                for msg_id in nodes_to_delete:
                    bot_client.msg_nodes.pop(msg_id, None)
            pass
            end_time = time.time()