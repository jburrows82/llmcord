import logging
import re
import time
from typing import Dict

import discord
from discord import app_commands

# Use google.genai.types
from google.genai import types as google_types

import httpx

from .constants import (
    MAX_EMBED_DESCRIPTION_LENGTH,
    AT_AI_PATTERN,
    GOOGLE_LENS_PATTERN,
    VISION_MODEL_TAGS,
    AVAILABLE_MODELS,
    DEEP_SEARCH_KEYWORDS,
    PROVIDERS_SUPPORTING_USERNAMES,
    MAX_PLAIN_TEXT_LENGTH,
    GROUNDING_SYSTEM_PROMPT_CONFIG_KEY,
    GEMINI_USE_THINKING_BUDGET_CONFIG_KEY,
    GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY,
    EMBED_COLOR_ERROR,
    # New config keys
    MAX_MESSAGE_NODES_CONFIG_KEY,
    GROUNDING_MODEL_CONFIG_KEY,
    FALLBACK_VISION_MODEL_CONFIG_KEY,
    DEEP_SEARCH_MODEL_CONFIG_KEY,
)

from . import models
from .rate_limiter import check_and_perform_global_reset, close_all_db_managers
from .utils import (
    extract_urls_with_indices,
    is_image_url,
    extract_text_from_pdf_bytes,
)
from .llm_handler import generate_response_stream # generate_response_stream is already imported
from .commands import (
    set_model_command,
    get_user_model_preference,
    set_system_prompt_command,
    get_user_system_prompt_preference,
    setgeminithinking,
    get_user_gemini_thinking_budget_preference,
    help_command,
    load_all_preferences,
)
from .permissions import is_message_allowed
from .external_content import (
    fetch_external_content,
    format_external_content,
)
from .grounding import (
    get_web_search_queries_from_gemini,
    fetch_and_format_searxng_results,
)
from .prompt_utils import prepare_system_prompt
from .response_sender import (
    send_initial_processing_message,
    handle_llm_response_stream,
    resend_imgur_urls,  # Also import this helper if direct calls are made from bot.py for exceptions outside stream
)
from .history_utils import build_message_history


# --- Discord Client Setup ---
class LLMCordClient(discord.Client):
    def __init__(
        self,
        *,
        intents: discord.Intents,
        activity: discord.CustomActivity,
        config: Dict,
    ):
        super().__init__(intents=intents, activity=activity)
        self.tree = app_commands.CommandTree(self)
        # Use the correct MsgNode class from the models module
        self.msg_nodes: Dict[int, models.MsgNode] = {}  # Message cache
        self.last_task_time: float = 0  # For stream editing delay
        self.config = config  # Store loaded config
        self.httpx_client = httpx.AsyncClient(
            timeout=20.0, follow_redirects=True
        )  # HTTP client for attachments/web

        # Initialize content fetcher modules that need config
        from .content_fetchers.youtube import (
            initialize_ytt_api,
            initialize_youtube_data_api,
        )
        from .content_fetchers.reddit import initialize_reddit_client

        initialize_ytt_api(self.config.get("proxy_config"))
        initialize_youtube_data_api(self.config.get("youtube_api_key"))
        initialize_reddit_client(
            client_id=self.config.get("reddit_client_id"),
            client_secret=self.config.get("reddit_client_secret"),
            user_agent=self.config.get("reddit_user_agent"),
        )

    async def setup_hook(self):
        """Sync slash commands and load preferences when the bot is ready."""
        # --- ADDED: Load user preferences ---
        await load_all_preferences()
        # --- END ADDED ---

        self.tree.add_command(
            app_commands.Command(
                name="model",
                description="Set your preferred LLM provider and model.",
                callback=set_model_command,
            )
        )
        # --- ADDED: Register /systemprompt command ---
        self.tree.add_command(
            app_commands.Command(
                name="systemprompt",
                description="Set your custom system prompt for the bot.",
                callback=set_system_prompt_command,
            )
        )
        # --- ADDED: Register /setgeminithinking command ---
        self.tree.add_command(
            app_commands.Command(
                name="setgeminithinking",
                description="Toggle usage of the 'thinkingBudget' parameter for Gemini models.",
                callback=setgeminithinking,
            )
        )
        # --- ADDED: Register /help command ---
        self.tree.add_command(
            app_commands.Command(
                name="help",
                description="Displays all available commands and how to use them.",
                callback=help_command,
            )
        )
        # Sync commands
        await self.tree.sync()
        logging.info(f"Synced slash commands for {self.user}.")

    async def on_ready(self):
        """Called when the bot is ready and logged in."""
        logging.info(f"Logged in as {self.user}")
        # Initial check/reset of rate limits
        await check_and_perform_global_reset(self.config)

    async def on_message(self, new_msg: discord.Message):
        """Handles incoming messages."""
        # --- Basic Checks and Trigger ---
        if new_msg.author.bot or new_msg.author == self.user:
            return
        start_time = time.time()

        is_dm = isinstance(new_msg.channel, discord.DMChannel)
        allow_dms = self.config.get("allow_dms", True)

        # Determine if the bot should process this message
        should_process = False
        mentions_bot = self.user.mentioned_in(new_msg)
        contains_at_ai = AT_AI_PATTERN.search(new_msg.content) is not None
        original_content_for_processing = new_msg.content

        if is_dm:
            if allow_dms:
                # In DMs, process unless it's a reply *not* to the bot and doesn't trigger
                is_reply_to_user = (
                    new_msg.reference
                    and new_msg.reference.resolved
                    and new_msg.reference.resolved.author != self.user
                )

                if is_reply_to_user and not (mentions_bot or contains_at_ai):
                    should_process = False  # Don't process replies to other users unless explicitly triggered
                else:
                    should_process = True  # Process direct messages, replies to bot, or explicitly triggered replies
            else:
                return  # Block DMs if not allowed
        elif mentions_bot or contains_at_ai:
            should_process = True

        if not should_process:
            return

        # --- Reload config & Check Global Reset ---
        # Config is now an instance variable, consider if reloading is needed per message
        # self.config = await get_config() # Uncomment if hot-reloading is desired, get_config is now async
        await check_and_perform_global_reset(self.config)

        # --- Permissions Check ---
        if not is_message_allowed(new_msg, self.config, is_dm):
            logging.warning(
                f"Blocked message from user {new_msg.author.id} in channel {new_msg.channel.id} due to permissions."
            )
            return

        # --- Send Initial "Processing" Message ---
        _use_plain_for_initial_status = self.config.get("use_plain_responses", False)
        processing_msg = await send_initial_processing_message(
            new_msg, _use_plain_for_initial_status
        )

        # --- Clean Content and Check for Google Lens ---
        cleaned_content = original_content_for_processing
        if not is_dm and mentions_bot:
            cleaned_content = cleaned_content.replace(self.user.mention, "").strip()
        cleaned_content = AT_AI_PATTERN.sub(" ", cleaned_content).strip()
        cleaned_content = re.sub(r"\s{2,}", " ", cleaned_content).strip()
        logging.debug(f"Cleaned content for keyword/URL check: '{cleaned_content}'")

        use_google_lens = False
        image_attachments = [
            att
            for att in new_msg.attachments
            if att.content_type and att.content_type.startswith("image/")
        ]
        user_warnings = set()

        if GOOGLE_LENS_PATTERN.match(cleaned_content) and image_attachments:
            serpapi_keys_ok = bool(self.config.get("serpapi_api_keys"))
            if not serpapi_keys_ok:
                # Updated log and warning messages
                logging.warning(
                    "Google Lens requested but SerpAPI keys are not configured."
                )
                user_warnings.add(
                    "⚠️ Google Lens requested but requires SerpAPI key configuration."
                )
            else:
                use_google_lens = True
                cleaned_content = GOOGLE_LENS_PATTERN.sub("", cleaned_content).strip()
                logging.info(f"Google Lens keyword detected for message {new_msg.id}.")

        # --- LLM Provider/Model Selection ---
        user_id = new_msg.author.id
        default_model_str = self.config.get("model", "google/gemini-2.0-flash")
        provider_slash_model = get_user_model_preference(user_id, default_model_str)

        # --- Override Model based on Keywords ---
        final_provider_slash_model = provider_slash_model
        if any(keyword in cleaned_content.lower() for keyword in DEEP_SEARCH_KEYWORDS):
            target_model_str = self.config.get(
                DEEP_SEARCH_MODEL_CONFIG_KEY, "x-ai/grok-3"
            )
            target_provider, target_model_name = target_model_str.split("/", 1)
            if use_google_lens:
                logging.warning(
                    f"Both 'deepsearch'/'deepersearch' and 'googlelens' detected. Prioritizing deep search ('{target_model_str}'). Google Lens disabled."
                )
                use_google_lens = False

            xai_provider_config = self.config.get("providers", {}).get(
                target_provider, {}
            )
            xai_keys = xai_provider_config.get("api_keys", [])
            is_target_available = (
                target_provider in AVAILABLE_MODELS
                and target_model_name in AVAILABLE_MODELS.get(target_provider, [])
            )

            if xai_provider_config and xai_keys and is_target_available:
                final_provider_slash_model = target_model_str
                logging.info(
                    f"Keywords {DEEP_SEARCH_KEYWORDS} detected. Overriding model to {final_provider_slash_model}."
                )
            else:
                warning_reason = ""
                if not xai_provider_config:
                    warning_reason = f"provider '{target_provider}' not configured"
                elif not xai_keys:
                    warning_reason = f"no API keys for provider '{target_provider}'"
                elif not is_target_available:
                    warning_reason = (
                        f"model '{target_model_str}' not listed in AVAILABLE_MODELS"
                    )
                logging.warning(
                    f"Keywords {DEEP_SEARCH_KEYWORDS} detected, but cannot use '{target_model_str}' ({warning_reason}). Using original: {provider_slash_model}"
                )
                user_warnings.add(
                    f"⚠️ Deep search requested, but model '{target_model_str}' unavailable ({warning_reason})."
                )

        # --- Validate Final Model Selection ---
        try:
            provider, model_name = final_provider_slash_model.split("/", 1)
            if (
                provider not in AVAILABLE_MODELS
                or model_name not in AVAILABLE_MODELS.get(provider, [])
            ):
                logging.warning(
                    f"Final model '{final_provider_slash_model}' is invalid/unavailable. Falling back to default: {default_model_str}"
                )
                final_provider_slash_model = default_model_str
                provider, model_name = final_provider_slash_model.split("/", 1)
        except ValueError:
            logging.error(
                f"Invalid model format for final selection '{final_provider_slash_model}'. Using hardcoded default."
            )
            final_provider_slash_model = "google/gemini-2.0-flash"  # Fallback
            provider, model_name = final_provider_slash_model.split("/", 1)

        logging.info(
            f"Final model selected for user {user_id}: '{final_provider_slash_model}'"
        )

        # --- Get Config for the FINAL Provider ---
        provider_config = self.config.get("providers", {}).get(provider, {})
        if not isinstance(provider_config, dict):
            logging.error(
                f"Configuration for provider '{provider}' is invalid or missing. Cannot proceed."
            )
            return
        all_api_keys = provider_config.get("api_keys", [])

        is_gemini = provider == "google"
        is_grok_model = provider == "x-ai"
        keys_required = provider not in [
            "ollama",
            "lmstudio",
            "vllm",
            "oobabooga",
            "jan",
        ]

        if keys_required and not all_api_keys:
            logging.error(
                f"No API keys configured for the selected provider '{provider}' in config.yaml."
            )
            return

        # --- Configuration Values ---
        accept_files = any(x in model_name.lower() for x in VISION_MODEL_TAGS)
        if provider == "openai" and provider_config.get("disable_vision", False):
            accept_files = False
            logging.info(
                f"Vision explicitly disabled for OpenAI model '{model_name}' via config."
            )

        has_potential_image_urls_in_text = False
        if cleaned_content:
            urls_in_text = extract_urls_with_indices(cleaned_content)
            if any(is_image_url(url_info[0]) for url_info in urls_in_text):
                has_potential_image_urls_in_text = True

        if (image_attachments or has_potential_image_urls_in_text) and not accept_files:
            original_model_for_warning = final_provider_slash_model
            fallback_model_str = self.config.get(
                FALLBACK_VISION_MODEL_CONFIG_KEY,
                "google/gemini-2.5-flash-preview-05-20",
            )
            logging.info(
                f"Query has images, but current model '{final_provider_slash_model}' does not support vision. Switching to '{fallback_model_str}'."
            )
            user_warnings.add(
                f"⚠️ Images detected. Switched from '{original_model_for_warning}' to '{fallback_model_str}'."
            )
            final_provider_slash_model = fallback_model_str
            try:
                provider, model_name = final_provider_slash_model.split("/", 1)
                # Re-fetch provider_config and re-evaluate accept_files for the new model
                provider_config = self.config.get("providers", {}).get(provider, {})
                if not isinstance(
                    provider_config, dict
                ):  # Major issue if fallback model's provider is not configured
                    logging.error(
                        f"Fallback provider '{provider}' for model '{final_provider_slash_model}' not configured. This is a critical error."
                    )
                    return
                all_api_keys = provider_config.get("api_keys", [])
                is_gemini = provider == "google"
                is_grok_model = provider == "x-ai"
                keys_required = provider not in [
                    "ollama",
                    "lmstudio",
                    "vllm",
                    "oobabooga",
                    "jan",
                ]
                if keys_required and not all_api_keys:
                    user_warnings.add(
                        f"⚠️ Fallback model '{final_provider_slash_model}' has no API keys configured."
                    )
                accept_files = any(x in model_name.lower() for x in VISION_MODEL_TAGS)
                if provider == "openai" and provider_config.get(
                    "disable_vision", False
                ):
                    accept_files = False
                if not accept_files:  # If fallback STILL doesn't accept files
                    user_warnings.add(
                        f"⚠️ Fallback model '{final_provider_slash_model}' also cannot process images. Check configuration."
                    )
            except ValueError:  # Error splitting fallback model string
                logging.error(
                    f"Invalid format for FALLBACK_VISION_MODEL_PROVIDER_SLASH_MODEL: '{fallback_model_str}'."
                )
                user_warnings.add(
                    f"⚠️ Error switching to vision model. Processing with '{original_model_for_warning}'."
                )
                # Revert to original provider/model if fallback string is bad
                provider, model_name = original_model_for_warning.split("/", 1)
                provider_config = self.config.get("providers", {}).get(provider, {})
                all_api_keys = provider_config.get("api_keys", [])
                is_gemini = provider == "google"
                is_grok_model = provider == "x-ai"
                keys_required = provider not in [
                    "ollama",
                    "lmstudio",
                    "vllm",
                    "oobabooga",
                    "jan",
                ]
                accept_files = any(
                    x in model_name.lower() for x in VISION_MODEL_TAGS
                )  # Re-evaluate for original
                if provider == "openai" and provider_config.get(
                    "disable_vision", False
                ):
                    accept_files = False

        max_files_per_message = self.config.get("max_images", 5)
        max_tokens_for_text_config = self.config.get(
            "max_text", 2000
        )  # Default to 2000 tokens if not in config
        max_messages = self.config.get("max_messages", 25)
        use_plain_responses = self.config.get("use_plain_responses", False)
        split_limit = (
            MAX_EMBED_DESCRIPTION_LENGTH
            if not use_plain_responses
            else MAX_PLAIN_TEXT_LENGTH
        )

        is_text_empty = not cleaned_content.strip()
        has_meaningful_attachments_final = any(
            att.content_type
            and (
                (
                    att.content_type.startswith("image/")
                    and (accept_files or use_google_lens)
                )
                or att.content_type.startswith("text/")
                or (
                    att.content_type == "application/pdf"
                    and ((is_gemini and accept_files) or not is_gemini)
                )
            )
            for att in new_msg.attachments
        )
        if (
            is_text_empty
            and not has_meaningful_attachments_final
            and not new_msg.reference
        ):
            logging.info(
                f"Empty query received from user {new_msg.author.id} in channel {new_msg.channel.id}. Message ID: {new_msg.id}"
            )
            error_message_text = "Your query is empty. Please reply to a message to reference it or don't send an empty query."
            use_plain = self.config.get("use_plain_responses", False)

            try:
                if processing_msg:
                    if use_plain:
                        await processing_msg.edit(
                            content=error_message_text, embed=None, view=None
                        )
                    else:
                        error_embed = discord.Embed(
                            description=error_message_text, color=EMBED_COLOR_ERROR
                        )
                        await processing_msg.edit(embed=error_embed, view=None)
                else:
                    # Fallback if processing_msg couldn't be sent
                    if use_plain:
                        await new_msg.reply(
                            error_message_text,
                            mention_author=False,
                            suppress_embeds=True,
                        )
                    else:
                        error_embed = discord.Embed(
                            description=error_message_text, color=EMBED_COLOR_ERROR
                        )
                        await new_msg.reply(embed=error_embed, mention_author=False)
            except discord.HTTPException as e:
                logging.error(f"Failed to send/edit empty query error message: {e}")
            except Exception as e:
                logging.error(
                    f"Unexpected error sending/editing empty query error message: {e}",
                    exc_info=True,
                )
            return

        combined_context = ""
        url_fetch_results = []
        all_urls_in_cleaned_content = extract_urls_with_indices(cleaned_content)
        user_has_provided_urls = bool(all_urls_in_cleaned_content)
        has_only_backticked_urls = False

        if user_has_provided_urls:
            all_backticked = True
            for url, index_pos in all_urls_in_cleaned_content:
                char_before_is_backtick = (
                    index_pos > 0 and cleaned_content[index_pos - 1] == "`"
                )
                char_after_is_backtick = (index_pos + len(url)) < len(
                    cleaned_content
                ) and cleaned_content[index_pos + len(url)] == "`"
                if not (char_before_is_backtick and char_after_is_backtick):
                    all_backticked = False
                    break
            if all_backticked:
                has_only_backticked_urls = True

        if use_google_lens:
            url_fetch_results = await fetch_external_content(
                cleaned_content,
                image_attachments,
                True,
                max_files_per_message,
                user_warnings,
                self.config,
                self.httpx_client,
            )
            if url_fetch_results:
                combined_context = format_external_content(url_fetch_results)
            logging.info(
                "Skipping Gemini grounding/SearxNG step because Google Lens is active."
            )
        elif (
            (not user_has_provided_urls or has_only_backticked_urls)
            and not is_gemini
            and not is_grok_model
        ):
            if has_only_backticked_urls:
                logging.info(
                    f"Target model '{final_provider_slash_model}' is non-Gemini/non-Grok, Google Lens is not active, and all user URLs are backticked. Attempting grounding pre-step for SearxNG."
                )
            else:  # No user URLs at all
                logging.info(
                    f"Target model '{final_provider_slash_model}' is non-Gemini/non-Grok, Google Lens is not active, and no user URLs detected. Attempting grounding pre-step for SearxNG."
                )
            history_for_gemini_grounding = await build_message_history(
                new_msg=new_msg,
                initial_cleaned_content=cleaned_content,
                combined_context="",
                max_messages=max_messages,
                max_tokens_for_text=max_tokens_for_text_config,
                max_files_per_message=max_files_per_message,
                accept_files=True,
                use_google_lens=False,
                is_target_provider_gemini=True,  # Grounding model is assumed to be Gemini-like for now
                target_provider_name=self.config.get(
                    GROUNDING_MODEL_CONFIG_KEY, "google/gemini-2.5-flash-preview-05-20"
                ).split("/", 1)[0],
                target_model_name=self.config.get(
                    GROUNDING_MODEL_CONFIG_KEY, "google/gemini-2.5-flash-preview-05-20"
                ).split("/", 1)[1],
                user_warnings=user_warnings,
                current_message_url_fetch_results=None,
                msg_nodes_cache=self.msg_nodes,
                bot_user_obj=self.user,
                # Pass the grounding system prompt for budgeting
                httpx_async_client=self.httpx_client,
                models_module=models,
                google_types_module=google_types,
                extract_text_from_pdf_bytes_func=extract_text_from_pdf_bytes,
                at_ai_pattern_re=AT_AI_PATTERN,
                providers_supporting_usernames_const=PROVIDERS_SUPPORTING_USERNAMES,
                system_prompt_text_for_budgeting=prepare_system_prompt(
                    True,  # Grounding model is assumed to be Gemini-like
                    self.config.get(
                        GROUNDING_MODEL_CONFIG_KEY,
                        "google/gemini-2.5-flash-preview-05-20",
                    ).split("/", 1)[0],
                    self.config.get(GROUNDING_SYSTEM_PROMPT_CONFIG_KEY),
                ),
            )
            if history_for_gemini_grounding:
                grounding_sp_text_from_config = self.config.get(
                    GROUNDING_SYSTEM_PROMPT_CONFIG_KEY
                )
                # This system_prompt_for_grounding is what's actually sent to the API
                system_prompt_for_grounding = prepare_system_prompt(
                    True,  # Grounding model is assumed to be Gemini-like
                    self.config.get(
                        GROUNDING_MODEL_CONFIG_KEY,
                        "google/gemini-2.5-flash-preview-05-20",
                    ).split("/", 1)[0],
                    grounding_sp_text_from_config,
                )
                web_search_queries = await get_web_search_queries_from_gemini(
                    history_for_gemini_grounding,
                    system_prompt_for_grounding,  # This is the prompt sent to the API
                    self.config,
                    generate_response_stream,
                )
                if web_search_queries:
                    searxng_derived_context = await fetch_and_format_searxng_results(
                        web_search_queries,
                        cleaned_content,
                        self.config,
                        self.httpx_client,
                    )
                    if searxng_derived_context:
                        combined_context = searxng_derived_context
                    else:
                        logging.info(
                            "Failed to generate context from SearxNG, or no results found."
                        )
                else:
                    logging.info(
                        "Gemini grounding did not yield any web search queries."
                    )
            else:
                logging.warning(
                    "Could not build history for Gemini grounding step. Skipping SearxNG."
                )
        elif user_has_provided_urls:
            url_fetch_results = await fetch_external_content(
                cleaned_content,
                image_attachments,
                False,
                max_files_per_message,
                user_warnings,
                self.config,
                self.httpx_client,
            )
            if url_fetch_results:
                combined_context = format_external_content(url_fetch_results)

        if url_fetch_results:
            successfully_fetched_image_urls = {
                res.url
                for res in url_fetch_results
                if res.type == "image_url_content" and res.content and not res.error
            }
            if successfully_fetched_image_urls:
                temp_cleaned_content = cleaned_content
                for img_url in successfully_fetched_image_urls:
                    temp_cleaned_content = temp_cleaned_content.replace(img_url, "")
                cleaned_content = re.sub(r"\s{2,}", " ", temp_cleaned_content).strip()
                logging.info(
                    f"Removed {len(successfully_fetched_image_urls)} successfully fetched image URLs from cleaned_content."
                )

        history_for_llm = await build_message_history(
            new_msg=new_msg,
            initial_cleaned_content=cleaned_content,
            combined_context=combined_context,
            max_messages=max_messages,
            max_tokens_for_text=max_tokens_for_text_config,
            max_files_per_message=max_files_per_message,
            accept_files=accept_files,
            use_google_lens=use_google_lens,
            is_target_provider_gemini=is_gemini,
            target_provider_name=provider,
            target_model_name=model_name,  # Uses the final model_name for tokenizer
            user_warnings=user_warnings,
            current_message_url_fetch_results=url_fetch_results
            if not use_google_lens
            else [],
            msg_nodes_cache=self.msg_nodes,
            bot_user_obj=self.user,
            httpx_async_client=self.httpx_client,
            models_module=models,
            google_types_module=google_types,
            extract_text_from_pdf_bytes_func=extract_text_from_pdf_bytes,
            at_ai_pattern_re=AT_AI_PATTERN,
            providers_supporting_usernames_const=PROVIDERS_SUPPORTING_USERNAMES,
            # Pass the main system prompt for budgeting
            system_prompt_text_for_budgeting=prepare_system_prompt(
                is_gemini,
                provider,
                get_user_system_prompt_preference(
                    new_msg.author.id, self.config.get("system_prompt")
                ),
            ),
            app_config=self.config,  # ADDED
            generate_response_stream_func=generate_response_stream,  # ADDED
        )

        if not history_for_llm:
            # ... (handle empty history)
            return

        logging.info(
            f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, history length: {len(history_for_llm)}, google_lens: {use_google_lens}, warnings: {user_warnings}):\n{new_msg.content}"
        )

        default_system_prompt_from_config = self.config.get("system_prompt")
        base_system_prompt_text = get_user_system_prompt_preference(
            new_msg.author.id, default_system_prompt_from_config
        )
        # This system_prompt_text is what's actually sent to the API
        system_prompt_text = prepare_system_prompt(
            is_gemini, provider, base_system_prompt_text
        )
        extra_api_params = self.config.get("extra_api_parameters", {}).copy()

        if is_gemini:
            global_use_thinking_budget = self.config.get(
                GEMINI_USE_THINKING_BUDGET_CONFIG_KEY, False
            )
            global_thinking_budget_value = self.config.get(
                GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY, 0
            )
            user_wants_thinking_budget = get_user_gemini_thinking_budget_preference(
                new_msg.author.id, global_use_thinking_budget
            )
            if user_wants_thinking_budget:
                extra_api_params["thinking_budget"] = global_thinking_budget_value

        (
            llm_call_successful,
            final_text,
            response_msgs,
        ) = await handle_llm_response_stream(
            client=self,
            new_msg=new_msg,
            processing_msg=processing_msg,
            provider=provider,
            model_name=model_name,  # Use the correctly split model_name here
            history_for_llm=history_for_llm,
            system_prompt_text=system_prompt_text,
            provider_config=provider_config,
            extra_api_params=extra_api_params,
            app_config=self.config,  # Pass app_config
            initial_user_warnings=user_warnings,
            use_plain_responses_config=use_plain_responses,
            split_limit_config=split_limit,
        )

        try:
            pass
        finally:
            if llm_call_successful and final_text:
                await resend_imgur_urls(new_msg, response_msgs, final_text)

            logging.debug(
                f"Entering finally block. response_msgs count: {len(response_msgs)}"
            )
            for response_msg in response_msgs:
                if response_msg and response_msg.id in self.msg_nodes:
                    node = self.msg_nodes[response_msg.id]
                    if llm_call_successful:
                        node.full_response_text = final_text
                    if node.lock.locked():
                        try:
                            node.lock.release()
                        except RuntimeError:
                            logging.warning(
                                f"Attempted to release an already unlocked lock for node {response_msg.id}"
                            )
                elif response_msg:
                    logging.warning(
                        f"Response message {response_msg.id} not found in msg_nodes during cleanup."
                    )

            max_nodes_from_config = self.config.get(MAX_MESSAGE_NODES_CONFIG_KEY, 500)
            if (num_nodes := len(self.msg_nodes)) > max_nodes_from_config:
                nodes_to_delete = sorted(self.msg_nodes.keys())[
                    : num_nodes - max_nodes_from_config
                ]
                for msg_id in nodes_to_delete:
                    node_to_delete = self.msg_nodes.get(msg_id)
                    if (
                        node_to_delete
                        and hasattr(node_to_delete, "lock")
                        and node_to_delete.lock.locked()
                    ):
                        try:
                            node_to_delete.lock.release()
                        except RuntimeError:
                            pass
                    self.msg_nodes.pop(msg_id, None)

            end_time = time.time()
            logging.info(
                f"Finished processing message {new_msg.id}. Success: {llm_call_successful}. Total time: {end_time - start_time:.2f} seconds."
            )

    # All internal methods that were moved out are now removed from the class definition.
    # _is_allowed, _fetch_external_content, _format_external_content,
    # _build_message_history, _find_parent_message, _prepare_system_prompt,
    # _get_web_search_queries_from_gemini, _fetch_and_format_searxng_results
    # are no longer defined here.

    async def close(self):
        """Clean up resources when the bot is shutting down."""
        logging.info("Closing HTTPX client...")
        await self.httpx_client.aclose()

        # Close Reddit client
        # Import here to avoid circular dependency at module level if reddit.py also imports from bot.py (though it doesn't seem to)
        from .content_fetchers.reddit import reddit_client_instance

        if reddit_client_instance:
            logging.info("Closing Reddit client...")
            try:
                await reddit_client_instance.close()
            except Exception as e:
                logging.error(f"Error closing Reddit client: {e}")

        logging.info("Closing database connections...")
        close_all_db_managers()
        await super().close()
