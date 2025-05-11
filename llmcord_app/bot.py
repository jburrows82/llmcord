# llmcord_app/bot.py
import asyncio
import base64
from datetime import datetime as dt, timezone
import logging
import os
import re # <-- Added import
import time
from typing import Dict, Optional, Set, List, Any
import traceback # Keep for detailed error logging if needed

import discord
from discord import app_commands
# Use google.genai.types
from google.genai import types as google_types

import httpx

# Import from our modules
# Remove the incorrect import of cfg
from .config import get_config
from .constants import (
    EMBED_COLOR_COMPLETE, EMBED_COLOR_INCOMPLETE, EMBED_COLOR_ERROR,
    STREAMING_INDICATOR, EDIT_DELAY_SECONDS, MAX_MESSAGE_NODES,
    MAX_EMBED_DESCRIPTION_LENGTH, AT_AI_PATTERN, GOOGLE_LENS_PATTERN,
    GOOGLE_LENS_KEYWORD, VISION_MODEL_TAGS, AVAILABLE_MODELS,
    DEEP_SEARCH_KEYWORDS, DEEP_SEARCH_MODEL, PROVIDERS_SUPPORTING_USERNAMES,
    AllKeysFailedError, IMGUR_HEADER, IMGUR_URL_PREFIX, IMGUR_URL_PATTERN, # <-- Added Imgur constants
    MAX_PLAIN_TEXT_LENGTH, # <-- Added plain text limit
    SEARXNG_BASE_URL_CONFIG_KEY, SEARXNG_NUM_RESULTS, # Added SearxNG constants
    SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY, # <-- ADDED
    GROUNDING_MODEL_PROVIDER, GROUNDING_MODEL_NAME, # Added Grounding model constants
    GROUNDING_SYSTEM_PROMPT_CONFIG_KEY, # <-- ADDED
    GEMINI_USE_THINKING_BUDGET_CONFIG_KEY, GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY, # Added Gemini Thinking Budget keys
    FALLBACK_VISION_MODEL_PROVIDER_SLASH_MODEL # <-- ADDED for configurable fallback vision model
)
# Corrected import: Use the models module directly
from . import models # Import the models module
from .rate_limiter import check_and_perform_global_reset, close_all_db_managers
from .ui import ResponseActionView
from .utils import (
    extract_urls_with_indices, is_youtube_url, is_reddit_url, is_image_url, # Added is_image_url
    extract_video_id, extract_reddit_submission_id, extract_text_from_pdf_bytes
)
from .content_fetchers import (
    fetch_youtube_data, fetch_reddit_data, fetch_general_url_content,
    process_google_lens_image, fetch_searxng_results # Added fetch_searxng_results
)
from .llm_handler import generate_response_stream
from .commands import ( # Import command logic and preference getter
    set_model_command, get_user_model_preference,
    set_system_prompt_command, get_user_system_prompt_preference,
    setgeminithinking, get_user_gemini_thinking_budget_preference # Added Gemini thinking budget command and getter
)


# --- Discord Client Setup ---
class LLMCordClient(discord.Client):
    def __init__(self, *, intents: discord.Intents, activity: discord.CustomActivity, config: Dict):
        super().__init__(intents=intents, activity=activity)
        self.tree = app_commands.CommandTree(self)
        # Use the correct MsgNode class from the models module
        self.msg_nodes: Dict[int, models.MsgNode] = {} # Message cache
        self.last_task_time: float = 0 # For stream editing delay
        self.config = config # Store loaded config
        self.httpx_client = httpx.AsyncClient(timeout=20.0, follow_redirects=True) # HTTP client for attachments/web

        # Initialize content fetcher modules that need config
        from .content_fetchers.youtube import initialize_ytt_api
        initialize_ytt_api(self.config.get("proxy_config"))

    async def setup_hook(self):
        """Sync slash commands when the bot is ready."""
        # Register the command function with the tree
        self.tree.add_command(app_commands.Command(
            name="model",
            description="Set your preferred LLM provider and model.",
            callback=set_model_command
        ))
        # --- ADDED: Register /systemprompt command ---
        self.tree.add_command(app_commands.Command(
            name="systemprompt",
            description="Set your custom system prompt for the bot.",
            callback=set_system_prompt_command
        ))
        # --- ADDED: Register /setgeminithinking command ---
        self.tree.add_command(app_commands.Command(
            name="setgeminithinking",
            description="Toggle usage of the 'thinkingBudget' parameter for Gemini models.",
            callback=setgeminithinking
        ))
        # Sync commands
        await self.tree.sync()
        logging.info(f'Synced slash commands for {self.user}.')

    async def on_ready(self):
        """Called when the bot is ready and logged in."""
        logging.info(f'Logged in as {self.user}')
        # Initial check/reset of rate limits
        check_and_perform_global_reset(self.config)

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
                is_reply_to_bot = new_msg.reference and new_msg.reference.resolved and new_msg.reference.resolved.author == self.user
                is_reply_to_user = new_msg.reference and new_msg.reference.resolved and new_msg.reference.resolved.author != self.user

                if is_reply_to_user and not (mentions_bot or contains_at_ai):
                    should_process = False # Don't process replies to other users unless explicitly triggered
                else:
                    should_process = True # Process direct messages, replies to bot, or explicitly triggered replies
            else:
                return # Block DMs if not allowed
        elif mentions_bot or contains_at_ai:
            should_process = True

        if not should_process:
            return

        # --- Reload config & Check Global Reset ---
        # Config is now an instance variable, consider if reloading is needed per message
        # self.config = get_config() # Uncomment if hot-reloading is desired
        check_and_perform_global_reset(self.config)
        # youtube_api_key = self.config.get("youtube_api_key") # Moved down, not needed this early
        # reddit_client_id = self.config.get("reddit_client_id")
        # reddit_client_secret = self.config.get("reddit_client_secret")
        # reddit_user_agent = self.config.get("reddit_user_agent")
        # custom_google_lens_config = self.config.get("custom_google_lens_config")

        # --- Permissions Check ---
        if not self._is_allowed(new_msg, is_dm):
            logging.warning(f"Blocked message from user {new_msg.author.id} in channel {new_msg.channel.id} due to permissions.")
            return

        # --- Send Initial "Processing" Message ---
        processing_msg: Optional[discord.Message] = None
        _use_plain_for_initial_status = self.config.get("use_plain_responses", False)
        try:
            if _use_plain_for_initial_status:
                processing_msg = await new_msg.reply("⏳ Processing request...", mention_author=False, suppress_embeds=True)
            else:
                processing_embed = discord.Embed(description="⏳ Processing request...", color=EMBED_COLOR_INCOMPLETE)
                processing_msg = await new_msg.reply(embed=processing_embed, mention_author=False)
        except discord.HTTPException as e:
            logging.warning(f"Failed to send initial 'Processing request...' message: {e}")
            processing_msg = None # Ensure it's None if sending failed
        except Exception as e: # Catch any other unexpected errors during initial reply
            logging.error(f"Unexpected error sending initial 'Processing request...' message: {e}", exc_info=True)
            processing_msg = None


        # --- Config values needed for further processing ---
        youtube_api_key = self.config.get("youtube_api_key")
        reddit_client_id = self.config.get("reddit_client_id")
        reddit_client_secret = self.config.get("reddit_client_secret")
        reddit_user_agent = self.config.get("reddit_user_agent")
        custom_google_lens_config = self.config.get("custom_google_lens_config")


        # --- Clean Content and Check for Google Lens ---
        cleaned_content = original_content_for_processing
        if not is_dm and mentions_bot:
            cleaned_content = cleaned_content.replace(self.user.mention, '').strip()
        cleaned_content = AT_AI_PATTERN.sub(' ', cleaned_content).strip()
        cleaned_content = re.sub(r'\s{2,}', ' ', cleaned_content).strip()
        logging.debug(f"Cleaned content for keyword/URL check: '{cleaned_content}'")

        use_google_lens = False
        image_attachments = [att for att in new_msg.attachments if att.content_type and att.content_type.startswith("image/")]
        user_warnings = set()

        if GOOGLE_LENS_PATTERN.match(cleaned_content) and image_attachments:
            custom_lens_ok = custom_google_lens_config and custom_google_lens_config.get("user_data_dir") and custom_google_lens_config.get("profile_directory_name")
            serpapi_keys_ok = bool(self.config.get("serpapi_api_keys"))
            if not serpapi_keys_ok and not custom_lens_ok:
                 logging.warning("Google Lens requested but neither SerpAPI keys nor custom implementation are configured.")
                 user_warnings.add("⚠️ Google Lens requested but requires configuration (SerpAPI or custom).")
            else:
                use_google_lens = True
                cleaned_content = GOOGLE_LENS_PATTERN.sub('', cleaned_content).strip()
                logging.info(f"Google Lens keyword detected for message {new_msg.id}.")

        # --- LLM Provider/Model Selection ---
        user_id = new_msg.author.id # user_id defined here
        default_model_str = self.config.get("model", "google/gemini-2.0-flash")
        provider_slash_model = get_user_model_preference(user_id, default_model_str) # Use function from commands.py

        # --- Override Model based on Keywords ---
        final_provider_slash_model = provider_slash_model
        deep_search_triggered = False
        if any(keyword in cleaned_content.lower() for keyword in DEEP_SEARCH_KEYWORDS):
            target_model_str = DEEP_SEARCH_MODEL
            target_provider, target_model_name = target_model_str.split("/", 1)
            if use_google_lens:
                logging.warning(f"Both 'deepsearch'/'deepersearch' and 'googlelens' detected. Prioritizing deep search ('{target_model_str}'). Google Lens disabled.")
                use_google_lens = False

            xai_provider_config = self.config.get("providers", {}).get(target_provider, {})
            xai_keys = xai_provider_config.get("api_keys", [])
            is_target_available = target_provider in AVAILABLE_MODELS and target_model_name in AVAILABLE_MODELS.get(target_provider, [])

            if xai_provider_config and xai_keys and is_target_available:
                final_provider_slash_model = target_model_str
                deep_search_triggered = True
                logging.info(f"Keywords {DEEP_SEARCH_KEYWORDS} detected. Overriding model to {final_provider_slash_model}.")
            else:
                warning_reason = ""
                if not xai_provider_config: warning_reason = f"provider '{target_provider}' not configured"
                elif not xai_keys: warning_reason = f"no API keys for provider '{target_provider}'"
                elif not is_target_available: warning_reason = f"model '{target_model_str}' not listed in AVAILABLE_MODELS"
                logging.warning(f"Keywords {DEEP_SEARCH_KEYWORDS} detected, but cannot use '{target_model_str}' ({warning_reason}). Using original: {provider_slash_model}")
                user_warnings.add(f"⚠️ Deep search requested, but model '{target_model_str}' unavailable ({warning_reason}).")

        # --- Validate Final Model Selection ---
        try:
            provider, model_name = final_provider_slash_model.split("/", 1)
            if provider not in AVAILABLE_MODELS or model_name not in AVAILABLE_MODELS.get(provider, []):
                 logging.warning(f"Final model '{final_provider_slash_model}' is invalid/unavailable. Falling back to default: {default_model_str}")
                 final_provider_slash_model = default_model_str
                 provider, model_name = final_provider_slash_model.split("/", 1)
        except ValueError:
            logging.error(f"Invalid model format for final selection '{final_provider_slash_model}'. Using hardcoded default.")
            final_provider_slash_model = "google/gemini-2.0-flash"
            provider, model_name = final_provider_slash_model.split("/", 1)
            # Consider clearing invalid user preference here if needed

        logging.info(f"Final model selected for user {user_id}: '{final_provider_slash_model}'")

        # --- Get Config for the FINAL Provider ---
        provider_config = self.config.get("providers", {}).get(provider, {})
        if not isinstance(provider_config, dict): # Ensure provider_config is a dict
            logging.error(f"Configuration for provider '{provider}' is invalid or missing. Cannot proceed.")
            error_text = f"⚠️ Configuration error for provider `{provider}`."
            if processing_msg:
                if _use_plain_for_initial_status:
                    await processing_msg.edit(content=error_text, embed=None, view=None)
                else:
                    await processing_msg.edit(content=None, embed=discord.Embed(description=error_text, color=EMBED_COLOR_ERROR), view=None)
            else:
                await new_msg.reply(error_text, mention_author=False)
            return
        all_api_keys = provider_config.get("api_keys", []) # Expecting a list

        is_gemini = provider == "google"
        # Determine if the target model is Grok
        is_grok_model = provider == "x-ai" # Assuming DEEP_SEARCH_MODEL is the only Grok model or x-ai is only Grok
        
        keys_required = provider not in ["ollama", "lmstudio", "vllm", "oobabooga", "jan"]

        if keys_required and not all_api_keys:
             logging.error(f"No API keys configured for the selected provider '{provider}' in config.yaml.")
             error_text = f"⚠️ No API keys configured for provider `{provider}`."
             if processing_msg:
                if _use_plain_for_initial_status:
                    await processing_msg.edit(content=error_text, embed=None, view=None)
                else:
                    await processing_msg.edit(content=None, embed=discord.Embed(description=error_text, color=EMBED_COLOR_ERROR), view=None)
             else:
                await new_msg.reply(error_text, mention_author = False)
             return

        # --- Configuration Values ---
        accept_files = any(x in model_name.lower() for x in VISION_MODEL_TAGS) # Renamed from accept_images for clarity

        # --- ADDED: Check OpenAI specific vision disabling ---
        if provider == "openai" and provider_config.get("disable_vision", False):
            accept_files = False
            logging.info(f"Vision explicitly disabled for OpenAI model '{model_name}' via config.")
        # --- END ADDED ---

        # --- ADDED: Switch to Gemini if images present (attachments OR URLs) and current model lacks vision ---
        # Preliminary check for image URLs in the text
        has_potential_image_urls_in_text = False
        if cleaned_content: # Only check if there's content
            urls_in_text = extract_urls_with_indices(cleaned_content)
            if any(is_image_url(url_info[0]) for url_info in urls_in_text):
                has_potential_image_urls_in_text = True
                logging.info("Potential image URLs detected in message content.")

        if (image_attachments or has_potential_image_urls_in_text) and not accept_files:
            original_model_for_warning = final_provider_slash_model
            fallback_model_str = FALLBACK_VISION_MODEL_PROVIDER_SLASH_MODEL
            logging.info(f"Query has images, but current model '{final_provider_slash_model}' does not support vision or has vision disabled. Switching to '{fallback_model_str}' for this query.")
            user_warnings.add(f"⚠️ Images detected. Switched from '{original_model_for_warning}' to '{fallback_model_str}' as the original model cannot process images.")

            final_provider_slash_model = fallback_model_str
            try:
                provider, model_name = final_provider_slash_model.split("/", 1)
            except ValueError:
                logging.error(f"Invalid format for FALLBACK_VISION_MODEL_PROVIDER_SLASH_MODEL: '{fallback_model_str}'. Cannot switch. Original model '{original_model_for_warning}' will be attempted.")
                # Keep original provider, model_name, provider_config, all_api_keys, is_gemini, is_grok_model, keys_required, accept_files
                # This means the original non-vision model will attempt to process, likely without images.
                # Add a specific warning for this failure.
                user_warnings.add(f"⚠️ Error: Could not switch to vision model '{fallback_model_str}' due to invalid format. Processing with '{original_model_for_warning}'.")
                # Skip the rest of the override logic
            else:
                # Re-fetch config for the NEW provider
                new_provider_config = self.config.get("providers", {}).get(provider, {})
                if not isinstance(new_provider_config, dict):
                    logging.error(f"Configuration for fallback provider '{provider}' is invalid or missing. Cannot proceed with switch.")
                    error_text = f"⚠️ Error: Could not switch to vision model due to configuration issue for '{provider}'. Original model '{original_model_for_warning}' will be attempted if possible, or processing may fail."
                    if processing_msg:
                        if _use_plain_for_initial_status:
                            await processing_msg.edit(content=error_text, embed=None, view=None)
                        else:
                            await processing_msg.edit(content=None, embed=discord.Embed(description=error_text, color=EMBED_COLOR_ERROR), view=None)
                    else:
                        await new_msg.reply(error_text, mention_author=False)
                    # Attempt to revert to original provider_config if new one is bad, or let it fail if original was also problematic
                    # For safety, we might just return or use the original provider_config if the new one is bad.
                    # However, the original model can't handle images, so this path is problematic.
                    # Best to inform user and potentially stop. For now, let's allow it to try to continue with original if new config is bad.
                    # This means provider_config remains the one for the *original* model.
                else:
                    provider_config = new_provider_config # Successfully updated provider_config

                    all_api_keys = provider_config.get("api_keys", [])

                    is_gemini = (provider == "google") # Update based on the new provider
                    is_grok_model = (provider == "x-ai") # Update based on the new provider

                    keys_required = provider not in ["ollama", "lmstudio", "vllm", "oobabooga", "jan"]
                    if keys_required and not all_api_keys:
                        logging.error(f"No API keys configured for the fallback vision provider '{provider}'. Cannot use vision model '{final_provider_slash_model}'.")
                        user_warnings.add(f"⚠️ Switched to '{final_provider_slash_model}' for images, but no API keys found for it. Image processing will likely fail.")

                    # Re-evaluate accept_files for the new model
                    accept_files = any(x in model_name.lower() for x in VISION_MODEL_TAGS)
                    # Check OpenAI specific disable_vision for the new provider_config
                    if provider == "openai" and provider_config.get("disable_vision", False):
                        accept_files = False
                        logging.info(f"Vision explicitly disabled for fallback OpenAI model '{model_name}' via config.")

                    # Final check: if after switching, the new model STILL doesn't accept files (e.g. misconfigured constant or keys missing for a vision model)
                    if not accept_files:
                        logging.warning(f"Fallback model '{final_provider_slash_model}' was selected but it still does not accept files (accept_files: {accept_files}). This might be due to missing keys or configuration. Images may not be processed.")
                        user_warnings.add(f"⚠️ Fallback model '{final_provider_slash_model}' cannot process images. Check configuration.")


                    logging.info(f"Switched to '{final_provider_slash_model}'. New provider: '{provider}', model: '{model_name}'. New accept_files: {accept_files}. Keys required: {keys_required}. API Keys found: {bool(all_api_keys)}")
        # --- END ADDED ---

        max_files_per_message = self.config.get("max_images", 5) # Using max_images config key for max files
        if use_google_lens:
            # max_images_for_lens = self.config.get("max_images", 5) # Retain original logic for lens if needed separate
            pass # For now, lens also uses max_files_per_message
        elif accept_files:
            # max_images_for_llm = self.config.get("max_images", 5)
            pass
        else:
            # max_images_for_llm = 0
            pass # max_files_per_message will be effectively 0 if accept_files is false later

        max_text = self.config.get("max_text", 100000)
        max_messages = self.config.get("max_messages", 25)
        use_plain_responses = self.config.get("use_plain_responses", False)
        split_limit = MAX_EMBED_DESCRIPTION_LENGTH if not use_plain_responses else MAX_PLAIN_TEXT_LENGTH

        # --- Check for Empty Query AGAIN ---
        is_text_empty = not cleaned_content.strip()
        has_meaningful_attachments_final = any(
            att.content_type and (
                (att.content_type.startswith("image/") and (accept_files or use_google_lens))
                or att.content_type.startswith("text/")
                # PDF is meaningful if Gemini can accept files OR if it's non-Gemini (text will be extracted)
                or (att.content_type == "application/pdf" and ((is_gemini and accept_files) or not is_gemini))
            )
            for att in new_msg.attachments
        )
        is_reply = bool(new_msg.reference)

        if is_text_empty and not has_meaningful_attachments_final and not is_reply:
            logging.info(f"Empty query detected from user {new_msg.author.id}. Not a reply and no meaningful attachments.")
            empty_query_text = "Your query is empty. Please provide text, attach relevant files/images, or reply to a message."
            if processing_msg:
                if _use_plain_for_initial_status:
                    await processing_msg.edit(content=empty_query_text, embed=None, view=None)
                else:
                    await processing_msg.edit(content=None, embed=discord.Embed(description=empty_query_text, color=EMBED_COLOR_ERROR), view=None)
            else:
                await new_msg.reply(empty_query_text, mention_author=False)
            return

        # --- URL Extraction and Content Fetching ---
        combined_context = ""
        url_fetch_results = [] # This will store all fetched results

        # --- Determine if any user-provided URLs exist ---
        # This check is now done upfront to influence whether SearxNG runs.
        user_has_provided_urls = bool(extract_urls_with_indices(cleaned_content))

        # --- Step 1: Handle Google Lens if explicitly triggered ---
        if use_google_lens:
            logging.info("Google Lens is active. Fetching Lens and user URL content.")
            # _fetch_external_content will handle both Lens images and any URLs in cleaned_content
            url_fetch_results = await self._fetch_external_content(
                cleaned_content, image_attachments, True, # use_google_lens is True
                max_files_per_message, user_warnings
            )
            if url_fetch_results:
                combined_context = self._format_external_content(url_fetch_results)
            
            # If Google Lens is active, we skip the Gemini grounding/SearxNG step.
            logging.info("Skipping Gemini grounding/SearxNG step because Google Lens is active.")

        # --- Step 2: If not Google Lens, AND NO user-provided URLs, attempt Gemini Grounding for SearxNG (for non-Gemini/non-Grok targets) ---
        elif not user_has_provided_urls and not is_gemini and not is_grok_model:
            # This block executes only if:
            # 1. use_google_lens is False
            # 2. there are NO URLs in cleaned_content
            # 3. the target model is non-Gemini/non-Grok.
            logging.info(f"Target model '{final_provider_slash_model}' is non-Gemini/non-Grok, Google Lens is not active, and no user URLs detected. Attempting grounding pre-step for SearxNG.")
            
            history_for_gemini_grounding = await self._build_message_history(
                new_msg, cleaned_content, "", # Empty combined_context for grounding history
                max_messages, max_text, max_files_per_message,
                True, # Assume Gemini can accept files for grounding call
                False, # Not using Google Lens for this internal Gemini call
                True, # It IS a Gemini call (for grounding_model_name)
                GROUNDING_MODEL_PROVIDER, # Provider for grounding
                GROUNDING_MODEL_NAME, # Model for grounding
                user_warnings # Pass user_warnings to collect any from history build
            )

            if history_for_gemini_grounding:
                grounding_sp_text_from_config = self.config.get(GROUNDING_SYSTEM_PROMPT_CONFIG_KEY)
                system_prompt_for_grounding = self._prepare_system_prompt(
                    True, GROUNDING_MODEL_PROVIDER, grounding_sp_text_from_config
                )
                
                web_search_queries = await self._get_web_search_queries_from_gemini(
                    history_for_gemini_grounding,
                    system_prompt_for_grounding
                )

                if web_search_queries:
                    searxng_derived_context = await self._fetch_and_format_searxng_results(
                        web_search_queries,
                        cleaned_content # For logging purposes
                    )
                    if searxng_derived_context:
                        logging.info("Successfully generated context from SearxNG results.")
                        combined_context = searxng_derived_context
                        # No user URLs were present if this block is reached, so no warning about skipping them.
                    else:
                        logging.info("Failed to generate context from SearxNG, or no results found.")
                else:
                    logging.info("Gemini grounding did not yield any web search queries.")
            else:
                logging.warning("Could not build history for Gemini grounding step. Skipping SearxNG.")
            
            # If SearxNG failed or didn't run, and there were truly no user URLs (as per the 'elif' condition),
            # combined_context will remain empty. No further URL processing is needed here.

        # --- Step 3: If Google Lens was not used, BUT user-provided URLs ARE present, process those URLs directly ---
        elif user_has_provided_urls: # This implies use_google_lens is False.
                                     # This block will catch cases where SearxNG was skipped because URLs were present.
            logging.info("Google Lens is not active, but user-provided URLs detected. Processing these URLs directly. SearxNG is skipped.")
            # _fetch_external_content will only process URLs in cleaned_content because use_google_lens is False here.
            url_fetch_results = await self._fetch_external_content(
                cleaned_content, image_attachments, False, # use_google_lens is False
                max_files_per_message, user_warnings
            )
            if url_fetch_results:
                combined_context = self._format_external_content(url_fetch_results)

        # Note: If use_google_lens was false, user_has_provided_urls was false, and SearxNG didn't produce context,
        # then combined_context will correctly be empty. This is the case for a direct query to any model without
        # explicit external context provided by the user or derived by the system.

        # --- ADDED: Remove successfully fetched image URLs from cleaned_content ---
        # This ensures the URL string itself isn't sent if the image bytes are.
        if url_fetch_results:
            successfully_fetched_image_urls = set()
            for res in url_fetch_results:
                if res.type == "image_url_content" and res.content and not res.error:
                    successfully_fetched_image_urls.add(res.url)
            
            if successfully_fetched_image_urls:
                temp_cleaned_content = cleaned_content
                for img_url in successfully_fetched_image_urls:
                    # Escape the URL for regex replacement to handle special characters
                    escaped_url = re.escape(img_url)
                    # Replace the URL, ensuring it's a whole word or surrounded by spaces/punctuation
                    # to avoid partial replacements in longer strings.
                    # Using \b (word boundary) might be too restrictive if URL is adjacent to punctuation.
                    # A simpler approach is to replace the exact string.
                    temp_cleaned_content = temp_cleaned_content.replace(img_url, "")
                
                # Clean up extra spaces that might result from removal
                cleaned_content = re.sub(r'\s{2,}', ' ', temp_cleaned_content).strip()
                logging.info(f"Removed {len(successfully_fetched_image_urls)} successfully fetched image URLs from cleaned_content.")
        # --- END ADDED ---

        # --- Build Message History (for the *target* LLM) ---
        # `combined_context` now holds the appropriate context based on the logic above.
        # Pass url_fetch_results to _build_message_history for the current message
        history_for_llm = await self._build_message_history(
            new_msg, cleaned_content, combined_context, max_messages, max_text, max_files_per_message,
            accept_files, use_google_lens, is_gemini, provider, model_name, user_warnings,
            url_fetch_results if not use_google_lens else [] # Pass fetched image URLs if not using Google Lens (Lens handles its own images)
        )

        if not history_for_llm:
            logging.warning(f"Message history is empty for message {new_msg.id}. Cannot proceed.")
            # Send warnings if any were generated during history building
            if user_warnings:
                warning_msg_text = f"Could not process request. Issues found:\n" + "\n".join(sorted(list(user_warnings)))
                if processing_msg:
                    if _use_plain_for_initial_status:
                        await processing_msg.edit(content=warning_msg_text, embed=None, view=None)
                    else:
                        await processing_msg.edit(content=None, embed=discord.Embed(description=warning_msg_text, color=EMBED_COLOR_ERROR), view=None)
                else:
                    await new_msg.reply(warning_msg_text, mention_author=False)
            else:
                error_text = "Could not process request (failed to build message history)."
                if processing_msg:
                    if _use_plain_for_initial_status:
                        await processing_msg.edit(content=error_text, embed=None, view=None)
                    else:
                        await processing_msg.edit(content=None, embed=discord.Embed(description=error_text, color=EMBED_COLOR_ERROR), view=None)
                else:
                    await new_msg.reply(error_text, mention_author=False)
            return

        logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, history length: {len(history_for_llm)}, google_lens: {use_google_lens}, warnings: {user_warnings}):\n{new_msg.content}")

        # --- Prepare API Call ---
        # --- MODIFIED: Get user-specific or default system prompt ---
        default_system_prompt_from_config = self.config.get("system_prompt")
        base_system_prompt_text = get_user_system_prompt_preference(new_msg.author.id, default_system_prompt_from_config)
        system_prompt_text = self._prepare_system_prompt(is_gemini, provider, base_system_prompt_text)
        extra_api_params = self.config.get("extra_api_parameters", {}).copy()

        # --- ADDED: Gemini Thinking Budget Logic ---
        if is_gemini:
            global_use_thinking_budget = self.config.get(GEMINI_USE_THINKING_BUDGET_CONFIG_KEY, False)
            global_thinking_budget_value = self.config.get(GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY, 0)

            user_wants_thinking_budget = get_user_gemini_thinking_budget_preference(new_msg.author.id, global_use_thinking_budget)

            # If the user/config indicates the thinking budget feature should be used,
            # pass the configured value (which can be 0 to disable thinking, or >0 to set a budget).
            # The llm_handler will apply this value to the ThinkingConfig if it's valid (0-24576).
            if user_wants_thinking_budget:
                extra_api_params["thinking_budget"] = global_thinking_budget_value
                logging.info(f"Passing Gemini thinking_budget value: {global_thinking_budget_value} for user {new_msg.author.id} to LLM handler.")
        # --- END ADDED ---

        # --- Generate and Send Response ---
        response_msgs: List[discord.Message] = [] # Ensure type hint
        final_text = ""
        llm_call_successful = False
        final_view = None
        grounding_metadata = None
        edit_task = None
        self.last_task_time = 0 # Reset last task time for this message

        embed = discord.Embed()
        embed.set_footer(text=f"Model: {final_provider_slash_model}")
        for warning in sorted(user_warnings):
            embed.add_field(name=warning, value="", inline=False)

        try:
            async with new_msg.channel.typing():
                stream_generator = generate_response_stream(
                    provider=provider,
                    model_name=model_name,
                    history_for_llm=history_for_llm,
                    system_prompt_text=system_prompt_text,
                    provider_config=provider_config,
                    extra_params=extra_api_params
                )

                async for text_chunk, finish_reason, chunk_grounding_metadata, error_message in stream_generator:
                    if error_message: # Handle non-retryable errors yielded by the generator
                        logging.error(f"LLM stream failed with non-retryable error: {error_message}")
                        # Update embed/message with the error
                        if not use_plain_responses and response_msgs:
                            try:
                                # Make sure there's an embed to modify
                                if response_msgs[-1].embeds:
                                    current_desc = response_msgs[-1].embeds[0].description or ""
                                else:
                                    current_desc = "" # Start fresh if no embed exists
                                    embed = discord.Embed(color=EMBED_COLOR_ERROR) # Create embed if needed

                                embed.description = current_desc.replace(STREAMING_INDICATOR, "").strip()
                                embed.description += f"\n\n⚠️ Error: {error_message}"
                                embed.description = embed.description[:MAX_EMBED_DESCRIPTION_LENGTH]
                                embed.color = EMBED_COLOR_ERROR
                                await response_msgs[-1].edit(embed=embed, view=None) # Edits the current message in stream
                            except Exception as edit_err:
                                logging.error(f"Failed to edit message to show stream error: {edit_err}")
                                # Fallback to sending new reply if edit fails
                                if processing_msg and processing_msg.id == response_msgs[-1].id: # If the message being edited was the initial processing_msg
                                    await processing_msg.reply(f"⚠️ Error during response generation: {error_message}", mention_author=False)
                                else: # Some other message in the stream
                                    await new_msg.reply(f"⚠️ Error during response generation: {error_message}", mention_author=False)
                        else: # Plain response or no response_msgs yet
                            error_text_plain = f"⚠️ Error during response generation: {error_message}"
                            if processing_msg:
                                if _use_plain_for_initial_status:
                                    await processing_msg.edit(content=error_text_plain, embed=None, view=None)
                                else: # This case implies use_plain_responses is false, but response_msgs is empty.
                                      # This means the very first attempt to send/edit failed before stream.
                                    await processing_msg.edit(content=None, embed=discord.Embed(description=error_text_plain, color=EMBED_COLOR_ERROR), view=None)
                            else:
                                await new_msg.reply(error_text_plain, mention_author=False)
                        llm_call_successful = False
                        break # Stop processing stream

                    if chunk_grounding_metadata:
                        grounding_metadata = chunk_grounding_metadata

                    if text_chunk:
                        final_text += text_chunk

                    # --- Real-time Editing Logic ---
                    if not use_plain_responses:
                        is_final_chunk = finish_reason is not None
                        if not final_text and not is_final_chunk: continue # Skip empty intermediate chunks

                        current_msg_index = (len(final_text) - 1) // split_limit if final_text else 0
                        start_next_msg = current_msg_index >= len(response_msgs)

                        # Check if enough time has passed OR if it's the final chunk OR starting a new message
                        ready_to_edit = (
                            (edit_task is None or edit_task.done()) and
                            (dt.now().timestamp() - self.last_task_time >= EDIT_DELAY_SECONDS)
                        )

                        if start_next_msg or ready_to_edit or is_final_chunk:
                            if edit_task is not None and not edit_task.done():
                                try:
                                    await edit_task # Wait for previous edit
                                except asyncio.CancelledError:
                                    logging.warning("Previous edit task cancelled.")
                                except Exception as e:
                                    logging.error(f"Error waiting for previous edit task: {e}")
                                edit_task = None # Reset task after waiting or if done

                            # Finalize previous message if splitting
                            if start_next_msg and response_msgs:
                                prev_msg_index = current_msg_index - 1
                                if prev_msg_index >= 0 and prev_msg_index < len(response_msgs):
                                    prev_msg_text = final_text[prev_msg_index * split_limit : current_msg_index * split_limit]
                                    prev_msg_text = prev_msg_text[:MAX_EMBED_DESCRIPTION_LENGTH + len(STREAMING_INDICATOR)]
                                    # Use a temporary embed object for modification
                                    temp_embed = discord.Embed.from_dict(response_msgs[prev_msg_index].embeds[0].to_dict()) if response_msgs[prev_msg_index].embeds else discord.Embed()
                                    temp_embed.description = prev_msg_text or "..."
                                    temp_embed.color = EMBED_COLOR_COMPLETE
                                    try:
                                        await response_msgs[prev_msg_index].edit(embed=temp_embed, view=None)
                                    except discord.HTTPException as e:
                                        logging.error(f"Failed to finalize previous message {prev_msg_index}: {e}")
                                else:
                                     logging.warning(f"Invalid prev_msg_index {prev_msg_index} while splitting.")


                            # Prepare current message segment
                            current_display_text = final_text[current_msg_index * split_limit : (current_msg_index + 1) * split_limit]
                            current_display_text = current_display_text[:MAX_EMBED_DESCRIPTION_LENGTH]

                            # Determine color and view
                            view_to_attach = None
                            # Check for successful finish (Gemini UNSPECIFIED is also success)
                            is_successful_finish = finish_reason and (finish_reason.lower() in ("stop", "end_turn") or (is_gemini and finish_reason == str(google_types.FinishReason.FINISH_REASON_UNSPECIFIED)))
                            is_blocked = finish_reason and finish_reason.lower() in ("safety", "recitation", "other")

                            # Use a temporary embed for the current segment
                            current_segment_embed = discord.Embed()
                            # Copy footer and fields from the initial embed
                            current_segment_embed.set_footer(text=embed.footer.text if embed.footer else "")
                            for field in embed.fields:
                                current_segment_embed.add_field(name=field.name, value=field.value, inline=field.inline)

                            current_segment_embed.description = (current_display_text or "...") if is_final_chunk else ((current_display_text or "...") + STREAMING_INDICATOR)

                            if is_final_chunk and is_blocked:
                                current_segment_embed.description = (current_display_text or "...").replace(STREAMING_INDICATOR, "").strip()
                                if finish_reason.lower() == "safety":
                                    current_segment_embed.description += "\n\n⚠️ Response blocked by safety filters."
                                elif finish_reason.lower() == "recitation":
                                    current_segment_embed.description += "\n\n⚠️ Response stopped due to recitation."
                                else: # Other
                                    current_segment_embed.description += "\n\n⚠️ Response blocked (Reason: Other)."
                                current_segment_embed.description = current_segment_embed.description[:MAX_EMBED_DESCRIPTION_LENGTH]
                                current_segment_embed.color = EMBED_COLOR_ERROR
                                view_to_attach = None
                            else:
                                current_segment_embed.color = EMBED_COLOR_COMPLETE if is_final_chunk and is_successful_finish else EMBED_COLOR_INCOMPLETE
                                if is_final_chunk and is_successful_finish: # Only add view on successful final chunk
                                    has_sources = grounding_metadata and (getattr(grounding_metadata, 'web_search_queries', None) or getattr(grounding_metadata, 'grounding_chunks', None) or getattr(grounding_metadata, 'search_entry_point', None))
                                    has_text_content = bool(final_text)
                                    if has_sources or has_text_content:
                                        view_to_attach = ResponseActionView(
                                            grounding_metadata=grounding_metadata,
                                            full_response_text=final_text,
                                            model_name=final_provider_slash_model
                                        )
                                        if not view_to_attach or len(view_to_attach.children) == 0:
                                            view_to_attach = None

                            # Create or Edit the current message
                            if start_next_msg: # Creating a new message segment
                                if not response_msgs and processing_msg: # This is the VERY FIRST segment of the actual response
                                    await processing_msg.edit(content=None, embed=current_segment_embed, view=view_to_attach) # Clear content, set embed
                                    response_msg = processing_msg # The processing_msg is now the first response_msg
                                    response_msgs.append(response_msg)
                                else: # Subsequent segments, or if processing_msg didn't exist/failed
                                    reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                                    response_msg = await reply_to_msg.reply(embed=current_segment_embed, view=view_to_attach, mention_author=False)
                                    response_msgs.append(response_msg)
                                # Use the correct MsgNode class
                                self.msg_nodes[response_msg.id] = models.MsgNode(parent_msg=new_msg)
                                if view_to_attach: view_to_attach.message = response_msg # Set message reference in view
                                await self.msg_nodes[response_msg.id].lock.acquire()
                            elif response_msgs and current_msg_index < len(response_msgs): # Editing an existing segment in response_msgs
                                # Ensure the message exists before creating task
                                target_msg = response_msgs[current_msg_index]
                                if target_msg:
                                    edit_task = asyncio.create_task(target_msg.edit(embed=current_segment_embed, view=view_to_attach))
                                    if view_to_attach: view_to_attach.message = target_msg # Update message ref
                                else:
                                     logging.error(f"Attempted to edit non-existent message at index {current_msg_index}")
                            elif not response_msgs and is_final_chunk: # Short final response, not starting a new segment
                                if processing_msg:
                                    await processing_msg.edit(content=None, embed=current_segment_embed, view=view_to_attach)
                                    response_msg = processing_msg
                                    response_msgs.append(response_msg)
                                else: # Short response, no processing_msg (should be rare)
                                    response_msg = await new_msg.reply(embed=current_segment_embed, view=view_to_attach, mention_author = False)
                                    response_msgs.append(response_msg)
                                # Common logic for new short response
                                self.msg_nodes[response_msg.id] = models.MsgNode(parent_msg=new_msg)
                                if view_to_attach: view_to_attach.message = response_msg
                                await self.msg_nodes[response_msg.id].lock.acquire()

                            self.last_task_time = dt.now().timestamp()

                    # Break loop if generation finished
                    if finish_reason:
                        # Check if finish reason indicates success
                        llm_call_successful = finish_reason.lower() in ("stop", "end_turn") or (is_gemini and finish_reason == str(google_types.FinishReason.FINISH_REASON_UNSPECIFIED))
                        break # Exit stream loop

            # --- End Stream Loop ---
            # Handle plain text responses (final output)
            if use_plain_responses and llm_call_successful: # use_plain_responses is the global config
                 final_messages_content = [final_text[i:i+split_limit] for i in range(0, len(final_text), split_limit)]
                 if not final_messages_content: final_messages_content.append("...") # Handle empty success

                 temp_response_msgs = []
                 for i, content_chunk in enumerate(final_messages_content):
                     if i == 0 and processing_msg: # First plain text chunk
                         await processing_msg.edit(content=content_chunk or "...", embed=None, view=None) # Clear embed
                         response_msg = processing_msg
                     else: # Subsequent chunks or no processing_msg
                         reply_to_msg = new_msg if not temp_response_msgs else temp_response_msgs[-1]
                         response_msg = await reply_to_msg.reply(content=content_chunk or "...", suppress_embeds=True, view=None, mention_author = False)
                     temp_response_msgs.append(response_msg)
                     # Create node and acquire lock immediately
                     # Use the correct MsgNode class
                     self.msg_nodes[response_msg.id] = models.MsgNode(parent_msg=new_msg)
                     node = self.msg_nodes[response_msg.id]
                     await node.lock.acquire()
                     # Store full text in the last node for plain responses
                     if i == len(final_messages_content) - 1:
                         node.full_response_text = final_text
                 response_msgs = temp_response_msgs # Update the main list


        except AllKeysFailedError as e:
            logging.error(f"LLM generation failed for message {new_msg.id}: {e}")
            error_text = f"⚠️ All API keys for provider `{e.service_name}` failed."
            last_err_str = str(e.errors[-1]) if e.errors else "Unknown reason."
            # Simplify error reporting for the user
            if "safety" in last_err_str.lower(): error_text += "\nLast error: Response blocked by safety filters."
            elif "recitation" in last_err_str.lower(): error_text += "\nLast error: Response stopped due to recitation."
            elif "no content received" in last_err_str.lower(): error_text += "\nLast error: No content received from API."
            elif "stream ended prematurely" in last_err_str.lower(): error_text += "\nLast error: Connection issue during response generation."
            else: error_text += f"\nLast error: `{last_err_str[:100]}{'...' if len(last_err_str) > 100 else ''}`"

            if processing_msg:
                if _use_plain_for_initial_status: # Check based on initial status message mode
                    await processing_msg.edit(content=error_text, embed=None, view=None)
                else:
                    # If an embed stream was active, try to update its last message (which might be processing_msg)
                    if response_msgs and response_msgs[-1].embeds:
                        target_edit_msg = response_msgs[-1]
                        embed_to_edit = discord.Embed.from_dict(target_edit_msg.embeds[0].to_dict())
                        current_desc = embed_to_edit.description or ""
                        embed_to_edit.description = current_desc.replace(STREAMING_INDICATOR, "").strip()
                        embed_to_edit.description += f"\n\n{error_text}"
                        embed_to_edit.description = embed_to_edit.description[:MAX_EMBED_DESCRIPTION_LENGTH]
                        embed_to_edit.color = EMBED_COLOR_ERROR
                        await target_edit_msg.edit(embed=embed_to_edit, view=None)
                    else: # No prior embed stream, edit processing_msg with a new error embed
                        await processing_msg.edit(content=None, embed=discord.Embed(description=error_text, color=EMBED_COLOR_ERROR), view=None)
            elif not _use_plain_for_initial_status and response_msgs and response_msgs[-1].embeds: # processing_msg failed, but stream started
                 # This is the existing logic if processing_msg wasn't there but stream was
                 target_edit_msg = response_msgs[-1]
                 embed_to_edit = discord.Embed.from_dict(target_edit_msg.embeds[0].to_dict())
                 current_desc = embed_to_edit.description or ""
                 embed_to_edit.description = current_desc.replace(STREAMING_INDICATOR, "").strip()
                 embed_to_edit.description += f"\n\n{error_text}"
                 embed_to_edit.description = embed_to_edit.description[:MAX_EMBED_DESCRIPTION_LENGTH]
                 embed_to_edit.color = EMBED_COLOR_ERROR
                 await target_edit_msg.edit(embed=embed_to_edit, view=None)
            else: # Fallback: No processing_msg and no stream to edit, send new reply
                await new_msg.reply(error_text, mention_author=False)
            llm_call_successful = False

        except Exception as outer_e:
            logging.exception(f"Unhandled error during message processing for {new_msg.id}.")
            error_text = f"⚠️ An unexpected error occurred: {type(outer_e).__name__}"
            if processing_msg:
                if _use_plain_for_initial_status:
                    await processing_msg.edit(content=error_text, embed=None, view=None)
                else:
                    await processing_msg.edit(content=None, embed=discord.Embed(description=error_text, color=EMBED_COLOR_ERROR), view=None)
            else:
                try:
                    await new_msg.reply(error_text, mention_author=False)
                except discord.HTTPException: pass # If even this fails, just log
            llm_call_successful = False

        finally: # --- Cleanup and Cache Management ---
            # --- MODIFIED: Imgur URL Resending Logic ---
            if llm_call_successful and final_text:
                lines = final_text.strip().split('\n')
                imgur_urls_to_resend = []
                found_header = False
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line == IMGUR_HEADER:
                        found_header = True
                        continue # Move to the next line after finding the header
                    if found_header and stripped_line.startswith(IMGUR_URL_PREFIX):
                        # Use the IMGUR_URL_PATTERN for validation
                        if IMGUR_URL_PATTERN.match(stripped_line):
                            imgur_urls_to_resend.append(stripped_line)
                        else:
                            # Stop if line starts with prefix but isn't a valid pattern
                            break
                    elif found_header and stripped_line: # If header was found but line is not imgur url and not empty, stop collecting
                        break # Stop collecting if a non-empty, non-imgur line is encountered after the header

                if imgur_urls_to_resend:
                    logging.info(f"Detected Imgur URLs in response to message {new_msg.id}. Resending without embeds.")
                    # Format URLs with spacing, no angle brackets
                    formatted_urls = imgur_urls_to_resend # Use raw URLs
                    max_chars = MAX_PLAIN_TEXT_LENGTH
                    messages_to_send_content = []
                    current_message_content = ""

                    for url_str in formatted_urls:
                        # Check if adding the next URL (plus double newline) exceeds the limit
                        needed_len = len(url_str) + (2 if current_message_content else 0) # +2 for '\n\n'
                        if len(current_message_content) + needed_len > max_chars:
                            if current_message_content: # Don't add empty messages
                                messages_to_send_content.append(current_message_content)
                            # Start new message, handle case where single URL is too long (unlikely for imgur)
                            if len(url_str) > max_chars:
                                logging.warning(f"Single Imgur URL too long to send: {url_str}")
                                messages_to_send_content.append(url_str[:max_chars-3] + "...")
                                current_message_content = ""
                            else:
                                current_message_content = url_str
                        else:
                            if current_message_content:
                                current_message_content += "\n\n" + url_str # Add double newline
                            else:
                                current_message_content = url_str

                    if current_message_content: # Add the last message segment
                        messages_to_send_content.append(current_message_content)

                    # Send the messages, replying to the last bot message
                    reply_target = response_msgs[-1] if response_msgs else new_msg
                    last_sent_msg = reply_target # Keep track of the last message sent in this sequence
                    for i, msg_content in enumerate(messages_to_send_content):
                        try:
                            # Reply to the previous message in the sequence or the original target
                            target_to_reply_to = last_sent_msg
                            # Ensure target is a Message object
                            if isinstance(target_to_reply_to, discord.Message):
                                sent_msg = await target_to_reply_to.reply(content=msg_content, mention_author=False) # REMOVED suppress_embeds=True
                                last_sent_msg = sent_msg # Update last sent message
                            else: # Fallback if target isn't a message (shouldn't happen often here)
                                 logging.warning(f"Invalid reply target type: {type(target_to_reply_to)}. Replying to original message.")
                                 sent_msg = await new_msg.reply(content=msg_content, mention_author=False) # REMOVED suppress_embeds=True
                                 last_sent_msg = sent_msg
                            await asyncio.sleep(0.1) # Small delay between sends
                        except discord.HTTPException as send_err:
                            logging.error(f"Failed to resend Imgur URL chunk {i+1}: {send_err}")
                            # Try sending to the original message as a fallback
                            try:
                                await new_msg.reply(f"(Error sending previous chunk)\n{msg_content}", mention_author=False) # REMOVED suppress_embeds=True
                            except discord.HTTPException as fallback_err:
                                logging.error(f"Failed fallback attempt to resend Imgur URL chunk {i+1}: {fallback_err}")
                        except Exception as e:
                             logging.error(f"Unexpected error resending Imgur URL chunk {i+1}: {e}")
            # --- END: Imgur URL Resending Logic ---


            # Release locks and store full text for *all* response messages created in this run if successful
            logging.debug(f"Entering finally block. response_msgs count: {len(response_msgs)}")
            for response_msg in response_msgs:
                if response_msg and response_msg.id in self.msg_nodes:
                    node = self.msg_nodes[response_msg.id]
                    # Store full text in the node if successful, regardless of which segment it is
                    if llm_call_successful:
                        node.full_response_text = final_text
                        logging.debug(f"Stored full_response_text ({len(final_text)} chars) in node {response_msg.id}")

                    if node.lock.locked():
                        try:
                            logging.debug(f"Releasing lock for message node {response_msg.id}")
                            node.lock.release()
                        except RuntimeError:
                            logging.warning(f"Attempted to release an already unlocked lock for node {response_msg.id}")
                elif response_msg:
                     logging.warning(f"Response message {response_msg.id} not found in msg_nodes during cleanup.")

            # Delete oldest MsgNodes from the cache
            if (num_nodes := len(self.msg_nodes)) > MAX_MESSAGE_NODES:
                nodes_to_delete = sorted(self.msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]
                logging.info(f"Cache limit reached ({num_nodes}/{MAX_MESSAGE_NODES}). Removing {len(nodes_to_delete)} oldest nodes.")
                for msg_id in nodes_to_delete:
                    node_to_delete = self.msg_nodes.get(msg_id)
                    if node_to_delete and hasattr(node_to_delete, 'lock') and node_to_delete.lock.locked():
                        try: node_to_delete.lock.release()
                        except RuntimeError: pass
                    self.msg_nodes.pop(msg_id, None)

            end_time = time.time()
            logging.info(f"Finished processing message {new_msg.id}. Success: {llm_call_successful}. Total time: {end_time - start_time:.2f} seconds.")

    def _is_allowed(self, message: discord.Message, is_dm: bool) -> bool:
        """Checks if the user and channel are allowed based on config."""
        permissions = self.config.get("permissions", {})
        user_perms = permissions.get("users", {"allowed_ids": [], "blocked_ids": []})
        role_perms = permissions.get("roles", {"allowed_ids": [], "blocked_ids": []})
        channel_perms = permissions.get("channels", {"allowed_ids": [], "blocked_ids": []})

        # Ensure lists exist and are lists
        allowed_user_ids = user_perms.get("allowed_ids", []) or []
        blocked_user_ids = user_perms.get("blocked_ids", []) or []
        allowed_role_ids = role_perms.get("allowed_ids", []) or []
        blocked_role_ids = role_perms.get("blocked_ids", []) or []
        allowed_channel_ids = channel_perms.get("allowed_ids", []) or []
        blocked_channel_ids = channel_perms.get("blocked_ids", []) or []

        role_ids = set(role.id for role in getattr(message.author, "roles", []))
        channel_ids = set(filter(None, (message.channel.id, getattr(message.channel, "parent_id", None), getattr(message.channel, "category_id", None))))

        # User check
        allow_all_users = not allowed_user_ids and not allowed_role_ids
        is_good_user = allow_all_users or message.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
        is_bad_user = (message.author.id in blocked_user_ids) or any(id in blocked_role_ids for id in role_ids)

        if is_bad_user or not is_good_user: # If blocked OR not explicitly allowed (when allow_all is false)
            return False

        # Channel check (only if not DM)
        if not is_dm:
            allow_all_channels = not allowed_channel_ids
            is_good_channel = allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
            is_bad_channel = any(id in blocked_channel_ids for id in channel_ids)

            if is_bad_channel or not is_good_channel: # If blocked OR not explicitly allowed (when allow_all is false)
                return False

        return True # Allowed if checks pass

    async def _fetch_external_content(self, cleaned_content: str, image_attachments: List[discord.Attachment], use_google_lens: bool, max_files_per_message: int, user_warnings: Set[str]) -> List[models.UrlFetchResult]:
        """Fetches content from URLs and Google Lens."""
        all_urls_with_indices = extract_urls_with_indices(cleaned_content)
        fetch_tasks = []
        processed_urls = set()
        url_fetch_results = []

        # Create tasks for non-Google Lens URLs
        for url, index in all_urls_with_indices:
            if url in processed_urls: continue
            processed_urls.add(url)

            if is_youtube_url(url):
                fetch_tasks.append(fetch_youtube_data(url, index, self.config.get("youtube_api_key")))
            elif is_reddit_url(url):
                sub_id = extract_reddit_submission_id(url)
                if sub_id:
                    fetch_tasks.append(fetch_reddit_data(url, sub_id, index, self.config.get("reddit_client_id"), self.config.get("reddit_client_secret"), self.config.get("reddit_user_agent")))
                else:
                    user_warnings.add(f"⚠️ Could not extract submission ID from Reddit URL: {url[:50]}...")
            elif is_image_url(url): # Check for image URLs
                # Define an async function to download the image and return a UrlFetchResult
                async def fetch_image_url_content(img_url: str, img_idx: int) -> models.UrlFetchResult:
                    try:
                        logging.info(f"Attempting to download image URL: {img_url}")
                        async with self.httpx_client.stream("GET", img_url, timeout=15.0) as response:
                            if response.status_code == 200:
                                content_type = response.headers.get("content-type", "").lower()
                                if content_type.startswith("image/"):
                                    img_bytes_list = []
                                    async for chunk in response.aiter_bytes():
                                        img_bytes_list.append(chunk)
                                    img_bytes = b"".join(img_bytes_list)
                                    logging.info(f"Successfully downloaded image from URL: {img_url} ({len(img_bytes)} bytes)")
                                    return models.UrlFetchResult(url=img_url, content=img_bytes, type="image_url_content", original_index=img_idx)
                                else:
                                    logging.warning(f"URL {img_url} is an image URL but content type is '{content_type}'. Skipping.")
                                    return models.UrlFetchResult(url=img_url, content=None, error=f"Not an image content type: {content_type}", type="image_url_content", original_index=img_idx)
                            else:
                                logging.warning(f"Failed to download image URL {img_url}. Status: {response.status_code}")
                                return models.UrlFetchResult(url=img_url, content=None, error=f"HTTP status {response.status_code}", type="image_url_content", original_index=img_idx)
                    except httpx.RequestError as e:
                        logging.warning(f"RequestError downloading image URL {img_url}: {e}")
                        return models.UrlFetchResult(url=img_url, content=None, error=f"Request error: {type(e).__name__}", type="image_url_content", original_index=img_idx)
                    except Exception as e:
                        logging.exception(f"Unexpected error downloading image URL {img_url}")
                        return models.UrlFetchResult(url=img_url, content=None, error=f"Unexpected error: {type(e).__name__}", type="image_url_content", original_index=img_idx)
                fetch_tasks.append(fetch_image_url_content(url, index))
            else: # General web page
                fetch_tasks.append(fetch_general_url_content(url, index))

        # Fetch non-Lens content concurrently
        if fetch_tasks:
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Unhandled exception during non-Lens URL fetch: {result}", exc_info=True)
                    user_warnings.add("⚠️ Unhandled error fetching URL content")
                # Use the correct UrlFetchResult class
                elif isinstance(result, models.UrlFetchResult):
                    url_fetch_results.append(result)
                    if result.error:
                        short_url = result.url[:40] + "..." if len(result.url) > 40 else result.url
                        user_warnings.add(f"⚠️ Error fetching {result.type} URL ({short_url}): {result.error}")
                else:
                     logging.error(f"Unexpected result type from non-Lens URL fetch: {type(result)}")

        # Process Google Lens Images Sequentially
        if use_google_lens:
            lens_images_to_process = image_attachments[:max_files_per_message] # Use max_files_per_message
            if len(image_attachments) > max_files_per_message:
                user_warnings.add(f"⚠️ Only processing first {max_files_per_message} images for Google Lens.")

            logging.info(f"Processing {len(lens_images_to_process)} Google Lens images sequentially...")
            for i, attachment in enumerate(lens_images_to_process):
                logging.info(f"Starting Google Lens processing for image {i+1}/{len(lens_images_to_process)}...")
                try:
                    # Pass self.config to process_google_lens_image
                    lens_result = await process_google_lens_image(attachment.url, i, self.config)
                    url_fetch_results.append(lens_result)
                    if lens_result.error:
                        logging.warning(f"Google Lens processing failed for image {i+1}: {lens_result.error}")
                        user_warnings.add(f"⚠️ Google Lens failed for image {i+1}: {lens_result.error[:100]}...")
                    else:
                        logging.info(f"Finished Google Lens processing for image {i+1}.")
                except Exception as e:
                    logging.exception(f"Unexpected error during sequential Google Lens processing for image {i+1}")
                    user_warnings.add(f"⚠️ Unexpected error processing Lens image {i+1}")
                    # Use the correct UrlFetchResult class
                    url_fetch_results.append(models.UrlFetchResult(
                        url=attachment.url, content=None,
                        error=f"Unexpected processing error: {type(e).__name__}",
                        type="google_lens_fallback_failed", original_index=i
                    ))

        return url_fetch_results

    def _format_external_content(self, url_fetch_results: List[models.UrlFetchResult]) -> str:
        """Formats fetched URL/Lens content into a string for the LLM."""
        if not url_fetch_results:
            return ""

        google_lens_parts = []
        other_url_parts = []
        other_url_counter = 1

        # Sort results by original position
        url_fetch_results.sort(key=lambda r: r.original_index)

        for result in url_fetch_results:
            if result.content: # Only include successful fetches
                if result.type == "google_lens_serpapi":
                    header = f"SerpAPI Google Lens results for image {result.original_index + 1}:\n"
                    google_lens_parts.append(header + str(result.content))
                elif result.type == "google_lens_custom":
                    header = f"Custom Google Lens fallback results for image {result.original_index + 1}:\n"
                    google_lens_parts.append(header + str(result.content))
                elif result.type == "youtube":
                    content_str = f"\nurl {other_url_counter}: {result.url}\nurl {other_url_counter} content:\n"
                    if isinstance(result.content, dict):
                        content_str += f"  title: {result.content.get('title', 'N/A')}\n"
                        content_str += f"  channel: {result.content.get('channel_name', 'N/A')}\n"
                        desc = result.content.get('description', 'N/A')
                        content_str += f"  description: {desc}\n"
                        transcript = result.content.get('transcript')
                        if transcript: content_str += f"  transcript: {transcript}\n"
                        comments = result.content.get("comments")
                        if comments: content_str += f"  top comments:\n" + "\n".join([f"    - {c}" for c in comments]) + "\n"
                    other_url_parts.append(content_str)
                    other_url_counter += 1
                elif result.type == "reddit":
                    content_str = f"\nurl {other_url_counter}: {result.url}\nurl {other_url_counter} content:\n"
                    if isinstance(result.content, dict):
                        content_str += f"  title: {result.content.get('title', 'N/A')}\n"
                        selftext = result.content.get('selftext')
                        if selftext: content_str += f"  content: {selftext}\n"
                        comments = result.content.get("comments")
                        if comments: content_str += f"  top comments:\n" + "\n".join([f"    - {c}" for c in comments]) + "\n"
                    other_url_parts.append(content_str)
                    other_url_counter += 1
                elif result.type == "general":
                    content_str = f"\nurl {other_url_counter}: {result.url}\nurl {other_url_counter} content:\n"
                    if isinstance(result.content, str): content_str += f"  {result.content}\n"
                    other_url_parts.append(content_str)
                    other_url_counter += 1
                # image_url_content types are handled as binary data and not formatted into combined_context text

        combined_context = ""
        if google_lens_parts or other_url_parts:
            combined_context = "Answer the user's query based on the following:\n\n"
            if google_lens_parts:
                combined_context += "\n\n".join(google_lens_parts) + "\n\n"
            if other_url_parts:
                combined_context += "".join(other_url_parts)

        return combined_context.strip()

    async def _build_message_history(
        self, new_msg: discord.Message, initial_cleaned_content: str, combined_context: str,
        max_messages: int, max_text: int, max_files_per_message: int,
        accept_files: bool, use_google_lens: bool, is_target_provider_gemini: bool,
        target_provider_name: str, target_model_name: str, user_warnings: Set[str],
        current_message_url_fetch_results: Optional[List[models.UrlFetchResult]] = None # Added parameter
    ) -> List[Dict[str, Any]]:
        history = []
        curr_msg = new_msg
        is_dm = isinstance(new_msg.channel, discord.DMChannel)

        while curr_msg is not None and len(history) < max_messages:
            if curr_msg.id not in self.msg_nodes:
                 # If node doesn't exist (e.g., message before bot restart), fetch it
                 logging.debug(f"Node for message {curr_msg.id} not in cache. Fetching message.")
                 try:
                     # Attempt to fetch the message if not the current one
                     if curr_msg.id != new_msg.id:
                         curr_msg = await new_msg.channel.fetch_message(curr_msg.id)
                         if not curr_msg: # If fetch failed or returned None
                              logging.warning(f"Failed to fetch message {curr_msg.id} for history building.")
                              user_warnings.add(f"⚠️ Couldn't fetch full history (message {curr_msg.id} missing).")
                              break # Stop building history if a message is missing
                     # If it's the current message, we already have it
                 except (discord.NotFound, discord.HTTPException) as fetch_err:
                      logging.warning(f"Failed to fetch message {curr_msg.id} for history building: {fetch_err}")
                      user_warnings.add(f"⚠️ Couldn't fetch full history (message {curr_msg.id} missing).")
                      break # Stop building history
                 # Create node after potentially fetching
                 self.msg_nodes[curr_msg.id] = models.MsgNode()

            curr_node = self.msg_nodes[curr_msg.id]

            async with curr_node.lock:
                is_current_message_node = (curr_msg.id == new_msg.id)
                current_role = "model" if curr_msg.author == self.user else "user"

                # Always set/update external_content for the *current message node* (new_msg)
                # if combined_context is provided for this specific history build.
                if is_current_message_node:
                    curr_node.external_content = combined_context if combined_context else None
                    logging.debug(f"Set external_content for node {curr_msg.id} to {'present' if combined_context else 'None'} for this history build.")

                # Determine if this node needs to be (re)populated for the current history build context.
                # Historical nodes are populated once. The current message node is (re)populated
                # each time _build_message_history is called for it.
                should_populate_node = (curr_node.text is None) or is_current_message_node

                if should_populate_node:
                    curr_node.has_bad_attachments = False # Initialize/Reset for this population pass
                    if is_current_message_node:
                        curr_node.api_file_parts = [] # Reset api_file_parts for re-evaluation if it's the current message

                    content_to_store = ""
                    if current_role == "model":
                        if curr_node.full_response_text:
                            content_to_store = curr_node.full_response_text
                            logging.debug(f"Using stored full_response_text for bot message {curr_msg.id}")
                        else:
                            if curr_msg.embeds and curr_msg.embeds[0].description:
                                content_to_store = curr_msg.embeds[0].description.replace(STREAMING_INDICATOR, "").strip()
                                logging.debug(f"Using embed description for bot message {curr_msg.id}")
                            else:
                                content_to_store = curr_msg.content
                                logging.debug(f"Falling back to curr_msg.content for bot message {curr_msg.id}")
                    else: # User message
                        content_to_store = initial_cleaned_content if curr_msg.id == new_msg.id else curr_msg.content
                        is_dm_current = isinstance(curr_msg.channel, discord.DMChannel)
                        if not is_dm_current and self.user.mentioned_in(curr_msg):
                             content_to_store = content_to_store.replace(self.user.mention, '').strip()
                        if curr_msg.id != new_msg.id:
                            content_to_store = AT_AI_PATTERN.sub(' ', content_to_store)
                        content_to_store = re.sub(r'\s{2,}', ' ', content_to_store).strip()

                    current_attachments = curr_msg.attachments
                    
                    MAX_ATTACHMENTS_TO_DOWNLOAD_IN_HISTORY = 5 
                    attachments_to_fetch = []
                    unfetched_unsupported_types = False

                    for att_idx, att in enumerate(current_attachments):
                        if len(attachments_to_fetch) >= MAX_ATTACHMENTS_TO_DOWNLOAD_IN_HISTORY:
                            curr_node.has_bad_attachments = True
                            logging.info(f"Message {curr_msg.id} has more than {MAX_ATTACHMENTS_TO_DOWNLOAD_IN_HISTORY} attachments. Only first {MAX_ATTACHMENTS_TO_DOWNLOAD_IN_HISTORY} considered for download.")
                            break
                        if att.content_type:
                            is_relevant_for_download = False
                            if att.content_type.startswith("text/"):
                                is_relevant_for_download = True
                            elif att.content_type.startswith("image/"):
                                # Images are relevant if the model can accept files (accept_files is true for vision models)
                                # or if it's the current message and Google Lens is active.
                                if accept_files or (curr_msg.id == new_msg.id and use_google_lens):
                                    is_relevant_for_download = True
                            elif att.content_type == "application/pdf":
                                # PDFs are relevant if:
                                # 1. Gemini model that accepts files (sent as bytes)
                                # 2. Non-Gemini model (text will be extracted from the downloaded PDF)
                                if (is_target_provider_gemini and accept_files) or (not is_target_provider_gemini):
                                    is_relevant_for_download = True
                            
                            if is_relevant_for_download:
                                attachments_to_fetch.append(att)
                            else: # Attachment type not relevant for download/API
                                unfetched_unsupported_types = True
                        # else: no content type, skip

                    if unfetched_unsupported_types: # If any attachment was skipped due to its type (not download limit)
                        curr_node.has_bad_attachments = True
                        logging.info(f"Message {curr_msg.id} has attachments of unsupported types that were not downloaded.")
                    
                    attachment_responses = await asyncio.gather(*[self.httpx_client.get(att.url, timeout=15.0) for att in attachments_to_fetch], return_exceptions=True)
                    
                    text_parts = [content_to_store] if content_to_store else []
                    if current_role == "user":
                        text_parts.extend(filter(None, (embed.title for embed in curr_msg.embeds)))
                        text_parts.extend(filter(None, (embed.description for embed in curr_msg.embeds)))

                    for att, resp in zip(attachments_to_fetch, attachment_responses):
                        if isinstance(resp, httpx.Response) and resp.status_code == 200 and att.content_type.startswith("text/"):
                            try:
                                attachment_text = resp.text
                                text_parts.append(attachment_text)
                            except Exception as e:
                                logging.warning(f"Failed to decode text attachment {att.filename} in history: {e}")
                                curr_node.has_bad_attachments = True # Error processing downloaded text
                        elif isinstance(resp, Exception) and att.content_type.startswith("text/"):
                            logging.warning(f"Failed to fetch text attachment {att.filename} in history: {resp}")
                            curr_node.has_bad_attachments = True # Error fetching text att

                    curr_node.text = "\n".join(filter(None, text_parts))

                    # --- NEW: PDF text extraction for non-Gemini models ---
                    if current_role == "user" and not is_target_provider_gemini: # Use is_target_provider_gemini
                        pdf_texts_to_append = []
                        for att, resp in zip(attachments_to_fetch, attachment_responses):
                            if att.content_type == "application/pdf":
                                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                                    try:
                                        pdf_bytes = resp.content
                                        # Use the new utility function
                                        extracted_pdf_text = await extract_text_from_pdf_bytes(pdf_bytes)
                                        if extracted_pdf_text:
                                            # Format similar to the user's example
                                            pdf_texts_to_append.append(f"\n\n--- Content from PDF: {att.filename} ---\n{extracted_pdf_text}\n--- End of PDF: {att.filename} ---")
                                        else:
                                            logging.warning(f"Could not extract text from PDF {att.filename} for non-Gemini model in msg {curr_msg.id}.")
                                            curr_node.has_bad_attachments = True
                                    except Exception as pdf_extract_err:
                                        logging.error(f"Error extracting text from PDF {att.filename} for non-Gemini model in msg {curr_msg.id}: {pdf_extract_err}")
                                        curr_node.has_bad_attachments = True
                                elif isinstance(resp, Exception): # Fetch failed
                                    logging.warning(f"Failed to fetch PDF attachment {att.filename} for text extraction (non-Gemini) in msg {curr_msg.id}: {resp}")
                                    curr_node.has_bad_attachments = True
                        
                        if pdf_texts_to_append:
                            # Append extracted PDF text to the existing node text
                            curr_node.text = (curr_node.text or "") + "".join(pdf_texts_to_append)
                            logging.info(f"Appended text from {len(pdf_texts_to_append)} PDF(s) to user query for non-Gemini model in message {curr_msg.id}")
                    # --- END NEW PDF text extraction ---

                    api_file_parts = [] # Renamed from image_parts
                    files_processed_for_api_count = 0 # Renamed from images_processed_count
                    is_lens_trigger_message = curr_msg.id == new_msg.id and use_google_lens

                    should_process_files_for_api = (current_role == "user" or is_lens_trigger_message) and \
                                           (accept_files or is_lens_trigger_message)

                    if should_process_files_for_api and not is_lens_trigger_message:
                        # Process Discord Attachments
                        for att, resp in zip(attachments_to_fetch, attachment_responses):
                            is_api_relevant_type = False
                            mime_type_for_api = att.content_type # Default to attachment's content type
                            file_bytes_for_api = None

                            if att.content_type.startswith("image/"):
                                is_api_relevant_type = True
                                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                                    file_bytes_for_api = resp.content
                                elif isinstance(resp, Exception):
                                    logging.warning(f"Failed to fetch image attachment {att.filename} for API: {resp}")
                                    curr_node.has_bad_attachments = True
                                    continue # Skip this attachment
                                else: # Non-200 response
                                    logging.warning(f"Non-200 response for image attachment {att.filename}: {resp.status_code if resp else 'N/A'}")
                                    curr_node.has_bad_attachments = True
                                    continue

                            elif att.content_type == "application/pdf" and is_target_provider_gemini and accept_files:
                                is_api_relevant_type = True
                                mime_type_for_api = "application/pdf" # Ensure correct MIME for PDF
                                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                                    file_bytes_for_api = resp.content
                                elif isinstance(resp, Exception):
                                    logging.warning(f"Failed to fetch PDF attachment {att.filename} for API: {resp}")
                                    curr_node.has_bad_attachments = True
                                    continue
                                else:
                                    logging.warning(f"Non-200 response for PDF attachment {att.filename}: {resp.status_code if resp else 'N/A'}")
                                    curr_node.has_bad_attachments = True
                                    continue
                            
                            if not is_api_relevant_type or file_bytes_for_api is None:
                                continue

                            if files_processed_for_api_count >= max_files_per_message:
                                curr_node.has_bad_attachments = True
                                logging.warning(f"Max {max_files_per_message} files (attachments + image URLs) processed for API for message {curr_msg.id}. Skipping attachment: {att.filename}")
                                break # Stop processing more attachments

                            try:
                                if is_target_provider_gemini:
                                    api_file_parts.append(google_types.Part.from_bytes(data=file_bytes_for_api, mime_type=mime_type_for_api))
                                else:
                                    api_file_parts.append(dict(type="image_url", image_url=dict(url=f"data:{mime_type_for_api};base64,{base64.b64encode(file_bytes_for_api).decode('utf-8')}")))
                                files_processed_for_api_count += 1
                            except Exception as e:
                                logging.error(f"Error preparing attachment {att.filename} for API in msg {curr_msg.id}: {e}")
                                curr_node.has_bad_attachments = True
                        
                        # Process successfully fetched image URLs for the current message
                        if curr_msg.id == new_msg.id and current_message_url_fetch_results:
                            for fetched_url_res in current_message_url_fetch_results:
                                if fetched_url_res.type == "image_url_content" and isinstance(fetched_url_res.content, bytes) and not fetched_url_res.error:
                                    if files_processed_for_api_count >= max_files_per_message:
                                        curr_node.has_bad_attachments = True
                                        user_warnings.add(f"⚠️ Max {max_files_per_message} files (attachments + image URLs) reached. Some image URLs may not be included.")
                                        logging.warning(f"Max {max_files_per_message} files reached. Skipping image URL: {fetched_url_res.url}")
                                        break # Stop processing more image URLs

                                    img_bytes = fetched_url_res.content
                                    # Try to infer mime type from URL extension or default
                                    url_lower = fetched_url_res.url.lower()
                                    mime_type = "image/png" # Default
                                    if url_lower.endswith(".jpg") or url_lower.endswith(".jpeg"): mime_type = "image/jpeg"
                                    elif url_lower.endswith(".gif"): mime_type = "image/gif"
                                    elif url_lower.endswith(".webp"): mime_type = "image/webp"
                                    elif url_lower.endswith(".bmp"): mime_type = "image/bmp"
                                    # Add more if needed

                                    try:
                                        if is_target_provider_gemini:
                                            api_file_parts.append(google_types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
                                        else:
                                            api_file_parts.append(dict(type="image_url", image_url=dict(url=f"data:{mime_type};base64,{base64.b64encode(img_bytes).decode('utf-8')}")))
                                        files_processed_for_api_count += 1
                                        logging.info(f"Added downloaded image from URL {fetched_url_res.url} to API parts.")
                                    except Exception as e:
                                        logging.error(f"Error preparing downloaded image URL {fetched_url_res.url} for API: {e}")
                                        user_warnings.add(f"⚠️ Error processing image from URL: {fetched_url_res.url[:50]}...")
                                        curr_node.has_bad_attachments = True


                    curr_node.api_file_parts = api_file_parts # Store prepared API parts
                    curr_node.role = current_role
                    curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
                    # The old check `curr_node.has_bad_attachments = curr_node.has_bad_attachments or (len(current_attachments) > len(good_attachments))` is removed.
                    # Its intent is covered by MAX_ATTACHMENTS_TO_DOWNLOAD_IN_HISTORY check and unfetched_unsupported_types flag.

                    if curr_node.parent_msg is None and not curr_node.fetch_parent_failed:
                        parent = await self._find_parent_message(curr_msg, is_dm)
                        if parent is None and curr_msg.reference and curr_msg.reference.message_id:
                            curr_node.fetch_parent_failed = True
                        curr_node.parent_msg = parent

                current_text_content = ""
                # Use the external_content specific to this node, which might have been set above for the current message,
                # or might be from a previous population for historical messages.
                if curr_node.external_content:
                    current_text_content += curr_node.external_content + "\n\nUser's query:\n"
                    logging.debug(f"Prepending external_content for node {curr_msg.id} in history build")

                node_text_to_use = ""
                if curr_node.role == "model":
                    node_text_to_use = curr_node.full_response_text or curr_node.text or ""
                    if not curr_node.full_response_text and curr_node.text:
                        logging.debug(f"Using fallback curr_node.text for bot message {curr_msg.id} in history build")
                else: 
                    node_text_to_use = curr_node.text or ""

                if node_text_to_use:
                    current_text_content += node_text_to_use

                current_text_content = current_text_content[:max_text] if current_text_content else ""
                
                # --- MODIFIED SECTION TO DETERMINE current_api_file_parts ---
                current_api_file_parts = [] # Initialize as empty
                if accept_files: # Check the 'accept_files' parameter for THE CURRENT _build_message_history call
                    # If files are allowed for *this* history construction,
                    # then use (and convert if necessary) the parts stored in the node.
                    raw_parts_from_node = curr_node.api_file_parts[:max_files_per_message]
                    
                    if is_target_provider_gemini: # Current call wants Gemini parts
                        for part_in_node in raw_parts_from_node:
                            if isinstance(part_in_node, google_types.Part):
                                current_api_file_parts.append(part_in_node)
                            elif isinstance(part_in_node, dict) and part_in_node.get("type") == "image_url": # Is OpenAI image dict
                                image_url_data = part_in_node.get("image_url", {})
                                data_url = image_url_data.get("url")
                                if isinstance(data_url, str) and data_url.startswith("data:image") and ";base64," in data_url:
                                    try:
                                        header, encoded_data = data_url.split(";base64,", 1)
                                        mime_parts = header.split(":")
                                        mime_type = mime_parts[1] if len(mime_parts) > 1 else "image/png"
                                        img_bytes = base64.b64decode(encoded_data)
                                        current_api_file_parts.append(google_types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
                                    except Exception as e:
                                        logging.warning(f"Error converting cached OpenAI image part to Gemini part for node {curr_msg.id}: {e}")
                            # else: skip unrecognized cached part type if it's not convertible
                    else: # Current call wants non-Gemini (e.g., OpenAI dict) parts
                        for part_in_node in raw_parts_from_node:
                            if isinstance(part_in_node, dict) and part_in_node.get("type") == "image_url":
                                current_api_file_parts.append(part_in_node) # Already correct format
                            elif isinstance(part_in_node, google_types.Part) and \
                                 hasattr(part_in_node, 'inline_data') and \
                                 part_in_node.inline_data and \
                                 hasattr(part_in_node.inline_data, 'data') and \
                                 hasattr(part_in_node.inline_data, 'mime_type'):
                                try:
                                    mime_type = part_in_node.inline_data.mime_type
                                    if mime_type.startswith("image/"): # Check if it's an image
                                        # inline_data.data from a Part created with from_bytes contains raw bytes
                                        raw_bytes = part_in_node.inline_data.data
                                        b64_encoded_data = base64.b64encode(raw_bytes).decode('utf-8')
                                        current_api_file_parts.append({
                                            "type": "image_url",
                                            "image_url": {"url": f"data:{mime_type};base64,{b64_encoded_data}"}
                                        })
                                        logging.debug(f"Converted cached Gemini image Part (mime: {mime_type}) to OpenAI part for node {curr_msg.id}.")
                                    else:
                                        # If it's not an image (e.g., a PDF that was cached from a Gemini call), skip it.
                                        # Its text content should have already been extracted and added to curr_node.text for non-Gemini models.
                                        logging.debug(f"Skipping cached non-image Gemini Part (mime: {mime_type}) for OpenAI history for node {curr_msg.id}.")
                                except Exception as e:
                                    logging.warning(f"Error converting cached Gemini Part to OpenAI image dict for node {curr_msg.id}: {e}")
                            # else: skip unrecognized cached part type if it's not convertible
                # --- END MODIFIED SECTION ---

                parts_for_api = []
                if is_target_provider_gemini: # Use is_target_provider_gemini
                    if current_text_content:
                        parts_for_api.append(google_types.Part.from_text(text=current_text_content))
                    parts_for_api.extend(current_api_file_parts)
                else: 
                    if current_text_content:
                        parts_for_api.append({"type": "text", "text": current_text_content})
                    parts_for_api.extend(current_api_file_parts) # This now correctly passes list of dicts for OpenAI

                if parts_for_api:
                    message_data = {"role": curr_node.role}
                    if is_target_provider_gemini: # Use is_target_provider_gemini
                        if not isinstance(parts_for_api, list): parts_for_api = [parts_for_api]
                        message_data["parts"] = parts_for_api
                    else: # Non-Gemini
                        if message_data["role"] == "model": message_data["role"] = "assistant"
                        
                        # Determine the content for the message_data
                        final_content_for_api = ""
                        if len(parts_for_api) == 1 and parts_for_api[0]["type"] == "text":
                            final_content_for_api = parts_for_api[0]["text"]
                        elif parts_for_api: # Multimodal content
                            final_content_for_api = parts_for_api
                        
                        message_data["content"] = final_content_for_api
                        
                        # ADDED/MODIFIED DETAILED LOGGING HERE
                        if curr_msg.id == new_msg.id and target_provider_name != "google":
                            logging.info(f"DEBUG MSG HISTORY (User: {curr_msg.author.name}, Target: {target_provider_name}/{target_model_name}):")
                            log_external_content = str(curr_node.external_content) if curr_node.external_content else "None"
                            log_node_text = str(curr_node.text) if curr_node.text else "None"
                            # current_text_content is the source for parts_for_api[0]['text']
                            log_assembled_text_for_parts = "N/A (multimodal or empty)"
                            if isinstance(final_content_for_api, str):
                                log_assembled_text_for_parts = final_content_for_api
                            elif isinstance(final_content_for_api, list) and final_content_for_api and final_content_for_api[0].get("type") == "text":
                                log_assembled_text_for_parts = final_content_for_api[0].get("text", "ERROR GETTING TEXT")


                            logging.info(f"  - Node Text (Original Query): '{log_node_text[:200]}{'...' if len(log_node_text) > 200 else ''}'")
                            logging.info(f"  - Node External Content (SearxNG/URL): '{log_external_content[:300]}{'...' if len(log_external_content) > 300 else ''}'")
                            logging.info(f"  - Assembled text for API parts: '{log_assembled_text_for_parts[:500]}{'...' if len(log_assembled_text_for_parts) > 500 else ''}'")
                            logging.info(f"  - Final message_data['content'] for API: '{str(message_data['content'])[:500]}{'...' if len(str(message_data['content'])) > 500 else ''}'")

                        if target_provider_name in PROVIDERS_SUPPORTING_USERNAMES and curr_node.role == "user" and curr_node.user_id is not None: # Use target_provider_name
                            message_data["name"] = str(curr_node.user_id)
                    history.append(message_data)

                if curr_node.text and len(curr_node.text) > max_text: # Check length of text content for this node
                    user_warnings.add(f"⚠️ Max {max_text:,} chars/msg node text")
                # Warning for hitting max_files_per_message is logged inside the loop.
                # The curr_node.has_bad_attachments flag handles general attachment issues.
                if curr_node.has_bad_attachments: # This flag is now more accurately set
                    user_warnings.add("⚠️ Some attachments might not have been processed (limit, error, or type).")
                if curr_node.fetch_parent_failed:
                     user_warnings.add(f"⚠️ Couldn't fetch full history")
                if curr_node.parent_msg is not None and len(history) >= max_messages:
                     user_warnings.add(f"⚠️ Only using last {max_messages} messages")

                curr_msg = curr_node.parent_msg
        return history[::-1]

    async def _find_parent_message(self, message: discord.Message, is_dm: bool) -> Optional[discord.Message]:
        """Determines the logical parent message for conversation history."""
        try:
            # Check if the current message explicitly triggers the bot
            mentions_bot_in_current = self.user.mentioned_in(message)
            contains_at_ai_in_current = AT_AI_PATTERN.search(message.content) is not None
            is_explicit_trigger = mentions_bot_in_current or contains_at_ai_in_current

            # 1. Explicit Reply always takes precedence
            if message.reference and message.reference.message_id:
                try:
                    # Prefer cached message, fetch if not cached
                    ref_msg = message.reference.cached_message
                    if not ref_msg:
                        # Use channel.fetch_message which is available on TextChannel, DMChannel, Thread etc.
                        ref_msg = await message.channel.fetch_message(message.reference.message_id)
                    # Ensure the referenced message is usable (not deleted system message etc.)
                    if ref_msg and ref_msg.type in (discord.MessageType.default, discord.MessageType.reply):
                        return ref_msg
                    else:
                         logging.debug(f"Referenced message {message.reference.message_id} is not usable type {getattr(ref_msg, 'type', 'N/A')}")
                except (discord.NotFound, discord.HTTPException) as e:
                    logging.warning(f"Could not fetch referenced message {message.reference.message_id}: {e}")
                    # Don't set fetch_parent_failed here, let the caller handle it based on return None

            # 2. Thread Start: If it's the first user message in a thread (not a reply)
            # Check if it's a thread and has a parent (to fetch starter message from)
            if isinstance(message.channel, discord.Thread) and message.channel.parent and not message.reference:
                 try:
                     # The starter message is the logical parent
                     starter_msg = message.channel.starter_message
                     if not starter_msg:
                         # Fetch starter message via parent channel and thread ID
                         starter_msg = await message.channel.parent.fetch_message(message.channel.id)
                     if starter_msg and starter_msg.type in (discord.MessageType.default, discord.MessageType.reply):
                         return starter_msg
                     else:
                          logging.debug(f"Thread starter message {message.channel.id} is not usable type {getattr(starter_msg, 'type', 'N/A')}")
                 except (discord.NotFound, discord.HTTPException, AttributeError) as e:
                     logging.warning(f"Could not fetch thread starter message for thread {message.channel.id}: {e}")

            # 3. Automatic Chaining (Only if NOT explicitly triggered)
            if not is_explicit_trigger:
                 prev_msg_in_channel = None
                 try:
                     async for m in message.channel.history(before=message, limit=1):
                         prev_msg_in_channel = m
                         break
                 except (discord.Forbidden, discord.HTTPException) as e:
                     logging.warning(f"Could not fetch history in channel {message.channel.id}: {e}")

                 if prev_msg_in_channel and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply):
                     # In DMs, chain if previous is from bot. In channels, chain if previous is from same user.
                     if (is_dm and prev_msg_in_channel.author == self.user) or \
                        (not is_dm and prev_msg_in_channel.author == message.author):
                         return prev_msg_in_channel

            # 4. No logical parent found
            return None

        except Exception as e:
            logging.exception(f"Error determining parent message for {message.id}")
            return None # Indicate failure to find parent

    def _prepare_system_prompt(self, is_gemini: bool, provider: str, base_prompt_text: Optional[str]) -> Optional[str]:
        """Constructs the system prompt string with dynamic elements based on the provided base text."""
        if not base_prompt_text:
            return None

        now_utc = dt.now(timezone.utc)
        hour_12 = now_utc.strftime('%I').lstrip('0')
        minute = now_utc.strftime('%M')
        am_pm = now_utc.strftime('%p')
        time_str = f"{hour_12}:{minute}\u202F{am_pm}" # Narrow no-break space
        date_str = now_utc.strftime('%A, %B %d, %Y')
        current_datetime_str = f"current date and time: {date_str} {time_str} Coordinated Universal Time (UTC)"

        system_prompt_extras = [current_datetime_str]
        if not is_gemini and provider in PROVIDERS_SUPPORTING_USERNAMES:
            system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")

        return "\n".join([base_prompt_text] + system_prompt_extras)

    async def _get_web_search_queries_from_gemini(
        self,
        history_for_gemini_grounding: List[Dict[str, Any]],
        system_prompt_text_for_grounding: Optional[str]
    ) -> Optional[List[str]]:
        """
        Calls Gemini to get web search queries from its grounding metadata.
        """
        logging.info("Attempting to get web search queries from Gemini for grounding...")
        gemini_provider_config = self.config.get("providers", {}).get(GROUNDING_MODEL_PROVIDER, {})
        if not gemini_provider_config or not gemini_provider_config.get("api_keys"):
            logging.warning(f"Cannot perform Gemini grounding step: Provider '{GROUNDING_MODEL_PROVIDER}' not configured with API keys.")
            return None

        all_web_search_queries = set() # Use a set to store unique queries

        try:
            # Use a minimal set of extra_params for the grounding call
            grounding_extra_params = {"temperature": 0.7, "thinking_budget": 0} # Could be configurable
            
            stream_generator = generate_response_stream(
                provider=GROUNDING_MODEL_PROVIDER,
                model_name=GROUNDING_MODEL_NAME,
                history_for_llm=history_for_gemini_grounding,
                system_prompt_text=system_prompt_text_for_grounding,
                provider_config=gemini_provider_config,
                extra_params=grounding_extra_params
            )

            async for _, _, chunk_grounding_metadata, error_message in stream_generator:
                if error_message:
                    logging.error(f"Error during Gemini grounding call: {error_message}")
                    # Don't immediately fail all, maybe some metadata was already processed
                    # Or, decide if any error here should abort query gathering
                    return None # Abort on first error for simplicity now

                if chunk_grounding_metadata:
                    if hasattr(chunk_grounding_metadata, 'web_search_queries') and chunk_grounding_metadata.web_search_queries:
                        for query in chunk_grounding_metadata.web_search_queries:
                            if isinstance(query, str) and query.strip():
                                all_web_search_queries.add(query.strip())
                                logging.debug(f"Gemini grounding produced search query: '{query.strip()}'")
            
            if all_web_search_queries:
                logging.info(f"Gemini grounding produced {len(all_web_search_queries)} unique search queries.")
                return list(all_web_search_queries)
            else:
                logging.info("Gemini grounding call completed but found no web_search_queries in metadata.")
                return None

        except AllKeysFailedError as e:
            logging.error(f"All API keys failed for Gemini grounding model ({GROUNDING_MODEL_PROVIDER}/{GROUNDING_MODEL_NAME}): {e}")
            return None
        except Exception as e:
            logging.exception(f"Unexpected error during Gemini grounding call to get search queries:")
            return None

    async def _fetch_and_format_searxng_results(
        self,
        queries: List[str],
        user_query_for_log: str # For logging context
    ) -> Optional[str]:
        """
        Fetches search results from SearxNG for given queries,
        then fetches content of those URLs (uniquely) and formats them.
        Uses specific fetchers for YouTube and Reddit URLs.
        """
        if not queries:
            return None

        searxng_base_url = self.config.get(SEARXNG_BASE_URL_CONFIG_KEY)
        if not searxng_base_url:
            logging.warning("SearxNG base URL not configured. Skipping web search enhancement.")
            return None

        logging.info(f"Fetching SearxNG results for {len(queries)} queries related to user query: '{user_query_for_log[:100]}...'")
        searxng_content_limit = self.config.get(SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY)

        # API keys for specific fetchers
        youtube_api_key = self.config.get("youtube_api_key")
        reddit_client_id = self.config.get("reddit_client_id")
        reddit_client_secret = self.config.get("reddit_client_secret")
        reddit_user_agent = self.config.get("reddit_user_agent")

        # Fetch SearxNG URLs for all queries concurrently
        searxng_tasks = []
        for query in queries:
            searxng_tasks.append(
                fetch_searxng_results(query, self.httpx_client, searxng_base_url, SEARXNG_NUM_RESULTS)
            )
        
        try:
            list_of_url_lists_per_query = await asyncio.gather(*searxng_tasks, return_exceptions=True)
        except Exception as e:
            logging.exception("Error gathering SearxNG results.")
            return None

        # Collect all unique URLs from all queries
        unique_urls_to_fetch_content = set()
        for i, query_urls_or_exc in enumerate(list_of_url_lists_per_query):
            query_str = queries[i]
            if isinstance(query_urls_or_exc, Exception):
                logging.error(f"Failed to get SearxNG results for query '{query_str}': {query_urls_or_exc}")
                continue
            if not query_urls_or_exc:
                logging.info(f"No URLs returned by SearxNG for query: '{query_str}'")
                continue
            for url_str in query_urls_or_exc:
                if isinstance(url_str, str): # Ensure it's a string
                    unique_urls_to_fetch_content.add(url_str)

        if not unique_urls_to_fetch_content:
            logging.info("No unique URLs found from SearxNG results to process content for.")
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
                        fetch_reddit_data(url_str, submission_id, idx, reddit_client_id, reddit_client_secret, reddit_user_agent)
                    )
                else:
                    logging.warning(f"Could not extract submission ID from SearxNG Reddit URL: {url_str}. Skipping.")
                    # Add a dummy failed result to maintain list structure if needed, or handle Nones later
                    async def dummy_failed_result():
                        return models.UrlFetchResult(url=url_str, content=None, error="Failed to extract Reddit submission ID.", type="reddit", original_index=idx)
                    url_content_processing_tasks.append(dummy_failed_result())
            else:
                url_content_processing_tasks.append(
                    fetch_general_url_content(url_str, idx, max_text_length=searxng_content_limit)
                )
        
        try:
            fetched_unique_content_results = await asyncio.gather(*url_content_processing_tasks, return_exceptions=True)
        except Exception as e:
            logging.exception("Error gathering content from unique SearxNG URLs.")
            return None
            
        # Create a map of URL string to its fetched content result for easy lookup
        url_to_content_map: Dict[str, models.UrlFetchResult] = {}
        for result_or_exc in fetched_unique_content_results:
            if isinstance(result_or_exc, models.UrlFetchResult):
                url_to_content_map[result_or_exc.url] = result_or_exc
            elif isinstance(result_or_exc, Exception):
                logging.error(f"Exception while processing a unique URL for content: {result_or_exc}")


        # Now, format the output using the pre-fetched unique content
        formatted_query_blocks = []
        query_counter = 1
        limit = searxng_content_limit # Use the configured limit for truncation

        for i, query_str in enumerate(queries):
            urls_for_this_query_or_exc = list_of_url_lists_per_query[i]
            
            if isinstance(urls_for_this_query_or_exc, Exception) or not urls_for_this_query_or_exc:
                continue

            current_query_url_content_parts = []
            url_counter_for_query = 1
            
            for url_from_searxng_for_query in urls_for_this_query_or_exc:
                if not isinstance(url_from_searxng_for_query, str):
                    continue

                content_result = url_to_content_map.get(url_from_searxng_for_query)
                
                if content_result and content_result.content and not content_result.error:
                    content_str_part = ""
                    if content_result.type == "youtube" and isinstance(content_result.content, dict):
                        data = content_result.content
                        title = data.get('title', 'N/A')
                        content_str_part += f"  title: {discord.utils.escape_markdown(title)}\n"
                        channel = data.get('channel_name', 'N/A')
                        content_str_part += f"  channel: {discord.utils.escape_markdown(channel)}\n"
                        
                        desc = data.get('description', 'N/A')
                        if limit and len(desc) > limit: desc = desc[:limit-3] + "..."
                        content_str_part += f"  description: {discord.utils.escape_markdown(desc)}\n"
                        
                        transcript = data.get('transcript')
                        if transcript:
                            if limit and len(transcript) > limit: transcript = transcript[:limit-3] + "..."
                            content_str_part += f"  transcript: {discord.utils.escape_markdown(transcript)}\n"
                            
                        comments_list = data.get("comments")
                        if comments_list:
                            # Join, then truncate the whole block if needed
                            comments_str = "\n".join([f"    - {discord.utils.escape_markdown(c)}" for c in comments_list])
                            if limit and len(comments_str) > limit: comments_str = comments_str[:limit-3] + "..."
                            content_str_part += f"  top comments:\n{comments_str}\n"

                    elif content_result.type == "reddit" and isinstance(content_result.content, dict):
                        data = content_result.content
                        title = data.get('title', 'N/A')
                        content_str_part += f"  title: {discord.utils.escape_markdown(title)}\n"

                        selftext = data.get('selftext')
                        if selftext:
                            if limit and len(selftext) > limit: selftext = selftext[:limit-3] + "..."
                            content_str_part += f"  content: {discord.utils.escape_markdown(selftext)}\n"

                        comments_list = data.get("comments")
                        if comments_list:
                            comments_str = "\n".join([f"    - {discord.utils.escape_markdown(c)}" for c in comments_list])
                            if limit and len(comments_str) > limit: comments_str = comments_str[:limit-3] + "..."
                            content_str_part += f"  top comments:\n{comments_str}\n"

                    elif content_result.type == "general" and isinstance(content_result.content, str):
                        # Already truncated by fetch_general_url_content if limit was passed
                        content_str_part = discord.utils.escape_markdown(content_result.content)
                    else: # Fallback for unknown types or if content is not string/dict as expected
                        raw_content_str = str(content_result.content)
                        if limit and len(raw_content_str) > limit: raw_content_str = raw_content_str[:limit-3] + "..."
                        content_str_part = discord.utils.escape_markdown(raw_content_str)
                    
                    current_query_url_content_parts.append(
                        f"URL {url_counter_for_query}: {content_result.url}\n" # URL itself should not be escaped
                        f"URL {url_counter_for_query} content:\n{content_str_part.strip()}\n"
                    )
                    url_counter_for_query += 1
                elif content_result and content_result.error:
                    logging.warning(f"Skipping SearxNG URL {content_result.url} for query '{query_str}' due to fetch error: {content_result.error}")
                elif not content_result:
                     logging.warning(f"Content for SearxNG URL {url_from_searxng_for_query} (query: '{query_str}') not found in pre-fetched map. It might have been invalid or failed very early.")
            
            if current_query_url_content_parts:
                query_block_header = f'Query {query_counter} ("{discord.utils.escape_markdown(query_str)}") search results:\n\n'
                formatted_query_blocks.append(query_block_header + "\n".join(current_query_url_content_parts))
                query_counter += 1
        
        if formatted_query_blocks:
            final_context = "Answer the user's query based on the following:\n\n" + "\n\n".join(formatted_query_blocks)
            return final_context.strip()

        logging.info("No content successfully fetched and formatted from SearxNG result URLs.")
        return None

    async def close(self):
        """Clean up resources when the bot is shutting down."""
        logging.info("Closing HTTPX client...")
        await self.httpx_client.aclose()
        logging.info("Closing database connections...")
        close_all_db_managers()
        await super().close()
