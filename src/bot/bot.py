import logging
import time
from typing import Dict

import discord
from discord import app_commands

# Use google.genai.types
from google.genai import types as google_types

import httpx

from ..core.constants import (
    MAX_EMBED_DESCRIPTION_LENGTH,
    AT_AI_PATTERN,
    GOOGLE_LENS_PATTERN,
    PROVIDERS_SUPPORTING_USERNAMES,
    MAX_PLAIN_TEXT_LENGTH,
    GEMINI_USE_THINKING_BUDGET_CONFIG_KEY,
    GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY,
    EMBED_COLOR_ERROR,
    # New config keys
    MAX_MESSAGE_NODES_CONFIG_KEY,
)

from ..core import models
from ..core.rate_limiter import check_and_perform_global_reset, close_all_db_managers
from ..core.config import get_max_text_for_model  # New
from ..core.utils import (
    extract_urls_with_indices,
    is_image_url,
    extract_text_from_pdf_bytes,
    extract_text_from_docx_bytes,
)
from ..llm.model_selector import determine_final_model  # Added import
from ..messaging.message_parser import (
    should_process_message,
    clean_message_content,
    check_google_lens_trigger,
)
from ..content.processor import process_content_and_grounding  # Added import
from .commands import (
    set_model_command,
    set_system_prompt_command,
    get_user_system_prompt_preference,
    setgeminithinking,
    get_user_gemini_thinking_budget_preference,
    help_command,
    load_all_preferences,
    enhance_prompt_command,  # Added for slash command
    _execute_enhance_prompt_logic,  # Added for prefix command
)
from .permissions import is_message_allowed
from ..services.prompt_utils import prepare_system_prompt
from ..messaging.response_sender import (
    send_initial_processing_message,
    handle_llm_response_stream,
    resend_imgur_urls,  # Also import this helper if direct calls are made from bot.py for exceptions outside stream
)
from ..messaging.history_utils import build_message_history


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
        # Optimized HTTP client for improved performance
        from ..core.constants import (
            HTTP_CLIENT_USE_HTTP2_CONFIG_KEY,
            DEFAULT_HTTP_CLIENT_USE_HTTP2,
        )

        # Enhanced connection pooling for better performance
        limits = httpx.Limits(
            max_keepalive_connections=30,  # Increased from 20 for better reuse
            max_connections=150,  # Increased from 100 for higher concurrency
            keepalive_expiry=30.0,  # Keep connections alive for 30 seconds
        )
        # Optimized timeout configuration
        timeout = httpx.Timeout(
            connect=10.0,  # Connection timeout
            read=20.0,  # Read timeout
            write=10.0,  # Write timeout
            pool=5.0,  # Pool timeout for getting a connection
        )
        use_http2 = config.get(
            HTTP_CLIENT_USE_HTTP2_CONFIG_KEY, DEFAULT_HTTP_CLIENT_USE_HTTP2
        )
        self.httpx_client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            limits=limits,
            http2=use_http2,  # Configurable HTTP/2 support
            # Enable connection pooling optimizations
            trust_env=True,  # Trust proxy environment variables
        )  # HTTP client for attachments/web

        # Initialize content fetcher modules that need config
        from ..content.fetchers.youtube import (
            initialize_ytt_api,
            initialize_youtube_data_api,
        )
        from ..content.fetchers.reddit import initialize_reddit_client

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
        # --- ADDED: Register /enhanceprompt command ---
        self.tree.add_command(
            app_commands.Command(
                name="enhanceprompt",
                description="Enhances a given prompt using an LLM.",
                callback=enhance_prompt_command,
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

        # Determine if the bot should process this message using the new parser function
        should_process, original_content_for_processing = should_process_message(
            new_msg, self.user, allow_dms, is_dm
        )

        if not should_process:
            # --- ADDED: Prefix command handling BEFORE should_process check for other types of messages ---
            # Check for !enhanceprompt prefix command
            ENHANCE_CMD_PREFIX = "!enhanceprompt"
            if new_msg.content.startswith(ENHANCE_CMD_PREFIX):
                if not is_message_allowed(
                    new_msg, self.config, is_dm
                ):  # Still check permissions
                    logging.warning(
                        f"Blocked {ENHANCE_CMD_PREFIX} from user {new_msg.author.id} in channel {new_msg.channel.id} due to permissions."
                    )
                    return

                # Extract text after the command prefix, if any. Handles "!enhanceprompt" and "!enhanceprompt text"
                text_content_from_message = new_msg.content[
                    len(ENHANCE_CMD_PREFIX) :
                ].strip()

                prompt_text_for_prefix_cmd = ""
                processed_from_file = False
                attachment_filename = None
                initial_status_message = "⏳ Enhancing your prompt..."  # Default

                if new_msg.attachments:
                    for attachment in new_msg.attachments:
                        # Prioritize text files based on content_type.
                        # Also consider common text file extensions if content_type is generic.
                        is_text_content_type = (
                            attachment.content_type
                            and attachment.content_type.startswith("text/")
                        )
                        is_common_text_extension = attachment.filename.lower().endswith(
                            (
                                ".txt",
                                ".md",
                                ".py",
                                ".js",
                                ".html",
                                ".css",
                                ".json",
                                ".xml",
                                ".csv",
                            )
                        )

                        # Check for DOCX files
                        is_docx_file = (
                            attachment.content_type
                            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            or attachment.filename.lower().endswith(".docx")
                        )

                        if is_text_content_type or (
                            attachment.content_type == "application/octet-stream"
                            and is_common_text_extension
                        ):
                            try:
                                file_bytes = await attachment.read()
                                prompt_text_for_prefix_cmd = file_bytes.decode(
                                    "utf-8", errors="replace"
                                )
                                attachment_filename = attachment.filename
                                processed_from_file = True
                                initial_status_message = f"⏳ Enhancing prompt from file `{discord.utils.escape_markdown(attachment_filename)}`..."
                                logging.info(
                                    f"{ENHANCE_CMD_PREFIX}: Processing content from attachment: {attachment.filename}"
                                )
                                break  # Process first valid text file
                            except Exception as e:
                                logging.error(
                                    f"Error reading attachment {attachment.filename} for {ENHANCE_CMD_PREFIX}: {e}"
                                )
                                await new_msg.reply(
                                    f"Sorry, I couldn't read the file: {attachment.filename}.",
                                    mention_author=False,
                                )
                                return
                        elif is_docx_file:
                            try:
                                file_bytes = await attachment.read()
                                prompt_text_for_prefix_cmd = (
                                    await extract_text_from_docx_bytes(file_bytes)
                                )
                                if prompt_text_for_prefix_cmd:
                                    attachment_filename = attachment.filename
                                    processed_from_file = True
                                    initial_status_message = f"⏳ Enhancing prompt from DOCX file `{discord.utils.escape_markdown(attachment_filename)}`..."
                                    logging.info(
                                        f"{ENHANCE_CMD_PREFIX}: Processing content from DOCX attachment: {attachment.filename}"
                                    )
                                    break  # Process first valid DOCX file
                                else:
                                    logging.warning(
                                        f"No text content found in DOCX attachment {attachment.filename} for {ENHANCE_CMD_PREFIX}"
                                    )
                            except Exception as e:
                                logging.error(
                                    f"Error reading DOCX attachment {attachment.filename} for {ENHANCE_CMD_PREFIX}: {e}"
                                )
                                await new_msg.reply(
                                    f"Sorry, I couldn't read the DOCX file: {attachment.filename}.",
                                    mention_author=False,
                                )
                                return

                if (
                    not processed_from_file
                ):  # No suitable file processed, or no attachments
                    prompt_text_for_prefix_cmd = text_content_from_message

                # After attempting to get from file or text, check if we have a prompt
                if not prompt_text_for_prefix_cmd.strip():
                    if (
                        new_msg.attachments and not processed_from_file
                    ):  # Attachments were present but not suitable
                        await new_msg.reply(
                            f"No suitable text file found in attachments for `{ENHANCE_CMD_PREFIX}`. Please provide a prompt as text or attach a recognized text file (.txt, .py, .md, etc.) or DOCX file.",
                            mention_author=False,
                        )
                    else:  # No text and no attachments or no suitable file
                        await new_msg.reply(
                            f"Please provide a prompt to enhance (either as text after `{ENHANCE_CMD_PREFIX}` or as a text file attachment).",
                            mention_author=False,
                        )
                    return

                try:
                    await new_msg.reply(initial_status_message, mention_author=False)
                except discord.HTTPException as e:
                    logging.warning(
                        f"Failed to send initial processing message for {ENHANCE_CMD_PREFIX}: {e}"
                    )

                # Call the refactored logic
                await _execute_enhance_prompt_logic(
                    new_msg, prompt_text_for_prefix_cmd, self
                )

                # If an initial reply was sent and the logic function sent more messages,
                # we might want to delete the initial "Enhancing..." message if it's now redundant.
                # However, _execute_enhance_prompt_logic now handles all replies.
                # For simplicity, we'll leave the initial "Enhancing..." message if it was sent.
                # Or, _execute_enhance_prompt_logic could take an optional initial_message_to_edit.
                # For now, this is okay. The main responses will be replies.
                return  # Handled as a prefix command

            return  # If not a prefix command and not should_process for other reasons

        # --- Reload config & Check Global Reset ---
        # Config is now an instance variable, consider if reloading is needed per message

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
        # Use the new parser functions
        cleaned_content = clean_message_content(
            original_content_for_processing,
            self.user.mention if self.user else None,  # Pass bot mention if available
            is_dm,
        )
        logging.debug(f"Cleaned content for keyword/URL check: '{cleaned_content}'")

        image_attachments = [
            att
            for att in new_msg.attachments
            if att.content_type and att.content_type.startswith("image/")
        ]
        user_warnings = set()

        use_google_lens, cleaned_content, lens_warning = check_google_lens_trigger(
            cleaned_content, image_attachments, self.config
        )
        if lens_warning:
            user_warnings.add(lens_warning)
        if use_google_lens:
            logging.info(f"Google Lens keyword detected for message {new_msg.id}.")

        user_id = new_msg.author.id

        # --- Determine Potential Image URLs in Text (before model selection modifies cleaned_content further) ---
        has_potential_image_urls_in_text = False
        if cleaned_content:  # Check if cleaned_content is not empty
            urls_in_text_for_image_check = extract_urls_with_indices(cleaned_content)
            if any(
                is_image_url(url_info[0]) for url_info in urls_in_text_for_image_check
            ):
                has_potential_image_urls_in_text = True

        # Also check for image URLs in replied-to messages
        if (
            not has_potential_image_urls_in_text
            and new_msg.reference
            and new_msg.reference.message_id
        ):
            try:
                referenced_msg = new_msg.reference.cached_message
                if not referenced_msg:
                    referenced_msg = await new_msg.channel.fetch_message(
                        new_msg.reference.message_id
                    )
                if referenced_msg and referenced_msg.content:
                    urls_in_replied_msg = extract_urls_with_indices(
                        referenced_msg.content
                    )
                    if any(
                        is_image_url(url_info[0]) for url_info in urls_in_replied_msg
                    ):
                        has_potential_image_urls_in_text = True
            except (discord.NotFound, discord.HTTPException, Exception):
                pass  # If we can't fetch the message, just continue without it

        # --- LLM Provider/Model Selection using the new selector function ---
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
            initial_cleaned_content=cleaned_content,  # Pass content before lens keyword removal
            image_attachments=image_attachments,
            has_potential_image_urls_in_text=has_potential_image_urls_in_text,
            config=self.config,
            user_warnings=user_warnings,
        )

        # If Google Lens was triggered, it might have modified cleaned_content.
        # The model selection should ideally happen *before* lens keyword removal if lens influences model choice.
        # For now, determine_final_model uses initial_cleaned_content.
        # If deep_search was triggered by determine_final_model, and lens was also triggered,
        # the model_selector's logic for deep_search might need to be aware of lens to avoid conflict,
        # or the bot.py logic needs to ensure lens doesn't run if deep_search model is chosen and incompatible.
        # Current determine_final_model doesn't explicitly disable lens if deep_search is chosen.
        # We might need to re-evaluate `use_google_lens` if `determine_final_model` chose a non-vision model due to deep_search.
        if (
            use_google_lens and not accept_files
        ):  # If lens was on, but final model is not vision capable
            logging.warning(
                f"Google Lens was active, but final model '{final_provider_slash_model}' does not support vision. Disabling Lens for this request."
            )
            use_google_lens = False  # Turn off lens if the final model (e.g. deep search override) can't use it.
            # Remove lens keyword from content if it wasn't already by check_google_lens_trigger
            # This is a bit redundant if check_google_lens_trigger already did it, but safe.
            if GOOGLE_LENS_PATTERN.match(
                cleaned_content
            ):  # Check again on potentially already modified content
                cleaned_content = GOOGLE_LENS_PATTERN.sub(
                    "", cleaned_content
                ).strip()  # Corrected indentation

        # --- Configuration Values based on final model selection ---
        max_files_per_message = self.config.get("max_images", 5)
        # Use the new helper function to get model-specific max_text
        max_tokens_for_text_config = get_max_text_for_model(
            self.config, final_provider_slash_model
        )
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

        # --- Content Processing and Grounding ---
        (
            formatted_user_urls_content,
            formatted_google_lens_content,
            searxng_derived_context_str,
            url_fetch_results,  # Contains results from fetch_external_content
            custom_search_queries_generated_flag,
            successful_api_results_count,
            cleaned_content,  # Potentially modified by content_processor if image URLs were removed
        ) = await process_content_and_grounding(
            new_msg=new_msg,
            cleaned_content=cleaned_content,  # Pass the already cleaned content
            image_attachments=image_attachments,
            use_google_lens=use_google_lens,
            provider=provider,  # Pass current provider
            model_name=model_name,  # Pass current model name
            is_gemini_provider=is_gemini,  # Pass boolean for current provider
            is_grok_model=is_grok_model,  # Pass boolean for current model
            config=self.config,
            user_warnings=user_warnings,
            httpx_client=self.httpx_client,
            max_messages=max_messages,
            max_tokens_for_text_config=max_tokens_for_text_config,
            max_files_per_message_config=max_files_per_message,
            current_model_accepts_files=accept_files,
            msg_nodes_cache=self.msg_nodes,
            bot_user_obj=self.user,
            models_module=models,
            google_types_module=google_types,
        )
        # `cleaned_content` is updated by the call above if image URLs were processed and removed.

        history_for_llm = await build_message_history(
            new_msg=new_msg,
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
            msg_nodes_cache=self.msg_nodes,
            bot_user_obj=self.user,
            httpx_async_client=self.httpx_client,
            models_module=models,
            google_types_module=google_types,
            extract_text_from_pdf_bytes_func=extract_text_from_pdf_bytes,
            at_ai_pattern_re=AT_AI_PATTERN,
            providers_supporting_usernames_const=PROVIDERS_SUPPORTING_USERNAMES,
            system_prompt_text_for_budgeting=prepare_system_prompt(
                is_gemini,
                provider,
                get_user_system_prompt_preference(
                    new_msg.author.id, self.config.get("system_prompt")
                ),
            ),
            config=self.config,
        )

        if not history_for_llm:
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
            custom_search_queries_generated=custom_search_queries_generated_flag,  # New
            successful_api_results_count=successful_api_results_count,  # New
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
                        async with node.lock:
                            node.full_response_text = final_text
                    # Explicit lock release removed as context manager handles it.
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
                    # Lock handling removed here. If a node is being deleted,
                    # any operations holding its lock should have completed or been cancelled.
                    # The context manager at the point of lock acquisition would handle release.
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

    async def retry_with_modified_content(
        self, original_message: discord.Message, content_suffix: str
    ):
        """
        Retry processing a message with additional content appended.

        Args:
            original_message: The original Discord message to retry
            content_suffix: Text to append to the original message content
        """
        try:
            # Create a new message-like object with modified content
            # We'll reuse the original message but with modified content for processing
            modified_content = f"{original_message.content} {content_suffix}"

            # Create a temporary message object that mimics the original
            # but with the modified content
            class RetryMessage:
                def __init__(self, original_msg, new_content):
                    # Copy essential attributes
                    self.author = original_msg.author
                    self.channel = original_msg.channel
                    self.guild = original_msg.guild
                    self.id = original_msg.id  # Keep the same ID to avoid duplication
                    self.attachments = original_msg.attachments
                    self.reference = original_msg.reference
                    self.content = new_content  # Use the modified content
                    self.created_at = original_msg.created_at
                    self.edited_at = original_msg.edited_at
                    self.mention_everyone = original_msg.mention_everyone
                    self.mentions = original_msg.mentions
                    self.channel_mentions = original_msg.channel_mentions
                    self.role_mentions = original_msg.role_mentions
                    self.pinned = original_msg.pinned
                    self.flags = original_msg.flags
                    self.type = original_msg.type
                    self.system_content = original_msg.system_content
                    self.clean_content = (
                        new_content  # Use modified content for clean_content too
                    )

                    # Add missing attributes that the message processing expects
                    self.embeds = getattr(original_msg, "embeds", [])
                    self.reactions = getattr(original_msg, "reactions", [])
                    self.webhook_id = getattr(original_msg, "webhook_id", None)
                    self.application_id = getattr(original_msg, "application_id", None)
                    self.activity = getattr(original_msg, "activity", None)
                    self.application = getattr(original_msg, "application", None)
                    self.stickers = getattr(original_msg, "stickers", [])
                    self.components = getattr(original_msg, "components", [])
                    self.thread = getattr(original_msg, "thread", None)
                    self.interaction = getattr(original_msg, "interaction", None)
                    self.role_subscription = getattr(
                        original_msg, "role_subscription", None
                    )
                    self.resolved = getattr(original_msg, "resolved", None)
                    self.position = getattr(original_msg, "position", None)
                    self.poll = getattr(original_msg, "poll", None)
                    self.call = getattr(original_msg, "call", None)

                def reply(self, *args, **kwargs):
                    return self.channel.send(*args, **kwargs)

            retry_message = RetryMessage(original_message, modified_content)

            # Process the message using the existing on_message logic
            await self.on_message(retry_message)

        except Exception as e:
            logging.error(f"Error in retry_with_modified_content: {e}", exc_info=True)
            # Send an error message to the channel
            try:
                await original_message.channel.send(
                    f"❌ Failed to retry the request: {str(e)}"
                )
            except Exception:
                pass  # Ignore errors when sending error messages
