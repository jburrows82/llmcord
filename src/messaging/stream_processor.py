import asyncio
from typing import List, Dict, Optional, Any, Set
import discord
from ..core.constants import (
    FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY,
    FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY,
    AllKeysFailedError,
)
from ..llm.handler import generate_response_stream
from ..services.prompt_utils import prepare_system_prompt
from .message_editor import MessageEditor
from .error_handler import StreamErrorHandler
from .image_handler import ImageHandler


class StreamProcessor:
    def __init__(self, client, app_config: Dict[str, Any]):
        self.client = client
        self.app_config = app_config
        self.message_editor = MessageEditor(client, app_config)
        self.error_handler = StreamErrorHandler(client, app_config)
        self.image_handler = ImageHandler(client, app_config)

    async def process_stream(
        self,
        new_msg: discord.Message,
        processing_msg: Optional[discord.Message],
        provider: str,
        model_name: str,
        history_for_llm: List[Dict[str, Any]],
        system_prompt_text: Optional[str],
        provider_config: Dict[str, Any],
        extra_api_params: Dict[str, Any],
        initial_user_warnings: Set[str],
        use_plain_responses_config: bool,
        split_limit_config: int,
        custom_search_queries_generated: bool,
        successful_api_results_count: int,
        deep_search_used: bool = False,
    ):
        """Process the LLM response stream with retry logic."""
        response_msgs: List[discord.Message] = []
        final_text_to_return = ""
        llm_call_successful_final = False
        # Store original parameters for retry
        original_params = {
            "provider": provider,
            "model_name": model_name,
            "provider_config": provider_config.copy(),
            "extra_api_params": extra_api_params.copy(),
            "system_prompt_text": system_prompt_text,
        }
        # Retry loop
        for attempt_num in range(2):
            retry_context = await self._setup_retry(
                attempt_num,
                original_params,
                initial_user_warnings,
                processing_msg,
                response_msgs,
                new_msg,
            )
            if not retry_context:
                break
            current_params, should_retry_flags = retry_context
            # Process single attempt
            attempt_result = await self._process_single_attempt(
                new_msg,
                processing_msg,
                response_msgs,
                current_params,
                initial_user_warnings,
                use_plain_responses_config,
                split_limit_config,
                custom_search_queries_generated,
                successful_api_results_count,
                deep_search_used,
                history_for_llm,
                attempt_num,
            )
            if attempt_result["should_retry"]:
                continue
            llm_call_successful_final = attempt_result["success"]
            final_text_to_return = attempt_result["text"]
            response_msgs = attempt_result["response_msgs"]
            break
        return llm_call_successful_final, final_text_to_return, response_msgs

    async def _setup_retry(
        self,
        attempt_num,
        original_params,
        initial_user_warnings,
        processing_msg,
        response_msgs,
        new_msg,
    ):
        """Setup retry parameters and clean up incomplete messages if needed."""
        if attempt_num == 0:
            return original_params, {}
        # Handle retry setup
        fallback_model_str = self.client.config.get(
            FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY,
            "google/gemini-2.5-flash",
        )
        # Clean up incomplete messages
        await self._cleanup_incomplete_messages(processing_msg, response_msgs)
        # Setup fallback parameters
        try:
            provider, model_name = fallback_model_str.split("/", 1)
        except ValueError:
            return None
        provider_config = self.client.config.get("providers", {}).get(provider, {})
        if not provider_config or not provider_config.get("api_keys"):
            return None
        extra_params = self.client.config.get("extra_api_parameters", {}).copy()
        system_prompt_text = prepare_system_prompt(
            provider == "google",
            provider,
            self.client.config.get(FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY),
        )
        current_params = {
            "provider": provider,
            "model_name": model_name,
            "provider_config": provider_config,
            "extra_api_params": extra_params,
            "system_prompt_text": system_prompt_text,
        }
        return current_params, {}

    async def _cleanup_incomplete_messages(self, processing_msg, response_msgs):
        """Clean up incomplete messages before retry."""
        if response_msgs:
            temp_msgs = list(response_msgs)
            response_msgs.clear()
            for msg_to_delete in reversed(temp_msgs):
                try:
                    await msg_to_delete.delete()
                    await asyncio.sleep(0.2)
                except discord.NotFound:
                    pass
                except discord.HTTPException:
                    pass
                finally:
                    if msg_to_delete.id in self.client.msg_nodes:
                        self.client.msg_nodes.pop(msg_to_delete.id, None)

    async def _process_single_attempt(
        self,
        new_msg,
        processing_msg,
        response_msgs,
        current_params,
        initial_user_warnings,
        use_plain_responses_config,
        split_limit_config,
        custom_search_queries_generated,
        successful_api_results_count,
        deep_search_used,
        history_for_llm,
        attempt_num,
    ):
        """Process a single stream attempt."""
        final_text = ""
        success = False
        should_retry = False
        # Check if this is an image generation model
        is_image_model = current_params["provider"] == "google" and (
            current_params["model_name"] == "gemini-2.0-flash-preview-image-generation"
            or current_params["model_name"].startswith("imagen-")
        )
        try:
            # Generate response stream
            stream_generator = generate_response_stream(
                provider=current_params["provider"],
                model_name=current_params["model_name"],
                history_for_llm=history_for_llm,
                system_prompt_text=current_params["system_prompt_text"],
                provider_config=current_params["provider_config"],
                extra_params=current_params["extra_api_params"],
                app_config=self.app_config,
            )
            # Process stream
            if is_image_model:
                result = await self.image_handler.process_image_stream(
                    stream_generator, new_msg, processing_msg, response_msgs, final_text
                )
            else:
                result = await self.message_editor.process_text_stream(
                    stream_generator,
                    new_msg,
                    processing_msg,
                    response_msgs,
                    current_params,
                    initial_user_warnings,
                    use_plain_responses_config,
                    split_limit_config,
                    custom_search_queries_generated,
                    successful_api_results_count,
                    deep_search_used,
                    attempt_num,
                )
            success = result.get("success", False)
            final_text = result.get("text", "")
            response_msgs = result.get("response_msgs", response_msgs)
            should_retry = result.get("should_retry", False)
        except AllKeysFailedError as e:
            pass
            if attempt_num == 0:
                should_retry = True
            else:
                await self.error_handler.handle_all_keys_failed(
                    new_msg,
                    processing_msg,
                    response_msgs,
                    e,
                    use_plain_responses_config,
                )
        except Exception as e:
            await self.error_handler.handle_unexpected_error(
                new_msg, processing_msg, response_msgs, e, use_plain_responses_config
            )
        return {
            "success": success,
            "text": final_text,
            "response_msgs": response_msgs,
            "should_retry": should_retry,
        }
