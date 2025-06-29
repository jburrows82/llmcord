import logging
import random
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple

from datetime import date  # Specifically for enhance_prompt_with_llm

# OpenAI specific imports

# Google Gemini specific imports

from ..core.constants import (
    AllKeysFailedError,
    PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY,  # New
)
from ..core.rate_limiter import get_db_manager, get_available_keys

from .handler_utils import (
    format_history_for_gemini,
    format_history_for_openai,
    handle_compression_retry,
)


async def generate_response_stream(
    provider: str,
    model_name: str,
    history_for_llm: List[Dict[str, Any]],
    system_prompt_text: Optional[str],
    provider_config: Dict[str, Any],
    extra_params: Dict[str, Any],
    app_config: Dict[str, Any],
) -> AsyncGenerator[
    Tuple[
        Optional[str],
        Optional[str],
        Optional[Any],
        Optional[str],
        Optional[bytes],
        Optional[str],
    ],
    None,
]:
    """
    Generates a response stream from the specified LLM provider.

    Handles API key selection, retries, payload preparation, streaming, and error handling.
    """
    # Initialize compression tracking
    compression_occurred = False
    final_quality = 100
    final_resize = 1.0

    # Determine API keys and validate configuration
    is_imagen_model = provider == "google" and model_name.startswith("imagen-")
    all_api_keys = provider_config.get(
        "billed_api_keys" if is_imagen_model else "api_keys", []
    )
    base_url = provider_config.get("base_url")

    keys_required = provider not in ["ollama", "lmstudio", "vllm", "oobabooga", "jan"]

    if keys_required and not all_api_keys:
        raise ValueError(
            f"No API keys configured for the selected provider '{provider}' which requires keys."
        )

    available_llm_keys = await get_available_keys(provider, all_api_keys, app_config)
    llm_db_manager = await get_db_manager(provider)

    if keys_required and not available_llm_keys:
        raise AllKeysFailedError(
            provider, ["No available (non-rate-limited) API keys."]
        )

    random.shuffle(available_llm_keys)
    keys_to_loop = available_llm_keys if keys_required else ["dummy_key"]
    llm_errors = []

    # Try each API key
    for key_index, current_api_key in enumerate(keys_to_loop):
        key_display = (
            f"...{current_api_key[-4:]}"
            if current_api_key != "dummy_key"
            else "N/A (keyless)"
        )
        logging.info(
            f"Attempting LLM request with provider '{provider}' using key {key_display} ({key_index + 1}/{len(keys_to_loop)})"
        )

        # Format history for current provider
        is_gemini = provider == "google"
        current_history_for_api_call = (
            format_history_for_gemini(history_for_llm)
            if is_gemini
            else format_history_for_openai(history_for_llm)
        )

        # Handle compression retry logic
        try:
            async for result in handle_compression_retry(
                current_api_key,
                key_display,
                provider,
                model_name,
                current_history_for_api_call,
                system_prompt_text,
                extra_params,
                app_config,
                base_url,
                is_gemini,
                llm_db_manager,
                llm_errors,
                compression_occurred,
                final_quality,
                final_resize,
            ):
                if result is None:
                    # This key failed, try next key
                    break

                yield result
            else:
                # The async for completed without breaking, meaning stream finished successfully
                return
        except Exception as e:
            logging.error(f"Error in compression retry for key {key_display}: {e}")
            llm_errors.append(f"Key {key_display}: Unexpected error in retry logic")
            continue

    # All keys failed
    logging.error(
        f"All LLM API keys failed for provider '{provider}'. Errors: {json.dumps(llm_errors)}"
    )
    raise AllKeysFailedError(provider, llm_errors)


async def enhance_prompt_with_llm(
    prompt_to_enhance: str,
    prompt_design_strategies_doc: str,
    prompt_guide_2_doc: str,
    prompt_guide_3_doc: str,
    provider: str,
    model_name: str,
    provider_config: Dict[str, Any],
    extra_params: Dict[str, Any],
    app_config: Dict[str, Any],
) -> AsyncGenerator[
    Tuple[Optional[str], Optional[str], Optional[Any], Optional[str]], None
]:
    """
    Enhances a given prompt using a specified LLM and provided documentation.
    """
    enhancement_user_prompt_content = f"""Improve the following user prompt so it follows the prompt design strategies and prompt guides provided below.
Output only the improved prompt, without any preamble or explanation.

<my prompt>
{prompt_to_enhance}
</my prompt>

<prompt design strategies>
{prompt_design_strategies_doc}
</prompt design strategies>

<prompt guide 2>
{prompt_guide_2_doc}
</prompt guide 2>

<prompt guide 3>
{prompt_guide_3_doc}
</prompt guide 3>

Improved Prompt:"""

    history_for_enhancement_llm = [
        {"role": "user", "content": enhancement_user_prompt_content}
    ]

    logging.info(
        f"Attempting to enhance prompt using {provider}/{model_name} with instructions as user prompt."
    )

    # Use the existing generate_response_stream function
    async for chunk_tuple in generate_response_stream(
        provider=provider,
        model_name=model_name,
        history_for_llm=history_for_enhancement_llm,
        system_prompt_text=(
            f"{app_config.get(PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY, '')}\n"
            f"Current date: {date.today().strftime('%Y-%m-%d')}"
        ),
        provider_config=provider_config,
        extra_params=extra_params,
        app_config=app_config,
    ):
        # Extract only the first 4 elements from the 6-tuple returned by generate_response_stream
        text_chunk, finish_reason, grounding_metadata, error_message = chunk_tuple[:4]
        yield text_chunk, finish_reason, grounding_metadata, error_message
