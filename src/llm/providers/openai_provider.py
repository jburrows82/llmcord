import logging
import json  # Added for payload printing
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple

from openai import (
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    BadRequestError,
    UnprocessableEntityError,
)  # Removed RateLimitError

from ...core.constants import PROVIDERS_SUPPORTING_USERNAMES  # Relative import
from ...core.utils import (
    _truncate_base64_in_payload,
    default_serializer,
)  # Added for payload printing


async def generate_openai_stream(
    api_key: Optional[str],  # Key can be None for keyless local providers
    base_url: str,
    model_name: str,
    history_for_api_call: List[Dict[str, Any]],  # Already formatted for OpenAI
    system_prompt_text: Optional[str],
    extra_params: Dict[str, Any],
    current_provider_name: str,  # To check against PROVIDERS_SUPPORTING_USERNAMES
    # app_config is not strictly needed here unless there are OpenAI specific global settings
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
    Generates a response stream from an OpenAI-compatible API.

    Args:
        api_key: The API key (can be None for keyless).
        base_url: The base URL of the API.
        model_name: The specific model name.
        history_for_api_call: Conversation history formatted for OpenAI API.
        system_prompt_text: System prompt string.
        extra_params: Dictionary of extra API parameters.
        current_provider_name: The name of the current provider being called.

    Yields:
        Tuple containing:
        - text_chunk (Optional[str]): A chunk of text from the response stream.
        - finish_reason (Optional[str]): The reason the generation finished.
        - grounding_metadata (Optional[Any]): Always None for OpenAI.
        - error_message (Optional[str]): An error message if one occurred.
        - image_data (Optional[bytes]): Always None for OpenAI (no image generation).
        - image_mime_type (Optional[str]): Always None for OpenAI (no image generation).
    """
    if not base_url:
        yield (
            None,
            None,
            None,
            "Base URL is required for OpenAI-compatible provider but not found.",
            None,
            None,
        )
        return

    llm_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    openai_messages = [msg.copy() for msg in history_for_api_call]
    if system_prompt_text:
        # Ensure system prompt is at the beginning
        if not openai_messages or openai_messages[0].get("role") != "system":
            openai_messages.insert(0, {"role": "system", "content": system_prompt_text})
        elif openai_messages[0].get("role") == "system":
            openai_messages[0]["content"] = system_prompt_text  # Update if exists

    # Remove "name" field if provider doesn't support it
    if current_provider_name not in PROVIDERS_SUPPORTING_USERNAMES:
        for msg_data in openai_messages:
            if msg_data.get("role") == "user" and "name" in msg_data:
                del msg_data["name"]
                logging.debug(
                    f"Removed 'name' field for user message for provider '{current_provider_name}'"
                )

    api_call_params = extra_params.copy()
    api_call_params["stream"] = True  # Ensure streaming is enabled

    # --- Payload Logging ---
    try:
        # Construct the payload that will be sent to the API
        payload_to_log = {
            "model": model_name,
            "messages": openai_messages,
            **api_call_params,  # Spread other parameters like temperature, max_tokens, etc.
        }
        # The 'stream' parameter is already in api_call_params if set, otherwise defaults in create()

        truncated_payload = _truncate_base64_in_payload(payload_to_log)
        logging.info(
            f"--- OpenAI Payload (Provider: {current_provider_name}, Base URL: {base_url}) ---\n"
            f"{json.dumps(truncated_payload, indent=2, default=default_serializer)}\n"
            f"----------------------"
        )
    except Exception as e_log:
        logging.error(f"Error during OpenAI payload logging: {e_log}", exc_info=True)
    # --- End Payload Logging ---

    try:
        stream_response = await llm_client.chat.completions.create(
            model=model_name,
            messages=openai_messages,
            **api_call_params,
        )

        async for chunk in stream_response:
            text_chunk = None
            finish_reason_str = None
            error_msg_chunk = None

            if chunk.choices:
                delta = chunk.choices[0].delta
                finish_reason_str = chunk.choices[0].finish_reason
                if delta and delta.content:
                    text_chunk = delta.content

            yield (
                text_chunk,
                finish_reason_str,
                None,
                error_msg_chunk,
                None,
                None,
            )  # No grounding metadata or images for OpenAI
            if finish_reason_str:
                return

    except (
        APIError
    ) as e:  # Catches various OpenAI API errors including 413, RateLimitError
        logging.error(f"OpenAI API Error: {type(e).__name__} - {e}")
        error_detail = str(e)
        if hasattr(e, "status_code"):
            error_detail = f"Status {e.status_code}: {e}"

        # Specific handling for 413 to allow compression retry by the caller
        if hasattr(e, "status_code") and e.status_code == 413:
            yield (
                None,
                None,
                None,
                "OPENAI_API_ERROR_413_PAYLOAD_TOO_LARGE",
                None,
                None,
            )  # Signal for compression
            return

        yield (
            None,
            None,
            None,
            f"OpenAI API Error: {type(e).__name__} - {error_detail[:200]}",
            None,
            None,
        )
    except APIConnectionError as e:
        logging.error(f"OpenAI Connection Error: {e}")
        yield None, None, None, f"OpenAI Connection Error: {e}", None, None
    except UnprocessableEntityError as e:  # Specific for 422
        logging.error(f"OpenAI Unprocessable Entity Error (422): {e}")
        yield (
            None,
            None,
            None,
            "OPENAI_UNPROCESSABLE_ENTITY_422",
            None,
            None,
        )  # Specific signal
    except BadRequestError as e:  # Specific for 400
        logging.error(f"OpenAI Bad Request Error (400): {e}")
        error_body = str(e.body) if hasattr(e, "body") else str(e)
        yield (
            None,
            None,
            None,
            f"OpenAI Bad Request (400): {error_body[:200]}",
            None,
            None,
        )
    except Exception as e:
        logging.exception("Unexpected error in generate_openai_stream")
        yield None, None, None, f"Unexpected error: {type(e).__name__}", None, None
