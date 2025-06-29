import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple

from google import genai as google_genai
from google.genai import types as google_types
from google.api_core import exceptions as google_api_exceptions

from ...core.constants import GEMINI_SAFETY_SETTINGS_CONFIG_KEY
from ...core.utils import _truncate_base64_in_payload, default_serializer


def _create_gemini_client(api_key: str) -> google_genai.Client:
    """Create a Gemini client with proper configuration."""
    return google_genai.Client(api_key=api_key)


def _process_history_to_contents(
    history_for_api_call: List[Dict[str, Any]],
) -> List[google_types.Content]:
    """Convert history format to Gemini Content objects."""
    gemini_contents = []

    for msg_data in history_for_api_call:
        role = msg_data["role"]
        parts_list = msg_data.get("parts", [])
        if not parts_list:
            logging.warning(f"Skipping message with no parts for Gemini: Role {role}")
            continue

        try:
            valid_parts = []
            for part_item in parts_list:
                if isinstance(part_item, google_types.Part):
                    valid_parts.append(part_item)
                elif isinstance(part_item, dict) and "text" in part_item:
                    valid_parts.append(google_types.Part.from_text(part_item["text"]))
                else:
                    logging.warning(
                        f"Skipping invalid part item for Gemini: {type(part_item)}"
                    )

            if valid_parts:
                gemini_contents.append(
                    google_types.Content(role=role, parts=valid_parts)
                )
            else:
                logging.warning(
                    f"Message with role '{role}' resulted in no valid parts for Gemini."
                )

        except Exception as e:
            logging.error(
                f"Failed to create google_types.Content for Gemini. Role: {role}",
                exc_info=True,
            )
            raise ValueError(f"Internal error creating Gemini content: {e}")

    if not gemini_contents:
        raise ValueError("No valid content to send to Gemini.")

    return gemini_contents


def _create_safety_settings(
    app_config: Dict[str, Any],
) -> List[google_types.SafetySetting]:
    """Create safety settings from app configuration."""
    safety_settings_from_config = app_config.get(GEMINI_SAFETY_SETTINGS_CONFIG_KEY, {})
    safety_settings_list = []

    for cat_str, thr_str in safety_settings_from_config.items():
        try:
            cat_enum = getattr(google_types.HarmCategory, cat_str.upper(), None)
            thr_enum = getattr(google_types.HarmBlockThreshold, thr_str.upper(), None)
            if cat_enum and thr_enum:
                safety_settings_list.append(
                    google_types.SafetySetting(category=cat_enum, threshold=thr_enum)
                )
        except Exception as e:
            logging.warning(f"Error processing safety setting {cat_str}={thr_str}: {e}")

    if not safety_settings_list:
        safety_settings_list = [
            google_types.SafetySetting(
                category=c, threshold=google_types.HarmBlockThreshold.BLOCK_NONE
            )
            for c in [
                google_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                google_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                google_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                google_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]
        ]

    return safety_settings_list


def _create_thinking_config(
    thinking_budget_val: Any,
) -> Optional[google_types.ThinkingConfig]:
    """Create thinking config from budget value."""
    if thinking_budget_val is None:
        return None

    budget_to_apply = None
    if isinstance(thinking_budget_val, int):
        budget_to_apply = thinking_budget_val
    elif isinstance(thinking_budget_val, str):
        try:
            budget_to_apply = int(thinking_budget_val)
        except ValueError:
            logging.warning(
                f"Non-integer string for thinking_budget ('{thinking_budget_val}')."
            )

    if budget_to_apply is not None and 0 <= budget_to_apply <= 24576:
        return google_types.ThinkingConfig(thinking_budget=budget_to_apply)
    elif budget_to_apply is not None:
        logging.warning(
            f"Invalid thinking_budget value ({budget_to_apply}). Must be 0-24576."
        )

    return None


def _log_payload(
    payload_to_log: Dict[str, Any], is_image_generation: bool = False
) -> None:
    """Log the API payload for debugging."""
    try:
        truncated_payload = _truncate_base64_in_payload(payload_to_log)

        if "config" in truncated_payload and isinstance(
            truncated_payload["config"], google_types.GenerateContentConfig
        ):
            config_dict = {}
            attrs = [
                "candidate_count",
                "stop_sequences",
                "max_output_tokens",
                "temperature",
                "top_p",
                "top_k",
                "safety_settings",
                "tools",
                "tool_config",
                "system_instruction",
                "thinking_config",
                "response_modalities",
            ]

            for attr_name in attrs:
                if hasattr(truncated_payload["config"], attr_name):
                    attr_value = getattr(truncated_payload["config"], attr_name)
                    if attr_value is not None:
                        config_dict[attr_name] = attr_value
            truncated_payload["config"] = config_dict

        prefix = "Gemini Image Generation" if is_image_generation else "Gemini"
        logging.info(
            f"--- {prefix} Payload ---\n{json.dumps(truncated_payload, indent=2, default=default_serializer)}\n{'-' * (len(prefix) + 15)}"
        )
    except Exception as e:
        logging.error(f"Error during Gemini payload logging: {e}", exc_info=True)


def _process_response_content(candidate) -> Tuple[str, Optional[bytes], Optional[str]]:
    """Extract text and image content from response candidate."""
    full_text = ""
    image_data = None
    image_mime_type = None

    if hasattr(candidate, "content") and candidate.content:
        if hasattr(candidate.content, "parts") and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    full_text += part.text
                elif hasattr(part, "inline_data") and part.inline_data:
                    image_data = part.inline_data.data
                    image_mime_type = part.inline_data.mime_type
                    logging.info(
                        f"Received image data: {len(image_data)} bytes, MIME type: {image_mime_type}"
                    )

    return full_text, image_data, image_mime_type


def _get_finish_reason(candidate) -> Optional[str]:
    """Extract finish reason from candidate."""
    if (
        hasattr(candidate, "finish_reason")
        and candidate.finish_reason
        and candidate.finish_reason
        != google_types.FinishReason.FINISH_REASON_UNSPECIFIED
    ):
        reason_map = {
            google_types.FinishReason.STOP: "stop",
            google_types.FinishReason.MAX_TOKENS: "length",
            google_types.FinishReason.SAFETY: "safety",
            google_types.FinishReason.RECITATION: "recitation",
            google_types.FinishReason.OTHER: "other",
        }
        return reason_map.get(candidate.finish_reason, str(candidate.finish_reason))
    return None


def _check_safety_ratings(candidate, has_content: bool = False) -> Optional[str]:
    """Check safety ratings and return error message if blocked."""
    if hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
        for rating in candidate.safety_ratings:
            if rating.probability in (
                google_types.HarmProbability.MEDIUM,
                google_types.HarmProbability.HIGH,
            ):
                if not has_content:
                    return f"Response Blocked (Safety Rating: {rating.category}={rating.probability})"
    return None


async def _cleanup_client(llm_client) -> None:
    """Clean up the Gemini client."""
    if llm_client is not None:
        try:
            if hasattr(llm_client, "_client") and hasattr(llm_client._client, "close"):
                await llm_client._client.close()
            elif hasattr(llm_client, "close"):
                await llm_client.close()
        except Exception as e:
            logging.warning(f"Error during Gemini client cleanup: {e}")


async def _make_gemini_request(
    llm_client,
    model_name: str,
    contents: List[google_types.Content],
    config: google_types.GenerateContentConfig,
) -> Any:
    """Make the actual API request to Gemini."""
    return await asyncio.wait_for(
        llm_client.aio.models.generate_content(
            model=model_name, contents=contents, config=config
        ),
        timeout=300.0,
    )


async def generate_gemini_stream(
    api_key: str,
    model_name: str,
    history_for_api_call: List[Dict[str, Any]],
    system_instruction_text: Optional[str],
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
    """Generate a response stream from the Google Gemini API."""
    llm_client = None
    try:
        llm_client = _create_gemini_client(api_key)
        gemini_contents = _process_history_to_contents(history_for_api_call)

        # Process parameters
        gemini_extra_params = extra_params.copy()
        if "max_tokens" in gemini_extra_params:
            gemini_extra_params["max_output_tokens"] = gemini_extra_params.pop(
                "max_tokens"
            )

        thinking_config = _create_thinking_config(
            gemini_extra_params.pop("thinking_budget", None)
        )
        safety_settings_list = _create_safety_settings(app_config)

        system_instruction_part = None
        if system_instruction_text:
            system_instruction_part = google_types.Part.from_text(
                text=system_instruction_text
            )

        api_generation_config = google_types.GenerateContentConfig(
            **gemini_extra_params,
            safety_settings=safety_settings_list,
            tools=[google_types.Tool(google_search=google_types.GoogleSearch())],
            thinking_config=thinking_config,
            system_instruction=system_instruction_part,
        )

        _log_payload(
            {
                "model": model_name,
                "contents": gemini_contents,
                "config": api_generation_config,
            }
        )

        response = await _make_gemini_request(
            llm_client, model_name, gemini_contents, api_generation_config
        )

        # Check for prompt feedback blocking
        if (
            hasattr(response, "prompt_feedback")
            and response.prompt_feedback
            and response.prompt_feedback.block_reason
        ):
            yield (
                None,
                "safety",
                None,
                f"Prompt Blocked ({response.prompt_feedback.block_reason})",
                None,
                None,
            )
            return

        # Process response
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            full_text, _, _ = _process_response_content(candidate)
            finish_reason_str = _get_finish_reason(candidate)
            grounding_meta = (
                getattr(candidate, "grounding_metadata", None)
                if hasattr(candidate, "grounding_metadata")
                else None
            )

            safety_error = _check_safety_ratings(candidate, bool(full_text))
            if safety_error:
                yield None, "safety", grounding_meta, safety_error, None, None
                return

            # Simulate streaming by yielding text in chunks
            if full_text:
                chunk_size = 50
                for i in range(0, len(full_text), chunk_size):
                    text_chunk = full_text[i : i + chunk_size]
                    is_final_chunk = i + chunk_size >= len(full_text)

                    yield (
                        text_chunk,
                        finish_reason_str if is_final_chunk else None,
                        grounding_meta if is_final_chunk else None,
                        None,
                        None,
                        None,
                    )
                    await asyncio.sleep(0.01)
            else:
                yield (
                    None,
                    finish_reason_str or "stop",
                    grounding_meta,
                    None,
                    None,
                    None,
                )

    except asyncio.TimeoutError:
        logging.error("Gemini request timed out after 5 minutes")
        yield None, None, None, "Request timeout error", None, None
    except ValueError as e:
        yield None, None, None, str(e), None, None
    except google_api_exceptions.GoogleAPIError as e:
        logging.error(f"Gemini API Error: {type(e).__name__} - {e}")
        yield None, None, None, f"Gemini API Error: {type(e).__name__}", None, None
    except Exception as e:
        logging.exception("Unexpected error in generate_gemini_stream")
        yield None, None, None, f"Unexpected error: {type(e).__name__}", None, None
    finally:
        await _cleanup_client(llm_client)


async def generate_gemini_image_stream(
    api_key: str,
    model_name: str,
    history_for_api_call: List[Dict[str, Any]],
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
    """Generate a response stream from the Gemini image generation API."""
    llm_client = None
    try:
        llm_client = _create_gemini_client(api_key)
        gemini_contents = _process_history_to_contents(history_for_api_call)

        # Process parameters (no thinking budget for image generation)
        gemini_extra_params = extra_params.copy()
        if "max_tokens" in gemini_extra_params:
            gemini_extra_params["max_output_tokens"] = gemini_extra_params.pop(
                "max_tokens"
            )
        gemini_extra_params.pop("thinking_budget", None)

        safety_settings_list = _create_safety_settings(app_config)

        api_generation_config = google_types.GenerateContentConfig(
            **gemini_extra_params,
            safety_settings=safety_settings_list,
            response_modalities=["TEXT", "IMAGE"],
        )

        _log_payload(
            {
                "model": model_name,
                "contents": gemini_contents,
                "config": api_generation_config,
            },
            is_image_generation=True,
        )

        response = await _make_gemini_request(
            llm_client, model_name, gemini_contents, api_generation_config
        )

        # Check for prompt feedback blocking
        if (
            hasattr(response, "prompt_feedback")
            and response.prompt_feedback
            and response.prompt_feedback.block_reason
        ):
            yield (
                None,
                "safety",
                None,
                f"Prompt Blocked ({response.prompt_feedback.block_reason})",
                None,
                None,
            )
            return

        # Process response
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            full_text, image_data, image_mime_type = _process_response_content(
                candidate
            )
            finish_reason_str = _get_finish_reason(candidate)

            safety_error = _check_safety_ratings(
                candidate, bool(full_text or image_data)
            )
            if safety_error:
                yield None, "safety", None, safety_error, None, None
                return

            # Yield complete response for image generation
            yield (
                full_text if full_text else None,
                finish_reason_str or "stop",
                None,
                None,
                image_data,
                image_mime_type,
            )

    except asyncio.TimeoutError:
        logging.error("Gemini image generation request timed out after 5 minutes")
        yield None, None, None, "Request timeout error", None, None
    except ValueError as e:
        yield None, None, None, str(e), None, None
    except google_api_exceptions.GoogleAPIError as e:
        logging.error(f"Gemini Image Generation API Error: {type(e).__name__} - {e}")
        yield (
            None,
            None,
            None,
            f"Gemini Image Generation API Error: {type(e).__name__}",
            None,
            None,
        )
    except Exception as e:
        logging.exception("Unexpected error in generate_gemini_image_stream")
        yield None, None, None, f"Unexpected error: {type(e).__name__}", None, None
    finally:
        await _cleanup_client(llm_client)
