import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple

from google import genai as google_genai
from google.genai import types as google_types
from google.api_core import exceptions as google_api_exceptions

from ..constants import GEMINI_SAFETY_SETTINGS_CONFIG_KEY # Relative import

async def generate_gemini_stream(
    api_key: str,
    model_name: str,
    history_for_api_call: List[Dict[str, Any]], # Already formatted for Gemini
    system_instruction_text: Optional[str], # Gemini uses system_instruction
    extra_params: Dict[str, Any],
    app_config: Dict[str, Any], # For safety settings, etc.
) -> AsyncGenerator[Tuple[Optional[str], Optional[str], Optional[Any], Optional[str]], None]:
    """
    Generates a response stream from the Google Gemini API.

    Args:
        api_key: The API key for Gemini.
        model_name: The specific Gemini model name.
        history_for_api_call: Conversation history formatted for Gemini API.
        system_instruction_text: System instruction for Gemini.
        extra_params: Dictionary of extra API parameters for the model.
        app_config: The main application configuration dictionary.

    Yields:
        Tuple containing:
        - text_chunk (Optional[str]): A chunk of text from the response stream.
        - finish_reason (Optional[str]): The reason the generation finished.
        - grounding_metadata (Optional[Any]): Grounding metadata.
        - error_message (Optional[str]): An error message if one occurred.
    """
    llm_client = google_genai.Client(api_key=api_key)
    gemini_contents = []

    for msg_data in history_for_api_call:
        role = msg_data["role"]
        parts_list = msg_data.get("parts", [])
        if not parts_list:
            logging.warning(f"Skipping message with no parts for Gemini: Role {role}")
            continue
        try:
            # Ensure all parts are valid google_types.Part instances
            valid_parts = []
            for part_item in parts_list:
                if isinstance(part_item, google_types.Part):
                    valid_parts.append(part_item)
                elif isinstance(part_item, dict) and "text" in part_item: # Simple text dict
                     valid_parts.append(google_types.Part.from_text(part_item["text"]))
                # Add more conversions if other dict formats are expected for parts
                else:
                    logging.warning(f"Skipping invalid part item for Gemini: {type(part_item)}")
            
            if valid_parts:
                gemini_contents.append(google_types.Content(role=role, parts=valid_parts))
            else:
                logging.warning(f"Message with role '{role}' resulted in no valid parts for Gemini.")

        except Exception as content_creation_error:
            logging.error(
                f"Failed to create google_types.Content for Gemini. Role: {role}, Parts: {parts_list}",
                exc_info=True,
            )
            yield None, None, None, f"Internal error creating Gemini content: {content_creation_error}"
            return

    if not gemini_contents:
        logging.error("Gemini contents list is empty after processing history. Cannot make API call.")
        yield None, None, None, "Internal error: No valid content to send to Gemini."
        return

    gemini_extra_params = extra_params.copy()
    if "max_tokens" in gemini_extra_params: # Gemini uses max_output_tokens
        gemini_extra_params["max_output_tokens"] = gemini_extra_params.pop("max_tokens")
    
    thinking_budget_val = gemini_extra_params.pop("thinking_budget", None)
    thinking_config = None
    if thinking_budget_val is not None:
        budget_to_apply: Optional[int] = None
        if isinstance(thinking_budget_val, int):
            budget_to_apply = thinking_budget_val
        elif isinstance(thinking_budget_val, str):
            try:
                budget_to_apply = int(thinking_budget_val)
            except ValueError:
                logging.warning(f"Non-integer string for thinking_budget ('{thinking_budget_val}').")
        # This 'else' was missing from the previous diff, which might have caused issues.
        # It should correctly handle the case where thinking_budget_val is neither int nor str.
        # However, the original code didn't have an explicit else for the outer if,
        # it was for the isinstance(thinking_budget_val, str)
        # The original logic was:
        # if isinstance(int): ...
        # elif isinstance(str): try/except
        # else: (this was missing, but implied by falling through)
        # The corrected logic below should be fine.
        if budget_to_apply is not None and 0 <= budget_to_apply <= 24576:
            thinking_config = google_types.ThinkingConfig(thinking_budget=budget_to_apply)
        elif budget_to_apply is not None:
            logging.warning(f"Invalid thinking_budget value ({budget_to_apply}). Must be 0-24576.")

    safety_settings_from_config = app_config.get(GEMINI_SAFETY_SETTINGS_CONFIG_KEY, {})
    safety_settings_list = []
    for cat_str, thr_str in safety_settings_from_config.items():
        try:
            cat_enum = getattr(google_types.HarmCategory, cat_str.upper(), None)
            thr_enum = getattr(google_types.HarmBlockThreshold, thr_str.upper(), None)
            if cat_enum and thr_enum:
                safety_settings_list.append(google_types.SafetySetting(category=cat_enum, threshold=thr_enum))
        except Exception as e_safe: # Ensure this except is aligned with its try
            logging.warning(f"Error processing safety setting {cat_str}={thr_str}: {e_safe}")
    
    if not safety_settings_list:  # Default if none valid from config
        safety_settings_list = [
            google_types.SafetySetting(category=c, threshold=google_types.HarmBlockThreshold.BLOCK_NONE)
            for c in [
                google_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                google_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                google_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                google_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]
        ]

    api_generation_config = google_types.GenerateContentConfig(
        **gemini_extra_params,
        safety_settings=safety_settings_list,
        tools=[google_types.Tool(google_search=google_types.GoogleSearch())],
        thinking_config=thinking_config,
    )
    
    # system_instruction is now part of GenerateContentConfig
    if system_instruction_text:
        api_generation_config.system_instruction = google_types.Part.from_text(text=system_instruction_text)

    try:
        stream_response = await llm_client.aio.models.generate_content_stream(
            model=model_name,
            contents=gemini_contents,
            config=api_generation_config, # Changed from generation_config and removed system_instruction
            # system_instruction=system_instruction_part, # Removed, it's in api_generation_config
        )

        async for chunk in stream_response:
            text_chunk = None
            finish_reason_str = None
            grounding_meta = None
            error_msg_chunk = None

            if hasattr(chunk, "prompt_feedback") and chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                error_msg_chunk = f"Prompt Blocked ({chunk.prompt_feedback.block_reason})"
                finish_reason_str = "safety" # Treat as a safety finish
                yield text_chunk, finish_reason_str, grounding_meta, error_msg_chunk
                return # Stop generation

            if hasattr(chunk, "text") and chunk.text:
                text_chunk = chunk.text
            
            if hasattr(chunk, "candidates") and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, "finish_reason") and candidate.finish_reason and \
                   candidate.finish_reason != google_types.FinishReason.FINISH_REASON_UNSPECIFIED:
                    reason_map = {
                        google_types.FinishReason.STOP: "stop",
                        google_types.FinishReason.MAX_TOKENS: "length",
                        google_types.FinishReason.SAFETY: "safety",
                        google_types.FinishReason.RECITATION: "recitation",
                        google_types.FinishReason.OTHER: "other",
                    }
                    finish_reason_str = reason_map.get(candidate.finish_reason, str(candidate.finish_reason))
                
                if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                    grounding_meta = candidate.grounding_metadata
                
                if hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                    for rating in candidate.safety_ratings:
                        if rating.probability in (google_types.HarmProbability.MEDIUM, google_types.HarmProbability.HIGH):
                            if not text_chunk: # If blocked before any content
                                error_msg_chunk = f"Response Blocked (Safety Rating: {rating.category}={rating.probability})"
                                finish_reason_str = "safety"
                                yield None, finish_reason_str, grounding_meta, error_msg_chunk
                                return # Stop generation

            yield text_chunk, finish_reason_str, grounding_meta, error_msg_chunk
            if finish_reason_str: # If a finish reason is determined, stop.
                return

    except google_api_exceptions.GoogleAPIError as e:
        logging.error(f"Gemini API Error: {type(e).__name__} - {e}")
        yield None, None, None, f"Gemini API Error: {type(e).__name__}"
    except Exception as e:
        logging.exception("Unexpected error in generate_gemini_stream")
        yield None, None, None, f"Unexpected error: {type(e).__name__}"