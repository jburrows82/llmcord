import asyncio
import base64
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator, Tuple
import httpx
from ...core.constants import (
    GROUNDING_MODEL_CONFIG_KEY,
    GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY,
    GROUNDING_MODEL_TOP_K_CONFIG_KEY,
    GROUNDING_MODEL_TOP_P_CONFIG_KEY,
    DEFAULT_GROUNDING_MODEL_TEMPERATURE,
    DEFAULT_GROUNDING_MODEL_TOP_K,
    DEFAULT_GROUNDING_MODEL_TOP_P,
    GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY,
    GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY,
    GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET,
    GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE,
    AllKeysFailedError,
Config = Dict[str, Any]
async def get_web_search_queries_from_gemini(
    history_for_gemini_grounding: List[Dict[str, Any]],
    system_prompt_text_for_grounding: Optional[str],
    config: Dict[str, Any],
    generate_response_stream_func: Callable[
        [
            str,
            str,
            List[Dict[str, Any]],
            Optional[str],
            Dict[str, Any],
            Dict[str, Any],
            Dict[str, Any],
        ],
        AsyncGenerator[
            Tuple[
                Optional[str],
                Optional[str],
                Optional[Any],
                Optional[str],
                Optional[bytes],
                Optional[str],
            ],
            None,
        ],
    ],
) -> Optional[List[str]]:
    """
    Calls Gemini to get web search queries from its grounding metadata.
    Returns search queries immediately when found, without waiting for full response.
    """
    grounding_model_str = config.get(
        GROUNDING_MODEL_CONFIG_KEY, "google/gemini-2.5-flash"
    try:
        grounding_provider, grounding_model_name = grounding_model_str.split("/", 1)
    except ValueError:
        return None
    gemini_provider_config = config.get("providers", {}).get(grounding_provider, {})
    if not gemini_provider_config or not gemini_provider_config.get("api_keys"):
        return None
    all_web_search_queries = set()  # Use a set to store unique queries
    try:
        # Parameters for the grounding model call.
        grounding_temperature = config.get(
            GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY, DEFAULT_GROUNDING_MODEL_TEMPERATURE
        grounding_top_k = config.get(
            GROUNDING_MODEL_TOP_K_CONFIG_KEY, DEFAULT_GROUNDING_MODEL_TOP_K
        grounding_top_p = config.get(
            GROUNDING_MODEL_TOP_P_CONFIG_KEY, DEFAULT_GROUNDING_MODEL_TOP_P
        grounding_extra_params = {
            "temperature": grounding_temperature,
            "top_k": grounding_top_k,
            "top_p": grounding_top_p,
        }
        pass
        # Add thinking budget if configured and model is Gemini
        grounding_use_thinking_budget = config.get(
            GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY,
            GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET,
        if grounding_use_thinking_budget and grounding_provider == "google":
            grounding_thinking_budget_value = config.get(
                GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY,
                GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE,
            # Ensure the model is Flash, as thinkingBudget is only supported in Gemini 2.5 Flash
            if "flash" in grounding_model_name.lower():
                grounding_extra_params["thinking_budget"] = (
                    grounding_thinking_budget_value
            else:
        stream_generator = generate_response_stream_func(
            provider=grounding_provider,
            model_name=grounding_model_name,
            history_for_llm=history_for_gemini_grounding,
            system_prompt_text=system_prompt_text_for_grounding,
            provider_config=gemini_provider_config,
            extra_params=grounding_extra_params,
            app_config=config,  # Pass app_config
        async for (
            _,
            _,
            chunk_grounding_metadata,
            error_message,
            _,
            _,
        ) in stream_generator:
            if error_message:
                return None  # Abort on first error
            if chunk_grounding_metadata:
                if (
                    hasattr(chunk_grounding_metadata, "web_search_queries")
                    and chunk_grounding_metadata.web_search_queries
                ):
                    for query in chunk_grounding_metadata.web_search_queries:
                        if isinstance(query, str) and query.strip():
                            all_web_search_queries.add(query.strip())
                    pass
                    # Return search queries immediately when found
                    if all_web_search_queries:
                        return list(all_web_search_queries)
        # If we get here, no search queries were found
        return None
    except AllKeysFailedError as e:
        return None
    except Exception:
        return None
async def get_web_search_queries_from_gemini_force_stop(
    history_for_gemini_grounding: List[Dict[str, Any]],
    system_prompt_text_for_grounding: Optional[str],
    config: Dict[str, Any],
    generate_response_stream_func: Callable[
        [
            str,
            str,
            List[Dict[str, Any]],
            Optional[str],
            Dict[str, Any],
            Dict[str, Any],
            Dict[str, Any],
        ],
        AsyncGenerator[
            Tuple[
                Optional[str],
                Optional[str],
                Optional[Any],
                Optional[str],
                Optional[bytes],
                Optional[str],
            ],
            None,
        ],
    ],
) -> Optional[List[str]]:
    """
    Calls Gemini to get web search queries from its grounding metadata.
    Forcefully stops the stream immediately after getting search queries to minimize API usage.
    """
    grounding_model_str = config.get(
        GROUNDING_MODEL_CONFIG_KEY, "google/gemini-2.5-flash"
    try:
        grounding_provider, grounding_model_name = grounding_model_str.split("/", 1)
    except ValueError:
        return None
    gemini_provider_config = config.get("providers", {}).get(grounding_provider, {})
    if not gemini_provider_config or not gemini_provider_config.get("api_keys"):
        return None
    all_web_search_queries = set()  # Use a set to store unique queries
    try:
        # Parameters for the grounding model call.
        grounding_temperature = config.get(
            GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY, DEFAULT_GROUNDING_MODEL_TEMPERATURE
        grounding_top_k = config.get(
            GROUNDING_MODEL_TOP_K_CONFIG_KEY, DEFAULT_GROUNDING_MODEL_TOP_K
        grounding_top_p = config.get(
            GROUNDING_MODEL_TOP_P_CONFIG_KEY, DEFAULT_GROUNDING_MODEL_TOP_P
        grounding_extra_params = {
            "temperature": grounding_temperature,
            "top_k": grounding_top_k,
            "top_p": grounding_top_p,
        }
        # Add thinking budget if configured and model is Gemini
        grounding_use_thinking_budget = config.get(
            GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY,
            GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET,
        if grounding_use_thinking_budget and grounding_provider == "google":
            grounding_thinking_budget_value = config.get(
                GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY,
                GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE,
            # Ensure the model is Flash, as thinkingBudget is only supported in Gemini 2.5 Flash
            if "flash" in grounding_model_name.lower():
                grounding_extra_params["thinking_budget"] = (
                    grounding_thinking_budget_value
            else:
        stream_generator = generate_response_stream_func(
            provider=grounding_provider,
            model_name=grounding_model_name,
            history_for_llm=history_for_gemini_grounding,
            system_prompt_text=system_prompt_text_for_grounding,
            provider_config=gemini_provider_config,
            extra_params=grounding_extra_params,
            app_config=config,  # Pass app_config
        # Use asyncio.wait_for with a timeout to force-stop if queries are found
        try:
            async for (
                _,
                _,
                chunk_grounding_metadata,
                error_message,
                _,
                _,
            ) in stream_generator:
                if error_message:
                    return None  # Abort on first error
                if chunk_grounding_metadata:
                    if (
                        hasattr(chunk_grounding_metadata, "web_search_queries")
                        and chunk_grounding_metadata.web_search_queries
                    ):
                        for query in chunk_grounding_metadata.web_search_queries:
                            if isinstance(query, str) and query.strip():
                                all_web_search_queries.add(query.strip())
                        pass
                        # Force stop by breaking and closing the generator
                        if all_web_search_queries:
                            # Try to close the generator to stop the stream
                            try:
                                await stream_generator.aclose()
                            except Exception as e:
                            return list(all_web_search_queries)
        except asyncio.CancelledError:
            if all_web_search_queries:
                return list(all_web_search_queries)
            return None
        # If we get here, no search queries were found
        return None
    except AllKeysFailedError as e:
        return None
    except Exception:
        return None
async def generate_search_queries_with_custom_prompt(
    latest_query: str,
    chat_history: List[Dict[str, Any]],
    config: Config,
    generate_response_stream_func: Callable[
        [
            str,
            str,
            List[Dict[str, Any]],
            Optional[str],
            Dict[str, Any],
            Dict[str, Any],
            Config,
        ],
        AsyncGenerator[
            Tuple[
                Optional[str],
                Optional[str],
                Optional[Any],
                Optional[str],
                Optional[bytes],
                Optional[str],
            ],
            None,
        ],
    ],
    current_model_id: str,
    httpx_client: httpx.AsyncClient,
    image_urls: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generates search queries using a custom prompt with a non-Gemini model, potentially with images.
    Returns a dict with 'web_search_required' and 'search_queries' keys.
    """
    alt_search_config_dict = config.get("alternative_search_query_generation", {})
    prompt_template = alt_search_config_dict.get(
        "search_query_generation_prompt_template", ""
    system_prompt_template = alt_search_config_dict.get(
        "search_query_generation_system_prompt", ""
    if not prompt_template:
        return None
    now = datetime.now()
    current_date_str = now.strftime("%Y-%m-%d")
    current_day_of_week_str = now.strftime("%A")
    current_time_str = now.strftime("%I:%M %p")
    # Prepare system prompt
    final_system_prompt_text = None
    if system_prompt_template:
        final_system_prompt_text = system_prompt_template.replace(
            "{current_date}", current_date_str
    # Format Chat History
    formatted_chat_history_parts = []
    history_to_format = chat_history[:-1]  # History leading up to the latest query
    for message_dict in history_to_format:
        role = message_dict.get("role")
        text_content = ""
        if role == "model" or role == "assistant":
            role_for_display = "assistant"
        elif role == "user":
            role_for_display = "user"
        else:
            continue
        # Extract text content from message_dict
        if "parts" in message_dict:  # Likely Gemini-style
            for part in message_dict["parts"]:
                if hasattr(part, "text") and part.text:
                    text_content = part.text
                    break
        elif "content" in message_dict:  # Likely OpenAI-style
            if isinstance(message_dict["content"], str):
                text_content = message_dict["content"]
            elif isinstance(message_dict["content"], list):
                for item_part in message_dict["content"]:
                    if isinstance(item_part, dict) and item_part.get("type") == "text":
                        text_content = item_part.get("text", "")
                        break
        if text_content.strip():
            formatted_chat_history_parts.append(
                f"{role_for_display}: {text_content.strip()}"
    formatted_chat_history_string = "\n\n".join(formatted_chat_history_parts)
    # Prepare the prompt text
    current_prompt_text = prompt_template.replace("{latest_query}", latest_query)
    current_prompt_text = current_prompt_text.replace("{current_date}", current_date_str)
    current_prompt_text = current_prompt_text.replace("{current_day_of_week}", current_day_of_week_str)
    current_prompt_text = current_prompt_text.replace("{current_time}", current_time_str)
    # Inject formatted chat history
    if "{chat_history}" in current_prompt_text:
        final_prompt_text = current_prompt_text.replace(
            "{chat_history}", formatted_chat_history_string
    else:
        final_prompt_text = current_prompt_text
    # Process Images
    processed_image_data_urls = []
    if image_urls and httpx_client:
        for img_url in image_urls:
            try:
                response = await httpx_client.get(img_url, timeout=10)
                response.raise_for_status()
                image_bytes = await response.aread()
                mime_type = response.headers.get("Content-Type", "image/jpeg")
                if not mime_type.startswith("image/"):
                    continue
                base64_encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                data_url = f"data:{mime_type};base64,{base64_encoded_image}"
                processed_image_data_urls.append(data_url)
            except Exception as e:
    # Construct user_prompt_content_parts for the API call
    user_prompt_content_parts_for_api = [{"type": "text", "text": final_prompt_text}]
    pass
    if processed_image_data_urls:
        for data_url in processed_image_data_urls:
            user_prompt_content_parts_for_api.append(
                {"type": "image_url", "image_url": {"url": data_url}}
    # Construct messages_for_llm as a single user turn
    messages_for_llm = [{"role": "user", "content": user_prompt_content_parts_for_api}]
    try:
        provider_name, model_name = current_model_id.split("/", 1)
    except ValueError:
        return None
    provider_config = config.get("providers", {}).get(provider_name, {})
    if not provider_config or not provider_config.get("api_keys"):
        return None
    final_response_text = ""
    try:
        extra_params = {}
        # Call the passed stream generation function
        stream_generator = generate_response_stream_func(
            provider=provider_name,
            model_name=model_name,
            history_for_llm=messages_for_llm,
            system_prompt_text=final_system_prompt_text,
            provider_config=provider_config,
            extra_params=extra_params,
            app_config=config,
        async for text_chunk, _, _, error_message, _, _ in stream_generator:
            if error_message:
                return None
            if text_chunk:
                final_response_text += text_chunk
        if not final_response_text:
            return None
        # Process the response
        content_to_process = final_response_text.strip()
        # Markdown stripping logic
        if content_to_process.startswith("```json"):
            content_to_process = content_to_process[len("```json") :].strip()
        elif content_to_process.startswith("```"):
            content_to_process = content_to_process[len("```") :].strip()
        if content_to_process.endswith("```"):
            content_to_process = content_to_process[: -len("```")].strip()
        # Parse as JSON and expect the new structure
        try:
            parsed_response = json.loads(content_to_process)
            if (
                isinstance(parsed_response, dict)
                and "web_search_required" in parsed_response
                and isinstance(parsed_response["web_search_required"], bool)
            ):
                if parsed_response["web_search_required"]:
                    search_queries = parsed_response.get("search_queries")
                    if isinstance(search_queries, list) and all(
                        isinstance(q, str) for q in search_queries
                    ):
                        return {
                            "web_search_required": True,
                            "search_queries": search_queries,
                        }
                    else:
                        return {
                            "web_search_required": True,
                            "search_queries": [],
                        }
                else:
                    return {"web_search_required": False}
            else:
                return {"web_search_required": False}
        except json.JSONDecodeError:
            return {"web_search_required": False}
    except AllKeysFailedError as e:
        return None
    except Exception:
        return None 