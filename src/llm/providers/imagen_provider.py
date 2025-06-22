import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple

from google import genai as google_genai
from google.genai import types as google_types
from google.api_core import exceptions as google_api_exceptions

StreamReturnType = Tuple[
    Optional[str],
    Optional[str],
    Optional[Any],
    Optional[str],
    Optional[bytes],
    Optional[str],
]


async def generate_imagen_image_stream(
    api_key: str,
    model_name: str,
    history_for_api_call: List[Dict[str, Any]],
    extra_params: Dict[str, Any],
    app_config: Dict[str, Any],
) -> AsyncGenerator[StreamReturnType, None]:
    """Generate images with Google Imagen 3 and yield results in the unified stream format.

    The Imagen API does **not** support server-side streaming, so this function makes a
    single request and then yields each returned image as an independent chunk so that
    downstream logic (already built around async generators) continues to work without
    modification.

    Args:
        api_key: Google Generative AI API key.
        model_name: Imagen model identifier (e.g. ``"imagen-3.0-generate-002"``).
        history_for_api_call: Chat history prepared by the handler. For Imagen we
            collapse this into a single text prompt (the latest user message).
        extra_params: Additional parameters such as ``number_of_images``,
            ``aspect_ratio`` or ``person_generation``. They are forwarded to
            ``GenerateImagesConfig`` when present.
        app_config: Complete application config (currently unused).
    """

    llm_client = google_genai.Client(api_key=api_key)

    prompt_text = ""
    for msg_data in reversed(history_for_api_call):
        if msg_data.get("role") == "user":
            parts = msg_data.get("parts", [])
            candidate_text_parts = []
            for part in parts:
                if isinstance(part, google_types.Part) and getattr(part, "text", None):
                    candidate_text_parts.append(part.text)
                elif isinstance(part, dict) and part.get("text") is not None:
                    candidate_text_parts.append(part["text"])
            if candidate_text_parts:
                prompt_text = " ".join(candidate_text_parts)
                break
    if not prompt_text:
        combined_parts = []
        for msg_data in history_for_api_call:
            parts = msg_data.get("parts", [])
            for part in parts:
                if isinstance(part, google_types.Part) and getattr(part, "text", None):
                    combined_parts.append(part.text)
                elif isinstance(part, dict) and part.get("text") is not None:
                    combined_parts.append(part["text"])
        prompt_text = " ".join(combined_parts)[:480]

    if not prompt_text:
        error_msg = "No suitable text found to build an Imagen prompt."
        logging.error(error_msg)
        yield None, None, None, error_msg, None, None
        return

    imagen_cfg_kwargs = {}
    for key in ("number_of_images", "aspect_ratio", "person_generation"):
        if key in extra_params and extra_params[key] is not None:
            imagen_cfg_kwargs[key] = extra_params[key]

    imagen_config_obj = (
        google_types.GenerateImagesConfig(**imagen_cfg_kwargs)
        if imagen_cfg_kwargs
        else None
    )

    try:
        response = None
        try:
            response = await llm_client.aio.models.generate_images(
                model=model_name,
                prompt=prompt_text,
                config=imagen_config_obj,
            )
        except AttributeError:
            import asyncio
            from typing import Any

            def _sync_generate() -> Any:
                return llm_client.models.generate_images(
                    model=model_name,
                    prompt=prompt_text,
                    config=imagen_config_obj,
                )

            response = await asyncio.to_thread(_sync_generate)

        for generated_image in response.generated_images:
            try:
                image_bytes = generated_image.image.image_bytes
                mime_type = getattr(generated_image.image, "mime_type", "image/png")
            except AttributeError:
                image_bytes = getattr(generated_image, "image_bytes", None)
                mime_type = "image/png"

            if not image_bytes:
                logging.warning(
                    "Received a generated_image without image bytes – skipping this entry."
                )
                continue

            logging.info(
                f"Imagen returned {len(image_bytes)} bytes (MIME type: {mime_type})."
            )
            yield None, "stop", None, None, image_bytes, mime_type

        return

    except google_api_exceptions.GoogleAPIError as e:
        logging.error(f"Imagen API Error: {type(e).__name__} - {e}")
        yield None, None, None, f"Imagen API Error: {type(e).__name__}", None, None
    except Exception as e:
        logging.exception("Unexpected error in generate_imagen_image_stream")
        yield None, None, None, f"Unexpected error: {type(e).__name__}", None, None
