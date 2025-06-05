import logging
import base64
import io
from typing import List, Dict, Any, Tuple
from PIL import Image
from google.genai import (
    types as google_types,
)  # For type hinting if needed, or pass specific types


async def compress_images_in_history(
    history: List[Dict[str, Any]],
    is_gemini_provider: bool,
    compression_quality: int,
    resize_factor: float,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Compresses images within the message history.
    Modifies the history in-place for OpenAI, returns new history for Gemini.
    Returns the modified history and a boolean indicating if any compression occurred.
    """
    modified_history = []
    compression_occurred_flag = False

    for msg_data in history:
        new_msg_data = msg_data.copy()  # Work on a copy

        if is_gemini_provider:
            if isinstance(new_msg_data.get("parts"), list):
                new_parts = []
                for part in new_msg_data["parts"]:
                    if (
                        isinstance(part, google_types.Part)
                        and hasattr(part, "inline_data")
                        and part.inline_data
                        and part.inline_data.mime_type.startswith("image/")
                    ):
                        try:
                            image_bytes = part.inline_data.data
                            img = Image.open(io.BytesIO(image_bytes))
                            original_format = img.format or (
                                part.inline_data.mime_type.split("/")[-1].upper()
                                if "/" in part.inline_data.mime_type
                                else "PNG"
                            )

                            if resize_factor < 1.0:
                                new_width = int(img.width * resize_factor)
                                new_height = int(img.height * resize_factor)
                                if new_width > 0 and new_height > 0:
                                    img = img.resize(
                                        (new_width, new_height),
                                        Image.Resampling.LANCZOS,
                                    )
                                    compression_occurred_flag = True

                            output_buffer = io.BytesIO()
                            save_params = {}
                            target_format = original_format
                            if original_format in ["JPEG", "WEBP"]:
                                save_params["quality"] = compression_quality
                                target_format = "JPEG"
                                if (
                                    img.mode == "RGBA"
                                    or img.mode == "LA"
                                    or (img.mode == "P" and "transparency" in img.info)
                                ):
                                    img = img.convert("RGB")
                                compression_occurred_flag = True
                            elif original_format == "PNG":
                                save_params["optimize"] = True  # Basic PNG optimization

                            img.save(output_buffer, format=target_format, **save_params)
                            compressed_bytes = output_buffer.getvalue()
                            img.close()

                            new_mime_type = f"image/{target_format.lower()}"
                            new_parts.append(
                                google_types.Part.from_bytes(
                                    data=compressed_bytes, mime_type=new_mime_type
                                )
                            )
                            logging.debug(
                                f"Gemini Compressed image. Original size: {len(image_bytes)}, New: {len(compressed_bytes)}"
                            )
                        except Exception as e:
                            logging.error(
                                f"Error compressing Gemini image: {e}", exc_info=True
                            )
                            new_parts.append(part)  # Add original back
                    else:
                        new_parts.append(part)
                new_msg_data["parts"] = new_parts
        else:  # OpenAI-compatible
            if new_msg_data.get("role") == "user" and isinstance(
                new_msg_data.get("content"), list
            ):
                new_content_parts = []
                for part in new_msg_data["content"]:
                    if part.get("type") == "image_url" and isinstance(
                        part.get("image_url"), dict
                    ):
                        image_data_url = part["image_url"].get("url", "")
                        if (
                            image_data_url.startswith("data:image")
                            and ";base64," in image_data_url
                        ):
                            try:
                                header, encoded_image_data = image_data_url.split(
                                    ";base64,", 1
                                )
                                mime_type = (
                                    header.split(":", 1)[1]
                                    if ":" in header
                                    else "image/png"
                                )
                                image_bytes = base64.b64decode(encoded_image_data)

                                img = Image.open(io.BytesIO(image_bytes))
                                original_format = img.format or (
                                    mime_type.split("/")[-1].upper()
                                    if "/" in mime_type
                                    else "PNG"
                                )

                                if resize_factor < 1.0:
                                    new_width = int(img.width * resize_factor)
                                    new_height = int(img.height * resize_factor)
                                    if new_width > 0 and new_height > 0:
                                        img = img.resize(
                                            (new_width, new_height),
                                            Image.Resampling.LANCZOS,
                                        )
                                        compression_occurred_flag = True

                                output_buffer = io.BytesIO()
                                save_params = {}
                                target_format = original_format
                                if original_format in ["JPEG", "WEBP"]:
                                    save_params["quality"] = compression_quality
                                    target_format = "JPEG"
                                    if (
                                        img.mode == "RGBA"
                                        or img.mode == "LA"
                                        or (
                                            img.mode == "P"
                                            and "transparency" in img.info
                                        )
                                    ):
                                        img = img.convert("RGB")
                                    compression_occurred_flag = True
                                elif original_format == "PNG":
                                    save_params["optimize"] = True

                                img.save(
                                    output_buffer, format=target_format, **save_params
                                )
                                compressed_bytes = output_buffer.getvalue()
                                img.close()

                                new_mime_type = f"image/{target_format.lower()}"
                                new_encoded_data = base64.b64encode(
                                    compressed_bytes
                                ).decode("utf-8")

                                new_part_item = part.copy()
                                new_part_item["image_url"]["url"] = (
                                    f"data:{new_mime_type};base64,{new_encoded_data}"
                                )
                                new_content_parts.append(new_part_item)
                                logging.debug(
                                    f"OpenAI Compressed image. Original base64 len: {len(encoded_image_data)}, New base64 len: {len(new_encoded_data)}"
                                )
                            except Exception as e:
                                logging.error(
                                    f"Error compressing OpenAI image: {e}",
                                    exc_info=True,
                                )
                                new_content_parts.append(part)  # Add original back
                        else:
                            new_content_parts.append(part)  # Not a data URL image
                    else:
                        new_content_parts.append(part)  # Not an image_url part
                new_msg_data["content"] = new_content_parts

        modified_history.append(new_msg_data)

    return modified_history, compression_occurred_flag
