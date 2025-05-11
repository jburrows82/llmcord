import re
import urllib.parse
import base64
import json
from typing import List, Tuple, Optional, Any
import discord
from google.genai import types as google_types # Use google.genai.types
import pypdfium2 as pdfium # Added import
import asyncio # Added import

from .constants import (
    GENERAL_URL_PATTERN, YOUTUBE_URL_PATTERN, REDDIT_URL_PATTERN, IMAGE_URL_PATTERN,
    MAX_EMBED_TOTAL_SIZE, MAX_EMBED_FIELD_VALUE_LENGTH, MAX_EMBED_FIELDS
)

# --- URL Parsing and Checking ---

def extract_urls_with_indices(text: str) -> List[Tuple[str, int]]:
    """Extracts all URLs from text along with their start index."""
    return [(match.group(0), match.start()) for match in GENERAL_URL_PATTERN.finditer(text)]

def get_domain(url: str) -> Optional[str]:
    """Extracts the domain name from a URL."""
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return None

def is_youtube_url(url: str) -> bool:
    """Checks if a URL is a YouTube URL."""
    domain = get_domain(url)
    return domain in ('youtube.com', 'www.youtube.com', 'youtu.be') and YOUTUBE_URL_PATTERN.search(url) is not None

def is_reddit_url(url: str) -> bool:
    """Checks if a URL is a Reddit submission URL."""
    domain = get_domain(url)
    return domain in ('reddit.com', 'www.reddit.com') and REDDIT_URL_PATTERN.search(url) is not None

def extract_video_id(url: str) -> Optional[str]:
    """Extracts the video ID from a YouTube URL."""
    match = YOUTUBE_URL_PATTERN.search(url)
    return match.group(2) if match else None

def extract_reddit_submission_id(url: str) -> Optional[str]:
    """Extracts the submission ID from a Reddit URL."""
    match = REDDIT_URL_PATTERN.search(url)
    return match.group(2) if match else None

def is_image_url(url: str) -> bool:
    """Checks if a URL matches the image URL pattern."""
    return IMAGE_URL_PATTERN.match(url) is not None

# --- Embed Utilities ---

def calculate_embed_size(embed: discord.Embed) -> int:
    """Calculates the approximate total size of an embed's text content."""
    size = 0
    if embed.title:
        size += len(embed.title)
    if embed.description:
        size += len(embed.description)
    if embed.footer and embed.footer.text:
        size += len(embed.footer.text)
    if embed.author and embed.author.name:
        size += len(embed.author.name)
    # Add sizes of fields
    for field in embed.fields:
        size += len(field.name) + len(field.value)
    # Add a small buffer for overhead
    size += 100
    return size

def add_field_safely(embed: discord.Embed, name: str, value: str, inline: bool, embeds_to_send: List[discord.Embed], current_embed: discord.Embed, embed_color: discord.Color) -> discord.Embed:
    """
    Adds a field to an embed, handling splitting into new embeds if limits are exceeded.

    Args:
        embed: The current discord.Embed object to potentially add to.
        name: The name of the field.
        value: The value of the field.
        inline: Whether the field should be inline.
        embeds_to_send: The list accumulating embeds to be sent.
        current_embed: The embed currently being built (might be the same as embed initially).
        embed_color: The color to use for new embeds.

    Returns:
        The embed to continue building on (might be a new one).
    """
    # Truncate value if it exceeds the per-field limit
    if len(value) > MAX_EMBED_FIELD_VALUE_LENGTH:
        value = value[:MAX_EMBED_FIELD_VALUE_LENGTH - 3] + "..."

    # Check if adding this field exceeds total size or field count limits
    potential_size = calculate_embed_size(current_embed) + len(name) + len(value)
    if potential_size > MAX_EMBED_TOTAL_SIZE or len(current_embed.fields) >= MAX_EMBED_FIELDS:
        # Current embed is full, finalize it and start a new one
        if current_embed.fields or current_embed.description or (current_embed.title and current_embed.title != "Grounding Sources (cont.)"): # Avoid adding empty embeds
            embeds_to_send.append(current_embed)

        new_embed = discord.Embed(title="Grounding Sources (cont.)", color=embed_color)
        new_embed.add_field(name=name, value=value, inline=inline)
        return new_embed # Return the new embed
    else:
        # Add to current embed
        current_embed.add_field(name=name, value=value, inline=inline)
        return current_embed # Return the same embed

# --- Payload Utilities ---

async def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Optional[str]:
    """Extracts text from PDF bytes using pypdfium2, run in a thread."""
    if not pdf_bytes:
        return None

    def sync_extract():
        try:
            pdf_doc = pdfium.PdfDocument(pdf_bytes)
            all_text = ""
            for page_index in range(len(pdf_doc)):
                page = pdf_doc.get_page(page_index)
                textpage = page.get_textpage()
                all_text += textpage.get_text_bounded() + "\n" # Add newline between pages
                textpage.close()
                page.close()
            pdf_doc.close()
            return all_text.strip()
        except Exception as e:
            # The caller (in bot.py) will log this exception.
            raise # Re-raise to be caught by asyncio.to_thread and then the caller

    try:
        return await asyncio.to_thread(sync_extract)
    except Exception: # Catch exceptions from sync_extract re-raised by to_thread
        raise # Re-raise for the caller in bot.py

def _truncate_base64_in_payload(payload: Any, max_len: int = 40, prefix_len: int = 10) -> Any:
    """
    Recursively creates a deep copy of the payload and truncates long base64 strings
    found in specific known structures (OpenAI image_url, Gemini inline_data).
    """
    if isinstance(payload, dict):
        new_dict = {}
        for key, value in payload.items():
            # Check for OpenAI image_url structure
            if key == "image_url" and isinstance(value, dict) and "url" in value:
                url_value = value.get("url") # Use .get for safety
                if isinstance(url_value, str) and url_value.startswith("data:image") and ";base64," in url_value:
                    try:
                        prefix, data = url_value.split(";base64,", 1)
                        if len(data) > max_len:
                            truncated_data = data[:prefix_len] + "..." + data[-prefix_len:] + f" (truncated {len(data)} chars)"
                            # Create a copy of the inner dict to modify
                            new_image_url_dict = value.copy()
                            new_image_url_dict["url"] = prefix + ";base64," + truncated_data
                            new_dict[key] = new_image_url_dict
                        else:
                            # No truncation needed, shallow copy the inner dict is fine here
                            new_dict[key] = value.copy()
                    except ValueError: # Handle potential split errors
                         # If split fails, recurse into the value dict itself
                         new_dict[key] = _truncate_base64_in_payload(value, max_len, prefix_len)
                else:
                    # Not a base64 data URL or structure is different, recurse normally
                    new_dict[key] = _truncate_base64_in_payload(value, max_len, prefix_len)

            # Check for Gemini inline_data structure
            elif key == "inline_data" and isinstance(value, dict) and "data" in value:
                 data_value = value.get("data") # Use .get for safety
                 if isinstance(data_value, str) and len(data_value) > max_len: # Check if it's a string and long
                     # Create a copy of the inner dict to modify
                     new_inline_data_dict = value.copy()
                     # Truncate the data string itself
                     new_inline_data_dict["data"] = data_value[:prefix_len] + "..." + data_value[-prefix_len:] + f" (truncated {len(data_value)} chars)"
                     new_dict[key] = new_inline_data_dict
                 else:
                     # No truncation needed or not a string, shallow copy the inner dict is fine
                     new_dict[key] = value.copy()

            else:
                # Recurse for other keys/values
                new_dict[key] = _truncate_base64_in_payload(value, max_len, prefix_len)
        return new_dict
    elif isinstance(payload, list):
        # Recurse for items in a list
        return [_truncate_base64_in_payload(item, max_len, prefix_len) for item in payload]
    else:
        # Return non-dict/list types as is
        return payload

def default_serializer(obj):
    """Custom JSON serializer for printing payloads, handling Pydantic and other types."""
    # Check for Pydantic models first
    if hasattr(obj, 'model_dump'):
        try:
            # Exclude fields that are None for cleaner output
            return obj.model_dump(mode='json', exclude_none=True)
        except Exception:
            pass # Fall through if model_dump fails for some reason
    # Handle specific known types like google_types.Part
    elif isinstance(obj, google_types.Part):
         part_dict = {}
         if hasattr(obj, 'text') and obj.text is not None:
             part_dict["text"] = obj.text
         # The truncation function handles inline_data, but keep this for structure
         if hasattr(obj, 'inline_data') and obj.inline_data:
             part_dict["inline_data"] = {
                 "mime_type": obj.inline_data.mime_type,
                 "data": "<base64_data_handled_by_truncation>" # Placeholder
             }
         # Handle function calls if present
         if hasattr(obj, 'function_call') and obj.function_call:
             part_dict["function_call"] = {
                 "name": obj.function_call.name,
                 "args": obj.function_call.args
             }
         # Handle function responses if present
         if hasattr(obj, 'function_response') and obj.function_response:
             part_dict["function_response"] = {
                 "name": obj.function_response.name,
                 "response": obj.function_response.response
             }
         return part_dict if part_dict else None
    elif isinstance(obj, bytes):
        return "<bytes_data>"
    # Fallback for other types
    try:
        # Use standard JSONEncoder default for basic types
        return json.JSONEncoder.default(None, obj)
    except TypeError:
        # Represent unserializable objects clearly
        return f"<unserializable_object: {type(obj).__name__}>"
