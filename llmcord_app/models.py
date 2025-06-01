import asyncio
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, Union
import discord


# --- Data Classes ---
@dataclass
class MsgNode:
    """Represents a node in the conversation history cache."""

    text: Optional[str] = None
    api_file_parts: list = field(
        default_factory=list
    )  # Renamed from 'images', stores prepared API parts (img, pdf)
    role: Literal["user", "assistant", "model"] = (
        "assistant"  # 'model' for Gemini assistant role
    )
    user_id: Optional[int] = None  # Discord user ID if role is 'user'
    has_bad_attachments: bool = (
        False  # Flag if unsupported/failed attachments were present
    )
    fetch_parent_failed: bool = False  # Flag if fetching the parent message failed
    parent_msg: Optional[discord.Message] = (
        None  # Reference to the parent discord.Message object
    )
    lock: asyncio.Lock = field(
        default_factory=asyncio.Lock
    )  # Lock for async access to this node
    full_response_text: Optional[str] = (
        None  # Stores the complete text from the LLM for this node (if it's an assistant response)
    )
    # --- Specific Formatted Content Fields for History Persistence ---
    user_provided_url_formatted_content: Optional[str] = (
        None  # Formatted content from URLs provided by the user in their message
    )
    google_lens_formatted_content: Optional[str] = (
        None  # Formatted content from Google Lens analysis
    )
    search_results_formatted_content: Optional[str] = (
        None  # Formatted content from web search results (e.g., SearxNG)
    )


@dataclass
class UrlFetchResult:
    """Represents the result of fetching content from a URL."""

    url: str
    content: Optional[
        Union[str, Dict[str, Any], bytes]
    ]  # Fetched content (text, dict for YouTube/Reddit, bytes for image_url)
    error: Optional[str] = None  # Error message if fetching failed
    type: Literal[
        "youtube",
        "reddit",
        "general",
        "google_lens_serpapi",
        "image_url_content",
        "general_crawl4ai",
        "general_jina",
    ] = "general"
    original_index: int = -1  # Original start index of the URL in the user's message or attachment index for Lens
