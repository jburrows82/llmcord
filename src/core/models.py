import asyncio
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, Union
import discord


@dataclass
class MsgNode:
    """Represents a node in the conversation history cache."""

    text: Optional[str] = None
    api_file_parts: list = field(default_factory=list)
    role: Literal["user", "assistant", "model"] = "assistant"
    user_id: Optional[int] = None
    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False
    parent_msg: Optional[discord.Message] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    full_response_text: Optional[str] = None
    user_provided_url_formatted_content: Optional[str] = None
    google_lens_formatted_content: Optional[str] = None
    search_results_formatted_content: Optional[str] = None


@dataclass
class UrlFetchResult:
    """Represents the result of fetching content from a URL."""

    url: str
    content: Optional[Union[str, Dict[str, Any], bytes]]
    error: Optional[str] = None
    type: Literal[
        "youtube",
        "reddit",
        "general",
        "google_lens_serpapi",
        "image_url_content",
        "general_crawl4ai",
        "general_jina",
    ] = "general"
    original_index: int = -1
