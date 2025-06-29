from typing import Optional, List, Dict, Any, Set, Tuple
import asyncio
import base64
import httpx
import tiktoken


def get_tokenizer_for_model(model_name: str):
    """Gets the tiktoken encoder. Always uses 'o200k_base'."""
    return tiktoken.get_encoding("o200k_base")


def truncate_text_by_tokens(text: str, tokenizer, max_tokens: int) -> Tuple[str, int]:
    """Truncates text to a maximum number of tokens and returns the truncated text and token count."""
    if not text:
        return "", 0
    tokens = tokenizer.encode(text)
    actual_token_count = len(tokens)
    if actual_token_count > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        try:
            truncated_text = tokenizer.decode(truncated_tokens)
        except UnicodeDecodeError:
            try:
                truncated_text = tokenizer.decode(truncated_tokens[:-1])
            except Exception:
                return "", actual_token_count
        if truncated_text and max_tokens > 0:
            truncated_text += "..."
        return truncated_text, actual_token_count
    return text, actual_token_count


def smart_truncate_external_content(
    user_provided_url_content: Optional[str],
    google_lens_content: Optional[str],
    search_results_content: Optional[str],
    tokenizer,
    max_tokens_for_external_content: int,
    user_warnings: Set[str],
) -> Tuple[Optional[str], Optional[str], Optional[str], int]:
    """Intelligently truncates external content, prioritizing web search results for truncation first."""
    if max_tokens_for_external_content <= 0:
        return None, None, None, 0

    # Calculate current token usage for each content type
    user_url_tokens = (
        len(tokenizer.encode(user_provided_url_content))
        if user_provided_url_content
        else 0
    )
    google_lens_tokens = (
        len(tokenizer.encode(google_lens_content)) if google_lens_content else 0
    )
    search_results_tokens = (
        len(tokenizer.encode(search_results_content)) if search_results_content else 0
    )
    total_tokens = user_url_tokens + google_lens_tokens + search_results_tokens

    # If within budget, return as-is
    if total_tokens <= max_tokens_for_external_content:
        return (
            user_provided_url_content,
            google_lens_content,
            search_results_content,
            total_tokens,
        )

    # Need to truncate - prioritize keeping user URLs and Google Lens over search results
    tokens_to_remove = total_tokens - max_tokens_for_external_content

    # Start by truncating search results first (they are lowest priority)
    truncated_search_results = search_results_content
    if search_results_tokens > 0 and tokens_to_remove > 0:
        tokens_to_remove_from_search = min(tokens_to_remove, search_results_tokens)
        remaining_search_tokens = search_results_tokens - tokens_to_remove_from_search
        if remaining_search_tokens <= 0:
            truncated_search_results = None
            user_warnings.add("⚠️ Web search results truncated due to length limits")
            tokens_to_remove -= search_results_tokens
        else:
            truncated_search_results, _ = truncate_text_by_tokens(
                search_results_content, tokenizer, remaining_search_tokens
            )
            user_warnings.add(
                "⚠️ Web search results partially truncated due to length limits"
            )
            tokens_to_remove = 0

    # If still need to truncate more, truncate Google Lens content next
    truncated_google_lens = google_lens_content
    if google_lens_tokens > 0 and tokens_to_remove > 0:
        tokens_to_remove_from_lens = min(tokens_to_remove, google_lens_tokens)
        remaining_lens_tokens = google_lens_tokens - tokens_to_remove_from_lens
        if remaining_lens_tokens <= 0:
            truncated_google_lens = None
            user_warnings.add("⚠️ Google Lens results truncated due to length limits")
            tokens_to_remove -= google_lens_tokens
        else:
            truncated_google_lens, _ = truncate_text_by_tokens(
                google_lens_content, tokenizer, remaining_lens_tokens
            )
            user_warnings.add(
                "⚠️ Google Lens results partially truncated due to length limits"
            )
            tokens_to_remove = 0

    # Finally, if still need to truncate, truncate user URL content (highest priority to keep)
    truncated_user_urls = user_provided_url_content
    if user_url_tokens > 0 and tokens_to_remove > 0:
        tokens_to_remove_from_urls = min(tokens_to_remove, user_url_tokens)
        remaining_url_tokens = user_url_tokens - tokens_to_remove_from_urls
        if remaining_url_tokens <= 0:
            truncated_user_urls = None
            user_warnings.add("⚠️ User URL content truncated due to length limits")
        else:
            truncated_user_urls, _ = truncate_text_by_tokens(
                user_provided_url_content, tokenizer, remaining_url_tokens
            )
            user_warnings.add(
                "⚠️ User URL content partially truncated due to length limits"
            )

    # Calculate final token usage
    final_user_url_tokens = (
        len(tokenizer.encode(truncated_user_urls)) if truncated_user_urls else 0
    )
    final_google_lens_tokens = (
        len(tokenizer.encode(truncated_google_lens)) if truncated_google_lens else 0
    )
    final_search_results_tokens = (
        len(tokenizer.encode(truncated_search_results))
        if truncated_search_results
        else 0
    )
    final_total_tokens = (
        final_user_url_tokens + final_google_lens_tokens + final_search_results_tokens
    )

    return (
        truncated_user_urls,
        truncated_google_lens,
        truncated_search_results,
        final_total_tokens,
    )


async def process_message_attachments(
    curr_msg,
    httpx_async_client,
    extract_text_from_pdf_bytes_func,
    current_role: str,
    is_current_message_node: bool,
    current_message_url_fetch_results,
    accept_files: bool,
    use_google_lens_for_current: bool,
    is_target_provider_gemini: bool,
    max_files_per_message: int,
    google_types_module,
    user_warnings: set,
) -> Tuple[List[str], List]:
    """Process message attachments and return text parts and API file parts."""
    MAX_ATTACHMENTS_TO_DOWNLOAD_IN_HISTORY = 5
    current_attachments = curr_msg.attachments
    attachments_to_fetch = []

    # Determine which attachments to fetch
    for att in current_attachments:
        if len(attachments_to_fetch) >= MAX_ATTACHMENTS_TO_DOWNLOAD_IN_HISTORY:
            break
        if att.content_type:
            is_relevant_for_download = False
            if att.content_type.startswith("text/"):
                is_relevant_for_download = True
            elif att.content_type.startswith("image/"):
                if (
                    accept_files
                    or (is_current_message_node and use_google_lens_for_current)
                    or current_role == "model"
                ):
                    is_relevant_for_download = True
            elif att.content_type == "application/pdf":
                if (
                    is_target_provider_gemini and accept_files
                ) or not is_target_provider_gemini:
                    is_relevant_for_download = True
            elif (
                att.content_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                is_relevant_for_download = True

            if is_relevant_for_download:
                attachments_to_fetch.append(att)

    # Download attachments
    attachment_responses = await asyncio.gather(
        *[
            httpx_async_client.get(
                att.url,
                timeout=httpx.Timeout(connect=8.0, read=15.0, write=8.0, pool=5.0),
            )
            for att in attachments_to_fetch
        ],
        return_exceptions=True,
    )

    # Extract text from attachments
    text_parts = []
    for att, resp in zip(attachments_to_fetch, attachment_responses):
        if (
            isinstance(resp, httpx.Response)
            and resp.status_code == 200
            and att.content_type.startswith("text/")
        ):
            try:
                text_parts.append(resp.text)
            except Exception:
                pass

    # Handle PDF text extraction for non-Gemini user messages
    if current_role == "user" and not is_target_provider_gemini:
        for att, resp in zip(attachments_to_fetch, attachment_responses):
            if att.content_type == "application/pdf":
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    try:
                        extracted_pdf_text = await extract_text_from_pdf_bytes_func(
                            resp.content
                        )
                        if extracted_pdf_text:
                            text_parts.append(
                                f"\n\n--- Content from PDF: {att.filename} ---\n{extracted_pdf_text}\n--- End of PDF: {att.filename} ---"
                            )
                    except Exception:
                        pass

    # Handle DOCX text extraction for all user messages
    if current_role == "user":
        for att, resp in zip(attachments_to_fetch, attachment_responses):
            if (
                att.content_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    try:
                        from .utils import extract_text_from_docx_bytes

                        extracted_docx_text = await extract_text_from_docx_bytes(
                            resp.content
                        )
                        if extracted_docx_text:
                            text_parts.append(
                                f"\n\n--- Content from DOCX: {att.filename} ---\n{extracted_docx_text}\n--- End of DOCX: {att.filename} ---"
                            )
                    except Exception:
                        pass

    # Build API file parts
    api_file_parts = await build_api_file_parts(
        attachments_to_fetch,
        attachment_responses,
        current_role,
        is_current_message_node,
        current_message_url_fetch_results,
        accept_files,
        use_google_lens_for_current,
        is_target_provider_gemini,
        max_files_per_message,
        google_types_module,
        user_warnings,
    )

    return text_parts, api_file_parts


async def build_api_file_parts(
    attachments_to_fetch,
    attachment_responses,
    current_role: str,
    is_current_message_node: bool,
    current_message_url_fetch_results,
    accept_files: bool,
    use_google_lens_for_current: bool,
    is_target_provider_gemini: bool,
    max_files_per_message: int,
    google_types_module,
    user_warnings: set,
) -> List:
    """Build API file parts from attachments and URL fetches."""
    api_file_parts = []
    files_processed_for_api_count = 0
    is_lens_trigger_message = is_current_message_node and use_google_lens_for_current
    should_process_files_for_api = (
        (current_role == "user" or current_role == "model") or is_lens_trigger_message
    ) and (accept_files or is_lens_trigger_message)

    if should_process_files_for_api and not is_lens_trigger_message:
        for att, resp in zip(attachments_to_fetch, attachment_responses):
            is_api_relevant_type = False
            mime_type_for_api = att.content_type
            file_bytes_for_api = None

            if att.content_type.startswith("image/"):
                is_api_relevant_type = True
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    file_bytes_for_api = resp.content
                else:
                    continue
            elif (
                att.content_type == "application/pdf"
                and is_target_provider_gemini
                and accept_files
                and current_role == "user"
            ):
                is_api_relevant_type = True
                mime_type_for_api = "application/pdf"
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    file_bytes_for_api = resp.content
                else:
                    continue

            if not is_api_relevant_type or file_bytes_for_api is None:
                continue
            if files_processed_for_api_count >= max_files_per_message:
                break

            try:
                if is_target_provider_gemini:
                    api_file_parts.append(
                        google_types_module.Part.from_bytes(
                            data=file_bytes_for_api, mime_type=mime_type_for_api
                        )
                    )
                else:
                    base64_encoded_image = await asyncio.to_thread(
                        lambda b: base64.b64encode(b).decode("utf-8"),
                        file_bytes_for_api,
                    )
                    api_file_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type_for_api};base64,{base64_encoded_image}"
                            },
                        }
                    )
                files_processed_for_api_count += 1
            except Exception:
                pass

        # Add fetched image URLs for current message
        if is_current_message_node and current_message_url_fetch_results:
            for fetched_url_res in current_message_url_fetch_results:
                if (
                    fetched_url_res.type == "image_url_content"
                    and isinstance(fetched_url_res.content, bytes)
                    and not fetched_url_res.error
                ):
                    if files_processed_for_api_count >= max_files_per_message:
                        user_warnings.add("⚠️ Max files reached.")
                        break

                    img_bytes = fetched_url_res.content
                    url_lower = fetched_url_res.url.lower()
                    mime_type = "image/png"
                    if url_lower.endswith((".jpg", ".jpeg")):
                        mime_type = "image/jpeg"
                    elif url_lower.endswith(".gif"):
                        mime_type = "image/gif"
                    elif url_lower.endswith(".webp"):
                        mime_type = "image/webp"
                    elif url_lower.endswith(".bmp"):
                        mime_type = "image/bmp"

                    try:
                        if is_target_provider_gemini:
                            api_file_parts.append(
                                google_types_module.Part.from_bytes(
                                    data=img_bytes, mime_type=mime_type
                                )
                            )
                        else:
                            base64_encoded_image = await asyncio.to_thread(
                                lambda b: base64.b64encode(b).decode("utf-8"), img_bytes
                            )
                            api_file_parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_encoded_image}"
                                    },
                                }
                            )
                        files_processed_for_api_count += 1
                    except Exception:
                        user_warnings.add(
                            f"⚠️ Error processing image URL: {fetched_url_res.url[:50]}..."
                        )

    return api_file_parts


def calculate_entry_tokens(
    entry_data: Dict[str, Any],
    tokenizer,
    is_target_provider_gemini: bool,
    google_types_module,
) -> int:
    """Calculate total token count for a history entry."""
    # Build text content for token counting
    current_entry_text_content = ""
    if entry_data.get("external_content"):
        current_entry_text_content = (
            "User's query:\n"
            + (entry_data["text"] or "")
            + "\n\nExternal Content:\n"
            + entry_data["external_content"]
        )
    elif entry_data["text"]:
        current_entry_text_content = "User's query:\n" + entry_data["text"]

    # Calculate text tokens
    text_only_token_count = len(tokenizer.encode(current_entry_text_content))
    image_token_cost = 0

    # Add fixed token cost for images
    if entry_data["files"]:
        num_images = 0
        for file_part in entry_data["files"]:
            if isinstance(file_part, dict) and file_part.get("type") == "image_url":
                img_url_dict = file_part.get("image_url", {})
                data_url_val = img_url_dict.get("url", "")
                if data_url_val.startswith("data:image") and ";base64," in data_url_val:
                    num_images += 1
            elif (
                is_target_provider_gemini
                and hasattr(file_part, "inline_data")
                and file_part.inline_data
                and file_part.inline_data.mime_type.startswith("image/")
            ):
                num_images += 1
        image_token_cost = num_images * 765

    return text_only_token_count + image_token_cost


def format_api_message_parts(
    final_api_message_parts: List[Dict[str, Any]],
    is_target_provider_gemini: bool,
    target_provider_name: str,
    providers_supporting_usernames_const: tuple,
    accept_files: bool,
    max_files_per_message: int,
    google_types_module,
) -> List[Dict[str, Any]]:
    """Format message parts for API consumption."""
    api_formatted_history = []

    for entry in final_api_message_parts:
        # Construct text content for API
        text_content_for_api = entry["text"] or ""

        if entry.get("role") == "user" and entry.get("external_content"):
            text_content_for_api = (
                "User's query:\n"
                + (entry["text"] or "")
                + "\n\nExternal Content:\n"
                + entry["external_content"]
            )

        # Prepare file parts within limits
        current_api_file_parts = []
        if accept_files:
            raw_parts_from_node = entry["files"][:max_files_per_message]
            if is_target_provider_gemini:
                for part_in_node in raw_parts_from_node:
                    if isinstance(part_in_node, google_types_module.Part):
                        current_api_file_parts.append(part_in_node)
                    elif (
                        isinstance(part_in_node, dict)
                        and part_in_node.get("type") == "image_url"
                    ):
                        try:
                            header, encoded_data = part_in_node["image_url"][
                                "url"
                            ].split(";base64,", 1)
                            mime_type = header.split(":")[1]
                            img_bytes = base64.b64decode(encoded_data)
                            current_api_file_parts.append(
                                google_types_module.Part.from_bytes(
                                    data=img_bytes, mime_type=mime_type
                                )
                            )
                        except Exception:
                            pass
            else:  # OpenAI format
                for part_in_node in raw_parts_from_node:
                    if isinstance(part_in_node, dict):
                        current_api_file_parts.append(part_in_node)
                    elif isinstance(part_in_node, google_types_module.Part) and hasattr(
                        part_in_node, "inline_data"
                    ):
                        try:
                            if part_in_node.inline_data.mime_type.startswith("image/"):
                                b64_data = base64.b64encode(
                                    part_in_node.inline_data.data
                                ).decode("utf-8")
                                current_api_file_parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{part_in_node.inline_data.mime_type};base64,{b64_data}"
                                        },
                                    }
                                )
                        except Exception:
                            pass

        # Build message parts
        parts_for_this_api_message = []
        if is_target_provider_gemini:
            if text_content_for_api:
                parts_for_this_api_message.append(
                    google_types_module.Part.from_text(text=text_content_for_api)
                )
            parts_for_this_api_message.extend(current_api_file_parts)
        else:  # OpenAI
            if text_content_for_api:
                parts_for_this_api_message.append(
                    {"type": "text", "text": text_content_for_api}
                )
            parts_for_this_api_message.extend(current_api_file_parts)

        if parts_for_this_api_message:
            message_data = {"role": entry["role"]}
            if is_target_provider_gemini:
                message_data["parts"] = (
                    parts_for_this_api_message
                    if isinstance(parts_for_this_api_message, list)
                    else [parts_for_this_api_message]
                )
            else:  # OpenAI
                if message_data["role"] == "model":
                    message_data["role"] = "assistant"
                message_data["content"] = (
                    parts_for_this_api_message[0]["text"]
                    if len(parts_for_this_api_message) == 1
                    and parts_for_this_api_message[0]["type"] == "text"
                    else parts_for_this_api_message
                )
                if (
                    target_provider_name in providers_supporting_usernames_const
                    and entry["role"] == "user"
                    and entry["user_id"]
                ):
                    message_data["name"] = str(entry["user_id"])
            api_formatted_history.append(message_data)

    return api_formatted_history
