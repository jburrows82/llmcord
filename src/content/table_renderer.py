"""
Module for detecting markdown tables in text and rendering them as images.
"""

import re
import logging
import io
import tempfile
from typing import List, Tuple, Optional
from pathlib import Path

import markdown2
from PIL import Image, ImageDraw, ImageFont
import discord

try:
    from html2image import Html2Image

    HTML2IMAGE_AVAILABLE = True
except ImportError:
    HTML2IMAGE_AVAILABLE = False
    logging.warning(
        "html2image not available. Table rendering will use basic PIL rendering."
    )

try:
    from playwright.async_api import async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning(
        "Playwright not available. Table rendering will fall back to html2image or PIL."
    )

# WeasyPrint removed due to compatibility issues
WEASYPRINT_AVAILABLE = False


def detect_markdown_tables(text: str) -> List[Tuple[str, int, int]]:
    """
    Detect markdown tables in text and return them with their positions.

    Args:
        text: The text to search for markdown tables

    Returns:
        List of tuples containing (table_markdown, start_pos, end_pos)
    """
    tables = []

    # Pattern to match markdown tables
    # This matches tables with header separators (|---| style)
    table_pattern = r"(\|.*\|[\r\n]+\|[\s]*:?-+:?[\s]*\|.*(?:[\r\n]+\|.*\|)*)"

    for match in re.finditer(table_pattern, text, re.MULTILINE):
        table_text = match.group(1).strip()
        start_pos = match.start()
        end_pos = match.end()

        # Validate that this is actually a proper table
        lines = table_text.split("\n")
        if len(lines) >= 2:  # At least header and separator
            # Check if second line is a separator line
            separator_line = lines[1].strip()
            if re.match(r"^\|[\s]*:?-+:?[\s]*(\|[\s]*:?-+:?[\s]*)*\|$", separator_line):
                tables.append((table_text, start_pos, end_pos))

    return tables


async def render_table_with_playwright(table_markdown: str) -> Optional[bytes]:
    """
    Render a markdown table as an image using Playwright.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return None

    try:
        html_content = markdown2.markdown(table_markdown, extras=["tables"])

        css_styles = """
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                padding: 20px;
                background-color: #2f3136;
                color: #dcddde;
                margin: 0;
                width: fit-content;
            }
            table {
                border-collapse: collapse;
                margin: 0 auto;
                background-color: #36393f;
                border-radius: 6px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
                font-size: 14px;
            }
            th, td {
                border: 1px solid #4f545c;
                padding: 12px 16px;
                text-align: left;
                vertical-align: top;
            }
            th {
                background-color: #5865f2;
                color: white;
                font-weight: 600;
            }
            tr:nth-child(even) td {
                background-color: #2f3136;
            }
            tr:nth-child(odd) td {
                background-color: #36393f;
            }
        </style>
        """

        full_html = f"<!DOCTYPE html><html><head>{css_styles}</head><body>{html_content}</body></html>"

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.set_content(full_html)
            await page.wait_for_load_state("networkidle")

            try:
                table_element = page.locator("table").first
                if await table_element.count() > 0:
                    screenshot_bytes = await table_element.screenshot(type="png")
                else:
                    body_element = page.locator("body").first
                    screenshot_bytes = await body_element.screenshot(type="png")
            except Exception as e:
                logging.warning(
                    f"Error taking element screenshot, falling back to full page: {e}"
                )
                screenshot_bytes = await page.screenshot(type="png", full_page=True)

            await browser.close()
            return screenshot_bytes

    except Exception as e:
        logging.error(f"Error rendering table with Playwright: {e}")

    return None


def render_table_with_html2image(table_markdown: str) -> Optional[bytes]:
    """
    Render a markdown table as an image using html2image.
    """
    if not HTML2IMAGE_AVAILABLE:
        return None

    try:
        html_content = markdown2.markdown(table_markdown, extras=["tables"])

        css_styles = """
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                padding: 20px;
                background-color: #2f3136;
                color: #dcddde;
                margin: 0;
            }
            table {
                border-collapse: collapse;
                margin: 0 auto;
                background-color: #36393f;
                border-radius: 6px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            }
            th, td {
                border: 1px solid #4f545c;
                padding: 12px 16px;
                text-align: left;
            }
            th {
                background-color: #5865f2;
                color: white;
                font-weight: 600;
            }
            tr:nth-child(even) {
                background-color: #2f3136;
            }
            tr:hover {
                background-color: #393c43;
            }
        </style>
        """

        full_html = f"<!DOCTYPE html><html><head>{css_styles}</head><body>{html_content}</body></html>"

        hti = Html2Image(size=(800, 600), output_path=tempfile.gettempdir())

        temp_filename = f"table_{hash(table_markdown) % 10000}.png"

        hti.screenshot(html_str=full_html, save_as=temp_filename, size=(800, 600))

        temp_path = Path(tempfile.gettempdir()) / temp_filename
        if temp_path.exists():
            with open(temp_path, "rb") as f:
                image_bytes = f.read()
            temp_path.unlink()
            return image_bytes

    except Exception as e:
        logging.error(f"Error rendering table with html2image: {e}")

    return None


def render_table_with_pil(table_markdown: str) -> Optional[bytes]:
    """
    Render a markdown table as an image using PIL (fallback method).
    """
    try:
        lines = [
            line.strip() for line in table_markdown.strip().split("\n") if line.strip()
        ]

        if len(lines) < 2:
            return None

        rows = []
        for i, line in enumerate(lines):
            if i == 1:  # Skip separator line
                continue
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if cells:
                rows.append(cells)

        if not rows:
            return None

        max_cols = max(len(row) for row in rows)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except (OSError, IOError):
                font = ImageFont.load_default()

        cell_padding = 10
        cell_heights = []
        cell_widths = [0] * max_cols

        for row in rows:
            row_height = 0
            for col_idx, cell in enumerate(row):
                if col_idx < max_cols:
                    bbox = font.getbbox(str(cell))
                    cell_width = bbox[2] - bbox[0] + 2 * cell_padding
                    cell_height = bbox[3] - bbox[1] + 2 * cell_padding

                    cell_widths[col_idx] = max(cell_widths[col_idx], cell_width)
                    row_height = max(row_height, cell_height)

            cell_heights.append(row_height)

        total_width = sum(cell_widths) + max_cols + 1
        total_height = sum(cell_heights) + len(rows) + 1

        img = Image.new("RGB", (total_width, total_height), color="#2f3136")
        draw = ImageDraw.Draw(img)

        y_pos = 0
        for row_idx, row in enumerate(rows):
            x_pos = 0
            row_height = cell_heights[row_idx]

            for col_idx in range(max_cols):
                cell_width = cell_widths[col_idx]

                if row_idx == 0:  # Header
                    bg_color = "#5865f2"
                    text_color = "#ffffff"
                elif row_idx % 2 == 0:
                    bg_color = "#36393f"
                    text_color = "#dcddde"
                else:
                    bg_color = "#2f3136"
                    text_color = "#dcddde"

                draw.rectangle(
                    [x_pos, y_pos, x_pos + cell_width, y_pos + row_height],
                    fill=bg_color,
                    outline="#4f545c",
                )

                if col_idx < len(row):
                    cell_text = str(row[col_idx])
                    bbox = font.getbbox(cell_text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    text_x = x_pos + (cell_width - text_width) // 2
                    text_y = y_pos + (row_height - text_height) // 2

                    draw.text((text_x, text_y), cell_text, fill=text_color, font=font)

                x_pos += cell_width

            y_pos += row_height

        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        return img_buffer.getvalue()

    except Exception as e:
        logging.error(f"Error rendering table with PIL: {e}")

    return None


async def render_markdown_table(table_markdown: str) -> Optional[bytes]:
    """
    Render a markdown table as an image using the best available method.

    Priority order:
    1. Playwright (best quality and reliability)
    2. html2image (good fallback)
    3. PIL (basic fallback, always available)
    """
    # Try Playwright first (best quality and reliability)
    if PLAYWRIGHT_AVAILABLE:
        result = await render_table_with_playwright(table_markdown)
        if result:
            logging.info("Table rendered successfully with Playwright")
            return result

    # Try html2image second
    if HTML2IMAGE_AVAILABLE:
        result = render_table_with_html2image(table_markdown)
        if result:
            logging.info("Table rendered successfully with html2image")
            return result

    # Fallback to PIL (always available)
    result = render_table_with_pil(table_markdown)
    if result:
        logging.info("Table rendered successfully with PIL")
        return result

    logging.warning("Failed to render table with any available method")
    return None


async def process_and_send_table_images(
    final_text: str, response_msgs: List[discord.Message], new_msg: discord.Message
) -> bool:
    """
    Detect markdown tables in the final response text and send them as images.

    Args:
        final_text: The complete response text to check for tables
        response_msgs: List of response messages sent by the bot
        new_msg: The original user message

    Returns:
        True if tables were found and processed, False otherwise
    """
    if not final_text:
        return False

    # Detect tables in the response
    tables = detect_markdown_tables(final_text)

    if not tables:
        return False

    logging.info(f"Detected {len(tables)} markdown table(s) in response")

    # Process each table
    table_images = []
    for i, (table_markdown, start_pos, end_pos) in enumerate(tables):
        try:
            # Render table as image
            image_bytes = await render_markdown_table(table_markdown)

            if image_bytes:
                # Create Discord file
                filename = f"table_{i + 1}.png"
                discord_file = discord.File(
                    fp=io.BytesIO(image_bytes), filename=filename
                )
                table_images.append((discord_file, i + 1))
            else:
                logging.warning(f"Failed to render table {i + 1}")

        except Exception as e:
            logging.error(f"Error processing table {i + 1}: {e}")

    # Send table images as follow-up messages
    if table_images:
        try:
            # Determine what to reply to
            reply_target = response_msgs[-1] if response_msgs else new_msg

            if len(table_images) == 1:
                # Single table
                file, table_num = table_images[0]
                await reply_target.reply(
                    content="📊 Here's the table from my response as an image:",
                    file=file,
                    mention_author=False,
                )
            else:
                # Multiple tables - send them in separate messages
                await reply_target.reply(
                    content=f"📊 Here are the {len(table_images)} tables from my response as images:",
                    mention_author=False,
                )

                for file, table_num in table_images:
                    await reply_target.reply(
                        content=f"Table {table_num}:", file=file, mention_author=False
                    )

            logging.info(f"Successfully sent {len(table_images)} table image(s)")
            return True

        except Exception as e:
            logging.error(f"Error sending table images: {e}")

    return False
