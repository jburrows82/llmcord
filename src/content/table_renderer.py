"""
Module for detecting markdown tables in text and rendering them as images.
"""

import re
import logging
import io
import tempfile
import os
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
    logging.warning("html2image not available. Table rendering will use basic PIL rendering.")

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning("Playwright not available. Table rendering will fall back to html2image or PIL.")

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
    table_pattern = r'(\|.*\|[\r\n]+\|[\s]*:?-+:?[\s]*\|.*(?:[\r\n]+\|.*\|)*)'
    
    for match in re.finditer(table_pattern, text, re.MULTILINE):
        table_text = match.group(1).strip()
        start_pos = match.start()
        end_pos = match.end()
        
        # Validate that this is actually a proper table
        lines = table_text.split('\n')
        if len(lines) >= 2:  # At least header and separator
            # Check if second line is a separator line
            separator_line = lines[1].strip()
            if re.match(r'^\|[\s]*:?-+:?[\s]*(\|[\s]*:?-+:?[\s]*)*\|$', separator_line):
                tables.append((table_text, start_pos, end_pos))
    
    return tables


async def render_table_with_playwright(table_markdown: str) -> Optional[bytes]:
    """
    Render a markdown table as an image using Playwright.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return None
    
    try:
        html_content = markdown2.markdown(table_markdown, extras=['tables'])
        
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
            await page.wait_for_load_state('networkidle')
            
            try:
                table_element = page.locator('table').first
                if await table_element.count() > 0:
                    screenshot_bytes = await table_element.screenshot(type='png')
                else:
                    body_element = page.locator('body').first
                    screenshot_bytes = await body_element.screenshot(type='png')
            except Exception as e:
                logging.warning(f"Error taking element screenshot, falling back to full page: {e}")
                screenshot_bytes = await page.screenshot(type='png', full_page=True)
            
            await browser.close()
            return screenshot_bytes
            
    except Exception as e:
        logging.error(f"Error rendering table with Playwright: {e}")
    
    return None


async def render_markdown_table(table_markdown: str) -> Optional[bytes]:
    """
    Render a markdown table as an image using the best available method.
    """
    # Try Playwright first (best quality and reliability)
    if PLAYWRIGHT_AVAILABLE:
        result = await render_table_with_playwright(table_markdown)
        if result:
            logging.info("Table rendered successfully with Playwright")
            return result
    
    logging.warning("Failed to render table with any available method")
    return None


async def process_and_send_table_images(final_text: str, response_msgs: List, new_msg) -> bool:
    """
    Detect markdown tables in the final response text and send them as images.
    """
    return False
