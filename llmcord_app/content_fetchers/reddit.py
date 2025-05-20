import logging
from typing import Optional

import asyncpraw
from asyncprawcore.exceptions import (
    NotFound,
    Redirect,
    Forbidden,
    RequestException as AsyncPrawRequestException,
)

from ..models import UrlFetchResult

# Global asyncpraw.Reddit instance
reddit_client_instance: Optional[asyncpraw.Reddit] = None


def initialize_reddit_client(
    client_id: Optional[str], client_secret: Optional[str], user_agent: Optional[str]
):
    """Initializes the global asyncpraw.Reddit client instance."""
    global reddit_client_instance
    if not all([client_id, client_secret, user_agent]):
        logging.warning(
            "Reddit API credentials not fully configured. Reddit client not initialized."
        )
        reddit_client_instance = None
        return

    try:
        reddit_client_instance = asyncpraw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            read_only=True,
            check_for_async=False,  # Suitable for environments where event loop might be running
        )
        logging.info("Async PRAW Reddit client initialized successfully.")
    except Exception:
        logging.exception("Failed to initialize Async PRAW Reddit client.")
        reddit_client_instance = None


async def fetch_reddit_data(
    url: str,
    submission_id: str,
    index: int,
    client_id: Optional[str],
    client_secret: Optional[str],
    user_agent: Optional[str],
) -> UrlFetchResult:
    """Fetches content for a single Reddit submission URL using asyncpraw."""
    if not all([client_id, client_secret, user_agent]):
        return UrlFetchResult(
            url=url,
            content=None,
            error="Reddit API credentials not configured.",
            type="reddit",
            original_index=index,
        )
    global reddit_client_instance
    if not reddit_client_instance:
        logging.error("Reddit client not initialized. Cannot fetch Reddit data.")
        # Optionally, could try a one-off initialization here if credentials are provided,
        # but ideally it should be initialized at startup.
        # For now, just return an error.
        return UrlFetchResult(
            url=url,
            content=None,
            error="Reddit client not initialized.",
            type="reddit",
            original_index=index,
        )

    # reddit = None # No longer initializing locally
    try:
        # Use the global reddit_client_instance
        submission = await reddit_client_instance.submission(id=submission_id)
        # Use fetch=True to ensure data is loaded, replaces await submission.load()
        await submission.load()  # Keep load() as it fetches comments too

        logging.debug(
            f"Fetched Reddit submission: {submission.title} using global client."
        )

        content_data = {"title": submission.title}
        if submission.selftext:
            content_data["selftext"] = submission.selftext

        # Fetch top-level comments
        top_comments_text = []
        comment_limit = 10  # Limit comments fetched
        await submission.comments.replace_more(limit=0)  # Load only top-level comments

        comment_count = 0
        for top_level_comment in submission.comments.list():
            if comment_count >= comment_limit:
                logging.debug(
                    f"Reached comment limit ({comment_limit}) for Reddit URL: {url}"
                )
                break
            # Check if comment exists and is not deleted/removed before accessing body
            if (
                hasattr(top_level_comment, "body")
                and top_level_comment.body
                and top_level_comment.body not in ("[deleted]", "[removed]")
            ):
                comment_body_cleaned = top_level_comment.body.replace(
                    "\n", " "
                ).replace("\r", "")
                top_comments_text.append(comment_body_cleaned)
                comment_count += 1

        if top_comments_text:
            content_data["comments"] = top_comments_text
            logging.debug(
                f"Fetched {len(top_comments_text)} comments for Reddit URL: {url}"
            )
        else:
            logging.debug(f"No comments found or fetched for Reddit URL: {url}")

        return UrlFetchResult(
            url=url, content=content_data, type="reddit", original_index=index
        )

    except (NotFound, Redirect):
        logging.warning(f"Reddit submission not found or invalid URL: {url}")
        return UrlFetchResult(
            url=url,
            content=None,
            error="Submission not found or invalid URL.",
            type="reddit",
            original_index=index,
        )
    except Forbidden as e:
        logging.warning(f"Reddit API Forbidden error for {url}: {e}")
        return UrlFetchResult(
            url=url,
            content=None,
            error="Reddit API access forbidden.",
            type="reddit",
            original_index=index,
        )
    except AsyncPrawRequestException as e:
        # Log specific details if available (e.g., response status)
        status = getattr(getattr(e, "response", None), "status", "N/A")
        logging.warning(f"Reddit API Request error for {url} (Status: {status}): {e}")
        return UrlFetchResult(
            url=url,
            content=None,
            error=f"Reddit API request error: {type(e).__name__}",
            type="reddit",
            original_index=index,
        )
    except Exception as e:
        logging.exception(f"Unexpected error fetching Reddit content for {url}")
        return UrlFetchResult(
            url=url,
            content=None,
            error=f"Unexpected error: {type(e).__name__}",
            type="reddit",
            original_index=index,
        )
    finally:
        # The global client is not closed here; it's closed when the bot shuts down.
        pass
