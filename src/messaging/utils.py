from contextlib import asynccontextmanager
import discord


@asynccontextmanager
async def safe_typing(channel: discord.abc.Messageable):
    """A drop-in replacement for `channel.typing()` that swallows 403 errors."""
    try:
        async with channel.typing():
            yield
    except discord.Forbidden:
        # Missing permission or Cloudflare block (error code 40333).
        # Continuing without it.
        yield
    except discord.HTTPException as e:
        # Catch any other 403 response that does not raise the concrete
        # Forbidden subtype (discord.py sometimes raises raw HTTPException).
        if getattr(e, "status", None) == 403 or getattr(e, "code", None) == 40333:
            yield
        else:
            # Re-raise unexpected HTTP errors so they are still surfaced.
            raise
