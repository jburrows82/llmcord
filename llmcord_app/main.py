import asyncio
import logging
import sys
import discord

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s", # Added logger name
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("discord.http").setLevel(logging.WARNING) # Reduce discord http noise
logging.getLogger("websockets.client").setLevel(logging.WARNING) # Reduce websocket noise
logging.getLogger("asyncprawcore").setLevel(logging.WARNING) # Reduce asyncpraw noise
logging.getLogger("filelock").setLevel(logging.WARNING) # Reduce playwright filelock noise


# Import necessary components using absolute paths from the package root
from llmcord_app.config import get_config
from llmcord_app.bot import LLMCordClient
from llmcord_app.rate_limiter import close_all_db_managers # Import cleanup function

async def main():
    """Main entry point for the bot."""
    # Load configuration
    cfg = get_config() # Load config once

    # Basic check for bot token after loading config
    bot_token = cfg.get("bot_token")
    if not bot_token:
        logging.critical("CRITICAL: bot_token not found in config.yaml. Exiting.")
        return # Exit if token is missing

    # Setup Discord intents and activity
    intents = discord.Intents.default()
    intents.message_content = True
    activity = discord.CustomActivity(name=(cfg.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])

    # Create and run the client
    client = LLMCordClient(intents=intents, activity=activity, config=cfg)

    # Display invite URL if client_id is present
    client_id = cfg.get("client_id")
    if client_id:
        invite_url = f"https://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot"
        logging.info(f"\n\nBOT INVITE URL:\n{invite_url}\n")
    else:
        logging.warning("client_id not found in config.yaml. Cannot generate invite URL.")


    try:
        logging.info("Starting bot...")
        await client.start(bot_token)
    except discord.LoginFailure:
        logging.critical("Failed to log in. Please check your bot_token in config.yaml.")
    except discord.PrivilegedIntentsRequired:
         logging.critical("Privileged Intents (Message Content) are not enabled for the bot in the Discord Developer Portal.")
    except Exception as e:
        logging.critical(f"Error starting Discord client: {e}", exc_info=True)
    finally:
        if not client.is_closed():
            await client.close() # Ensure client resources are cleaned up
        logging.info("Bot has been shut down.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user (KeyboardInterrupt).")
    finally:
        # Ensure DB connections are closed even if asyncio loop is interrupted
        close_all_db_managers()