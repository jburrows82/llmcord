import asyncio
import logging
import sys
import discord

# Import necessary components using absolute paths from the package root
from llmcord_app.config import get_config
from llmcord_app.bot import LLMCordClient
from llmcord_app.rate_limiter import close_all_db_managers  # Import cleanup function
from llmcord_app.output_server import (
    stop_output_server,
    cleanup_shared_html_dir,
)  # Import output server cleanup

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",  # Added logger name
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("discord.http").setLevel(logging.WARNING)  # Reduce discord http noise
logging.getLogger("websockets.client").setLevel(
    logging.WARNING
)  # Reduce websocket noise
logging.getLogger("asyncprawcore").setLevel(logging.WARNING)  # Reduce asyncpraw noise


async def main():
    """Main entry point for the bot."""
    # Load configuration
    cfg = get_config()  # Load config once

    # Basic check for bot token after loading config
    bot_token = cfg.get("bot_token")
    if not bot_token:
        logging.critical("CRITICAL: bot_token not found in config.yaml. Exiting.")
        return  # Exit if token is missing

    # Setup Discord intents and activity
    intents = discord.Intents.default()
    intents.message_content = True
    activity = discord.CustomActivity(
        name=(cfg.get("status_message") or "github.com/jakobdylanc/llmcord")[:128]
    )

    # Create and run the client
    client = LLMCordClient(intents=intents, activity=activity, config=cfg)

    # Display invite URL if client_id is present
    client_id = cfg.get("client_id")
    if client_id:
        invite_url = f"https://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot"
        logging.info(f"\n\nBOT INVITE URL:\n{invite_url}\n")
    else:
        logging.warning(
            "client_id not found in config.yaml. Cannot generate invite URL."
        )

    try:
        logging.info("Starting bot...")
        await client.start(bot_token)
    except discord.LoginFailure:
        logging.critical(
            "Failed to log in. Please check your bot_token in config.yaml."
        )
    except discord.PrivilegedIntentsRequired:
        logging.critical(
            "Privileged Intents (Message Content) are not enabled for the bot in the Discord Developer Portal."
        )
    except Exception as e:
        logging.critical(f"Error starting Discord client: {e}", exc_info=True)
    finally:
        if not client.is_closed():
            await client.close()  # Ensure client resources are cleaned up

        # Stop output server if it was running
        logging.info("Attempting to stop output server...")
        try:
            await asyncio.to_thread(stop_output_server)
        except Exception as e:
            logging.error(f"Error during output server stop: {e}", exc_info=True)

        # Conditionally cleanup shared HTML directory
        # Access cfg directly as it's in scope of main()
        output_sharing_cfg = cfg.get("output_sharing", {})
        cleanup_enabled = output_sharing_cfg.get("cleanup_on_shutdown", True)

        if cleanup_enabled:
            logging.info(
                "Attempting to cleanup shared HTML directory as cleanup_on_shutdown is true..."
            )
            try:
                await asyncio.to_thread(cleanup_shared_html_dir)
            except Exception as e:
                logging.error(
                    f"Error during shared HTML directory cleanup: {e}", exc_info=True
                )
        else:
            logging.info(
                "Skipping shared HTML directory cleanup as cleanup_on_shutdown is false."
            )

        logging.info("Bot has been shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user (KeyboardInterrupt).")
    finally:
        # Ensure DB connections are closed even if asyncio loop is interrupted
        close_all_db_managers()
        # Also ensure output server is stopped on broader script exit
        # This is a secondary cleanup, primary is in client.close() or main's finally
        logging.info("Ensuring output server is stopped on script exit...")
        try:
            # Running synchronous stop_output_server directly here as asyncio loop might be closed
            # If main's finally is called after loop closure, to_thread won't work.
            # For simplicity, and given it's a cleanup, direct call is acceptable.
            # If issues, this might need a separate synchronous cleanup handler or atexit.
            # However, the call within main's async finally block with to_thread is preferred.
            # For this specific location (outer finally), direct call is more robust if loop is gone.
            # Re-evaluating: The asyncio.run(main()) means this finally block is outside the async context.
            # So, a direct synchronous call to a potentially async-dependent function is tricky.
            # The call within the async main() function's finally block is the correct place.
            # This secondary call here might be redundant or problematic.
            # Let's rely on the cleanup within the async main() function's `finally` block.
            # If that fails, the atexit in output_server.py is a last resort.
            pass  # Relying on cleanup within async main()
        except Exception as e:
            logging.error(
                f"Error during final output server stop attempt: {e}", exc_info=True
            )
