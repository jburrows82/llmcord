import asyncio
import logging

from src.core.logging_config import setup_logging
from src.core.app_lifecycle import AppLifecycleManager


async def main():
    """Main entry point for the LLMCord bot."""
    # Initialize logging first (but it's already disabled for performance)
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting LLMCord bot...")

    print("🤖 LLMCord bot is starting...")

    lifecycle_manager = AppLifecycleManager()

    try:
        # Initialize configuration
        if not await lifecycle_manager.initialize_config():
            return 1

        # Create and configure Discord client (stored in lifecycle_manager)
        lifecycle_manager.create_discord_client()

        # Log invite URL
        lifecycle_manager.log_invite_url()

        # Start the bot
        await lifecycle_manager.start_bot()

    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt)")
        print("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"Critical error in main: {e}", exc_info=True)
        print(f"❌ Critical error: {e}")
        return 1
    finally:
        # Graceful shutdown
        await lifecycle_manager.shutdown()

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("🛑 Application interrupted by user")
        logging.info("Application interrupted by user")
        exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        exit(1)
