import asyncio
import logging
from typing import Dict, Any, Optional

import discord

from .config_loader import get_config
from .rate_limiter import close_all_db_managers
from ..bot.bot import LLMCordClient
from ..ui.sharing import stop_output_server, cleanup_shared_html_dir


logger = logging.getLogger(__name__)


class AppLifecycleManager:
    """Manages the application lifecycle including startup, shutdown, and cleanup."""

    def __init__(self):
        self.client: Optional[LLMCordClient] = None
        self.config: Optional[Dict[str, Any]] = None

    async def initialize_config(self) -> bool:
        """
        Initialize and validate configuration.

        Returns:
            bool: True if config loaded successfully, False otherwise
        """
        try:
            self.config = await get_config()
            if not self.config:
                logger.critical("Configuration could not be loaded")
                return False

            # Validate essential config
            if not self.config.get("bot_token"):
                logger.critical("bot_token not found in config.yaml")
                return False

            return True

        except FileNotFoundError:
            logger.critical(
                "config.yaml not found. Please copy config-example.yaml to config.yaml and configure it"
            )
            return False
        except Exception as e:
            logger.critical(f"Unexpected error loading config: {e}")
            return False

    def create_discord_client(self) -> LLMCordClient:
        """Create and configure the Discord client."""
        if not self.config:
            raise RuntimeError("Configuration must be initialized first")

        intents = discord.Intents.default()
        intents.message_content = True

        activity = discord.CustomActivity(
            name=(
                self.config.get("status_message") or "github.com/jakobdylanc/llmcord"
            )[:128]
        )

        self.client = LLMCordClient(
            intents=intents, activity=activity, config=self.config
        )

        return self.client

    def log_invite_url(self):
        """Log the bot invite URL if client_id is configured."""
        if not self.config:
            return

        client_id = self.config.get("client_id")
        if client_id:
            invite_url = (
                f"https://discord.com/api/oauth2/authorize?"
                f"client_id={client_id}&permissions=412317273088&scope=bot"
            )
            logger.info(f"\n\nBOT INVITE URL:\n{invite_url}\n")
        else:
            logger.warning(
                "client_id not found in config.yaml. Cannot generate invite URL"
            )

    async def start_bot(self) -> None:
        """Start the Discord bot with proper error handling."""
        if not self.client or not self.config:
            raise RuntimeError("Client and config must be initialized first")

        bot_token = self.config.get("bot_token")
        if not bot_token:
            raise ValueError("Bot token not found in configuration")

        try:
            logger.info("Starting bot...")
            await self.client.start(bot_token)
        except discord.LoginFailure:
            logger.critical(
                "Failed to log in. Please check your bot_token in config.yaml"
            )
            raise
        except discord.PrivilegedIntentsRequired:
            logger.critical(
                "Privileged Intents (Message Content) are not enabled for the bot "
                "in the Discord Developer Portal"
            )
            raise
        except Exception as e:
            logger.critical(f"Error starting Discord client: {e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        """Gracefully shutdown the application and cleanup resources."""
        logger.info("Starting graceful shutdown...")

        # Close Discord client
        if self.client and not self.client.is_closed():
            logger.info("Closing Discord client...")
            try:
                await self.client.close()
            except Exception as e:
                logger.error(f"Error closing Discord client: {e}", exc_info=True)

        # Stop output server
        logger.info("Stopping output server...")
        try:
            await asyncio.to_thread(stop_output_server)
        except Exception as e:
            logger.error(f"Error stopping output server: {e}", exc_info=True)

        # Cleanup shared HTML directory if configured
        await self._cleanup_shared_html()

        # Close database managers
        logger.info("Closing database managers...")
        try:
            close_all_db_managers()
        except Exception as e:
            logger.error(f"Error closing database managers: {e}", exc_info=True)

        logger.info("Shutdown complete")

    async def _cleanup_shared_html(self) -> None:
        """Cleanup shared HTML directory based on configuration."""
        if not self.config:
            return

        output_sharing_cfg = self.config.get("output_sharing", {})
        cleanup_enabled = output_sharing_cfg.get("cleanup_on_shutdown", True)

        if cleanup_enabled:
            logger.info("Cleaning up shared HTML directory...")
            try:
                await cleanup_shared_html_dir()
            except Exception as e:
                logger.error(
                    f"Error cleaning up shared HTML directory: {e}", exc_info=True
                )
        else:
            logger.info("Skipping shared HTML directory cleanup (disabled in config)")
