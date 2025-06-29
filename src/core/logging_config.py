import logging
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str = "logs/llmcord_errors.log"):
    """
    Configure logging for LLMCord bot.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file for warnings and errors
    """
    # Ensure logs directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure log formatter
    log_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console Handler - only show warnings, errors, and critical messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # File Handler for warnings, errors and critical messages
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.WARNING)  # Log warnings, errors and critical messages to file
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except (OSError, IOError) as e:
        print(f"Could not create log file {log_file}: {e}")  # Use print since logging might not be available yet

    # Suppress noisy third-party loggers
    logging.getLogger("discord.http").setLevel(logging.WARNING)
    logging.getLogger("websockets.client").setLevel(logging.WARNING)
    logging.getLogger("asyncprawcore").setLevel(logging.WARNING)

    logging.info("Logging configuration initialized")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)
