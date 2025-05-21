import os
import sqlite3
import time
import logging
import asyncio  # Added asyncio
from datetime import timedelta
from typing import Set, Dict, List, Any  # Added Any
import aiofiles
import aiofiles.os as aio_os

from .constants import (
    DB_FOLDER,
    RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY,
    GLOBAL_RESET_FILE,
)


# --- Rate Limit Database Manager ---
class RateLimitDBManager:
    """Manages the SQLite database for tracking rate-limited API keys."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        # self.conn is no longer an instance variable managed across calls.

    def _create_table_if_not_exists(self, conn: sqlite3.Connection):
        """Creates the rate_limited_keys table if it doesn't exist using the provided connection."""
        try:
            with conn:  # Use the passed connection for a transaction
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS rate_limited_keys (
                        api_key TEXT PRIMARY KEY,
                        ratelimit_timestamp REAL NOT NULL
                    )
                """)
        except sqlite3.Error as e:
            logging.error(f"Error creating table in {self.db_path}: {e}")
            raise  # Re-raise to be caught by the calling sync method

    def _get_connection_and_ensure_table(self) -> sqlite3.Connection:
        """Establishes a new SQLite connection and ensures the table exists."""
        # Ensure the directory for the specific DB file exists.
        # This is a synchronous call, acceptable as it's within a sync helper.
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        self._create_table_if_not_exists(conn)
        return conn

    async def add_key(self, api_key: str):
        """Marks an API key as rate-limited with the current timestamp."""
        if not api_key:
            logging.warning(
                f"Attempted to add empty API key to rate limit DB {self.db_path}"
            )
            return

        timestamp = time.time()

        def _sync_add_key():
            conn = None
            try:
                conn = self._get_connection_and_ensure_table()
                with conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO rate_limited_keys (api_key, ratelimit_timestamp)
                        VALUES (?, ?)
                        """,
                        (api_key, timestamp),
                    )
                # conn.commit() is handled by `with conn:`
                logging.info(
                    f"Key ending with ...{api_key[-4:]} marked as rate-limited in {os.path.basename(self.db_path)}."
                )
            except sqlite3.Error as e:
                logging.error(
                    f"Error adding key ...{api_key[-4:]} to {self.db_path}: {e}"
                )
            finally:
                if conn:
                    conn.close()

        await asyncio.to_thread(_sync_add_key)

    async def get_limited_keys(self, cooldown_seconds: int) -> Set[str]:
        """Returns the set of API keys currently within the cooldown period."""
        cutoff_time = time.time() - cooldown_seconds

        def _sync_get_limited_keys() -> Set[str]:
            conn = None
            limited_keys_set = set()
            try:
                conn = self._get_connection_and_ensure_table()
                cursor = conn.execute(
                    """
                    SELECT api_key FROM rate_limited_keys WHERE ratelimit_timestamp >= ?
                    """,
                    (cutoff_time,),
                )
                rows = cursor.fetchall()
                limited_keys_set = {row[0] for row in rows}
            except sqlite3.Error as e:
                logging.error(f"Error getting limited keys from {self.db_path}: {e}")
            finally:
                if conn:
                    conn.close()
            return limited_keys_set

        return await asyncio.to_thread(_sync_get_limited_keys)

    async def reset_db(self):
        """Removes all entries from the rate limit table."""

        def _sync_reset_db():
            conn = None
            try:
                conn = self._get_connection_and_ensure_table()
                with conn:
                    conn.execute("DELETE FROM rate_limited_keys")
                # conn.commit() is handled by `with conn:`
                logging.info(
                    f"Rate limit database {os.path.basename(self.db_path)} reset."
                )
            except sqlite3.Error as e:
                logging.error(f"Error resetting database {self.db_path}: {e}")
            finally:
                if conn:
                    conn.close()

        await asyncio.to_thread(_sync_reset_db)

    # close() method is removed as connections are now managed per operation.


# --- Global Rate Limit Management ---
db_managers: Dict[str, RateLimitDBManager] = {}


async def get_db_manager(service_name: str) -> RateLimitDBManager:
    """Gets or creates a DB manager for a specific service."""
    global db_managers
    if service_name not in db_managers:
        # Ensure the top-level DB_FOLDER exists.
        # This is an async call, appropriate for an async function.
        if not await aio_os.path.exists(DB_FOLDER):
            await aio_os.makedirs(DB_FOLDER, exist_ok=True)

        db_path = os.path.join(
            DB_FOLDER, f"ratelimit_{service_name.lower().replace('-', '_')}.db"
        )
        # RateLimitDBManager __init__ is synchronous.
        db_managers[service_name] = RateLimitDBManager(db_path)

    # No need to explicitly check/re-establish connection here,
    # as each operation in RateLimitDBManager now handles its own connection.
    return db_managers[service_name]


async def check_and_perform_global_reset(cfg: Dict):
    """Checks if 24 hours have passed since the last reset and resets all DBs if so. (Async file ops)"""
    now = time.time()
    last_reset_time = 0.0
    try:
        if await aio_os.path.exists(GLOBAL_RESET_FILE):
            async with aiofiles.open(GLOBAL_RESET_FILE, "r") as f:
                content = await f.read()
                content = content.strip()
                if content:  # Ensure file is not empty
                    last_reset_time = float(content)
                else:
                    logging.warning(f"{GLOBAL_RESET_FILE} is empty. Forcing reset.")
                    last_reset_time = 0.0
    except (IOError, ValueError) as e:
        logging.warning(
            f"Could not read last reset timestamp from {GLOBAL_RESET_FILE}: {e}. Forcing reset."
        )
        last_reset_time = 0.0  # Force reset if file is invalid or missing

    cooldown_hours = cfg.get(RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY, 24)
    cooldown_seconds = cooldown_hours * 60 * 60
    if now - last_reset_time >= cooldown_seconds:
        logging.info(
            f"Performing global {cooldown_hours}-hour rate limit database reset."
        )

        if not await aio_os.path.exists(DB_FOLDER):
            await aio_os.makedirs(DB_FOLDER)

        services_in_config = set()
        providers = cfg.get("providers", {})
        for provider_name, provider_cfg in providers.items():
            if isinstance(provider_cfg, dict) and provider_cfg.get("api_keys"):
                services_in_config.add(provider_name)
        if cfg.get("serpapi_api_keys"):
            services_in_config.add("serpapi")

        for service_name in services_in_config:
            manager = await get_db_manager(service_name)  # await here
            await manager.reset_db()  # await here

        # Reset any other managers that might exist but are not in current config
        # (e.g., if a service was removed from config but its DB file still exists)
        current_managers_copy = list(db_managers.values())  # Iterate over a copy
        for manager in current_managers_copy:
            await manager.reset_db()  # await here

        try:
            async with aiofiles.open(GLOBAL_RESET_FILE, "w") as f:
                await f.write(str(now))
        except IOError as e:
            logging.error(
                f"Could not write last reset timestamp to {GLOBAL_RESET_FILE}: {e}"
            )
    else:
        time_since_reset = timedelta(seconds=now - last_reset_time)
        next_reset_in = timedelta(seconds=cooldown_seconds - (now - last_reset_time))
        logging.info(
            f"Time since last global reset: {time_since_reset}. Next reset in approx {next_reset_in}."
        )


async def get_available_keys(
    service_name: str, all_keys: List[str], app_config: Dict[str, Any]
) -> List[str]:
    """Gets available (non-rate-limited) keys for a service."""
    if not all_keys:
        return []

    if not await aio_os.path.exists(DB_FOLDER) and service_name not in db_managers:
        await aio_os.makedirs(DB_FOLDER)

    db_manager = await get_db_manager(service_name)  # await here
    cooldown_hours = app_config.get(RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY, 24)
    cooldown_seconds = cooldown_hours * 60 * 60
    limited_keys = await db_manager.get_limited_keys(  # await here
        cooldown_seconds
    )
    available_keys = [key for key in all_keys if key not in limited_keys]

    if not available_keys and all_keys:
        logging.warning(
            f"All keys for service '{service_name}' are currently rate-limited. Resetting DB and using full list for this attempt."
        )
        await db_manager.reset_db()  # await here
        return all_keys

    return available_keys


def close_all_db_managers():  # This remains synchronous
    """Closes all active database manager connections."""
    global db_managers
    logging.info("Closing database connections...")
    # Connections are managed per-operation, so no explicit close needed for managers.
    # We just clear the global dictionary.
    db_managers = {}
    logging.info("Database manager references cleared.")
