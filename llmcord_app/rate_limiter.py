import os
import sqlite3
import time
import logging
from datetime import timedelta
from typing import Set, Dict, List
import aiofiles
import aiofiles.os as aio_os

from .constants import DB_FOLDER, RATE_LIMIT_COOLDOWN_SECONDS, GLOBAL_RESET_FILE


# --- Rate Limit Database Manager ---
class RateLimitDBManager:
    """Manages the SQLite database for tracking rate-limited API keys."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._connect()

    def _connect(self):  # This remains synchronous due to sqlite3
        """Establishes connection to the database and creates table if needed."""
        try:
            # Synchronous os.makedirs is fine here as it's part of a sync method
            # If this whole class were to be made async with aiosqlite, this would change.
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.conn = sqlite3.connect(self.db_path)
            self._create_table()
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database {self.db_path}: {e}")
            self.conn = None  # Ensure conn is None if connection fails

    def _create_table(self):
        """Creates the rate_limited_keys table if it doesn't exist."""
        if not self.conn:
            return
        try:
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS rate_limited_keys (
                        api_key TEXT PRIMARY KEY,
                        ratelimit_timestamp REAL NOT NULL
                    )
                """)
        except sqlite3.Error as e:
            logging.error(f"Error creating table in {self.db_path}: {e}")

    def add_key(self, api_key: str):
        """Marks an API key as rate-limited with the current timestamp."""
        if not self.conn:
            logging.error(f"Cannot add key, no connection to {self.db_path}")
            return
        if not api_key:  # Prevent adding empty keys
            logging.warning(
                f"Attempted to add empty API key to rate limit DB {self.db_path}"
            )
            return
        timestamp = time.time()
        try:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO rate_limited_keys (api_key, ratelimit_timestamp)
                    VALUES (?, ?)
                """,
                    (api_key, timestamp),
                )
            logging.info(
                f"Key ending with ...{api_key[-4:]} marked as rate-limited in {os.path.basename(self.db_path)}."
            )
        except sqlite3.Error as e:
            logging.error(f"Error adding key ...{api_key[-4:]} to {self.db_path}: {e}")

    def get_limited_keys(self, cooldown_seconds: int) -> Set[str]:
        """Returns the set of API keys currently within the cooldown period."""
        if not self.conn:
            logging.error(f"Cannot get limited keys, no connection to {self.db_path}")
            return set()
        cutoff_time = time.time() - cooldown_seconds
        limited_keys = set()
        try:
            with self.conn:
                cursor = self.conn.execute(
                    """
                    SELECT api_key FROM rate_limited_keys WHERE ratelimit_timestamp >= ?
                """,
                    (cutoff_time,),
                )
                limited_keys = {row[0] for row in cursor.fetchall()}
        except sqlite3.Error as e:
            logging.error(f"Error getting limited keys from {self.db_path}: {e}")
        return limited_keys

    def reset_db(self):
        """Removes all entries from the rate limit table."""
        if not self.conn:
            logging.error(f"Cannot reset DB, no connection to {self.db_path}")
            return
        try:
            with self.conn:
                self.conn.execute("DELETE FROM rate_limited_keys")
            logging.info(f"Rate limit database {os.path.basename(self.db_path)} reset.")
        except sqlite3.Error as e:
            logging.error(f"Error resetting database {self.db_path}: {e}")

    def close(self):
        """Closes the database connection."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                logging.debug(f"Closed database connection: {self.db_path}")
            except sqlite3.Error as e:
                logging.error(f"Error closing database {self.db_path}: {e}")


# --- Global Rate Limit Management ---
db_managers: Dict[str, RateLimitDBManager] = {}


def get_db_manager(service_name: str) -> RateLimitDBManager:
    """Gets or creates a DB manager for a specific service."""
    global db_managers
    if service_name not in db_managers:
        # Ensure DB_FOLDER exists before creating RateLimitDBManager
        # This is a synchronous operation, acceptable during initialization.
        if not os.path.exists(DB_FOLDER):
            os.makedirs(DB_FOLDER, exist_ok=True)

        db_path = os.path.join(
            DB_FOLDER, f"ratelimit_{service_name.lower().replace('-', '_')}.db"
        )
        db_managers[service_name] = RateLimitDBManager(db_path)
    # Ensure connection is alive (e.g., if it failed initially)
    if db_managers[service_name].conn is None:
        db_managers[service_name]._connect()  # This is a sync connect
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

    if now - last_reset_time >= RATE_LIMIT_COOLDOWN_SECONDS:
        logging.info("Performing global 24-hour rate limit database reset.")

        # Ensure DB_FOLDER exists before any DB operations
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
            manager = get_db_manager(service_name)  # get_db_manager is sync
            manager.reset_db()  # reset_db is sync

        for manager in list(db_managers.values()):
            manager.reset_db()  # reset_db is sync

        try:
            async with aiofiles.open(GLOBAL_RESET_FILE, "w") as f:
                await f.write(str(now))
        except IOError as e:
            logging.error(
                f"Could not write last reset timestamp to {GLOBAL_RESET_FILE}: {e}"
            )
    else:
        time_since_reset = timedelta(seconds=now - last_reset_time)
        next_reset_in = timedelta(
            seconds=RATE_LIMIT_COOLDOWN_SECONDS - (now - last_reset_time)
        )
        logging.info(
            f"Time since last global reset: {time_since_reset}. Next reset in approx {next_reset_in}."
        )


async def get_available_keys(service_name: str, all_keys: List[str]) -> List[str]:
    """Gets available (non-rate-limited) keys for a service. (DB ops remain sync)"""
    if not all_keys:
        return []

    # Ensure DB_FOLDER exists before get_db_manager is called, if it might create a new DB
    # This is a bit redundant if check_and_perform_global_reset already ran, but safe.
    if not await aio_os.path.exists(DB_FOLDER) and service_name not in db_managers:
        await aio_os.makedirs(DB_FOLDER)

    db_manager = get_db_manager(service_name)  # get_db_manager is sync
    limited_keys = db_manager.get_limited_keys(
        RATE_LIMIT_COOLDOWN_SECONDS
    )  # get_limited_keys is sync
    available_keys = [key for key in all_keys if key not in limited_keys]

    if (
        not available_keys and all_keys
    ):  # Check all_keys to avoid resetting if none were configured
        logging.warning(
            f"All keys for service '{service_name}' are currently rate-limited. Resetting DB and using full list for this attempt."
        )
        db_manager.reset_db()  # reset_db is sync
        return all_keys  # Return the full list after reset

    return available_keys


def close_all_db_managers():  # This remains synchronous
    """Closes all active database manager connections."""
    global db_managers
    logging.info("Closing database connections...")
    for manager in db_managers.values():
        manager.close()
    db_managers = {}  # Clear the dictionary
    logging.info("Database connections closed.")
