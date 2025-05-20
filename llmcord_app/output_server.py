import logging
import os

# import shutil  # Replaced by aio_os operations
import aiofiles  # Added import
import aiofiles.os as aio_os
import time
import uuid
from typing import Optional, Dict, Any
import http.server  # Remains synchronous
import socketserver  # Remains synchronous
import threading  # Remains synchronous
import httpx

from pyngrok import ngrok, conf as ngrok_conf  # Remains synchronous
from pyngrok.exception import PyngrokError
import markdown2

from .constants import (
    OUTPUT_SHARING_CONFIG_KEY,
    NGROK_ENABLED_CONFIG_KEY,
    NGROK_AUTHTOKEN_CONFIG_KEY,
    GRIP_PORT_CONFIG_KEY,  # This will now be the port for our Python HTTP server
    DEFAULT_GRIP_PORT,  # Default port for our Python HTTP server
    NGROK_STATIC_DOMAIN_CONFIG_KEY,
    URL_SHORTENER_ENABLED_CONFIG_KEY,
    URL_SHORTENER_SERVICE_CONFIG_KEY,
    # OUTPUT_FILENAME, # No longer a single output file
)

# New constant for the directory to store HTML files
SHARED_HTML_DIR = "shared_html_outputs"

# Module-level variables
_http_server_thread: Optional[threading.Thread] = None
_http_server_instance: Optional[socketserver.TCPServer] = None
_ngrok_tunnel: Optional[ngrok.NgrokTunnel] = None
_server_port: int = DEFAULT_GRIP_PORT  # Reuse constant, now for Python HTTP server


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SHARED_HTML_DIR, **kwargs)


def _start_http_server_thread(port: int):
    global _http_server_instance
    try:
        # Ensure the directory exists before starting the server
        os.makedirs(SHARED_HTML_DIR, exist_ok=True)

        _http_server_instance = socketserver.TCPServer(("", port), Handler)
        logging.info(
            f"Python HTTP server starting on port {port}, serving from {SHARED_HTML_DIR}"
        )
        _http_server_instance.serve_forever()
        logging.info(f"Python HTTP server on port {port} has shut down.")
    except Exception as e:
        logging.error(
            f"Python HTTP server failed to start or run on port {port}: {e}",
            exc_info=True,
        )
    finally:
        _http_server_instance = None  # Clear instance if server stops


def is_http_server_running() -> bool:
    global _http_server_thread
    return _http_server_thread is not None and _http_server_thread.is_alive()


def _is_ngrok_tunnel_active() -> bool:
    global _ngrok_tunnel
    if _ngrok_tunnel:
        try:
            active_tunnels = ngrok.get_tunnels()
            return any(t.public_url == _ngrok_tunnel.public_url for t in active_tunnels)
        except PyngrokError:
            return False
    return False


def _shorten_url_tinyurl(long_url: str) -> Optional[str]:
    """Shortens a URL using TinyURL's API."""
    if not long_url:
        return None
    try:
        # TinyURL API does not require an API key for basic usage
        api_url = f"http://tinyurl.com/api-create.php?url={long_url}"
        response = httpx.get(api_url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors 4xx/5xx
        short_url = response.text.strip()
        # Basic validation: check if it looks like a URL
        if short_url.startswith("http://") or short_url.startswith("https://"):
            logging.info(
                f"Successfully shortened {long_url} to {short_url} using TinyURL."
            )
            return short_url
        else:
            logging.warning(
                f"TinyURL returned an unexpected response: '{short_url}' for URL: {long_url}"
            )
            return None
    except httpx.RequestError as e:
        logging.error(f"Error requesting TinyURL for {long_url}: {e}")
        return None
    except httpx.HTTPStatusError as e:
        logging.error(
            f"TinyURL API returned an error for {long_url}: {e.response.status_code} - {e.response.text}"
        )
        return None
    except Exception as e:
        logging.error(
            f"Unexpected error shortening URL {long_url} with TinyURL: {e}",
            exc_info=True,
        )
        return None


async def start_output_server(
    text_content: str, config: Dict[str, Any]
) -> Optional[str]:
    """
    Converts markdown text_content to HTML, saves it to a unique file,
    ensures a local HTTP server is running to serve these files,
    and exposes the server via an ngrok tunnel.

    Returns the public URL to the specific HTML file if successful, otherwise None.
    This function is synchronous.
    """
    global _http_server_thread, _ngrok_tunnel, _server_port

    output_sharing_cfg = config.get(OUTPUT_SHARING_CONFIG_KEY, {})
    ngrok_enabled = output_sharing_cfg.get(NGROK_ENABLED_CONFIG_KEY, False)
    ngrok_authtoken = output_sharing_cfg.get(NGROK_AUTHTOKEN_CONFIG_KEY)
    ngrok_static_domain = output_sharing_cfg.get(
        NGROK_STATIC_DOMAIN_CONFIG_KEY
    )  # Get static domain
    server_port_from_config = output_sharing_cfg.get(
        GRIP_PORT_CONFIG_KEY, DEFAULT_GRIP_PORT
    )

    if not ngrok_enabled:
        logging.info("Output sharing via ngrok is disabled in config.")
        # No need to stop server here, as it might be serving other files.
        # stop_output_server() called from main.py will handle full shutdown.
        return None

    # If port in config changed, we might need to restart server.
    # For simplicity now, we use the port set at module load or first call.
    # A more robust solution would handle port changes by restarting the server.
    _server_port = server_port_from_config

    try:
        await aio_os.makedirs(SHARED_HTML_DIR, exist_ok=True)

        # Generate unique filename
        unique_filename = f"{uuid.uuid4().hex}.html"
        html_filepath = os.path.join(SHARED_HTML_DIR, unique_filename)

        # Convert Markdown to HTML
        html_content = markdown2.markdown(
            text_content, extras=["fenced-code-blocks", "tables", "strike", "task_list"]
        )

        # Basic HTML structure for better viewing
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Output</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; line-height: 1.6; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        code {{ font-family: monospace; }}
        table {{ border-collapse: collapse; width: auto; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

        async with aiofiles.open(html_filepath, "w", encoding="utf-8") as f:
            await f.write(full_html)
        logging.info(f"LLM output converted to HTML and saved to {html_filepath}")

        # Start HTTP server if not already running
        if not is_http_server_running():
            logging.info(
                f"HTTP server thread not running. Starting new one on port {_server_port}."
            )
            _http_server_thread = threading.Thread(
                target=_start_http_server_thread, args=(_server_port,), daemon=True
            )
            _http_server_thread.start()
            time.sleep(1)  # Give server a moment to start

            if not is_http_server_running():
                logging.error(
                    f"Python HTTP server thread failed to start on port {_server_port}."
                )
                if await aio_os.path.exists(html_filepath):
                    await aio_os.remove(html_filepath)  # Clean up generated file
                return None

        # Configure ngrok authtoken if provided (should only be needed once)
        # Pyngrok typically persists this config.
        if ngrok_authtoken:
            try:
                # Check current authtoken to avoid unnecessary calls if possible, though pyngrok handles this.
                current_ngrok_config = ngrok_conf.get_default()
                if (
                    not current_ngrok_config
                    or current_ngrok_config.auth_token != ngrok_authtoken
                ):
                    ngrok.set_auth_token(ngrok_authtoken)
                    logging.info("Ngrok authtoken configured.")
            except PyngrokError as e:
                logging.warning(
                    f"Failed to set ngrok authtoken: {e}. May impact tunnel stability."
                )
            except Exception as e:  # Catch any other error during authtoken set
                logging.warning(f"Unexpected error setting ngrok authtoken: {e}")

        # Start ngrok tunnel if not already active or if port changed (though port change not fully handled yet)
        # For now, we assume one ngrok tunnel for the HTTP server's port.
        # If _ngrok_tunnel exists but points to a different port, it should be closed.
        # This simplified logic just checks if any tunnel is active for the current _server_port.

        active_tunnels = ngrok.get_tunnels()
        existing_tunnel_for_port = None
        for t in active_tunnels:
            if t.config["addr"].endswith(str(_server_port)):
                existing_tunnel_for_port = t
                break

        if existing_tunnel_for_port:
            _ngrok_tunnel = existing_tunnel_for_port
            logging.info(
                f"Reusing existing ngrok tunnel for port {_server_port}: {_ngrok_tunnel.public_url}"
            )
        else:
            if _ngrok_tunnel:  # If a tunnel exists but not for current port (or died)
                try:
                    ngrok.disconnect(_ngrok_tunnel.public_url)
                except Exception:
                    pass
                _ngrok_tunnel = None
            try:
                logging.info(
                    f"Starting new ngrok tunnel for Python HTTP server on port {_server_port}..."
                )
                pyngrok_config = ngrok_conf.get_default()  # Get current config

                connect_kwargs = {"pyngrok_config": pyngrok_config}
                if ngrok_static_domain:
                    connect_kwargs["hostname"] = ngrok_static_domain
                    logging.info(
                        f"Attempting to use static domain: {ngrok_static_domain}"
                    )

                _ngrok_tunnel = ngrok.connect(_server_port, "http", **connect_kwargs)
                logging.info(f"Ngrok tunnel established: {_ngrok_tunnel.public_url}")
            except PyngrokError as e:
                logging.error(f"Failed to start ngrok tunnel: {e}")
                # Don't stop the HTTP server here, it might be needed for other files.
                if await aio_os.path.exists(html_filepath):
                    await aio_os.remove(html_filepath)
                return None
            except Exception as e:
                logging.error(
                    f"Unexpected error starting ngrok tunnel: {e}", exc_info=True
                )
                if await aio_os.path.exists(html_filepath):
                    await aio_os.remove(html_filepath)
                return None

        if not _ngrok_tunnel or not _ngrok_tunnel.public_url:
            logging.error("Ngrok tunnel could not be established or has no public URL.")
            if await aio_os.path.exists(html_filepath):
                await aio_os.remove(html_filepath)
            return None

        public_file_url = f"{_ngrok_tunnel.public_url}/{unique_filename}"
        logging.info(f"Raw public URL for HTML output: {public_file_url}")

        # URL Shortening
        shortener_enabled = output_sharing_cfg.get(
            URL_SHORTENER_ENABLED_CONFIG_KEY, False
        )
        shortener_service = output_sharing_cfg.get(
            URL_SHORTENER_SERVICE_CONFIG_KEY, "tinyurl"
        )

        final_url_to_share = public_file_url

        if shortener_enabled and public_file_url:
            logging.info(f"URL shortener enabled, service: {shortener_service}")
            if shortener_service.lower() == "tinyurl":
                shortened_url = _shorten_url_tinyurl(public_file_url)
                if shortened_url:
                    final_url_to_share = shortened_url
                else:
                    logging.warning(
                        f"Failed to shorten URL with TinyURL, using original: {public_file_url}"
                    )
            # Add other shortener services here with 'elif shortener_service == "otherservice":'
            else:
                logging.warning(
                    f"Unsupported URL shortener service: '{shortener_service}'. Using original URL."
                )

        logging.info(f"Final URL to share: {final_url_to_share}")
        return final_url_to_share

    except Exception as e:
        logging.error(f"Error in start_output_server: {e}", exc_info=True)
        # Don't stop the HTTP server here.
        return None


def stop_output_server():
    """
    Stops the ngrok tunnel and the Python HTTP server.
    This function is synchronous.
    """
    global _ngrok_tunnel, _http_server_instance, _http_server_thread

    if _ngrok_tunnel:
        try:
            logging.info(f"Closing ngrok tunnel: {_ngrok_tunnel.public_url}")
            ngrok.disconnect(_ngrok_tunnel.public_url)
        except PyngrokError as e:
            logging.error(
                f"Error disconnecting ngrok tunnel {_ngrok_tunnel.public_url}: {e}"
            )
        except Exception as e:
            logging.error(
                f"Unexpected error disconnecting ngrok tunnel: {e}", exc_info=True
            )
        finally:
            _ngrok_tunnel = None

    # Ensure all ngrok processes are killed
    try:
        ngrok.kill()
        logging.info("Ngrok process killed.")
    except PyngrokError as e:
        if "must be running" not in str(e).lower():  # It's ok if it wasn't running
            logging.warning(f"Error killing ngrok process: {e}")
    except Exception as e:
        logging.warning(f"Unexpected error killing ngrok process: {e}", exc_info=True)

    if _http_server_instance:
        try:
            logging.info(f"Shutting down Python HTTP server on port {_server_port}...")
            _http_server_instance.shutdown()  # Signal server to stop serve_forever loop
            _http_server_instance.server_close()  # Close the server socket
            logging.info("Python HTTP server shutdown initiated.")
        except Exception as e:
            logging.error(f"Error shutting down Python HTTP server: {e}", exc_info=True)
        finally:
            _http_server_instance = None

    if _http_server_thread and _http_server_thread.is_alive():
        try:
            logging.info("Waiting for Python HTTP server thread to join...")
            _http_server_thread.join(timeout=5)
            if _http_server_thread.is_alive():
                logging.warning("Python HTTP server thread did not join in time.")
            else:
                logging.info("Python HTTP server thread joined.")
        except Exception as e:
            logging.error(
                f"Error joining Python HTTP server thread: {e}", exc_info=True
            )
        finally:
            _http_server_thread = None

    logging.info("Output server cleanup process complete.")


async def async_rmtree(path: str):
    """Asynchronously removes a directory tree."""
    if not await aio_os.path.isdir(path):
        logging.warning(f"async_rmtree: {path} is not a directory or does not exist.")
        return
    try:
        entries = await aio_os.listdir(path)
        for entry in entries:
            entry_path = os.path.join(
                path, entry
            )  # os.path.join is fine for path manipulation
            if await aio_os.path.isdir(entry_path):
                await async_rmtree(entry_path)
            else:
                await aio_os.remove(entry_path)
        await aio_os.rmdir(path)
        logging.debug(f"Successfully removed directory tree: {path}")
    except Exception as e:
        logging.error(f"Error in async_rmtree for {path}: {e}", exc_info=True)
        # Depending on desired behavior, you might want to re-raise or handle more gracefully


async def cleanup_shared_html_dir():
    """Asynchronously removes the shared HTML directory and its contents."""
    if await aio_os.path.exists(SHARED_HTML_DIR):
        try:
            await async_rmtree(SHARED_HTML_DIR)
            logging.info(
                f"Asynchronously removed shared HTML directory: {SHARED_HTML_DIR}"
            )
        except Exception as e:  # Catching broader exceptions from async_rmtree if any
            logging.error(
                f"Error asynchronously removing shared HTML directory {SHARED_HTML_DIR}: {e}",
                exc_info=True,
            )
    else:
        logging.info(
            f"Shared HTML directory {SHARED_HTML_DIR} does not exist. No cleanup needed."
        )


# No atexit here, cleanup should be managed by main.py
