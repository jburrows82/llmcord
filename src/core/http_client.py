# NOTE: Module docstring must precede any future-import statements for PEP 236 compliance.
# The `annotations` future import is now placed immediately after the docstring.

"""Shared HTTPX client utilities.

This module exposes a single `get_httpx_client` function that returns a **singleton**
`httpx.AsyncClient`.  Re-using one client across the whole application avoids the
TCP/TLS handshake overhead incurred by creating a new client for every request and
allows httpx to keep connections alive and efficiently multiplex HTTP/2 streams.

A matching `close_httpx_client` coroutine is provided for graceful shutdown.

Typical usage::

    from src.core.http_client import get_httpx_client

    async def fetch(url: str) -> str:
        client = get_httpx_client()
        r = await client.get(url)
        r.raise_for_status()
        return r.text

The first call to `get_httpx_client()` builds the client using reasonable defaults
which can optionally be overridden by passing in a configuration dict containing
`http_client_use_http2` (bool).  Subsequent calls always return the **same**
instance, ignoring further config values.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import httpx

# Internal module-level cache for the singleton client
_shared_async_client: Optional[httpx.AsyncClient] = None

# Default performance-oriented settings – these were tuned empirically
_LIMITS = httpx.Limits(
    max_keepalive_connections=60,
    max_connections=300,
    keepalive_expiry=45.0,  # seconds
)
_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=30.0,
    write=10.0,
    pool=10.0,
)


def get_httpx_client(config: Optional[Dict[str, Any]] = None) -> httpx.AsyncClient:
    """Return a shared :class:`httpx.AsyncClient` instance.

    The instance is created lazily on first call and then cached.  Supplying a
    *config* dict the first time allows consumer code to control whether HTTP/2
    should be enabled via the ``http_client_use_http2`` key.  Further calls will
    always return the already-created client (and ignore *config*).
    """

    global _shared_async_client

    if _shared_async_client is None:
        use_http2 = True
        if config is not None:
            # Local import to avoid a hard dependency cycle at import time.
            from src.core.constants import (
                HTTP_CLIENT_USE_HTTP2_CONFIG_KEY,
                DEFAULT_HTTP_CLIENT_USE_HTTP2,
            )

            use_http2 = config.get(
                HTTP_CLIENT_USE_HTTP2_CONFIG_KEY, DEFAULT_HTTP_CLIENT_USE_HTTP2
            )

        _shared_async_client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            limits=_LIMITS,
            follow_redirects=True,
            http2=use_http2,
            trust_env=True,
        )

    return _shared_async_client


async def close_httpx_client() -> None:
    """Close and dispose of the shared HTTPX client (if it exists)."""

    global _shared_async_client
    if _shared_async_client is not None:
        try:
            await _shared_async_client.aclose()
        finally:
            _shared_async_client = None
