import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
# Simple in-memory cache for API responses
_api_response_cache: Dict[str, Tuple[Dict[str, Any], datetime, int]] = {}
def get_cache_key(
    query: str,
    api_url: str,
    api_max_results: int,
    max_char_per_url: Optional[int] = None,
) -> str:
    """Generate a cache key for API requests."""
    key_data = f"{query}|{api_url}|{api_max_results}|{max_char_per_url}"
    return hashlib.md5(key_data.encode()).hexdigest()
def get_cached_response(
    cache_key: str, cache_ttl_minutes: int
) -> Optional[Dict[str, Any]]:
    """Get cached response if it exists and hasn't expired."""
    if cache_key in _api_response_cache:
        cached_response, cache_time, stored_ttl = _api_response_cache[cache_key]
        # Use the stored TTL from when the item was cached, not the current call's TTL
        if datetime.now() - cache_time < timedelta(minutes=stored_ttl):
            return cached_response
        else:
            # Remove expired cache entry
            del _api_response_cache[cache_key]
    return None
def cache_response(
    cache_key: str, response: Dict[str, Any], cache_ttl_minutes: int
) -> None:
    """Cache an API response."""
    _api_response_cache[cache_key] = (response, datetime.now(), cache_ttl_minutes)
    # Periodically clean up expired entries (every 100 cache operations)
    if len(_api_response_cache) > 0 and len(_api_response_cache) % 100 == 0:
        cleanup_expired_cache()
def cleanup_expired_cache() -> None:
    """Remove expired entries from the cache."""
    current_time = datetime.now()
    expired_keys = [
        key
        for key, (_, cache_time, stored_ttl) in _api_response_cache.items()
        if current_time - cache_time >= timedelta(minutes=stored_ttl)
    ]
    for key in expired_keys:
        del _api_response_cache[key]
    if expired_keys:
        pass