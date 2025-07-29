"""Simple in-memory cache utility."""

from functools import lru_cache


@lru_cache(maxsize=128)
def cached_profile(username: str):
    # Dummy cache for user profile
    return username
