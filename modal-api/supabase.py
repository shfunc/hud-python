from __future__ import annotations

import asyncio
import logging
from json import JSONDecodeError
from typing import Any

from supabase import AsyncClient

from hud.settings import settings

LOGGER = logging.getLogger(__name__)




class _RetryableClient(AsyncClient):
    @staticmethod
    def _init_postgrest_client(*args: Any, **kwargs: Any):
        """
        Initializes a Postgrest client with a retry mechanism.
        """
        client = AsyncClient._init_postgrest_client(*args, **kwargs)
        original_request = client.session.request

        async def request_with_retry(*args: Any, **kwargs: Any):
            max_retries = 3
            retry_delay = 0.5
            retries = 0

            while True:
                try:
                    return await original_request(*args, **kwargs)
                except JSONDecodeError as e:
                    # Automatically retry on JSONDecodeError 502 errors to handle Cloudflare issues.
                    # Trying to parse the HTML to check specific error is likely not worth the overhead, so just retry.
                    retries += 1
                    if retries >= max_retries:
                        LOGGER.exception(
                            "JSON decode error after %s retries: %s", max_retries
                        )
                        raise
                    LOGGER.warning(
                        "JSON decode error, retrying (%s/%s): %s",
                        retries,
                        max_retries,
                        e,
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue

        client.session.request = request_with_retry
        return client


async def get_supabase_client() -> AsyncClient:
    """
    Creates and caches a Supabase client instance.
    Uses settings.supabase_url and settings.supabase_key.

    Returns:
        AsyncClient: A Supabase client instance

    Raises:
        ValueError: If supabase_url or supabase_key is not configured
    """
    if not settings.supabase_url or not settings.supabase_key:
        raise ValueError("Supabase credentials not configured")

    return await _RetryableClient.create(settings.supabase_url, settings.supabase_key)
