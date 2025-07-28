"""
Helper utilities for progressive initialization in MCP servers.

This module provides HudMcpContext, an extended context class that adds
convenient progress reporting during initialization.
"""
from __future__ import annotations

import logging
from typing import Optional

from mcp.shared.context import RequestContext

logger = logging.getLogger(__name__)


class HudMcpContext(RequestContext):
    """
    Extended MCP context with built-in progress reporting.
    
    This context adds a report_progress method that handles both 
    MCP notifications and logging fallback.
    
    Usage:
        Simply cast the context to HudMcpContext in your handler:
        
        @mcp.custom_request_handler(InitializeRequest)
        async def handle_initialize(ctx: RequestContext) -> InitializeResult:
            ctx = HudMcpContext(ctx)
            await ctx.report_progress(0, "Starting services...")
            # ... do work ...
            await ctx.report_progress(50, "Halfway there...")
            # ... more work ...
            await ctx.report_progress(100, "Ready!")
    """
    
    def __init__(self, base_ctx: RequestContext):
        """Initialize from a base RequestContext."""
        # Copy all attributes from base context
        self.__dict__.update(base_ctx.__dict__)
        self._progress_token: Optional[str] = None
        self._extract_progress_token()
    
    def _extract_progress_token(self) -> None:
        """Extract progress token from request metadata."""
        if hasattr(self.request.params, '_meta') and self.request.params._meta:
            self._progress_token = self.request.params._meta.get('progressToken')
    
    async def report_progress(self, percentage: float, message: str) -> None:
        """
        Report initialization progress.
        
        Sends MCP progress notifications if a progress token is available,
        always logs the progress message.
        
        Args:
            percentage: Progress percentage (0-100)
            message: Human-readable progress message
        """
        # Try to send MCP notification
        if self._progress_token and hasattr(self, 'session'):
            try:
                await self.session.send_progress_notification(
                    progress_token=self._progress_token,
                    progress=percentage,
                    total=100,
                    message=message
                )
            except Exception as e:
                logger.warning(f"Failed to send progress notification: {e}")
        
        # Always log progress
        logger.info(f"[{int(percentage)}%] {message}") 