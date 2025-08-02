from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from wrapt import wrap_function_wrapper

from hud.telemetry.context import (
    create_notification_record,
    create_request_record,
    create_response_record,
    get_current_task_run_id,
)
from hud.telemetry.mcp_models import DirectionType, MCPCallType, StatusType

logger = logging.getLogger(__name__)


class MCPInstrumentor:
    """
    Context-aware instrumentor for MCP calls.
    Only instruments MCP methods when there's an active trace context.
    """

    def __init__(self) -> None:
        self._installed = False

    def install(self) -> None:
        """Install instrumentation for MCP - but only activate when trace context exists."""
        logger.debug("MCPInstrumentor: install() called (context-aware mode)")
        if self._installed:
            logger.debug("MCP instrumentation already installed")
            return

        try:
            # Import and wrap the main session methods
            import mcp.shared.session  # noqa: F401

            wrap_function_wrapper(
                "mcp.shared.session",
                "BaseSession.send_request",
                self._context_aware_send_request_wrapper,
            )
            wrap_function_wrapper(
                "mcp.shared.session",
                "BaseSession.send_notification",
                self._context_aware_send_notification_wrapper,
            )

            logger.debug("Successfully wrapped BaseSession methods for context-aware telemetry")
        except ImportError:
            logger.debug("mcp.shared.session not available yet")
        except Exception as e:
            logger.warning("Failed to wrap BaseSession methods: %s", e)

        self._installed = True
        logger.debug("MCP instrumentation installed (context-aware mode)")

    async def _context_aware_send_request_wrapper(
        self,
        wrapped: Callable[[Any, Any, Any, Any], Awaitable[Any]],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        """Context-aware send request wrapper."""
        hud_task_run_id = get_current_task_run_id()

        if not hud_task_run_id:
            # No active trace - pass through without instrumentation
            return await wrapped(*args, **kwargs)

        start_time = time.time()
        request = args[0] if args else None

        # Extract method name from the request
        method_name = "unknown_method"
        message_id = None
        request_data = None

        if request:
            try:
                # Get model dump for Pydantic models
                if hasattr(request, "model_dump"):
                    request_data = request.model_dump(mode="json", exclude_none=True)
                    # Extract method from the dump
                    if isinstance(request_data, dict) and "method" in request_data:
                        method_name = request_data["method"]
                # Fallback to direct method attribute
                elif hasattr(request, "method"):
                    method_name = str(request.method)
                    request_data = {"method": method_name}
            except Exception as e:
                logger.debug("Failed to extract method: %s", e)

        # Create the request record
        create_request_record(
            method=method_name,
            message_id=message_id,
            call_type=MCPCallType.SEND_REQUEST,
            direction=DirectionType.SENT,
            status=StatusType.STARTED,
            start_time=start_time,
            request_data=request_data,
        )

        try:
            # Call the wrapped method and get the result
            result = await wrapped(*args, **kwargs)

            # Get the actual message_id from the session after sending
            if hasattr(instance, "_request_id"):
                # Current request ID minus 1 (since it was incremented)
                message_id = getattr(instance, "_request_id", 1) - 1

            # Update request record with completion
            create_request_record(
                method=method_name,
                message_id=message_id,
                call_type=MCPCallType.SEND_REQUEST,
                direction=DirectionType.SENT,
                status=StatusType.COMPLETED,
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                request_data=request_data,
            )

            # Capture the response
            if result is not None:
                response_data = None
                try:
                    if hasattr(result, "model_dump"):
                        response_data = result.model_dump(mode="json", exclude_none=True)
                    else:
                        response_data = {"_type": type(result).__name__}
                except Exception as e:
                    logger.debug("Failed to serialize response data: %s", e)
                    response_data = {"_type": type(result).__name__, "_error": str(e)}

                create_response_record(
                    method=method_name,
                    related_request_id=message_id,
                    is_error=False,
                    message_id=message_id,
                    direction=DirectionType.RECEIVED,
                    call_type=MCPCallType.RECEIVE_RESPONSE,
                    response_data=response_data,
                    timestamp=time.time(),
                )

            return result

        except Exception as e:
            # Log the error
            create_request_record(
                method=method_name,
                message_id=message_id,
                call_type=MCPCallType.SEND_REQUEST,
                direction=DirectionType.SENT,
                status=StatusType.ERROR,
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                error=str(e),
                error_type=type(e).__name__,
                request_data=request_data,
            )

            # Also record error response
            create_response_record(
                method=method_name,
                related_request_id=message_id,
                is_error=True,
                message_id=message_id,
                direction=DirectionType.RECEIVED,
                call_type=MCPCallType.RECEIVE_RESPONSE,
                status=StatusType.ERROR,
                error=str(e),
                error_type=type(e).__name__,
                timestamp=time.time(),
            )

            raise

    async def _context_aware_send_notification_wrapper(
        self,
        wrapped: Callable[[Any, Any, Any, Any], Awaitable[Any]],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        """Context-aware send notification wrapper."""
        hud_task_run_id = get_current_task_run_id()

        if not hud_task_run_id:
            # No active trace - pass through without instrumentation
            return await wrapped(*args, **kwargs)

        start_time = time.time()
        notification = args[0] if args else None

        # Extract method name
        method_name = "unknown_method"
        notification_data = None

        if notification:
            try:
                # Get model dump for Pydantic models
                if hasattr(notification, "model_dump"):
                    notification_data = notification.model_dump(mode="json", exclude_none=True)
                    # Extract method from the dump
                    if isinstance(notification_data, dict) and "method" in notification_data:
                        method_name = notification_data["method"]
                # Fallback to direct method attribute
                elif hasattr(notification, "method"):
                    method_name = str(notification.method)
                    notification_data = {"method": method_name}
            except Exception as e:
                logger.debug("Failed to extract notification method: %s", e)

        record_data_base = {
            "method": method_name,
            "message_id": None,  # Notifications don't have IDs
            "call_type": MCPCallType.SEND_NOTIFICATION,
            "direction": DirectionType.SENT,
            "notification_data": notification_data,
        }

        try:
            create_notification_record(
                **record_data_base, status=StatusType.STARTED, start_time=start_time
            )

            result = await wrapped(*args, **kwargs)

            create_notification_record(
                **record_data_base,
                status=StatusType.COMPLETED,
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
            )
            return result
        except Exception as e:
            create_notification_record(
                **record_data_base,
                status=StatusType.ERROR,
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
