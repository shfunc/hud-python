from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, cast

from wrapt import ObjectProxy, register_post_import_hook, wrap_function_wrapper

from hud.telemetry.context import buffer_mcp_call, get_current_task_run_id

logger = logging.getLogger(__name__)


class MCPInstrumentor:
    """
    Instrumentor for MCP calls.
    This class wraps MCP functions to capture telemetry data without importing MCP directly.
    It uses wrapt's post-import hooks to instrument MCP modules when they are imported.
    """

    def __init__(self):
        self._installed = False

    def install(self) -> None:
        """Install instrumentation for MCP"""
        if self._installed:
            logger.debug("MCP instrumentation already installed")
            return

        # Register post-import hooks for MCP modules
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.sse", "sse_client", self._transport_wrapper
            ),
            "mcp.client.sse",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.sse", "SseServerTransport.connect_sse", self._transport_wrapper
            ),
            "mcp.server.sse",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.stdio", "stdio_client", self._transport_wrapper
            ),
            "mcp.client.stdio",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.stdio", "stdio_server", self._transport_wrapper
            ),
            "mcp.server.stdio",
        )
        
        # Also instrument the server session to propagate context
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.session", "ServerSession.__init__", self._base_session_init_wrapper
            ),
            "mcp.server.session",
        )
        
        # Directly instrument any functions that might already be imported
        try:
            import mcp.client.sse
            wrap_function_wrapper("mcp.client.sse", "sse_client", self._transport_wrapper)
        except ImportError:
            pass
            
        try:
            import mcp.client.stdio
            wrap_function_wrapper("mcp.client.stdio", "stdio_client", self._transport_wrapper)
        except ImportError:
            pass

        self._installed = True
        logger.debug("MCP instrumentation installed")

    @asynccontextmanager
    async def _transport_wrapper(
        self, wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
    ) -> AsyncGenerator[Tuple[InstrumentedStreamReader, InstrumentedStreamWriter], None]:
        """Wrapper for MCP transport functions"""
        # Record the start of the MCP call
        start_time = time.time()
        task_run_id = get_current_task_run_id()
        
        # Get function details for telemetry
        module_name = wrapped.__module__
        function_name = wrapped.__name__
        call_type = f"{module_name}.{function_name}"
        
        # Call the original function
        try:
            async with wrapped(*args, **kwargs) as (read_stream, write_stream):
                # Create instrumented streams that will record telemetry data
                instrumented_reader = InstrumentedStreamReader(read_stream, task_run_id)
                instrumented_writer = InstrumentedStreamWriter(write_stream, task_run_id)
                
                # Buffer basic call info
                buffer_mcp_call({
                    "task_run_id": task_run_id,
                    "call_type": call_type,
                    "start_time": start_time,
                    "args": self._safe_serialize(args),
                    "kwargs": self._safe_serialize(kwargs),
                    "status": "started"
                })
                
                # Yield the instrumented streams
                yield instrumented_reader, instrumented_writer
        except Exception as e:
            # Record error information
            end_time = time.time()
            duration = end_time - start_time
            
            buffer_mcp_call({
                "task_run_id": task_run_id,
                "call_type": call_type,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "args": self._safe_serialize(args),
                "kwargs": self._safe_serialize(kwargs),
                "error": str(e),
                "error_type": type(e).__name__,
                "status": "error"
            })
            raise
        else:
            # Record successful completion
            end_time = time.time()
            duration = end_time - start_time
            
            buffer_mcp_call({
                "task_run_id": task_run_id,
                "call_type": call_type,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "args": self._safe_serialize(args),
                "kwargs": self._safe_serialize(kwargs),
                "status": "completed"
            })

    def _base_session_init_wrapper(
        self, wrapped: Callable[..., None], instance: Any, args: Any, kwargs: Any
    ) -> None:
        """Wrapper for ServerSession.__init__ to instrument message streams"""
        wrapped(*args, **kwargs)
        
        # Get the stream reader and writer from the instance
        reader = getattr(instance, "_incoming_message_stream_reader", None)
        writer = getattr(instance, "_incoming_message_stream_writer", None)
        
        # Replace with our instrumented versions if they exist
        if reader and writer:
            task_run_id = get_current_task_run_id()
            setattr(
                instance, "_incoming_message_stream_reader", InstrumentedStreamReader(reader, task_run_id)
            )
            setattr(
                instance, "_incoming_message_stream_writer", InstrumentedStreamWriter(writer, task_run_id)
            )

    def _safe_serialize(self, obj: Any) -> Any:
        """Safely serialize objects for telemetry, handling common types."""
        if obj is None:
            return None
        
        if isinstance(obj, (str, int, float, bool)):
            return obj
            
        if isinstance(obj, dict):
            return {self._safe_serialize(k): self._safe_serialize(v) for k, v in obj.items()}
            
        if isinstance(obj, (list, tuple)):
            return [self._safe_serialize(item) for item in obj]
            
        # For other types, just record their type
        return f"<{type(obj).__name__}>"


class InstrumentedStreamReader(ObjectProxy):
    """Wrapper for MCP stream readers that captures telemetry data"""
    
    def __init__(self, wrapped: Any, task_run_id: Optional[str] = None):
        super().__init__(wrapped)
        self._self_task_run_id = task_run_id
        
    # Support async context manager protocol
    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        """Instrument stream iteration to capture message details"""
        async for item in self.__wrapped__:
            try:
                # Try to extract message details
                message_info = self._extract_message_info(item)
                
                if message_info and self._self_task_run_id:
                    # Add task_run_id and record the message
                    message_info["task_run_id"] = self._self_task_run_id
                    message_info["direction"] = "received"
                    message_info["timestamp"] = time.time()
                    buffer_mcp_call(message_info)
            except Exception as e:
                # Log but don't fail if there's an error in instrumentation
                logger.debug(f"Error instrumenting MCP message: {e}")
                
            # Always yield the original item
            yield item

    def _extract_message_info(self, item: Any) -> Dict[str, Any]:
        """Extract relevant information from an MCP message"""
        try:
            # Try to extract session message and request details
            # This is a simplified version that can be expanded based on actual MCP message structure
            message_type = type(item).__name__
            
            message_info = {
                "message_type": message_type
            }
            
            # Try to extract method and params for JSON-RPC requests
            if hasattr(item, "message") and hasattr(item.message, "root"):
                root = item.message.root
                if hasattr(root, "method"):
                    message_info["method"] = root.method
                if hasattr(root, "params"):
                    message_info["params_summary"] = str(type(root.params))
                    
            return message_info
        except Exception:
            # Return minimal info if extraction fails
            return {"message_type": str(type(item))}


class InstrumentedStreamWriter(ObjectProxy):
    """Wrapper for MCP stream writers that captures telemetry data"""
    
    def __init__(self, wrapped: Any, task_run_id: Optional[str] = None):
        super().__init__(wrapped)
        self._self_task_run_id = task_run_id
        
    # Support async context manager protocol
    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def send(self, item: Any) -> Any:
        """Instrument message sending to capture telemetry data"""
        try:
            # Try to extract message details
            message_info = self._extract_message_info(item)
            
            if message_info and self._self_task_run_id:
                # Add task_run_id and record the message
                message_info["task_run_id"] = self._self_task_run_id
                message_info["direction"] = "sent"
                message_info["timestamp"] = time.time()
                buffer_mcp_call(message_info)
        except Exception as e:
            # Log but don't fail if there's an error in instrumentation
            logger.debug(f"Error instrumenting MCP message: {e}")
            
        # Always send the original item
        return await self.__wrapped__.send(item)

    def _extract_message_info(self, item: Any) -> Dict[str, Any]:
        """Extract relevant information from an MCP message"""
        try:
            # Try to extract session message and request details
            # This is a simplified version that can be expanded based on actual MCP message structure
            message_type = type(item).__name__
            
            message_info = {
                "message_type": message_type
            }
            
            # Try to extract method and params for JSON-RPC requests
            if hasattr(item, "message") and hasattr(item.message, "root"):
                root = item.message.root
                if hasattr(root, "method"):
                    message_info["method"] = root.method
                if hasattr(root, "params"):
                    message_info["params_summary"] = str(type(root.params))
                    
            return message_info
        except Exception:
            # Return minimal info if extraction fails
            return {"message_type": str(type(item))} 