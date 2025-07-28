"""Helper for MCP server initialization with progress notifications.

Example:
    ```python
    from hud.tools.helper import mcp_intialize_wrapper
    
    @mcp_intialize_wrapper
    async def initialize_environment(session=None, progress_token=None):
        # Send progress if available
        if session and progress_token:
            await session.send_progress_notification(
                progress_token=progress_token,
                progress=0,
                total=100,
                message="Starting services..."
            )
        
        # Your initialization code works with or without session
        start_services()
        
    # Create and run server - initialization happens automatically
    mcp = FastMCP("My Server")
    mcp.run()
    ```
"""

import logging
from typing import Callable, Awaitable, Optional
from mcp.server.session import ServerSession
from mcp.shared.session import RequestResponder
import mcp.types as types

# Store the original _received_request method
_original_received_request = ServerSession._received_request
_init_function: Optional[Callable] = None
_initialized = False


async def _patched_received_request(
    self, 
    responder: RequestResponder[types.ClientRequest, types.ServerResult]
):
    """Intercept initialization to run custom setup with progress notifications."""
    global _initialized, _init_function
    
    # Check if this is an initialization request
    if isinstance(responder.request.root, types.InitializeRequest):
        params = responder.request.root.params
        
        # Extract progress token if present
        progress_token = None
        if hasattr(params, '_meta') and params._meta and hasattr(params._meta, 'progressToken'):
            progress_token = params._meta.progressToken
            logging.info(f"Captured progress token for initialization: {progress_token}")
        
        # Run our initialization function if provided and not already done
        if _init_function and not _initialized:
            logging.info("Running custom initialization with progress support...")
            try:
                await _init_function(session=self, progress_token=progress_token)
                _initialized = True
                logging.info("Custom initialization completed successfully")
            except Exception as e:
                logging.error(f"Custom initialization failed: {e}")
                if progress_token:
                    await self.send_progress_notification(
                        progress_token=progress_token,
                        progress=0,
                        total=100,
                        message=f"Initialization failed: {str(e)}"
                    )
                raise
    
    # Call the original handler to send the InitializeResult
    return await _original_received_request(self, responder)


def mcp_intialize_wrapper(
    init_function: Optional[Callable[[Optional[ServerSession], Optional[str]], Awaitable[None]]] = None
) -> Callable:
    """Decorator to enable progress notifications during MCP server initialization.
    
    Your init function receives optional session and progress_token parameters.
    If provided, use them to send progress updates. If not, the function still works.
    
    Usage:
        @mcp_intialize_wrapper
        async def initialize(session=None, progress_token=None):
            if session and progress_token:
                await session.send_progress_notification(...)
            # Your init code here
    
    Must be applied before creating FastMCP instance or calling mcp.run().
    """
    global _init_function
    
    def decorator(func):
        # Store the initialization function
        _init_function = func
        
        # Apply the monkey patch if not already applied
        if ServerSession._received_request != _patched_received_request:
            ServerSession._received_request = _patched_received_request
            logging.info("Enabled initialization progress support via ServerSession patching")
        
        return func
    
    # If called with a function directly
    if init_function is not None:
        return decorator(init_function)
    
    # If used as @decorator
    return decorator


def reset_initialization():
    """Reset the initialization state and restore original ServerSession behavior.
    
    This is mainly useful for testing or if you need to disable the patching.
    """
    global _initialized, _init_function
    
    # Restore original method
    ServerSession._received_request = _original_received_request
    
    # Reset state
    _initialized = False
    _init_function = None
    
    logging.info("Reset initialization state and restored original ServerSession") 