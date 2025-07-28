# Simple Browser MCP Environment

A browser automation environment for the HUD platform with MCP (Model Context Protocol) support. This environment provides GUI control via X11/VNC and includes progress notifications during initialization.

## Architecture

```
start.sh → hud-controller mcp → server.py (FastMCP with tools)
    ↓
    └→ Xvfb (X11 display server)
       └→ x11vnc + websockify (VNC access)
          └→ Browser + Apps
```

## Key Features

- **X11 Display**: Virtual display for GUI automation
- **VNC Access**: Remote desktop via websockify on port 8080
- **Progress Notifications**: Real-time initialization feedback
- **Tool Registration**: Computer control and app management tools
- **Clean stdout**: All logs go to stderr, keeping stdout for JSON-RPC

## File Structure

```
simple_browser/
├── Dockerfile
├── start.sh              # Starts X11 then runs MCP server
├── pyproject.toml        # Defines hud-controller entry point
└── src/
    └── hud_controller/
        ├── __main__.py   # Entry point
        ├── server.py     # MCP server with tools
        ├── services.py   # Service management
        └── browser.py    # Browser control
```

## How It Works

1. **start.sh** starts Xvfb (X11) and waits for it to be ready
2. **server.py** uses `@mcp_intialize_wrapper` to enable progress notifications
3. During initialization:
   - X11, VNC, and websockify services are started
   - HudComputerTool is registered after X11 is ready
   - Progress notifications are sent to the client
   - Browser is launched with initial URL
4. After initialization, tools are available via MCP protocol

## MCP Protocol Flow

```
Client → Server: InitializeRequest (with optional progressToken)
Server → Client: Progress notifications (0%, 20%, 40%, ...)
Server → Client: InitializeResult
Client → Server: InitializedNotification 
Client → Server: tools/list              
```

## Available Tools

- **computer**: HUD computer control tool for GUI automation
- **launch_app**: Dynamically launch apps (todo, chat, etc.)
- **api_request**: Make HTTP API requests
- **query_database**: Execute database queries (mock)

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PYTHONUNBUFFERED` | `1` | Prevent Python output buffering |
| `DISPLAY` | `:1` | X11 display number |
| `LAUNCH_APPS` | `""` | Comma-separated apps to launch |
| `BROWSER_URL` | `https://google.com` | Initial browser URL |

## Testing

### Build the image
```bash
docker build -t hud-browser .
```

### Test with complete initialization
```bash
# Create test file with proper MCP sequence
cat > test.json << EOF
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1.0"},"_meta":{"progressToken":"init-123"}}}
{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
EOF

# Run test
cat test.json | docker run -i --rm hud-browser
```

### Access VNC viewer
```bash
docker run -d --rm \
  -p 8080:8080 \
  --name hud-browser \
  hud-browser

# Open http://localhost:8080/vnc.html in browser
```

## Common Issues

1. **"Invalid request parameters" for tools/list**
   - Ensure you send `InitializedNotification` after receiving `InitializeResult`
   
2. **Xlib.xauth warnings on stdout**
   - These come from X11/pyautogui, redirect stderr to avoid stdout contamination
   
3. **No tools registered**
   - Check that HudComputerTool is imported AFTER X11 is ready
   - Verify initialization completes before restoring handler

## Implementation Details

### Progress Notifications

The `@mcp_intialize_wrapper` decorator (from `hud.tools.helper`) enables progress notifications by:
1. Monkey-patching `ServerSession._received_request`
2. Intercepting `InitializeRequest` to extract `progressToken`
3. Running custom initialization with progress updates
4. Restoring original handler after initialization

### Tool Registration Timing

Tools are registered at different times:
- `@mcp.tool()` decorators: Registered when module loads
- `register_instance_tool()`: Called during initialization for tools needing X11

### stdout/stderr Separation

Critical for MCP stdio transport:
- All logging configured to use `sys.stderr`
- Xvfb output redirected to `/dev/null` in start.sh
- Only JSON-RPC messages go to stdout

## Development Tips

- Add new tools with `@mcp.tool()` decorator
- Use `logger.info()` for debugging (goes to stderr)
- Test initialization sequence with proper MCP flow
- Monitor stderr for service startup logs 