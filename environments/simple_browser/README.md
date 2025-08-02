# Simple Browser MCP Environment

A browser automation environment for the HUD platform demonstrating best practices for building MCP (Model Context Protocol) environments with evaluation systems.

## Quick Start

### Build & Deploy
```bash
# Build the Docker image
cd environments/simple_browser
docker build -t hud-browser .

# Run with stdio (recommended for HUD SDK v3)
docker run --rm -i -p 8080:8080 -e LAUNCH_APPS=todo hud-browser

# Run with HTTP transport (for testing)
docker-compose up -d
```

## Deployment to Registry

### 1. Publish to Docker Registry

#### Docker Hub
```bash
# Build and push to Docker Hub
docker build -t your-username/hud-browser:latest .
docker push your-username/hud-browser:latest
```

#### GitHub Container Registry (GHCR)
```bash
# Login and push to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u your-username --password-stdin
docker build -t ghcr.io/your-org/hud-browser:latest .
docker push ghcr.io/your-org/hud-browser:latest
```

### 2. Use via HUD Cloud Orchestrator

```python
import os
import hud
from mcp_use import MCPClient
from hud.mcp_agent import ClaudeMCPAgent

BASE_URL = "https://orchestrator-v3.up.railway.app"
HUD_API_KEY = os.getenv("HUD_API_KEY")

async def main():
    with hud.trace() as run_id:
        # Configure MCP client to connect to the cloud orchestrator
        config = {
            "mcpServers": {
                "browser": {
                    "url": f"{BASE_URL}/v3/mcp",
                    "headers": {
                        "Authorization": f"Bearer {HUD_API_KEY}",
                        "Mcp-Image": "your-username/hud-browser:latest",  # Your published image
                        "Run-Id": run_id,
                    },
                }
            }
        }

        client = MCPClient.from_dict(config)
        agent = ClaudeMCPAgent(
            client=client,
            model="claude-sonnet-4-20250514",
            allowed_tools=["computer", "setup", "evaluate"]
        )

        # Use your environment in the cloud
        await agent.run("Set up the todo app and evaluate completion")
```

### 3. Local Development (stdio)

For local testing and development:

```python
# Local Docker with stdio transport
config = {
    "mcpServers": {
        "browser": {
            "command": "docker",
            "args": ["run", "--rm", "-i", "-p", "8080:8080", "-e", "LAUNCH_APPS=todo", "hud-browser"]
        }
    }
}
```

See `examples/environments/simple_browser_example.py` for a complete working example.

## Environment Design Strategy

### Core Principles for MCP Environments

1. **Clean Protocol Separation**
   - All logging to stderr, only JSON-RPC to stdout
   - Use `@mcp_intialize_wrapper` for progress notifications
   - Separate tool registration timing (static vs dynamic)

2. **Evaluation System Architecture**
   - **Class-based problems** with inheritance for reusability
   - **App-centric evaluation** - backends provide `/api/eval/*` endpoints
   - **Registry pattern** for discovery (`@evaluator`, `@setup`, `@problem`)
   - **MCP resources** for runtime introspection

3. **Service Management**
   - Start core services (X11, VNC) before MCP initialization
   - Use progress notifications for long-running setup
   - Graceful error handling with meaningful messages

4. **Tool Design**
   - `setup` and `evaluate` tools with dual interfaces:
     - Direct: `{"function": "tool_name", "args": {...}}`
     - Problem-based: `{"name": "problem_name"}`
   - Environment context objects for unified API access
   - Factory patterns for runtime instantiation

### Environment Variables Strategy

Set these in your environment/Docker configuration:

#### Required
- `DISPLAY=:1` - X11 display number for GUI automation
- `PYTHONUNBUFFERED=1` - Prevent output buffering for real-time logs

#### Optional
- `LAUNCH_APPS` - Comma-separated apps to auto-launch (`todo,chat`)
- `BROWSER_URL` - Initial navigation URL (default: google.com)
- `LOG_LEVEL` - Logging verbosity (`DEBUG`, `INFO`, `WARNING`)

#### MCP-Specific
- Set in client configuration, not container:
  - Progress tokens for initialization feedback
  - Tool allowlists for security
  - Transport method (stdio vs HTTP)

### Common Pitfalls & Solutions

1. **stdout Contamination**
   ```bash
   # ❌ Wrong - X11 warnings go to stdout
   Xvfb :1 -screen 0 1024x768x24
   
   # ✅ Correct - Redirect stderr
   Xvfb :1 -screen 0 1024x768x24 2>/dev/null
   ```

2. **"Invalid request parameters" for tools/list**
   - Ensure your client sends `InitializedNotification` after receiving `InitializeResult`
   - The MCP protocol requires this handshake before tools are available

3. **No tools registered**
   - Check that tools requiring X11 are registered AFTER X11 is ready
   - Use `register_instance_tool()` in initialization, not at module load
   - Verify initialization completes before restoring handler

4. **Evaluation fails unexpectedly**
   - Ensure apps are launched before running evaluations
   - Check that app backend APIs are responding (`/api/eval/health`)
   - Verify problem setup completed successfully before evaluation
   - Use clean state between test runs (`reset` then `seed`)

5. **Progress notifications not working**
   - Ensure `progressToken` is provided in `InitializeRequest`
   - Use `@mcp_intialize_wrapper` decorator correctly
   - Send progress updates between 0-100 with meaningful messages

### Architecture Pattern

```
Docker Container
├── start.sh                 # Service startup orchestration
├── MCP Server (FastMCP)     # Protocol implementation
│   ├── Tools                # setup, evaluate, computer, etc.
│   └── Resources           # Dynamic registry discovery
├── Services
│   ├── X11 (Xvfb)          # Virtual display
│   ├── VNC + Websockify    # Remote access
│   └── Apps                # Web applications
└── Evaluation System
    ├── Evaluators          # @evaluator decorated classes
    ├── Setup Tools         # @setup decorated classes
    ├── Problems            # @problem decorated classes
    └── Context             # Unified environment API
```

## File Structure Overview

```
simple_browser/
├── Dockerfile              # Multi-stage build with optimization
├── start.sh                # Service startup script
├── docker-compose.yml      # HTTP transport testing
├── apps/                   # Launchable web applications
│   └── todo/              # Example app with evaluation APIs
├── src/hud_controller/     # MCP server implementation
│   ├── server.py          # FastMCP server + resource definitions
│   ├── runtime.py         # setup/evaluate tool implementations
│   ├── services.py        # Service management
│   ├── evaluators/        # Evaluation system
│   ├── setup/            # Setup system
│   └── problems/         # Problem definitions
└── README.md             # This file
```

## Development Workflow

1. **Start with apps** - Build your web applications independently
2. **Add evaluation APIs** - Extend app backends with `/api/eval/*` endpoints
3. **Create evaluators** - Build `@evaluator` classes that consume app APIs
4. **Build setup tools** - Create `@setup` classes for environment preparation
5. **Define problems** - Combine setup + evaluation using inheritance
6. **Test integration** - Use MCP tools to verify evaluation flow
7. **Containerize** - Package in Docker with proper service orchestration

## Testing & Debugging

### Local Testing
```bash
# Test app evaluation APIs directly
curl http://localhost:5000/api/eval/health
curl http://localhost:5000/api/eval/stats

# Test MCP evaluation flow
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"setup","arguments":{"config":{"name":"todo_basic_usage"}}}}' | docker run -i --rm hud-browser
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"evaluate","arguments":{"config":{"name":"todo_basic_usage"}}}}' | docker run -i --rm hud-browser
```

### Debug Logging
```bash
# Monitor all logs (stderr)
docker run -i --rm hud-browser 2>debug.log

# Monitor specific services
docker run -i --rm hud-browser 2>&1 | grep -E "(X11|VNC|MCP)"
```

### VNC Access
```bash
# Launch with VNC access for GUI debugging
docker run -d --rm -p 8080:8080 --name browser-debug hud-browser
# Open http://localhost:8080/vnc.html
```

## Extending to New Environments

When creating new MCP environments:

1. **Copy this structure** as a template
2. **Replace apps/** with your domain-specific applications
3. **Implement your evaluators** following the `@evaluator` pattern
4. **Create domain setup tools** with the `@setup` pattern
5. **Define problems** using class inheritance for reusability
6. **Update service dependencies** in `services.py` as needed
7. **Extend Dockerfile** with your environment's requirements

See `src/hud_controller/README.md` for detailed implementation guidance. 