# Browser Environment

A browser automation environment for HUD that provides GUI access and web app interaction capabilities. This environment supports hot-reloading during development while maintaining persistent state.

## Quick Start

### Interactive Development
```bash
# 1. Configure your API keys (optional - only needed for evaluation)
# Edit .env file to add your HUD_API_KEY and ANTHROPIC_API_KEY

# 2. Start the environment (optional: with inspector)
hud dev --build --inspector

# 3. Choose your preferred way to test:

# Option A: Run the task with Claude (requires ANTHROPIC_API_KEY)
hud eval tasks.json --agent claude

# Option B: Interactive notebook test_env.ipynb (great for learning!)
# Requires installation:
pip install hud-python[agents]

# Option C: Simple Python script (runs all tasks from tasks.json)
python test_task.py
```

## How HUD Environments Work

The environment is split into two components:

- **`env.py`** - Stateful logic that persists across reloads
- **`server.py`** - MCP server with tools (reloads on file changes)

This separation is crucial for `hud dev` - it allows you to modify the MCP tools and see changes immediately without losing the environment state. The environment runs as a separate process and communicates via socket, while the server can be restarted freely.

If you are ever seeing issues with the environment itself, running `hud dev --full-reload` will reload both the environment and the server.

## Publishing Your Environment

Once your environment is ready, you can share it with the community:

### 1. Push to Registry
```bash
# Build and push your environment (requires docker hub login and hud api key)
hud build
hud push
```

### 2. Create a Dataset

Create a dataset on HuggingFace with your tasks:

**Option A: Upload manually**
1. Upload your `tasks.json` to HuggingFace
2. Make sure it's **public** to appear on leaderboards

**Option B: Use the SDK**
```python
from hud.datasets import save_tasks
import json

# Load your tasks
with open("tasks.json") as f:
    tasks = json.load(f)

# Push to HuggingFace
save_tasks(tasks, repo_id="your-org/your-dataset")
```

### 3. Run and Track Performance

```bash
# Run Claude on your benchmark
hud eval "your-org/your-dataset" --agent claude

# View results at:
# hud.so/leaderboards/your-org/your-dataset
```

**Note**: Only public HuggingFace datasets appear as leaderboards!

📚 Learn more: [Creating Benchmarks](https://docs.hud.so/evaluate-agents/create-benchmarks) | [Leaderboards](https://docs.hud.so/evaluate-agents/leaderboards)

## Architecture Overview

The browser environment uses a two-process architecture:

1. **Context Server** (`context.py`): Long-running process that maintains persistent state
2. **MCP Server** (`server.py`): Hot-reloadable process that handles tool requests

### Key Components

- **BrowserContext**: Stores persistent state (running apps, ports, playwright instance)
- **ServiceManager**: Manages X11, VNC, and app processes
- **BaseHub Tools**: Setup and evaluate tools organized by app (2048, todo)
- **Multiprocessing Proxy**: Enables state sharing between processes

### 1. Tool Implementation Pattern

All setup and evaluate tools should follow this pattern:

```python
@setup.tool("tool_name")
async def tool_name(param1: type, param2: type):
    """Tool description."""
    try:
        # Get persistent context
        persistent_ctx = setup.env  # or evaluate.env
        
        # Get app ports
        backend_port = persistent_ctx.get_app_backend_port("app_name")
        
        # Make HTTP request
        url = f"http://localhost:{backend_port}/api/endpoint"
        async with httpx.AsyncClient() as client:
            response = await client.method(url, json=data)
            response.raise_for_status()
            result = response.json()
        
        # Return result
        return TextContent(
            text=f"Success message",
            type="text"
        )
    except Exception as e:
        logger.error(f"tool_name failed: {e}")
        return TextContent(
            text=f"Failed: {str(e)}",
            type="text"
        )
```

### 2. App Launch Pattern

When launching apps, ensure ports are stored in the persistent context:

```python
# In launch_app tool
app_info = await service_manager.launch_app(app_name)

# Store ports in persistent context for later access
try:
    backend_port = service_manager.get_app_port(app_name)
    frontend_port = service_manager.get_app_frontend_port(app_name)
    persistent_ctx.set_app_ports(app_name, frontend_port, backend_port)
except Exception as e:
    logger.error(f"Failed to store ports: {e}")

# Track app in persistent context
persistent_ctx.add_running_app(app_name)
```

### 3. Import Organization

Keep imports at module level:

```python
# At top of file
import logging
import httpx
from mcp.types import TextContent
from . import setup

# Not inside functions
```

## Development Workflow

1. **Start the environment**: `hud dev`
2. **Make changes**: Edit tools in `src/hud_controller/`
3. **Test immediately**: The MCP server hot-reloads automatically
4. **Check logs**: Look for serialization or proxy errors

## Adding New Apps

1. Create app directory in `apps/`
2. Add setup tools in `src/hud_controller/setup/app_name.py`
3. Add evaluate tools in `src/hud_controller/evaluate/app_name.py`
4. Follow the HTTP pattern - no `call_app_api` usage
5. Store app ports in persistent context when launching

## Key Files

- `context.py`: Persistent state management
- `server.py`: MCP server and tool definitions
- `services.py`: Process management for X11, VNC, apps
- `setup/`: Setup tools organized by app
- `evaluate/`: Evaluation tools organized by app

Remember: When in doubt, make direct HTTP calls and store state in the persistent context!

