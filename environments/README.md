# How to Build HUD-Compatible MCP Environments

This document is a step-by-step guide for turning *any* piece of software that can run in a Docker container into a **Model Context Protocol (MCP)** environment that the HUD SDK can evaluate or control.  We’ll move through six short phases, each with a clear checkpoint.

The official MCP lifecycle specification is an excellent companion reference – skim it now, keep it open while you work: [modelcontextprotocol.io › Lifecycle](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle).

---

## Phase Overview

| Phase | Goal |
|-------|------|
| 1 | A Docker image that *starts* and prints to **stderr** |
| 2 | A minimal MCP server that responds to `initialize` over **stdio** |
| 3 | Working `setup`, `evaluate`, and **interaction** tools |
| 4 | Image launches remotely on the HUD platform & exposes live telemetry |
| 5 | Fast local iteration with **cursor-mcp** and a tiny `mcp.json` |
| 6 | Optional polish – registries, optimisation, security, creative ideas |

Take the phases one at a time; do **not** jump ahead.  Each stage’s checkpoint is the foundation for the next.

---

## Phase 1 – Write a *Simple* Dockerfile

**Goal →** the container starts, prints a message to **stderr**, and exits cleanly.  Nothing else.

Why stderr?  In Phase 2 the MCP server will reserve **stdout** for JSON-RPC traffic, so *all* human-readable logs should already go to the other stream.

### Minimal example

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

# Optional: install requirements
# RUN pip install --no-cache-dir -r requirements.txt

# ‼️  Send logs to stderr (stdout remains untouched for MCP)
CMD [
  "python",
  "-c",
  "import sys, time; print('hello from the container', file=sys.stderr); time.sleep(1)"
]
```

Build & run:

```bash
docker build -t my-environment .
docker run --rm -it my-environment     # look for the log line on stderr
```

• **One Dockerfile only** – no docker-compose.  
• If you’re building a GUI environment, start from `hudpython/novnc-base:latest` instead and leave VNC configuration for later phases.

Checkpoint reached?  Congratulations – move on.

Need inspiration?  Skim the real Dockerfiles used in the example browser environments:
• [`simple_browser/Dockerfile`](./simple_browser/Dockerfile)
• [`remote_browser/Dockerfile`](./remote_browser/Dockerfile)
They follow the exact same pattern – a single file, logs to stderr, nothing fancy.

---

## Phase 2 – Create the MCP Server

**Goal →** a Python process that:
1. Speaks MCP over **stdio**.
2. Responds correctly to the `initialize` request.
3. Logs everything to **stderr**.

The MCP lifecycle is *initialize → operate → shutdown* (see spec link above).

### Skeleton server (FastMCP)

```python
import sys
import logging
from mcp.server.fastmcp import FastMCP

# 1️⃣  Always log to stderr – stdout is reserved for JSON-RPC
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | %(name)s | %(message)s'
)

mcp = FastMCP("My Environment")

from hud.tools.helper import mcp_intialize_wrapper

@mcp_intialize_wrapper()
async def initialize_environment():
    """Heavy one-time setup – start databases, launch background apps, etc."""
    logging.info("starting core services…")
    await start_services()            # your coroutine
    logging.info("services ready")

if __name__ == "__main__":
    mcp.run()
```

*(Replace `start_services()` with whatever takes noticeable startup time – browsers, DBs, X servers, …)*

### Adapt Dockerfile

At the end of your Dockerfile, you must launch the MCP server as the container's main process, ensuring it communicates over stdio (stdin/stdout). This is typically done by setting the `CMD` or `ENTRYPOINT` to run your server module directly, for example:


```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

# Optional: install requirements
# RUN pip install --no-cache-dir -r requirements.txt

CMD ["uv", "pip", "run", "python", "-m", "your_module_name"]  # Replace 'your_module_name' with your actual entrypoint module
```

### Three validation steps (run them **in order**)

| # | What you do | Why it matters |
|---|-------------|----------------|
| 1 | **Direct stdio test** – pipe the JSON below into your script | Proves the Python code handles `initialize` without any client or Docker noise |
| 2 | **MCP Inspector** – `npx @modelcontextprotocol/inspector python -m my_package.server` | Lets you click around: view capabilities, tools, resources |
| 3 | **Inside Docker** – rebuild the image and run it | This is *exactly* how HUD will execute the server |

#### JSON for step 1

```jsonc
{ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
  "protocolVersion": "2024-11-05",
  "capabilities": {"roots": {"listChanged": true}},
  "clientInfo": {"name": "DevClient", "title": "Dev", "version": "0.0.0"}
}}
```

Pipe it:

```bash
echo '<the-json-above>' | python -m my_package.server
```

If all three validations succeed, you have a real MCP server – time to make it useful.

---

## Phase 3 – Add Setup / Evaluate / Interaction Tools

**Goal →** tools are discoverable in the Inspector *and* callable from the HUD SDK.

1. Write **`setup`** and **`evaluate`** tools first – they are *lifecycle* tools and never shown to the LLM.
2. Register at least one **interaction** tool (`computer`, `playwright`, or your own).

### Example

```python
from hud.tools.helper import register_instance_tool
from hud.tools import HudComputerTool

@mcp.tool()
async def setup(config: dict) -> dict:
    ...               # prepare environment

@mcp.tool()
async def evaluate(config: dict) -> dict:
    ...               # return {"reward": <0-1>, "done": bool}

@mcp.initialize()
async def init():
    register_instance_tool(mcp, "computer", HudComputerTool())
```

### Test workflow

1. **Inspector first** – restart the server, refresh the *Tools* tab, confirm the new tools appear.  
2. **Rebuild the image** – `docker build -t my-environment .`.  
3. **HUD SDK test** – run a short script like the one below.  GUI environments built from `hudpython/novnc-base` still expose a VNC viewer on <http://localhost:8080/vnc.html> – keep it open while testing.

```python
import asyncio
from hud import Task
from hud.mcp import ClaudeMCPAgent
from hud.telemetry import trace
from mcp_use import MCPClient

async def main():
    # `trace` captures *everything* that happens and sends it to app.hud.so
    with trace("local_test"):
        cfg = {
            "mcp_config": {
                "local": {"command": "docker", "args": ["run", "--rm", "-i", "my-environment:latest"]}
            }
        }
        client = MCPClient.from_dict(cfg)

        agent = ClaudeMCPAgent(
            client=client,
            model="claude-3-sonnet-20241022",
            allowed_tools=["computer"]
        )

        task = Task(
            prompt="Mark two todo items as done",
            setup={"function": "todo_seed", "args": {"num_items": 5}},
            evaluate={"function": "todo_completed", "args": {"expected_count": 2}}
        )

        result = await agent.run(task)
        print(result)

    await client.close_all_sessions()

asyncio.run(main())
```

The `trace` context manager sends a full timeline of agent actions, tool calls, and rewards to app.hud.so – perfect for debugging.

See `examples/agents_tools/simple_task_example.py` and `examples/environments/gmail_local.py` for larger end-to-end demos.

---

## Phase 4 – Remote Deployment & HUD Runner

**Goal →** the exact same image runs in parallel on hundreds of instances, and exposes more telemetry so the app.hud.so can visualise the whole lifecycle.

### 1. Publish your image

Log in to Docker Hub (or any registry HUD can pull from) and push a tagged build:

```bash
docker tag my-environment yourdockerhubuser/my-environment:latest
docker push yourdockerhubuser/my-environment:latest
```

*(If you’re using a private registry, make sure the HUD worker has pull credentials.)*

### 2. Launch it remotely (gmail_remote pattern)

`examples/environments/gmail_remote.py` shows the canonical pattern – a remote MCP server entry that simply runs **the same Docker image**:

```python
from hud import settings
# Your image is in a registry, now tell HUD to pull & run it on demand
config = {
    "mcp_config": {
        "hud": {
            "url": settings.mcp_url,  # Provided by HUD when you create an evaluation run
            "headers": {
                "Authorization": f"Bearer {settings.api_key}",
                "Mcp-Image": "yourdockerhubuser/my-environment:latest",  # which image to launch
            },
        }
    }
}

client = MCPClient.from_dict(config)
```

_Steps 3 and 4 below are **optional but highly recommended** once the image boots successfully._

Spin up **many** agents in parallel by just launching multiple tasks – HUD will queue and start as many containers as resources allow.

### 3. Progress updates during `initialize` (Optional)

At remote scale it can take 10-30 s for heavy services to boot.  Use `mcp_intialize_wrapper()` with a *progress token* to stream status messages:

```python
from hud.tools.helper import mcp_intialize_wrapper

@mcp_intialize_wrapper()
async def initialize_environment(session=None, progress_token=None):
    async def send(p, msg):
        if session and progress_token:
            await session.send_progress_notification(progress_token, p, 100, msg)
    await send(10, "starting X11…")
    await start_x11()
    await send(50, "launching browser…")
    await launch_browser()
    await send(100, "ready")
```

Those messages are displayed live on app.hud.so alongside resource graphs – perfect feedback while you wait.

### 4. Live telemetry (`telemetry://live`) (Optional)

Expose a resource named `telemetry://live` exactly like in `environments/simple_browser/src/hud_controller/server.py` to return live url to be displayed on app.hud.so.

Once all of the above works you can unleash *hundreds* of concurrent agents on your new environment.

---

## Phase 5 – Automated Iteration with *cursor-mcp*

[`cursor-mcp`](https://github.com/hud-evals/cursor-mcp) turns the edit → build → restart → test loop into a single key-press and adds tools to Cursor Agent that can drive the whole workflow for you. The agent reads the MCP spec, your code, and the live server state, then proposes fixes or new tests on its own. It then has access to the MCP tools the environment provides, enabling it to test all functionality, which completes the iteration loop.

1. Add an entry to `.cursor/mcp.json`:

```jsonc
{
  "mcp_config": {
    "env": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "my-environment:latest"]
    },
    "cursor-manager": {
      "command": "uvx",
      "args": ["cursor-mcp"]
    }
  }
}
```

2. Follow the cursor rules below: rebuild, refresh, test, reflect, repeat.
3. Keep the agent open for any messages or issues.

### Cursor rules – paste this once

Inside `.cursor/rules/mcp_environment_iteration.mdc` add (or verify) the following so the agent always knows the expected loop:

```mdc
---
description: When making an environment that launches and MCP server this is the iteration loop
alwaysApply: false
---
Setting up (also refer to environments/README.md):
1. Follow each environment's README.md or any other steps to set it up for the MCP server to be able to directly launch it (such as building the dockerfile)
2. Run local tests to make sure the initialize without immediate errors and stays alive until properly closed. If the server crashes within the first few seconds then the manager will not pick up on it. In this case please go back and either debug the docker run directly, or the mcp server by piping an initialization request.
3. When the server initialization is stable, use the cursor-manager tool to see the current list of tools and add it if necessary. Take note of the name.
4. When working, tell the user to send another message to refresh your list of tools.

After setting up, when iterating (will not require a user message ever):
1. Look at the environment project and refine/edit/fix files
2. Follow its README to set it up for the MCP server (such as building the dockerfile)
3. Use the cursor-manager tool to refresh this server (by name)
4. See its status using cursor-manager, if it's running then follow with step 5. If it fails, then check the logs using cursor-manager and go back to step 1, but ask the user to reset.
5. Use the tools from that server (by name) to test the functionality and edge cases, reflect on the success of your TODOs and think of new things to fix. If the tools are unavailable but the status is running, then ask the user to refresh the user message.
6. Review your TODOs, update with new TODOs
7. Repeat until reached user's high level goals, or generally extremely happy with the final result

In general:
1. Try to avoid running direct docker or mcp commands and use the tools. If you want to run a docker command or python mcp server command then ask permission and only use if otherwise completely impossible.
2. If at any point the docker build starts breaking on initialize, return to setting up properly 
```

The result: fast, autonomous turnaround times even for complex GUI environments.

---

## Phase 6 – Optional Polish & Extensions

### Deeper dive into registries

An environment often needs *structured knowledge* about tasks, evaluation logic, or problem definitions.  The browser examples keep these in three explicit registries:

| Registry | Purpose | Example resource URI |
|----------|---------|----------------------|
| **Setup** | How to seed the environment before the agent starts | `setup://registry` & `setup://{env}` |
| **Evaluators** | Functions that decide success & reward | `evaluators://registry` |
| **Problems** | Bundled benchmarks / tasks with their own setup & evaluate pairs | `problems://registry` |

Each registry is just a dictionary mapping a *name* to a *class*.  Use a **decorator** to register classes:

```python
from .registry import setup, evaluator, problem

@setup("todo_seed")
class TodoSeed:
    ...

@evaluator("todo_completed")
class TodoCompleted:
    ...

@problem("todo_basic", description="Complete two todo items", difficulty="easy")
class TodoBasic:
    def get_setup(self):
        return {"function": "todo_seed", "args": {"num_items": 5}}
    def get_evaluation(self):
        return {"function": "todo_completed", "args": {"expected_count": 2}}
```

Decorators keep registration *next to the implementation* and avoid manual bookkeeping.  The server simply exposes the combined metadata through an MCP **resource**.  Follow `environments/simple_browser/src/hud_controller/problems/registry.py` as a template and expose the JSON with `@mcp.resource("problems://registry")`.

### Other finishing touches

* **Performance** – lazy-load heavy resources, pool DB connections, cache expensive calls.
* **Security** – sandbox untrusted code, keep secrets in env vars, audit-log every tool call.
* **Creative ideas** – API simulators, network test-beds, game worlds… if it fits in Docker it can be an MCP environment.

---

## Summary

1. Start with a *plain* Dockerfile – verify it runs.  
2. Add a minimal FastMCP server – verify with stdio, Inspector, Docker.  
3. Implement tools – verify discovery + execution.  
4. Run the same image remotely – verify telemetry.  
5. Automate the loop with cursor-mcp.  
6. Polish and extend as inspiration strikes.

Happy building – and remember: **stderr is your friend, stdout belongs to MCP.** 🚀
