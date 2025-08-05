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

### One-command sanity check (`docker_debug.py`)

While you move through the phases it’s handy to run the **interactive checker** to make sure nothing broke:

```bash
python environments/docker_debug.py my-environment:latest
```

The script walks the *same* checklist and prints coloured, human-friendly hints whenever something fails.

| What it validates | Phase |
|-------------------|-------|
| Container starts & logs to **stderr** | 1 |
| MCP server responds to an `initialize` request | 2 |
| Discovers `setup`, `evaluate`, and interaction tools | 3 |
| Calls `setup` / `evaluate`, checks telemetry & startup time | 4 |
| Spawns three concurrent clients to stress-test resources | 5 |

💡 **Run it after finishing each phase.** If the checker exits with a red ❌, scroll up for the gold-coloured *hint* block – it usually points directly to the root cause.

---

## Phase 1 – Write a *Simple* Dockerfile

**Goal →** the container starts, prints a message to **stderr**, and exits cleanly.  Nothing else.

Why stderr?  In Phase 2 the MCP server will reserve **stdout** for JSON-RPC traffic, so *all* human-readable logs should already go to the other stream.

### Minimal example

```dockerfile
FROM python:3.11-slim

WORKDIR /apphello

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

### Recommended Environment Structure

For Python-based MCP environments, use this standard structure:

```
my-environment/
├── Dockerfile
├── pyproject.toml
├── README.md
└── src/
    └── my_module/           # Your Python package
        ├── __init__.py
        ├── server.py        # MCP server (Phase 2)
        ├── setup/           # Setup functions (Phase 3)
        ├── evaluators/      # Evaluation logic (Phase 3)
        └── problems/        # Problem definitions (Phase 3)
```

This structure enables:
- Clean separation of concerns
- Easy volume mounting for development (Phase 5)
- Standard Python packaging with `pip install -e .`

• **One Dockerfile only** – no docker-compose.  
• If you're building a GUI environment, start from `hudpython/novnc-base:latest` instead and leave VNC configuration for later phases.

Checkpoint reached?  Congratulations – move on.

👉 Quick sanity check: `python environments/docker_debug.py my-environment:latest` (verifies Phase 1 automatically)

Need inspiration?  Skim the real Dockerfiles used in the example browser environments:
• [`browser/Dockerfile`](./browser/Dockerfile)
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
| 4 | **Run `docker_debug.py`** – `python environments/docker_debug.py my-environment:latest` | Combines the above checks & points out common mistakes |

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

👉 After wiring in the tools, confirm with `python environments/docker_debug.py my-environment:latest` – it now checks for their presence and basic execution.

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

Expose a resource named `telemetry://live` exactly like in `environments/browser/src/hud_controller/server.py` to return live url to be displayed on app.hud.so.

Once all of the above works you can unleash *hundreds* of concurrent agents on your new environment.

---

## Phase 5 – Takeoff: Automatic environment improvement with Cursor Agent

To enable rapid development without constant Docker rebuilds, use the unified Dockerfile's development mode. This allows you to edit code locally and see changes immediately in the running MCP server, and use Cursor Agent to automate iteration.

### Setting up Development Mode

#### 1. Update Your Dockerfile

First, modify your Dockerfile to support a `DEV_MODE` build argument to simplify transitioning between dev and build:

```dockerfile
# Add this at the top of your Dockerfile
ARG DEV_MODE=false

# ... your existing setup ...

# Conditionally handle source for dev mode -- this should reflect your environment structure
RUN if [ "$DEV_MODE" = "true" ]; then \
        mkdir -p /app/src/your_module && \
        echo "# Stub for editable install" > /app/src/your_module/__init__.py; \
    fi

# Copy source (will be overridden by volume mount in dev mode but necessary for the build in the Phase 1 recommended setup)
COPY src/ ./src/

# Install in editable mode still works!
RUN pip install -e .

# ... your existing setup ...
```

The key insight: In dev mode, we create stub files so the package can be installed, but the actual source will come from the volume mount.

#### 2. Build the Development Image

```bash
docker build --build-arg DEV_MODE=true -t my-environment:dev .
```

#### 3. Configure Cursor Agent for development

Add a development configuration to `.cursor/mcp.json` that includes the volume mount:

```jsonc
{
  "mcp_config": {
    // If your production config looks like this,
    "my-environment": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "my-environment:latest"]
    },
    // This is how you make the dev mode config:
    "my-environment-dev": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "%cd%/src:/app/src:rw",  // Windows
        // "-v", "$(pwd)/src:/app/src:rw",  // Linux/Mac
        "-e", "PYTHONPATH=/app/src",  // Required for module imports in the Phase 1 like setup
        // Add your environment variables here
        "my-environment:dev" // dev instead of latest!
      ]
    }
  }
}
```

2. Follow the cursor rules below: rebuild, refresh, test, reflect, repeat.
3. Keep the agent open for any messages or issues.

### 4. Cursor rules – paste this once

Inside `.cursor/rules/hud_environment_iteration.mdc` add (or verify) the following so the agent always knows the expected iteration loop:

```mdc
---
description: Improve an MCP environment
alwaysApply: false
---
Setup
1. Make sure the user has set up the mcp config for the environment by seeing if you have access to the tools by the given name (i.e. my-environment-dev), and make sure the title is in dev mode. If not, ask the user to make a dev version!
2. Make sure you can find the source folder for this environment. Explore its contents and README.
3. Clarify the objectives and ask follow up questions on the initial query to determine precise implementation details.

Iteration
1. Use the exposed tools by the environment to interact with it. This means navigating around with a computer, editing, launching commands, whatever means accessible to you. If there are any exposed resources, try to access them to determine the structure of the calls.
2. Based on the objectives, test and verify the functionality of different tools and parts of the environment. If any tool call responds with an error, note it down. If any interaction with the environment is wrong, unexpected, incomplete, or parts of the environment are not developed fully, note it down. If any new problem sets up wrong or evaluation does not match the expected outcome, note it down. All of these inconsistencies you should note down in your TODOs.
3. Then, based on the TODOs, view the source folder and find the places where those errors would occur. Think about the system and how to fix it. Then fix it.
4. After you've fixed your TODO items, go back to step 2 and test them. Test through all of your available tools, and use feedback (such as screenshots) to determine your progress. If they now work as expected, mark them as complete. If not, continue the loop from step 2. Be extremely careful, scrupolous and attentive to all details. Never assume something is working unless you've tested it fully for all of its edge cases.
5. The only time you can exit this iteration loop is if you're adding a *new* tool, a new import package to the environment, need additional environment variables, or if there is no feasible way to create input conditions to test something. In this case, ask the user for help and recap your progress. If you're simply changing tools, changing code, and still have more realistic TODOs, the environment will refresh automatically and you should continue working. In *all* other cases, you must continue this iteration loop until you can come up with no more TODOs. You must not halt.
```

### 5. Prompt the agent

```txt
Context: In the my-environment folder, I have a browser app environment. I've built a tool to interact with it called my-environment-dev.
Interaction: There are multiple tools to setup and evaluate the environment. There are also interaction tools for you to be able to move around it, and a screenshot tool to see the state. Use all of the available tools.
Objective: Please test if all setup, evaluation functions are working. This means you should come up with new problem definitions to test all functionality on. Be creative in how you pick edge cases to test on.
Rules: @hud_environment_iteration.mdc
```

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

Decorators keep registration *next to the implementation* and avoid manual bookkeeping.  The server simply exposes the combined metadata through an MCP **resource**.  Follow `environments/browser/src/hud_controller/problems/registry.py` as a template and expose the JSON with `@mcp.resource("problems://registry")`.

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
