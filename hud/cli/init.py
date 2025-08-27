"""Initialize new HUD environments with minimal templates."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.panel import Panel
from rich.syntax import Syntax

from hud.utils.design import HUDDesign

# Embedded templates
DOCKERFILE_TEMPLATE = """FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Start context server in background, then MCP server
CMD ["sh", "-c", "python -m hud_controller.env & sleep 1 && exec python -m hud_controller.server"]
"""  # noqa: E501

PYPROJECT_TEMPLATE = """[project]
name = "{name}"
version = "0.1.0"
description = "A minimal HUD environment"
requires-python = ">=3.11"
dependencies = [
    "hud-python",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hud]
image = "{name}:dev"

[tool.hatch.metadata]
allow-direct-references = true

[tool.hatch.build.targets.wheel]
packages = ["src/hud_controller"]
"""

ENV_TEMPLATE = '''"""Minimal environment that persists across hot-reloads."""
from hud.server.context import run_context_server
import asyncio

class Environment:
    """Simple counter environment."""
    
    def __init__(self):
        self.count = 0
    
    def act(self):
        """Increment the counter."""
        self.count += 1
        return self.count
    
    def get_count(self):
        """Get current counter."""
        return self.count
    
    def reset(self):
        """Reset counter to zero."""
        self.count = 0

if __name__ == "__main__":
    asyncio.run(run_context_server(Environment(), sock_path="/tmp/hud_ctx.sock"))
'''

SERVER_TEMPLATE = '''"""Minimal MCP server for HUD."""
import sys
import logging
from hud.server import MCPServer
from hud.server.context import attach_context
from hud.tools.types import EvaluationResult

# Configure logging to stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | %(name)s | %(message)s'
)

mcp = MCPServer(name="{name}")
env = None

@mcp.initialize
async def init(ctx):
    global env
    env = attach_context("/tmp/hud_ctx.sock")
    logging.info("Connected to context server")

@mcp.shutdown
async def cleanup():
    global env
    env = None

@mcp.tool()
async def act() -> str:
    """Perform an action that changes the environment state."""
    if env is None:
        raise RuntimeError("Context not initialized")
    count = env.act()
    return f"Action #{{count}} performed. Current count: {{count}}"

@mcp.tool()
async def setup() -> str:
    """Reset the environment to initial state."""
    if env is None:
        raise RuntimeError("Context not initialized")
    env.reset()
    return "Counter reset to 0"

@mcp.tool()
async def evaluate(target: int = 10) -> EvaluationResult:
    """Check if the counter reached the target value."""
    if env is None:
        raise RuntimeError("Context not initialized")
    current_count = env.get_count()
    
    # Calculate reward as progress towards target
    reward = min(current_count / target, 1.0) if target > 0 else 0.0
    done = current_count >= target
    
    return EvaluationResult(
        reward=reward,
        done=done,
        content=f"Counter at {{current_count}}/{{target}}"
    )

if __name__ == "__main__":
    mcp.run()
'''

TASKS_JSON_TEMPLATE = '''[
  {{
    "prompt": "Increment the counter to reach 10",
    "mcp_config": {{
      "{name}": {{
        "url": "http://localhost:8765/mcp"
      }}
    }},
    "setup_tool": {{
      "name": "setup",
      "arguments": {{}}
    }},
    "evaluate_tool": {{
      "name": "evaluate",
      "arguments": {{
        "target": 10
      }}
    }}
  }}
]
'''

TEST_TASK_TEMPLATE = '''#!/usr/bin/env python
"""Simple example of running tasks from tasks.json.

Make sure to run 'hud dev --build' in another terminal first!
"""

import asyncio
import json
from hud.datasets import Task
from hud.clients import MCPClient


async def run_task(task_data: dict):
    task = Task(**task_data)
    client = MCPClient(mcp_config=task.mcp_config)

    try:
        print("Initializing client...")
        await client.initialize()

        result = await client.call_tool(task.setup_tool) # type: ignore
        print(f"✅ Setup: {{result.content}}")
        
        print("\\n🔄 Performing actions:")
        for _ in range(10):
            result = await client.call_tool(name="act", arguments={{}})
            print(f"  {{result.content}}")
        
        result = await client.call_tool(task.evaluate_tool) # type: ignore
        print(f"\\n📊 Evaluation: {{result.content}}")
        
        return result.content
    except Exception as e:
        if "connection" in str(e).lower():
            print("❌ Could not connect. Make sure 'hud dev --build' is running in another terminal.")
        else:
            raise e
    finally:
        await client.shutdown()


async def main():
    for task_data in json.load(open("tasks.json")):
        await run_task(task_data)

if __name__ == "__main__":
    asyncio.run(main())
'''

NOTEBOOK_TEMPLATE = '''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "### Step 1: Create a Task\\n",
    "\\n",
    "A Task combines:\\n",
    "- **Prompt**: What we want an agent to accomplish\\n",
    "- **MCP Config**: How to spawn the environment\\n",
    "- **Setup Tool**: How to prepare the environment\\n",
    "- **Evaluate Tool**: How to check if the task succeeded"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "from hud.datasets import Task\\n",
    "from hud.types import MCPToolCall\\n",
    "\\n",
    "# Create a task that uses our {name} environment\\n",
    "# See tasks.json for how to build a loadable task dataset\\n",
    "task = Task(\\n",
    "    prompt=\\"Increment the counter to reach 10\\",\\n",
    "    mcp_config={{\\n",
    "        \\"{name}\\": {{\\n",
    "            \\"url\\": \\"http://localhost:8765/mcp\\"\\n",
    "        }},\\n",
    "    }},\\n",
    "    setup_tool=MCPToolCall(name=\\"setup\\", arguments={{}}),\\n",
    "    evaluate_tool=MCPToolCall(name=\\"evaluate\\", arguments={{\\"target\\": 10}}),\\n",
    ")"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "### Step 2: Initialize MCP Client\\n",
    "\\n",
    "Run `hud dev --build` before this cell to intialize the server at `http://localhost:8765/mcp`"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "from hud.clients import MCPClient\\n",
    "\\n",
    "# Create the client\\n",
    "client = MCPClient(mcp_config=task.mcp_config, auto_trace=False)\\n",
    "\\n",
    "# Initialize it (this connects to our dev server)\\n",
    "await client.initialize()"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "### Step 3: Run Setup\\n",
    "\\n",
    "Call the setup tool to prepare the environment according to the task."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Run the setup from our task\\n",
    "setup_result = await client.call_tool(task.setup_tool) # type: ignore\\n",
    "print(f\\"Setup result: {{setup_result}}\\")"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "### Step 4: Perform Actions\\n",
    "\\n",
    "Now we'll manually perform actions to complete the task. In a real scenario, an AI agent would figure out what actions to take."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Increment the counter 10 times\\n",
    "for i in range(10):\\n",
    "    result = await client.call_tool(name=\\"act\\", arguments={{}})\\n",
    "    print(f\\"Step {{i+1}}: {{result.content}}\\")"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## Step 5: Evaluate Success\\n",
    "\\n",
    "Check if we completed the task according to the evaluation criteria."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Run the evaluation from our task\\n",
    "eval_result = await client.call_tool(task.evaluate_tool) # type: ignore\\n",
    "\\n",
    "# The result is a list with one TextContent item containing JSON\\n",
    "print(eval_result)"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "### Step 6: Cleanup\\n",
    "\\n",
    "Always shut down the client when done to stop the Docker container. Either stop hud dev in the terminal, or run this command:"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "await client.shutdown()"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "### Bonus: Running with an AI Agent\\n",
    "\\n",
    "Instead of manually calling tools, you can have an AI agent solve the task automatically."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Uncomment to run with Claude (requires ANTHROPIC_API_KEY)\\n",
    "from hud.agents import ClaudeAgent\\n",
    "\\n",
    "# Create an agent\\n",
    "agent = ClaudeAgent(\\n",
    "    model=\\"claude-sonnet-4-20250514\\",\\n",
    "    allowed_tools=[\\"act\\"]  # Only allow the act tool\\n",
    ")\\n",
    "\\n",
    "# Run the task\\n",
    "result = await agent.run(task)\\n",
    "print(f\\"Final reward: {{result.reward}}\\")"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "### Next Steps\\n",
    "\\n",
    "1. **Create your own evaluators**: Add new evaluation functions to `server.py`\\n",
    "2. **Build complex environments**: Replace the simple counter with your actual application\\n",
    "3. **Test with agents**: Use different AI models to solve your tasks\\n",
    "\\n",
    "For more examples, check out:\\n",
    "- `environments/text_2048/` - A complete 2048 game environment\\n",
    "- `environments/browser/` - A full browser automation environment with GUI"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": []
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "codemirror_mode": {{
    "name": "ipython",
    "version": 3
   }},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}
'''

README_TEMPLATE = '''# {title}

A minimal HUD environment demonstrating the Task pattern with a simple counter.

## Quick Start

### Interactive Development
```bash
# 1. Start the environment (optional: with inspector)
hud dev --build --inspector

# 2. Choose your preferred way to test:

# Option A: Interactive notebook test_env.ipynb (great for learning!)

# Option B: Simple Python script (runs all tasks from tasks.json)
python test_task.py
```

### Run with an Agent
```bash
# Run the task with Claude
hud eval tasks.json --agent claude
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
# Build and push your environment (this requires docker hub login and hud api key)
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
# app.hud.so/leaderboards/your-org/your-dataset
```

**Note**: Only public HuggingFace datasets appear as leaderboards!

📚 Learn more: [Creating Benchmarks](https://docs.hud.so/evaluate-agents/create-benchmarks) | [Leaderboards](https://docs.hud.so/evaluate-agents/leaderboards)
'''


def sanitize_name(name: str) -> str:
    """Convert a name to a valid Python package name."""
    # Replace spaces and hyphens with underscores
    name = name.replace(" ", "_").replace("-", "_")
    # Remove any non-alphanumeric characters except underscores
    name = "".join(c for c in name if c.isalnum() or c == "_")
    # Ensure it doesn't start with a number
    if name and name[0].isdigit():
        name = f"env_{name}"
    return name.lower()


def create_environment(name: str | None, directory: str, force: bool) -> None:
    """Create a new HUD environment from templates."""

    design = HUDDesign()

    # Determine environment name
    if name is None:
        # Use current directory name
        current_dir = Path.cwd()
        name = current_dir.name
        target_dir = current_dir
        design.info(f"Using current directory name: {name}")
    else:
        # Create new directory
        target_dir = Path(directory) / name

    # Sanitize name for Python package
    package_name = sanitize_name(name)
    if package_name != name:
        design.warning(f"Package name adjusted for Python: {name} → {package_name}")

    # Check if directory exists
    if target_dir.exists() and any(target_dir.iterdir()):
        if not force:
            design.error(f"Directory {target_dir} already exists and is not empty")
            design.info("Use --force to overwrite existing files")
            raise typer.Exit(1)
        else:
            design.warning(f"Overwriting existing files in {target_dir}")

    # Create directory structure
    src_dir = target_dir / "src" / "hud_controller"
    src_dir.mkdir(parents=True, exist_ok=True)

    # Write files with proper formatting
    files_created = []

    # Dockerfile
    dockerfile_path = target_dir / "Dockerfile"
    dockerfile_path.write_text(DOCKERFILE_TEMPLATE.strip() + "\n", encoding="utf-8")
    files_created.append("Dockerfile")

    # pyproject.toml
    pyproject_path = target_dir / "pyproject.toml"
    pyproject_content = PYPROJECT_TEMPLATE.format(name=package_name).strip() + "\n"
    pyproject_path.write_text(pyproject_content, encoding="utf-8")
    files_created.append("pyproject.toml")

    # README.md
    readme_path = target_dir / "README.md"
    readme_content = README_TEMPLATE.format(name=package_name, title=name).strip() + "\n"
    readme_path.write_text(readme_content, encoding="utf-8")
    files_created.append("README.md")

    # Python files
    # __init__.py
    init_path = src_dir / "__init__.py"
    init_path.write_text('"""HUD Controller Package"""\n', encoding="utf-8")
    files_created.append("src/hud_controller/__init__.py")

    # env.py
    env_path = src_dir / "env.py"
    env_path.write_text(ENV_TEMPLATE.strip() + "\n", encoding="utf-8")
    files_created.append("src/hud_controller/env.py")

    # server.py (need to escape the double braces for .format())
    server_path = src_dir / "server.py"
    server_content = SERVER_TEMPLATE.format(name=package_name).strip() + "\n"
    server_path.write_text(server_content, encoding="utf-8")
    files_created.append("src/hud_controller/server.py")

    # tasks.json
    tasks_path = target_dir / "tasks.json"
    tasks_content = TASKS_JSON_TEMPLATE.format(name=package_name).strip() + "\n"
    tasks_path.write_text(tasks_content, encoding="utf-8")
    files_created.append("tasks.json")

    # test_task.py
    test_task_path = target_dir / "test_task.py"
    test_task_path.write_text(TEST_TASK_TEMPLATE.strip() + "\n", encoding="utf-8")
    files_created.append("test_task.py")

    # notebook.ipynb
    notebook_path = target_dir / "test_env.ipynb"
    notebook_content = NOTEBOOK_TEMPLATE.format(name=package_name).strip() + "\n"
    notebook_path.write_text(notebook_content, encoding="utf-8")
    files_created.append("test_env.ipynb")

    # Success message
    design.header(f"Created HUD Environment: {name}")

    design.section_title("Files created")
    for file in files_created:
        design.status_item(file, "created")

    design.section_title("Next steps")

    # Show commands based on where we created the environment
    if target_dir == Path.cwd():
        design.info("1. Start development server (with MCP inspector):")
        design.command_example("hud dev --inspector")
    else:
        design.info("1. Enter the directory:")
        design.command_example(f"cd {target_dir}")
        design.info("\n2. Start development server (with MCP inspector):")
        design.command_example("hud dev --inspector")

    design.info("\n3. Connect from Cursor or test via the MCP inspector:")
    design.info("   Follow the instructions shown by hud dev --inspector")

    design.info("\n4. Test your environment:")
    design.command_example("python test_task.py")

    design.info("\n5. Customize your environment:")
    design.info("   - Add tools to src/hud_controller/server.py")
    design.info("   - Add state to src/hud_controller/env.py")
    design.info("   - Modify tasks in tasks.json")
    design.info("   - Experiment in run_eval.ipynb")

    # Show a sample of the server code
    design.section_title("Your MCP server")
    sample_code = '''@mcp.tool()
async def act() -> str:
    """Perform an action that changes the environment state."""
    if env is None:
        raise RuntimeError("Context not initialized")
    count = env.act()
    return f"Action #{count} performed. Current count: {count}"'''

    syntax = Syntax(sample_code, "python", theme="monokai", line_numbers=False)
    design.console.print(Panel(syntax, border_style="dim"))
