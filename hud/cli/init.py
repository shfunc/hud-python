"""Initialize new HUD environments with minimal templates."""
import os
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

# Embedded templates
DOCKERFILE_TEMPLATE = '''FROM python:3.11-slim

WORKDIR /app

# Install git for hud-python dependency
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Set logging to stderr
ENV HUD_LOG_STREAM=stderr

# Start context server in background, then MCP server
CMD ["sh", "-c", "python -m hud_controller.context & sleep 1 && exec python -m hud_controller.server"]
'''

PYPROJECT_TEMPLATE = '''[project]
name = "{name}"
version = "0.1.0"
description = "A minimal HUD environment"
requires-python = ">=3.11"
dependencies = [
    "hud-python @ git+https://github.com/hud-evals/hud-python.git",
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
'''

CONTEXT_TEMPLATE = '''"""Minimal context that persists across hot-reloads."""
from hud.server.context import run_context_server
import asyncio

class Context:
    def __init__(self):
        self.count = 0
    
    def act(self):
        self.count += 1
        return self.count
    
    def get_count(self):
        return self.count

if __name__ == "__main__":
    asyncio.run(run_context_server(Context()))
'''

SERVER_TEMPLATE = '''"""Minimal MCP server for HUD."""
from hud.server import MCPServer
from hud.server.context import attach_context

mcp = MCPServer(name="{name}")
ctx = None

@mcp.initialize
async def init(init_ctx):
    global ctx
    ctx = attach_context("/tmp/hud_ctx.sock")

@mcp.shutdown
async def cleanup():
    global ctx
    ctx = None

@mcp.tool()
async def act() -> str:
    """Perform an action."""
    return f"Action #{{ctx.act()}}"

@mcp.tool()
async def setup() -> str:
    """Required for HUD environments."""
    return "Ready"

@mcp.tool() 
async def evaluate() -> dict:
    """Required for HUD environments."""
    return {{"count": ctx.get_count()}}

if __name__ == "__main__":
    mcp.run()
'''

README_TEMPLATE = '''# {title}

A minimal HUD environment created with `hud init`.

## Quick Start

```bash
# Build and run locally
hud dev

# Or build first
docker build -t {name}:dev .
hud dev --image {name}:dev
```

## Structure

- `src/hud_controller/server.py` - MCP server with tools
- `src/hud_controller/context.py` - Persistent state across hot-reloads
- `Dockerfile` - Container configuration
- `pyproject.toml` - Python dependencies

## Adding Tools

Add new tools to `server.py`:

```python
@mcp.tool()
async def my_tool(param: str) -> str:
    """Tool description."""
    return f"Result: {{param}}"
```

## Adding State

Extend the `Context` class in `context.py`:

```python
class Context:
    def __init__(self):
        self.count = 0
        self.data = {{}}  # Add your state
```

## Learn More

- [HUD Documentation](https://docs.hud.so)
- [MCP Specification](https://modelcontextprotocol.io)
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


def create_environment(name: Optional[str], directory: str, force: bool) -> None:
    """Create a new HUD environment from templates."""
    from hud.utils.design import HUDDesign
    
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
    dockerfile_path.write_text(DOCKERFILE_TEMPLATE.strip() + "\n")
    files_created.append("Dockerfile")
    
    # pyproject.toml
    pyproject_path = target_dir / "pyproject.toml"
    pyproject_content = PYPROJECT_TEMPLATE.format(name=package_name).strip() + "\n"
    pyproject_path.write_text(pyproject_content)
    files_created.append("pyproject.toml")
    
    # README.md
    readme_path = target_dir / "README.md"
    readme_content = README_TEMPLATE.format(name=package_name, title=name).strip() + "\n"
    readme_path.write_text(readme_content)
    files_created.append("README.md")
    
    # Python files
    # __init__.py
    init_path = src_dir / "__init__.py"
    init_path.write_text('"""HUD Controller Package"""\n')
    files_created.append("src/hud_controller/__init__.py")
    
    # context.py
    context_path = src_dir / "context.py"
    context_path.write_text(CONTEXT_TEMPLATE.strip() + "\n")
    files_created.append("src/hud_controller/context.py")
    
    # server.py (need to escape the double braces for .format())
    server_path = src_dir / "server.py"
    server_content = SERVER_TEMPLATE.format(name=package_name).strip() + "\n"
    server_path.write_text(server_content)
    files_created.append("src/hud_controller/server.py")
    
    # Success message
    design.header(f"Created HUD Environment: {name}")
    
    design.section_title("Files created")
    for file in files_created:
        console.print(f"  ✓ {file}")
    
    design.section_title("Next steps")
    
    # Show commands based on where we created the environment
    if target_dir == Path.cwd():
        console.print("1. Start development server:")
        console.print("   [cyan]hud dev[/cyan]")
    else:
        console.print("1. Enter the directory:")
        console.print(f"   [cyan]cd {target_dir.relative_to(Path.cwd())}[/cyan]")
        console.print("\n2. Start development server:")
        console.print("   [cyan]hud dev[/cyan]")
    
    console.print("\n3. Connect from Cursor:")
    console.print("   Follow the instructions shown by [cyan]hud dev[/cyan]")
    
    console.print("\n4. Customize your environment:")
    console.print("   - Add tools to [cyan]src/hud_controller/server.py[/cyan]")
    console.print("   - Add state to [cyan]src/hud_controller/context.py[/cyan]")
    
    # Show a sample of the server code
    design.section_title("Your MCP server")
    sample_code = '''@mcp.tool()
async def act() -> str:
    """Perform an action."""
    return f"Action #{ctx.act()}"'''
    
    syntax = Syntax(sample_code, "python", theme="monokai", line_numbers=False)
    console.print(Panel(syntax, border_style="dim"))
