# HUD MCP Environment Requirements

Quick guide for creating HUD-compatible MCP environments.

## Required MCP Tools

### 1. `setup` Tool
```python
@mcp.tool()
async def setup(config: dict) -> dict:
    """Initialize environment from task.setup config (any format)."""
    # Handle config however your environment needs
    return {"status": "success"}  # Return format is flexible
```

### 2. `evaluate` Tool  
```python
@mcp.tool()
async def evaluate(config: dict) -> dict:
    """Evaluate task completion from task.evaluate config."""
    # Your evaluation logic
    return {
        "reward": 1.0,    # Required: 0.0-1.0 score
        "done": True,     # Required: completion flag
        "info": {}        # Optional: metadata
    }
```

### 3. Interaction Tool(s)
At least one tool for agent interaction during task execution:

```python
# Option A: Use HUD computer tool
from hud.tools import HudComputerTool
from hud.tools.helper import register_instance_tool

register_instance_tool(mcp, "computer", HudComputerTool())

# Option B: Custom API tool
@mcp.tool()
async def api_request(url: str, method: str = "GET", data: dict = None) -> dict:
    # Your API logic
    pass
```

**Note**: `setup` and `evaluate` are **lifecycle** MCP tools that:
- Are **automatically discovered** by the MCP client (always available to framework)  
- Are **filtered out** from LLM conversation (not in `allowed_tools`)
- Are **called programmatically** by the agent during task execution

Only include **interaction tools** in `allowed_tools`: `computer`, `anthropic_computer`, `api_request`, etc.

## Config Flexibility

Task configs can be **any format**:
```python
task.setup = {"function": "reset", "args": {}}
task.setup = {"id": "task_123"}
task.setup = {"name": "problem_name"}
task.setup = "simple_string"
task.setup = ["step1", "step2"]
# Your environment decides what formats to support
```

## Minimal Example

```python
from fastmcp import FastMCP
from hud.tools import HudComputerTool
from hud.tools.helper import register_instance_tool

mcp = FastMCP("My Environment")

@mcp.tool()
async def setup(config: dict) -> dict:
    return {"status": "success"}

@mcp.tool() 
async def evaluate(config: dict) -> dict:
    return {"reward": 1.0, "done": True, "info": {}}

@mcp.initialize()
async def init():
    register_instance_tool(mcp, "computer", HudComputerTool())

if __name__ == "__main__":
    mcp.run()
```

## Testing

### Option 1: Direct Agent Interface (Simplified)

```python
from hud.mcp_agent import ClaudeMCPAgent
from hud import Task
from mcp_use import MCPClient

# Setup MCP client
config = {"mcpServers": {"env": {"command": "python", "args": ["my_env.py"]}}}
client = MCPClient.from_dict(config)

# Create agent
agent = ClaudeMCPAgent(
    client=client,
    model="claude-sonnet-4-20250514",
    allowed_tools=["computer_anthropic"]  # Only interaction tools, setup/evaluate called internally
)

# Simple query
result = await agent.run("Take a screenshot and tell me what you see")

# Full task with setup/evaluate
task = Task(
    prompt="Complete the todo app test",
    setup={"function": "todo_seed", "args": {"num_items": 3}},
    evaluate={"function": "todo_completed", "args": {"expected_count": 1}}
)
eval_result = await agent.run(task)  # Returns {"reward": 1.0, "done": True, "info": {...}}
```

### Option 2: Traditional Job Interface

```python
from hud import Task, run_job
from hud.mcp_agent import ClaudeMCPAgent

task = Task(
    prompt="Test task",
    setup={"test": True},
    evaluate={"test": True}
)

await run_job(
    ClaudeMCPAgent, 
    task, 
    "test_job",
    agent_kwargs={"allowed_tools": ["computer_anthropic"]}
)
```

### Option 3: Basic MCP Testing

```python
from mcp_use import MCPClient

config = {"mcpServers": {"env": {"command": "python", "args": ["my_env.py"]}}}
client = MCPClient.from_dict(config)
session = await client.create_session("env")

# Test tools directly
await session.connector.call_tool("setup", {})
await session.connector.call_tool("evaluate", {})
```

## Key Features

✨ **Simplified Interface**: Use `agent.run(task)` for both simple queries and full task lifecycle  
📋 **Flexible Config**: Support any setup/evaluate config format your environment needs  
🔧 **Easy Integration**: Import HUD tools with `register_instance_tool()`  

## Examples

### Environment Examples
- [`simple_browser/`](./simple_browser/) - Computer tool + GUI automation
- [`qa_controller/`](./qa_controller/) - Text-based environment

### Usage Examples  
- [`simple_task_example.py`](../examples/agents_tools/simple_task_example.py) - Full demo with simple_browser
