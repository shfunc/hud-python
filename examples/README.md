# Examples

A collection of examples demonstrating HUD SDK usage patterns.

## Quick Start Examples

### 00_agent_env.py
Minimal MCP server and client in one file. Shows the basic agent-environment communication pattern.

```bash
python examples/00_agent_env.py
```

### 01_hello_2048.py
Complete agent evaluation on the 2048 environment using Claude.

```bash
python examples/01_hello_2048.py
```

> | Requires Docker and `ANTHROPIC_API_KEY` environment variable.

## Core Patterns

### 02_agent_lifecycle.py
Demonstrates the full agent lifecycle with telemetry and state management.
- Task creation and configuration
- Trace context for debugging
- State persistence between runs

### run_evaluation.py
Generic dataset evaluation runner supporting multiple agents.

```bash
# Run single task
python examples/run_evaluation.py hud-evals/SheetBench-50

# Run full dataset
python examples/run_evaluation.py hud-evals/SheetBench-50 --full
```

## Integration Examples

### claude_agent.py
Direct usage of Claude agent without environments.

### integration_mcp_use.py
Using the legacy `mcp_use` client for multi-server setups.

### integration_otel.py
Custom OpenTelemetry backend integration (e.g., Jaeger).

## Prerequisites

| Requirement | Used For |
|-------------|----------|
| Docker | Running environment containers |
| `HUD_API_KEY` | Cloud deployments and telemetry |
| `ANTHROPIC_API_KEY` | Claude agent examples |

## Common Pattern

All examples follow this structure:

```python
import asyncio, hud
from hud.datasets import Task
from hud.agents import ClaudeAgent

async def main():
    with hud.trace("example-name"):
        task = Task(
            prompt="Your task here",
            mcp_config={...}
        )
        
        agent = ClaudeAgent()
        result = await agent.run(task)
        print(f"Reward: {result.reward}")

asyncio.run(main())
```

> | The agent automatically creates an MCP client from `task.mcp_config` if none is provided.