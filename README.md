<div align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo.svg">
    <img src="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo.svg" alt="HUD" width="150" style="margin-bottom: 20px;"/>
  </picture>
</div>

Evaluate and improve agents. Wrap software as environments, run benchmarks, and train with RL – locally or at scale.

[![PyPI version](https://img.shields.io/pypi/v/hud-python?style=flat-square)](https://pypi.org/project/hud-python/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Built on MCP](https://img.shields.io/badge/Built%20on-MCP-blueviolet?style=flat-square)](https://modelcontextprotocol.io)
[![Discord](https://img.shields.io/discord/1327447144772407390?label=Discord&logo=discord&style=flat-square)](https://discord.gg/wkjtmHYYjm)
[![X Follow](https://img.shields.io/twitter/follow/hud_evals?style=social)](https://x.com/intent/user?screen_name=hud_evals)
[![Shop](https://img.shields.io/badge/shop-white.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAJCAYAAAAywQxIAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAACxMAAAsTAQCanBgAAAF6SURBVChTlZA9ixNhFIWf8yaTpFHRRMXCKpAZhCAYFvwoLHZhwUKw9A9YCJb+Bq0sxGbBQrTxX1j41dvIRAjGZbdwRUUGIzPMeyw2swS3WZ/ynHvP5VylafoAWAd+5Xm+wX+SpukmcMf29RDCZrD9BViz3f53+CjYngKZpD5A2/Y7SQBMJpOkKIprdV1vdzqdHzHGblmW9Ww2+5pl2TmAxWKxmM/nP8fj8cmqqtZijJ9sb0u6ABBWjh0riuIt8CqE8LGu66e2d5MkeQ8QY3xme7fb7T4ZjUbrZVl+jjFuSXoEXGxCDgIl9WzfAO5LSmzvNB771R6vzG4Bx0MIt/M8vwV8aLyDQNt70+n0G1AspaTxVln+aghQluVsKbvxVysflT9NQK/XO7R/SGiQ9Nt2aftElmWXJd1kv0kbeANQVdWl4XB4XtJouXaqNRgMHkrqS+r0+/3XwD1JXdungRfAVWBi+6WkK8D3EMJz22cl3W21WgNgx3YAzvwFd0Chdq03gKUAAAAASUVORK5CYII=&style=flat-square)](https://shop.hud.so)

### Are you a startup building agents?

[📅 Hop on a call](https://cal.com/jay-ram-z6st6w/demo) or [📧 founders@hud.so](mailto:founders@hud.so)

## Highlights

- 🚀 **MCP-native connectivity** – any language model can call any hud environment.
- ⚡️ **Live telemetry** – inspect every tool call, observation, and reward in real time.
- 🗂️ **Public benchmarks** – SheetBench-50, OSWorld, and more.
- 🌱 **Reinforcement learning built-in** – Verifiers gym and ART pipelines for training.
- 🌐 **Cloud browsers** – AnchorBrowser, Steel, BrowserBase integrations.
- 🛠️ **Hot-reload dev loop** – edit environments live inside Cursor Agent.

> | We welcome contributors and feature requests – open an issue to discuss improvements.

## Installation

```bash
# Core installation - MCP servers, CLI, basic tools for environment design
pip install hud-python

# Agent installation - Adds AI providers, telemetry, datasets
pip install "hud-python[agent]"

# CLI utilities (inside isolated env)
uv tool install hud-python

# From source (latest)
git clone https://github.com/hud-evals/hud-python
pip install -e "hud-python[dev]"
```

> | See [docs.hud.so](https://docs.hud.so) for full documentation.

## Quick start

```python
import asyncio, hud, os
from hud.clients import MCPClient
from hud.agents import ClaudeAgent
from hud.datasets import Task

async def main() -> None:
    with hud.trace("Quick Start 2048"): # All telemetry works for any MCP agent
        task = Task(
            prompt="Reach 64 in 2048.",
            mcp_config={
                "hud": {
                    "url": "https://mcp.hud.so", # All hud environments work with any MCP client
                    "headers": {
                        "Authorization": f"Bearer {os.getenv('HUD_API_KEY')}",
                        "Mcp-Image": "hudpython/hud-text-2048:v1.1"
                    }
                }
            },
            evaluate_tool={"name": "evaluate", "arguments": {"name": "max_number", "target": 64}},
        )

        client = MCPClient(mcp_config=task.mcp_config)
        agent = ClaudeAgent(
            mcp_client=client,
            model="claude-3-7-sonnet-20250219",  # requires ANTHROPIC_API_KEY
        )

        result = await agent.run(task)
        print(f"Reward: {result.reward}")
        await client.close()

asyncio.run(main())
```

> | Every HUD environment is MCP-based and interactable from anywhere. Requires `HUD_API_KEY` and `ANTHROPIC_API_KEY`.

## Architecture

![HUD architecture](docs/images/architecture.svg)

```bash
# Explore any HUD-compatible image
hud analyze hudpython/hud-text-2048:v1.1
```

> | `hud analyze` launches the container, performs an MCP handshake, and lists tools/resources. Requires Docker.

## Reinforcement learning

```bash
git clone https://github.com/hud-evals/hud-python
cd hud-python
python rl/verifiers/train_2048.py
```

![RL curve](docs/images/rl_curve.png)

> | The repository ships runnable RL scripts; this example trains a policy on the 2048 environment using Verifiers.

## Agent evaluation

```bash
# Evaluate full dataset from command line
python examples/run_evaluation.py hud-evals/SheetBench-50 --full --agent claude

# Or use the run_dataset API
```

```python
import asyncio
from hud.datasets import run_dataset
from hud.agents import ClaudeAgent

results = await run_dataset(
    name="My SheetBench-50 Evaluation",
    dataset="hud-evals/SheetBench-50",
    agent_class=ClaudeAgent,
    agent_config={"model": "claude-3-7-sonnet-20250219"},
    max_concurrent=50,
    max_steps=50,
)
print(f"Average reward: {sum(r.reward for r in results) / len(results):.2f}")
```

![Trace screenshot](docs/images/trace_overview.png)

> | Batch evaluation streams results to the HUD platform for analysis and leaderboard submission.

## Custom environments (MCP)

```python
from hud.server import MCPServer

mcp = MCPServer("My Environment")

@mcp.tool()
async def click(x: int, y: int) -> None:
    ...

if __name__ == "__main__":
    mcp.run()
```

![Debug phases](docs/images/debug_progress.png)

> | Wrap any program as an MCP server with a few decorators; `hud debug` validates five phases automatically.

## Leaderboards & benchmarks

```bash
python examples/run_evaluation.py SheetBench-50 --dataset
```

![Leaderboard](docs/images/leaderboard_sheetbench.png)

> | `run_dataset` evaluates an agent on every task and uploads results for leaderboard comparison.

## CLI reference

| Command                 | Purpose                                    |
| ----------------------- | ------------------------------------------ |
| `hud analyze <image>` | Discover tools, resources, and metadata.   |
| `hud debug <image>`   | Five-phase health check of an environment. |
| `hud mcp`             | Expose analysis & debug as an MCP server.  |

## Roadmap & community

- Integrations with every agent.
- Evaluation environment registry.
- Native RL training to hud environments.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines. Open a PR or [issue](https://github.com/hud-evals/hud-python/issues).

Thanks to all our contributors!

<a href="https://github.com/hud-evals/hud-python/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=hud-evals/hud-python&max=50" />
</a>


Projects using HUD for agent evaluation and development:

<!-- gh-dependents-info-used-by-start -->
<!-- gh-dependents-info-used-by-end -->

## Citation

```bibtex
@software{hud2025agentevalplatform,
  author = {HUD and Jay Ram and Lorenss Martinsons and Parth Patel and Oskars Putans and Govind Pimpale and Mayank Singamreddy and Nguyen Nhat Minh},
  title  = {HUD: An Evaluation Platform for Agents},
  date   = {2025-04},
  url    = {https://github.com/hud-evals/hud-python},
  langid = {en}
}
```

## License

HUD is released under the MIT License – see the [LICENSE](LICENSE) file for details.
