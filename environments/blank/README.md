# test-test

A minimal HUD environment demonstrating the Task pattern with a simple counter.

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
# app.hud.so/leaderboards/your-org/your-dataset
```

**Note**: Only public HuggingFace datasets appear as leaderboards!

📚 Learn more: [Creating Benchmarks](https://docs.hud.so/evaluate-agents/create-benchmarks) | [Leaderboards](https://docs.hud.so/evaluate-agents/leaderboards)
