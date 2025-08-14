# HUD + ART Integration

Train agents on HUD MCP environments using ART's reinforcement learning framework.

## Overview

This integration combines:
- **HUD's MCP infrastructure** - Docker containers, SSH connections, tool discovery
- **HUD's evaluation tools** - Task-specific rewards from evaluate tools
- **ART's training framework** - GRPO training, LoRA fine-tuning

The result: Train custom agents for any HUD environment using HUD's built-in rewards.

## Quick Start

### 1. Install Dependencies

```bash
cd hud-python/rl
uv sync  # or: pip install -e .
```

### 2. Train on 2048 Environment

```bash
# Basic training
python -m art_integration.train_hud_art --env 2048

# With custom settings
python -m art_integration.train_hud_art \
    --env 2048 \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --scenarios 20 \
    --steps 10 \
    --learning-rate 1e-5
```

### 3. Train on Custom Environment

```python
from art_integration import train_hud_with_art

await train_hud_with_art(
    environment="browser",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    num_training_scenarios=24,
)
```

## Architecture

### Components

1. **`ARTTrainingAgent`** - Bridges HUD's MCPAgent with ART's Trajectory collection
   - Uses HUD's MCPClient for all MCP communication
   - Collects messages in ART's format during execution
   - Returns trajectories for training

2. **Task Loading** - Uses HUD's HuggingFace datasets
   - Loads from `hud-evals/*` datasets on HuggingFace
   - Automatically converts to TaskConfig objects
   - Consistent interface for all environments

3. **Training Loop** - GRPO training with HUD rewards
   - Collects multiple rollouts per scenario
   - Uses HUD's evaluate tool for rewards
   - Trains with ART's backend

### Data Flow

```
HUD MCPClient → ARTTrainingAgent → ART Trajectory
       ↓                                  ↓
   MCP Server                    HUD Evaluate Tool
       ↓                                  ↓
   Tool Results                    GRPO Training
```

## Configuration

Edit `config.yaml` to customize:

```yaml
environments:
  my_env:
    mcp_config:
      local:
        command: "docker"
        args: ["run", "my-mcp-server"]
    max_turns: 15

training:
  default:
    learning_rate: 1e-5
    rollouts_per_group: 4
```

## Examples

### Train on Remote Server

```yaml
# config.yaml
environments:
  remote_env:
    mcp_config:
      ssh:
        command: "ssh"
        args: ["user@host", "docker", "run", "mcp-server"]
```

```bash
python -m art_integration.train_hud_art \
    --env remote_env \
    --config config.yaml
```

### Load HUD Task Configurations

```python
from datasets import load_dataset
from hud.datasets import to_taskconfigs

# Load from HuggingFace datasets
dataset = load_dataset("hud-evals/2048-taskset", split="train")
tasks = to_taskconfigs(dataset)

# Available datasets:
# - hud-evals/2048-taskset
# - hud-evals/SheetBench-50
# - hud-evals/gmail-taskset
```

### Use HUD's Evaluation Tools

```python
# HUD's evaluate tools provide task-specific rewards
# For 2048: max_number evaluator gives reward based on tile reached
# For browser: task completion evaluator checks success

# Rewards are automatically collected during trajectory execution
trajectory = await agent.run_with_trajectory(task)
print(f"Reward: {trajectory.reward}")  # From HUD's evaluate tool
```

## Environment Support

### Available Environments

All environments load from HuggingFace datasets:

- **2048** (`hud-evals/2048-taskset`)
  - Text-based 2048 game
  - Tools: setup, move, evaluate
  - Rewards: Reaching target tiles

- **Browser** (`hud-evals/SheetBench-50`)
  - Web automation tasks
  - Tools: click, type, screenshot
  - Rewards: Task completion

- **Gmail** (`hud-evals/gmail-taskset`)
  - Email management tasks
  - Tools: Gmail API operations
  - Rewards: Task completion

### Adding New Environments

1. Upload your task dataset to HuggingFace
2. Add mapping in `train_hud_art.py`:
   ```python
   dataset_map = {
       "my_env": "my-org/my-taskset",
       ...
   }
   ```
3. Train: `python -m art_integration.train_hud_art --env my_env`

## Training Tips

### For Best Results

1. **More rollouts = better comparison**
   ```bash
   --rollouts-per-group 6  # Compare 6 trajectories
   ```

2. **Diverse scenarios improve generalization**
   ```bash
   --scenarios 50  # More training data
   ```

3. **Adjust learning rate for model size**
   - 3B models: `1e-5`
   - 7B models: `5e-6`
   - 14B+ models: `1e-6`

### GPU Memory

For limited VRAM, adjust in code:
```python
art_model._internal_config = art.dev.InternalModelConfig(
    init_args=art.dev.InitArgs(
        max_seq_length=4096,  # Reduce from 8192
    ),
    engine_args=art.dev.EngineArgs(
        gpu_memory_utilization=0.7,  # Reduce from 0.9
    ),
)
```

## API Reference

### Core Functions

```python
async def train_hud_with_art(
    environment: str,           # Environment name
    base_model: str,            # Model to fine-tune
    num_training_scenarios: int,  # Training data size
    rollouts_per_group: int,    # Trajectories per scenario
    learning_rate: float,       # Learning rate
    use_ruler: bool,           # Enable RULER scoring
) -> None
```

```python
async def generate_scenarios_from_hud(
    mcp_client: MCPClient,      # HUD MCP client
    num_scenarios: int,         # Number to generate
) -> list[TaskConfig]
```

### Classes

```python
class ARTTrainingAgent(MCPAgent):
    """HUD agent that collects ART trajectories."""
    
    async def run_with_trajectory(
        self, task: TaskConfig
    ) -> art.Trajectory
```

## Troubleshooting

### "No MCP configuration found"
- Check environment name matches config.yaml
- Ensure Docker container exists for built-in envs

### "Unknown environment"
- Check that the environment name matches the dataset_map in train_hud_art.py
- Available environments: 2048, browser, gmail

### Out of memory
- Reduce `max_seq_length` in training script
- Use smaller base model
- Reduce `rollouts_per_group`

### Training not improving
- Increase `num_training_scenarios`
- Check that HUD's evaluate tool is returning proper rewards
- Verify the task's evaluate_tool configuration

## Advanced Usage

### Custom Reward Functions

Override HUD's evaluate rewards:
```python
trajectory.reward = custom_reward_function(trajectory)
```

### SkyPilot Cloud Training

```python
from art.skypilot import SkyPilotBackend

backend = await SkyPilotBackend().initialize_cluster(
    cluster_name="hud-training",
    gpu="H100-SXM",
)
```

### Export Trained Model

```python
# Model saved to .art/project/models/model_name/
# Load with vLLM or convert to HuggingFace format
```

## License

Same as HUD and ART parent projects.