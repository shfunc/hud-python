# HUD environments for Reinforcement Learning

[hud-vf-gym](https://github.com/hud-evals/hud-vf-gym) module exposes CUA environments built with HUD MCP for training and evaluating RL agents using the [Verifiers](https://github.com/willccbb/verifiers) framework. It provides a standardized interface for agents to interact with computer interfaces through tool calls.


Need help? Join the Discord.
[![Discord](https://img.shields.io/discord/1327447144772407390?label=Discord&logo=discord&style=flat-square)](https://discord.gg/wkjtmHYYjm)


## Installation

You can directly install hud-vf-gym in this workspace `hud-python/rl`

```bash
# Clone hud-vf-gym
git clone https://github.com/hud-evals/hud-vf-gym.git

# Install dependencies (we recommend using uv for managing python envs)
uv sync

# Activate venv
source .venv/bin/activate

# Export environment variables
export OPENAI_API_KEY="YOUR_API_KEY"  # for running evals in openai models
export HUD_API_KEY="YOUR_API_KEY"   # for telemetry
```

if you don't have a hud api key, you can get one through the [HUD platform](https://app.hud.so).

## Running Evaluations

Use the Verifiers CLI to run evaluations such as hud-evals/2048-taskset.

For this, first build the base docker image locally:

```bash
cd ../environments/text_2048/
docker build -t hud-text-2048 .
```

Switch back to the workspace,

```bash
cd ../../rl
```

This will load in the taskset and run the gym via the config at ./configs/2048.yaml:
```bash
vf-eval hud-vf-gym \
    --model gpt-4.1-mini \
    --env-args '{"taskset": "hud-evals/2048-taskset", "config_path": "./configs/2048.yaml"}' \
    --num-examples 2 \
    --rollouts-per-example 3
```

Or use a custom config with a custom taskset:
```bash
# Use a custom config with custom taskset
vf-eval hud-vf-gym \
    --env-args '{"taskset": "your-org/your-taskset", "config_path": "custom_config.yaml"}' \
    --model gpt-4.1-mini \
    --num-examples 5 \
    --rollouts-per-example 3
```

## Training with GRPO

Verifier's GRPOtrainer is optimized for at least 2 GPUs. You can rent GPUs on marketplaces for [<$1/hr](https://app.primeintellect.ai).

HUD Gym supports training with the GRPO (Group Relative Policy Optimization) trainer:

Make sure you have the training dependencies installed:

```python
uv pip install 'verifiers[train]' && uv pip install flash-attn --no-build-isolation
```

Either just run:

```bash
python train_2048.py
```

Or configure your own training:
```python
from verifiers.trainers import GRPOTrainer, GRPOConfig
from verifiers import load_environment

# Load environment (both taskset and config_path are required)
env = load_environment(
    taskset="hud-evals/gmail-taskset",
    config_path="./configs/default.yaml"
)

# Configure training
config = GRPOConfig(
    model_name_or_path="your-model",
    per_device_train_batch_size=4,
    # ... other training parameters
)

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    env=env,
    args=config,
    processing_class=tokenizer,
)

# Train
trainer.train()
```

To train the 2048 agent on 2 A100 GPUs, use `train_2048.py` with one GPU for inference and one for training (see script header for setup commands).

For any issues related to Verifiers, see [their docs](https://verifiers.readthedocs.io/en/latest/training.html).

## Configuration

### Environment Configuration

HUD VF Gym uses a config-driven architecture where each environment defines its tools, mappings, and behavior through YAML configuration files.

#### Job Configuration

HUD VF Gym automatically creates a HUD job for each training/evaluation run to track all rollouts. Configure the job metadata in your YAML file:

```yaml
job:
  name: "2048 Training Run"
  metadata:
    dataset: "hud-evals/2048-taskset"
    experiment: "baseline"
```

This creates a unique job for each environment instance, and all rollouts during that training/evaluation run will be associated with the same job ID. You can view all traces from a training run grouped together on the [HUD platform](https://app.hud.so).

#### Available Configurations

The configuration files are now included in the hud-python/rl package:

- `./configs/default.yaml` - Browser/computer interaction environment
- `./configs/2048.yaml` - 2048 game environment

Note: `config_path` is now required when creating environments. There is no default fallback.

#### Configuration Structure

##### System Prompt
Defines the instructions and available tools presented to the agent:

```yaml
system_prompt: |
  You are an AI assistant that can interact with [environment description]...
  
  You have access to the following tools:
  - tool_name: Description
    Usage: <tool>tool_name(args)</tool>
  ...
```

##### Thinking Mode
Controls whether agents should use reasoning tags:

```yaml
parser:
  use_thinking: true   # Enable/disable <think> tag parsing (default: true)
  xml_weight: 0.6      # Weight for XML format validation
  action_weight: 0.4   # Weight for action syntax validation
```

When `use_thinking: false`, agents should output only tool calls without thinking tags (useful for production or when reasoning isn't needed).

##### Action Mappings
The core of the config-driven architecture. Maps agent-facing tools to underlying MCP tools:

```yaml
action_mappings:
  # Agent calls screenshot(), maps to computer tool
  screenshot:
    _parser:
      positional: []  # No arguments expected
    _tool: "computer"  # MCP tool to call
    action: "screenshot"  # Parameter for computer tool
  
  # Agent calls click(x, y), maps to computer tool  
  click:
    _parser:
      positional: ["x", "y"]  # Positional argument names
    _tool: "computer"
    action: "click"
    x:
      from_arg: "x"  # Map from parsed argument
    y:
      from_arg: "y"
  
  # Agent calls key("ctrl+a"), maps to computer tool with transform
  key:
    _parser:
      positional: ["key"]
    _tool: "computer"
    action: "press"
    keys:
      from_arg: "key"
      transform: "lambda x: x.split('+')"  # Split into key array
```

Key fields:
- `_parser.positional`: Defines expected positional arguments
- `_tool`: Specifies which MCP tool to call (required)
- `action`: For computer tool, specifies the action parameter
- `transform`: Lambda string for argument transformation
- `static`: Static value instead of from argument
- `from_arg`: Maps from parsed argument name

##### Transforms
Transforms are defined as lambda strings evaluated in a safe context:

```yaml
# Simple transforms
transform: "lambda x: x.upper()"
transform: "lambda x: int(x * 1000)"

# Context-aware transforms (access other arguments)
transform: "lambda d, ctx: ctx.get('amount', 3) if d == 'right' else -ctx.get('amount', 3)"
use_context: true
```

#### Parser and Rubric Configuration

```yaml
parser:
  xml_weight: 0.6    # Weight for XML format validation
  action_weight: 0.4  # Weight for action syntax validation

rubric:
  weights:
    task_completion: 0.8       # Primary task completion
    tool_execution: 0.1        # Successful tool execution rate
    format_compliance: 0.1     # XML format and action syntax
```

## Environment Examples

### Browser/Computer Environment (default.yaml)

Provides tools for interacting with computer interfaces:

```yaml
# Agent-facing tools (defined in system prompt)
screenshot()           → MCP: computer(action="screenshot")
click(100, 200)       → MCP: computer(action="click", x=100, y=200)
type("hello")         → MCP: computer(action="type", text="hello")
key("ctrl+a")         → MCP: computer(action="press", keys=["ctrl", "a"])
scroll("down", 3)     → MCP: computer(action="scroll", scroll_y=3, ...)
wait(2.5)             → MCP: computer(action="wait", time=2500)
done()                → Task completion signal
```

### 2048 Game Environment (2048.yaml)

Provides directional movement tools for the game:

```yaml
# Agent-facing tools (defined in system prompt)
left()   → MCP: move(direction="left")
right()  → MCP: move(direction="right")
up()     → MCP: move(direction="up")
down()   → MCP: move(direction="down")
done()   → Task completion signal
```

The configuration maps each directional tool to the same underlying `move` MCP tool:

```yaml
action_mappings:
  left:
    _parser:
      positional: []  # No arguments
    _tool: "move"     # MCP tool
    direction:
      static: "left"  # Static direction value
```

### Tool Format

All environments use the same XML format for tool calls:

```xml
<think>Reasoning about the task...</think>
<tool>tool_name(arguments)</tool>
```

## Dataset Format

HUD Gym uses HuggingFace datasets with hud.Task format:

```python
{
    "id": "task-001",
    "prompt": "Click on the submit button",
    "mcp_config": {...},  # MCP configuration as JSON string
    "setup_tool": {...},   # Setup tool call as JSON string
    "evaluate_tool": {...}, # Evaluation tool call as JSON string
    "metadata": {...}       # Additional metadata as JSON string
}
```

## Reward Functions

The base rubric system combines three core reward functions:

1. **Task Completion** (80%) - Primary reward from HUD evaluation tool
2. **Tool Execution** (10%) - Success rate of tool calls
3. **Format Compliance** (10%) - Proper XML format and action syntax

Note: The base rubric (`HUDBaseRubric`) contains only generic components for extensibility. Environment-specific behaviors (like screenshot requirements or thinking quality) can be added by extending the base rubric class.

## Custom Environments

Extend HUD Gym for custom tasks:

```python
from hud_vf_gym import HUDGym

class CustomGym(HUDGym):
    def setup_state(self, state, **kwargs):
        state = super().setup_state(state, **kwargs)
        # Add custom state tracking
        state["custom_metric"] = 0
        return state
    
    def env_response(self, messages, state, **kwargs):
        # Custom response logic
        return super().env_response(messages, state, **kwargs)
```

### Adding new HUD Environments

To create a new environment:

1. **Create a config file** (`configs/my_env.yaml`):
```yaml
system_prompt: |
  Instructions for the agent...
  
  You have access to these tools:
  - my_tool(arg): Description
    Usage: <tool>my_tool("value")</tool>

action_mappings:
  my_tool:
    _parser:
      positional: ["arg"]  # Expected arguments
    _tool: "mcp_tool_name"  # MCP tool to call
    param_name:
      from_arg: "arg"
      transform: "lambda x: x.upper()"  # Optional transform
```

2. **Use the config**:
```bash
vf-eval hud-vf-gym \
    --env-args '{"taskset": "your-org/your-taskset", "config_path": "configs/my_env.yaml"}' \
    --model gpt-4o-mini
```

### Adding New Tools to Existing Environments

1. **Update the system prompt** to describe the tool to agents

2. **Add action mapping**:
```yaml
action_mappings:
  new_tool:
    _parser:
      positional: ["arg1", "arg2"]  # Define positional arguments
    _tool: "target_mcp_tool"  # Required: which MCP tool to call
    # Map arguments with optional transforms
    param1:
      from_arg: "arg1"
    param2:
      from_arg: "arg2"
      transform: "lambda x: x * 2"
```

### Creating Datasets

Convert tasks to HuggingFace format:

```python
from datasets import Dataset
import json

# Load your tasks
tasks = [...]

# Convert to HF format
dataset_dict = {
    "id": [t["id"] for t in tasks],
    "prompt": [t["prompt"] for t in tasks],
    "mcp_config": [json.dumps(t["mcp_config"]) for t in tasks],
    "setup_tool": [json.dumps(t["setup_tool"]) for t in tasks],
    "evaluate_tool": [json.dumps(t["evaluate_tool"]) for t in tasks],
    "metadata": [json.dumps(t.get("metadata", {})) for t in tasks],
}

dataset = Dataset.from_dict(dataset_dict)
dataset.push_to_hub("your-org/your-dataset")
```

## Troubleshooting

### Common Issues

1. **"Unknown tool" errors**: Ensure action mappings are correctly configured
2. **XML parsing failures**: Check that agents use proper `<tool>` and `<think>` tags
3. **MCP connection issues**: Verify MCP configuration in dataset
4. **Low rewards**: Review rubric weights and ensure evaluation tool returns grades

## License

See LICENSE file in the hud-vf-gym directory.
