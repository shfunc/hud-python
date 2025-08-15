"""End-to-end RL training script using ART framework

This script demonstrates training an agent on the text-2048 environment using:
- ArtHUDAgent for agent implementation
- Dockerized text-2048 environment
- Tasks loaded from HuggingFace dataset (hud-evals/2048-taskset)
- Environment rewards

Run with:
    uv run python examples/train_art_rl_agent.py
"""
from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import art
import dotenv
import weave
from art.local import LocalBackend
from art.utils import iterate_dataset

from hud.agents import ArtHUDAgent
from hud.client import MCPClient
from hud.datasets import TaskConfig

if TYPE_CHECKING:
    from collections.abc import Awaitable

dotenv.load_dotenv()

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MODEL_NAME = os.environ.get("MODEL_NAME", "mcprl-1.5b-2048")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "2048-mcp-rl")

MAX_STEPS = int(os.environ.get("MAX_STEPS", "10"))

TRAINING_CONFIG = {
    "num_training_inputs": int(os.environ.get("NUM_INPUTS", "8")),
    "groups_per_step": int(os.environ.get("GROUPS_PER_STEP", "1")),
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "1")),
    "rollouts_per_group": int(os.environ.get("ROLLOUTS_PER_GROUP", "2")),
    "learning_rate": float(os.environ.get("LEARNING_RATE", "1e-5")),
}

@weave.op()
async def rollout(model: art.Model, task_dict: dict) -> art.Trajectory:
    """Generate one trajectory via ArtHUDAgent."""
    import hud
    
    with hud.trace(f"Art Rollout {task_dict['id']}", root=True, task_id=task_dict.get("id")):
        task_config = TaskConfig(**task_dict)
        
        mcp_client = MCPClient(task_config.mcp_config)
        agent = ArtHUDAgent(model, mcp_client, allowed_tools=["move"])
        await agent.initialize()

        trace = await agent.run(task_config, max_steps=MAX_STEPS)

        traj = art.Trajectory(
            messages_and_choices=agent.messages_and_choices,
            tools=agent.get_tool_schemas(),
            reward=trace.reward,
            metadata={"task": task_dict},
            metrics={
                "task_completed": trace.done,
                "success": trace.reward > 0,
                "reward": trace.reward,
                "ran_out_of_steps": not trace.done,
            },
        )

        await mcp_client.close()
    
        return traj.finish()


async def main() -> None:
    random.seed(42)

    # ---- Optional W&B ----
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
    if WANDB_API_KEY:
        weave.init(PROJECT_NAME)
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY

    # ---- Build / register model ----
    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=BASE_MODEL,
    )

    backend = LocalBackend(in_process=True, path="./.art")
    await model.register(backend)

    print("Model created and registered")

    # ---- Generate training scenarios  --------------------------------------------------
    from datasets import load_dataset

    dataset = load_dataset("hud-evals/2048-taskset", split="train")
    dataset = list(dataset.shuffle(seed=42))[:TRAINING_CONFIG["num_training_inputs"]]

    train_iterator = iterate_dataset(
        dataset,
        groups_per_step=TRAINING_CONFIG["groups_per_step"],
        num_epochs=TRAINING_CONFIG["num_epochs"],
        initial_step=await model.get_step(),
    )

    # ---- Training loop ----
    for batch in train_iterator:
        print(f"\n=== Training step {batch.step} ===")

        groups: list[Awaitable[art.TrajectoryGroup]] = []
        for task_dict in batch.items:
            group = art.TrajectoryGroup(
                rollout(model, task_dict)
                for _ in range(TRAINING_CONFIG["rollouts_per_group"])
            )
            groups.append(group)

        print("Gathering trajectory groups…")
        gathered = await art.gather_trajectory_groups(groups)

        # We already have rewards from environment evaluation
        await model.train(
            gathered,
            config=art.TrainConfig(learning_rate=TRAINING_CONFIG["learning_rate"]),
        )
        print("✅ step complete, model checkpoint saved\n")

    print("Training finished, checkpoints stored in ./.art/")


if __name__ == "__main__":
    asyncio.run(main())
