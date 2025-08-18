"""Evaluator for highest tile."""

import math
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate


@evaluate.tool("max_number")
async def evaluate_max_number(ctx: Context, target: int):
    game = evaluate.env
    highest_tile = game.get_state().get("highest_tile", 0)

    highest_tile = highest_tile - 1

    # Logarithmic reward scale
    # Reward is proportional to log(highest_tile) / log(target), capped at 1.0
    reward = min(1.0, math.log(highest_tile) / math.log(target)) if target > 0 else 0.0
    done = highest_tile >= target

    return EvaluationResult(
        reward=reward,
        done=done,
        content=f"Target: {target}, Highest: {highest_tile}",
        info={"target": target, "highest_tile": highest_tile},
    )
