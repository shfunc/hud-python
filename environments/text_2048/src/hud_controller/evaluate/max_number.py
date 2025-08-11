"""Evaluator for highest tile."""

import math
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate


@evaluate.tool("max_number")
async def evaluate_max_number(ctx: Context, target: int):
    game = evaluate.env
    highest_tile = game.get_state().get("highest_tile", 0)

    # Logarithmic reward scale
    # Since tiles are powers of 2, use log2
    tile_log = math.log2(highest_tile) if highest_tile > 1 else 0
    target_log = math.log2(target) if target > 1 else 0

    # Progress in log space gives more balanced rewards
    # e.g., 128->256 gives same reward increase as 1024->2048
    reward = min(1.0, tile_log / target_log) if target_log > 0 else 0.0
    done = highest_tile >= target

    return EvaluationResult(
        reward=reward,
        done=done,
        content=f"Target: {target}, Highest: {highest_tile}",
        info={"target": target, "highest_tile": highest_tile},
    )
