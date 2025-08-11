"""Evaluator for move efficiency."""

import math
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate


@evaluate.tool(name="efficiency", description="Evaluate game efficiency based on score/moves ratio")
async def evaluate_efficiency(ctx: Context, min_ratio: float):
    game = evaluate.env
    state = game.get_state()
    score = state.get("score", 0)
    moves = state.get("moves", 0)

    # Calculate ratio with logarithmic score
    # This gives diminishing returns for higher scores
    log_score = math.log2(score + 1) if score > 0 else 0
    log_ratio = log_score / moves if moves > 0 else 0.0

    # Also calculate the target ratio in log space
    # Assuming min_ratio is in linear space, we need to convert it
    # For consistency, we'll use the actual ratio but with log-based normalization
    ratio = score / moves if moves > 0 else 0.0

    # Logarithmic reward based on ratio
    # Use log to compress the scale
    log_ratio_actual = math.log2(ratio + 1) if ratio > 0 else 0
    log_min_ratio = math.log2(min_ratio + 1) if min_ratio > 0 else 0

    reward = min(1.0, log_ratio_actual / log_min_ratio) if log_min_ratio > 0 else 0.0
    done = ratio >= min_ratio

    return EvaluationResult(
        reward=reward,
        done=done,
        content=f"Efficiency: {ratio:.2f} (target: {min_ratio})",
        info={"score": score, "moves": moves, "ratio": ratio},
    )
