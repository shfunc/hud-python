"""Evaluator functions for 2048 environment."""

import logging
from hud.tools import BaseEvaluator, EvaluationResult
from . import evaluator

logger = logging.getLogger(__name__)


@evaluator("max_number", "Evaluates the highest tile achieved on the board")
class MaxNumberEvaluator(BaseEvaluator):
    """Evaluates the highest tile achieved on the board."""

    async def __call__(self, context, target: int | None = None, use_log: bool = True, **kwargs) -> EvaluationResult:
        """Evaluate based on the highest tile.

        Args:
            context: The game instance
            target: Target tile value (default 2048)
            use_log: Use logarithmic scaling for rewards (default True)
            **kwargs: Additional arguments

        Returns:
            Evaluation result with reward, done, and info
        """
        import math
        
        # Context is the game itself
        game_state = context.get_state()
        highest_tile = game_state.get("highest_tile", 0)

        if use_log and highest_tile > 2:
            # Logarithmic scaling - more balanced for exponential tile growth
            log_highest = math.log2(highest_tile)
            
            if target is None:
                # Normalize to ~1.0 for 2048 (log2(2048) = 11)
                reward = log_highest / 11.0
                done = True
            else:
                log_target = math.log2(max(target, 2))
                reward = min(1.0, log_highest / log_target)
                done = highest_tile >= target
        else:
            # Original linear scaling
            if target is None:
                reward = highest_tile
                done = True
            else:
                reward = min(1.0, highest_tile / target)
                done = highest_tile >= target

        return {
            "reward": float(reward),
            "done": bool(done),
            "info": {
                "success": True,
                "message": f"Target: {target}, Highest: {highest_tile}",
                "target": int(target) if target is not None else None,
                "highest_tile": int(highest_tile),
                "use_log": bool(use_log),
            },
        }


@evaluator("efficiency", "Evaluates move efficiency (score per move ratio)")
class EfficiencyEvaluator(BaseEvaluator):
    """Evaluates move efficiency based on score per move ratio."""

    async def __call__(self, context, min_ratio: float | None = None, **kwargs) -> EvaluationResult:
        """Evaluate based on score per move efficiency.

        Args:
            context: The game instance
            min_ratio: Minimum acceptable score/move ratio (default 10.0)
            **kwargs: Additional arguments

        Returns:
            Evaluation result with reward, done, and info
        """
        # Context is the game itself
        game_state = context.get_state()
        score = game_state.get("score", 0)
        moves = game_state.get("moves", 0)

        if moves == 0:
            ratio = 0.0
        else:
            ratio = score / moves

        if min_ratio is None:
            reward = ratio
            done = True
        else:
            reward = min(1.0, ratio / min_ratio) if min_ratio > 0 else 0.0
            done = ratio >= min_ratio

        return {
            "reward": float(reward),
            "done": bool(done),
            "info": {
                "success": True,
                "message": f"Efficiency: {ratio:.2f} (target: {min_ratio})",
                "score": int(score),
                "moves": int(moves),
                "ratio": float(ratio),
                "min_ratio": float(min_ratio) if min_ratio is not None else None,
            },
        }
