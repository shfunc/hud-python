"""2048 game evaluators."""

import logging
import math
import httpx
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("game_2048_max_number")
async def game_2048_max_number(target: int):
    """Check if player reached target tile value.

    Uses logarithmic reward scaling to match text-2048 implementation.

    Args:
        target: The target tile value to reach

    Returns:
        EvaluationResult with logarithmic reward
    """
    try:
        # Get the persistent context
        persistent_ctx = evaluate.env

        # Get the backend port and fetch game state
        try:
            backend_port = persistent_ctx.get_app_backend_port("2048")
        except:
            backend_port = 3001  # Fallback port

        url = f"http://localhost:{backend_port}/api/game/state"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            game_state = response.json()

        highest_tile = game_state.get("highest_tile", 0)
        initial_highest_tile = game_state.get("initial_highest_tile", 0)
        score = game_state.get("score", 0)

        # Only give reward if progress has been made
        # Score > 0 means merges have happened (real progress)
        # OR highest_tile > initial means we've created a higher tile
        if score == 0 and highest_tile <= initial_highest_tile:
            reward = 0.0
        elif target > 1 and highest_tile > 1:
            # Logarithmic reward scale
            reward = min(1.0, math.log(highest_tile) / math.log(target))
        else:
            reward = 0.0

        done = highest_tile >= target

        return EvaluationResult(
            reward=reward,
            done=done,
            content=f"Target: {target}, Highest: {highest_tile}",
            info={"target": target, "highest_tile": highest_tile, "progress": reward},
        )
    except Exception as e:
        logger.error(f"game_2048_max_number failed: {e}")
        return EvaluationResult(
            reward=0.0, done=False, content=f"Failed to evaluate max number: {str(e)}", isError=True
        )


@evaluate.tool("game_2048_efficiency")
async def game_2048_efficiency(min_ratio: float):
    """Evaluate game efficiency based on score/moves ratio.

    Matches text-2048 implementation.

    Args:
        min_ratio: The minimum efficiency ratio to achieve

    Returns:
        EvaluationResult with linear reward based on efficiency
    """
    try:
        # Get the persistent context
        persistent_ctx = evaluate.env

        # Get the backend port and fetch game state
        try:
            backend_port = persistent_ctx.get_app_backend_port("2048")
        except:
            backend_port = 3001  # Fallback port

        url = f"http://localhost:{backend_port}/api/game/state"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            game_state = response.json()
        score = game_state.get("score", 0)
        moves = game_state.get("moves", 0)

        # Calculate the efficiency ratio directly (no logarithmic scaling)
        ratio = score / moves if moves > 0 else 0.0

        # Linear reward: proportional to ratio / min_ratio, capped at 1.0
        reward = min(1.0, ratio / min_ratio) if min_ratio > 0 else 0.0
        done = ratio >= min_ratio

        return EvaluationResult(
            reward=reward,
            done=done,
            content=f"Efficiency: {ratio:.2f} (target: {min_ratio})",
            info={"score": score, "moves": moves, "ratio": ratio, "target_ratio": min_ratio},
        )
    except Exception as e:
        logger.error(f"game_2048_efficiency failed: {e}")
        return EvaluationResult(
            reward=0.0, done=False, content=f"Failed to evaluate efficiency: {str(e)}", isError=True
        )


@evaluate.tool("game_2048_score_reached")
async def game_2048_score_reached(target_score: int):
    """Check if player reached target score.

    Args:
        target_score: The target score to reach

    Returns:
        EvaluationResult
    """
    try:
        # Get the persistent context
        persistent_ctx = evaluate.env

        # Get the backend port and fetch game state
        try:
            backend_port = persistent_ctx.get_app_backend_port("2048")
        except:
            backend_port = 3001  # Fallback port

        url = f"http://localhost:{backend_port}/api/game/state"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            game_state = response.json()
        score = game_state.get("score", 0)

        # Linear reward based on score progress
        reward = min(1.0, score / target_score) if target_score > 0 else 0.0
        done = score >= target_score

        return EvaluationResult(
            reward=reward,
            done=done,
            content=f"Score: {score} (target: {target_score})",
            info={"score": score, "target_score": target_score, "progress": reward},
        )
    except Exception as e:
        logger.error(f"game_2048_score_reached failed: {e}")
        return EvaluationResult(
            reward=0.0, done=False, content=f"Failed to evaluate score: {str(e)}", isError=True
        )


@evaluate.tool("game_2048_game_won")
async def game_2048_game_won():
    """Check if game is won (reached target tile).

    Returns:
        EvaluationResult
    """
    try:
        # Get the persistent context
        persistent_ctx = evaluate.env

        # Get the backend port and fetch game state
        try:
            backend_port = persistent_ctx.get_app_backend_port("2048")
        except:
            backend_port = 3001  # Fallback port

        url = f"http://localhost:{backend_port}/api/game/state"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            game_state = response.json()
        won = game_state.get("won", False)
        highest_tile = game_state.get("highest_tile", 0)
        target_tile = game_state.get("target_tile", 2048)

        return EvaluationResult(
            reward=1.0 if won else 0.0,
            done=won,
            content=f"Game {'won' if won else 'in progress'} (highest: {highest_tile}, target: {target_tile})",
            info={"won": won, "highest_tile": highest_tile, "target_tile": target_tile},
        )
    except Exception as e:
        logger.error(f"game_2048_game_won failed: {e}")
        return EvaluationResult(
            reward=0.0, done=False, content=f"Failed to check win status: {str(e)}", isError=True
        )


@evaluate.tool("game_2048_game_over")
async def game_2048_game_over():
    """Check if game is over (no valid moves).

    Returns:
        EvaluationResult
    """
    try:
        # Get the persistent context
        persistent_ctx = evaluate.env

        # Get the backend port and fetch game state
        try:
            backend_port = persistent_ctx.get_app_backend_port("2048")
        except:
            backend_port = 3001  # Fallback port

        url = f"http://localhost:{backend_port}/api/game/state"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            game_state = response.json()
        game_over = game_state.get("game_over", False)
        score = game_state.get("score", 0)
        moves = game_state.get("moves", 0)

        return EvaluationResult(
            reward=0.0,  # Game over is not a positive outcome
            done=game_over,
            content=f"Game {'over' if game_over else 'active'} (score: {score}, moves: {moves})",
            info={"game_over": game_over, "score": score, "moves": moves},
        )
    except Exception as e:
        logger.error(f"game_2048_game_over failed: {e}")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content=f"Failed to check game over status: {str(e)}",
            isError=True,
        )


@evaluate.tool("game_2048_moves_made")
async def game_2048_moves_made(min_moves: int):
    """Check if minimum number of moves were made.

    Args:
        min_moves: The minimum number of moves to make

    Returns:
        EvaluationResult
    """
    try:
        # Get the persistent context
        persistent_ctx = evaluate.env

        # Get the backend port and fetch game state
        try:
            backend_port = persistent_ctx.get_app_backend_port("2048")
        except:
            backend_port = 3001  # Fallback port

        url = f"http://localhost:{backend_port}/api/game/state"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            game_state = response.json()
        moves = game_state.get("moves", 0)

        # Linear reward based on move progress
        reward = min(1.0, moves / min_moves) if min_moves > 0 else 1.0
        done = moves >= min_moves

        return EvaluationResult(
            reward=reward,
            done=done,
            content=f"Moves: {moves} (minimum: {min_moves})",
            info={"moves": moves, "min_moves": min_moves, "progress": reward},
        )
    except Exception as e:
        logger.error(f"game_2048_moves_made failed: {e}")
        return EvaluationResult(
            reward=0.0, done=False, content=f"Failed to evaluate moves: {str(e)}", isError=True
        )
