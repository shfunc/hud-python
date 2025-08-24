"""Todo app evaluators."""

import logging
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("todo_completed")
async def todo_completed(ctx: Context, expected_count: int):
    """Check if expected number of todos are completed.

    Args:
        expected_count: The expected number of completed todos

    Returns:
        Evaluation result
    """
    try:
        # Use the app-centric approach: call todo backend API
        env = evaluate.env  # Get BrowserEnvironmentContext from hub
        stats = await env.call_app_api("todo", "/api/eval/stats")
        completed_count = stats.get("completed_items", 0)
        total_count = stats.get("total_items", 0)

        success = completed_count >= expected_count
        reward = (
            1.0 if success else (completed_count / expected_count if expected_count > 0 else 0.0)
        )

        return EvaluationResult(
            reward=reward,
            done=success,  # Only done when target is reached
            info={
                "success": success,
                "completed_count": completed_count,
                "total_count": total_count,
                "expected_count": expected_count,
                "message": f"Found {completed_count} completed todos (expected {expected_count})",
            },
        )
    except Exception as e:
        logger.error(f"todo_completed failed: {e}")
        return EvaluationResult(
            reward=0.0,
            done=True,
            info={
                "success": False,
                "error": str(e),
                "message": "Failed to evaluate completed todos",
            },
        )


@evaluate.tool("todo_exists")
async def todo_exists(ctx: Context, title: str):
    """Check if a todo with specific title exists.

    Args:
        title: The title of the todo to check

    Returns:
        Evaluation result
    """
    try:
        # Call the app's API to get all todos
        env = evaluate.env  # Get BrowserEnvironmentContext from hub
        todos = await env.call_app_api("todo", "/api/eval/todos")

        # Check if any todo has the expected title
        exists = any(todo.get("title") == title for todo in todos)

        return EvaluationResult(
            reward=1.0 if exists else 0.0,
            done=exists,
            info={
                "success": exists,
                "title": title,
                "total_todos": len(todos),
                "message": f"Todo '{title}' {'exists' if exists else 'does not exist'}",
            },
        )
    except Exception as e:
        logger.error(f"todo_exists failed: {e}")
        return EvaluationResult(
            reward=0.0,
            done=True,
            info={
                "success": False,
                "error": str(e),
                "message": f"Failed to check if todo '{title}' exists",
            },
        )


@evaluate.tool("todo_completion_rate")
async def todo_completion_rate(ctx: Context, target_rate: float = 0.5):
    """Check if completion rate meets target.

    Args:
        target_rate: The target completion rate (default 0.5)

    Returns:
        Evaluation result
    """
    try:
        # Get stats from the app
        env = evaluate.env  # Get BrowserEnvironmentContext from hub
        stats = await env.call_app_api("todo", "/api/eval/stats")
        total_count = stats.get("total_items", 0)
        completed_count = stats.get("completed_items", 0)

        if total_count == 0:
            # No todos, consider this as 0% completion
            actual_rate = 0.0
        else:
            actual_rate = completed_count / total_count

        success = actual_rate >= target_rate
        reward = min(1.0, actual_rate / target_rate) if target_rate > 0 else 1.0

        return EvaluationResult(
            reward=reward,
            done=success,
            info={
                "success": success,
                "actual_rate": actual_rate,
                "target_rate": target_rate,
                "completed_count": completed_count,
                "total_count": total_count,
                "message": f"Completion rate: {actual_rate:.1%} (target: {target_rate:.1%})",
            },
        )
    except Exception as e:
        logger.error(f"todo_completion_rate failed: {e}")
        return EvaluationResult(
            reward=0.0,
            done=True,
            info={
                "success": False,
                "error": str(e),
                "message": "Failed to evaluate completion rate",
            },
        )


@evaluate.tool("todo_total_count")
async def todo_total_count(ctx: Context, min_count: int = 1):
    """Check if total todo count meets minimum.

    Args:
        min_count: The minimum number of todos expected (default 1)

    Returns:
        Evaluation result
    """
    try:
        # Get stats from the app
        env = evaluate.env  # Get BrowserEnvironmentContext from hub
        stats = await env.call_app_api("todo", "/api/eval/stats")
        total_count = stats.get("total_items", 0)

        success = total_count >= min_count
        reward = 1.0 if success else (total_count / min_count if min_count > 0 else 0.0)

        return EvaluationResult(
            reward=reward,
            done=success,
            info={
                "success": success,
                "total_count": total_count,
                "min_count": min_count,
                "message": f"Found {total_count} todos (minimum: {min_count})",
            },
        )
    except Exception as e:
        logger.error(f"todo_total_count failed: {e}")
        return EvaluationResult(
            reward=0.0,
            done=True,
            info={
                "success": False,
                "error": str(e),
                "message": "Failed to evaluate total count",
            },
        )


@evaluate.tool("todo_all_completed")
async def todo_all_completed(ctx: Context):
    """Check if all todos are completed.

    Returns:
        Evaluation result
    """
    try:
        # Get stats from the app
        env = evaluate.env  # Get BrowserEnvironmentContext from hub
        stats = await env.call_app_api("todo", "/api/eval/stats")
        total_count = stats.get("total_items", 0)
        completed_count = stats.get("completed_items", 0)

        if total_count == 0:
            # No todos, consider this as success
            success = True
            message = "No todos exist"
        else:
            success = completed_count == total_count
            message = f"{completed_count}/{total_count} todos completed"

        return EvaluationResult(
            reward=1.0 if success else (completed_count / total_count if total_count > 0 else 0.0),
            done=success,
            info={
                "success": success,
                "completed_count": completed_count,
                "total_count": total_count,
                "message": message,
            },
        )
    except Exception as e:
        logger.error(f"todo_all_completed failed: {e}")
        return EvaluationResult(
            reward=0.0,
            done=True,
            info={
                "success": False,
                "error": str(e),
                "message": "Failed to evaluate if all todos are completed",
            },
        )
