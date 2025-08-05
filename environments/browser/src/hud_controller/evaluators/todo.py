"""Todo app evaluators."""

import logging
from typing import Dict, Any, List
from .registry import evaluator

logger = logging.getLogger(__name__)


@evaluator(
    "todo_completed", app="todo", description="Check if expected number of todos are completed"
)
class TodoCompletedEvaluator:
    """Evaluator that checks if the expected number of todos are completed."""

    async def __call__(self, context, expected_count: int) -> Dict[str, Any]:
        """Check if at least expected_count todos are completed."""
        try:
            # Use the app-centric approach: call todo backend API
            stats = await context.call_app_api("todo", "/api/eval/stats")
            completed_count = stats.get("completed_count", 0)

            success = completed_count >= expected_count
            reward = (
                1.0
                if success
                else (completed_count / expected_count if expected_count > 0 else 0.0)
            )

            return {
                "reward": reward,
                "done": True,
                "info": {
                    "success": success,
                    "completed_count": completed_count,
                    "expected_count": expected_count,
                    "message": f"Found {completed_count} completed todos (expected {expected_count})",
                    "evaluator": "todo_completed",
                },
            }
        except Exception as e:
            logger.error(f"TodoCompletedEvaluator failed: {e}")
            return {
                "reward": 0.0,
                "done": True,
                "info": {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to evaluate completed todos",
                    "evaluator": "todo_completed",
                },
            }


@evaluator("todo_exists", app="todo", description="Check if todos containing specific text exist")
class TodoExistsEvaluator:
    """Evaluator that checks if todos containing specific text exist."""

    async def __call__(self, context, text: str) -> Dict[str, Any]:
        """Check if any todo contains the specified text."""
        try:
            # Use the app-centric approach: call todo backend API
            exists_response = await context.call_app_api("todo", f"/api/eval/has_todo?text={text}")
            exists = exists_response.get("exists", False)

            return {
                "reward": 1.0 if exists else 0.0,
                "done": True,
                "info": {
                    "success": exists,
                    "search_text": text,
                    "found": exists,
                    "message": f"Todo containing '{text}' {'found' if exists else 'not found'}",
                    "evaluator": "todo_exists",
                },
            }
        except Exception as e:
            logger.error(f"TodoExistsEvaluator failed: {e}")
            return {
                "reward": 0.0,
                "done": True,
                "info": {
                    "success": False,
                    "error": str(e),
                    "search_text": text,
                    "message": f"Failed to check for todo containing '{text}'",
                    "evaluator": "todo_exists",
                },
            }


@evaluator(
    "todo_completion_rate",
    app="todo",
    description="Check if todo completion rate meets minimum threshold",
)
class TodoCompletionRateEvaluator:
    """Evaluator that checks if todo completion rate meets a minimum threshold."""

    async def __call__(self, context, min_rate: float) -> Dict[str, Any]:
        """Check if completion rate meets minimum threshold."""
        try:
            # Use the app-centric approach: call todo backend API
            rate_response = await context.call_app_api("todo", "/api/eval/completion_rate")
            completion_rate = rate_response.get("completion_rate", 0.0)

            success = completion_rate >= min_rate
            # Proportional reward based on how close we are to the target
            reward = min(completion_rate / min_rate, 1.0) if min_rate > 0 else 1.0

            return {
                "reward": reward,
                "done": True,
                "info": {
                    "success": success,
                    "completion_rate": completion_rate,
                    "min_rate": min_rate,
                    "message": f"Completion rate: {completion_rate:.1%} (min: {min_rate:.1%})",
                    "evaluator": "todo_completion_rate",
                },
            }
        except Exception as e:
            logger.error(f"TodoCompletionRateEvaluator failed: {e}")
            return {
                "reward": 0.0,
                "done": True,
                "info": {
                    "success": False,
                    "error": str(e),
                    "min_rate": min_rate,
                    "message": f"Failed to evaluate completion rate",
                    "evaluator": "todo_completion_rate",
                },
            }


# === ADVANCED EVALUATORS ===


@evaluator("todo_db_direct", app="todo", description="Direct database access evaluator")
class TodoDbDirectEvaluator:
    """Evaluator that demonstrates direct database/API access for specific items."""

    async def __call__(self, context, todo_id: str, expected_completed: bool) -> Dict[str, Any]:
        """Check specific todo item state via direct API call."""
        try:
            # Direct API call to specific item
            item_data = await context.call_app_api("todo", f"/api/items/{todo_id}")
            actual_completed = item_data.get("completed", False)

            success = actual_completed == expected_completed

            return {
                "reward": 1.0 if success else 0.0,
                "done": True,
                "info": {
                    "success": success,
                    "todo_id": todo_id,
                    "expected_completed": expected_completed,
                    "actual_completed": actual_completed,
                    "item_data": item_data,
                    "message": f"Todo {todo_id} completion state: {actual_completed} (expected: {expected_completed})",
                    "evaluator": "todo_db_direct",
                },
            }
        except Exception as e:
            logger.error(f"TodoDbDirectEvaluator failed: {e}")
            return {
                "reward": 0.0,
                "done": True,
                "info": {
                    "success": False,
                    "error": str(e),
                    "todo_id": todo_id,
                    "expected_completed": expected_completed,
                    "message": f"Failed to access todo {todo_id}",
                    "evaluator": "todo_db_direct",
                },
            }


@evaluator(
    "composite_evaluate",
    app="todo",
    description="Composite evaluator with weighted sub-evaluations",
)
class CompositeEvaluator:
    """Evaluator that combines multiple sub-evaluations with weights (like psyop-bench Grade.from_subscores)."""

    async def __call__(self, context, evaluators: List[Dict]) -> Dict[str, Any]:
        """Run multiple evaluators and combine their scores with weights.

        Args:
            evaluators: List of evaluator specs with weights
                [{"function": "todo_completed", "args": {...}, "weight": 0.6}, ...]
        """
        try:
            results = []
            total_weight = 0.0
            weighted_score = 0.0

            for eval_spec in evaluators:
                function_name = eval_spec.get("function")
                args = eval_spec.get("args", {})
                weight = eval_spec.get("weight", 1.0)

                # Use context to execute the sub-evaluation
                sub_result = await context.execute_evaluation(
                    {"function": function_name, "args": args}
                )

                sub_score = sub_result.get("reward", 0.0)
                weighted_score += sub_score * weight
                total_weight += weight

                results.append(
                    {
                        "function": function_name,
                        "args": args,
                        "weight": weight,
                        "score": sub_score,
                        "result": sub_result,
                    }
                )

            # Normalize by total weight
            final_score = weighted_score / total_weight if total_weight > 0 else 0.0
            overall_success = final_score >= 0.7  # 70% threshold

            return {
                "reward": final_score,
                "done": True,
                "info": {
                    "success": overall_success,
                    "final_score": final_score,
                    "total_weight": total_weight,
                    "weighted_score": weighted_score,
                    "sub_evaluations": results,
                    "threshold": 0.7,
                    "message": f"Composite evaluation score: {final_score:.2f}",
                    "evaluator": "composite_evaluate",
                },
            }
        except Exception as e:
            logger.error(f"CompositeEvaluator failed: {e}")
            return {
                "reward": 0.0,
                "done": True,
                "info": {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to run composite evaluation",
                    "evaluator": "composite_evaluate",
                },
            }
