"""MCP tools for setup and evaluation."""

import logging
from mcp.server.fastmcp import Context
from .services import ServiceManager
from .evaluators import EvaluatorRegistry, BrowserEnvironmentContext
from .setup import SetupRegistry
from .problems import ProblemRegistry

logger = logging.getLogger(__name__)


def get_environment_context(service_manager: ServiceManager):
    """Get the environment context with current service manager and tools."""
    playwright_tool = None
    try:
        from hud.tools import PlaywrightTool

        playwright_tool = PlaywrightTool()
    except Exception as e:
        logger.warning(f"Could not get PlaywrightTool for context: {e}")

    return BrowserEnvironmentContext(service_manager, playwright_tool)


async def setup_tool(
    function: str, args: dict, name: str, ctx: Context, service_manager: ServiceManager
) -> dict:
    """Setup the environment based on configuration.

    Args:
        function: Setup function name (e.g. 'todo_seed')
        args: Arguments for the setup function
        name: Problem name to lookup setup from problem registry
        ctx: FastMCP context
        service_manager: Service manager instance

    Returns:
        Setup result dictionary
    """
    function_name = function
    problem_name = name
    args = args or {}

    await ctx.info(f"Setup - function: {function_name}, problem: {problem_name}")

    # Problem registry lookup
    if problem_name and not function_name:
        try:
            problem_instance = ProblemRegistry.create_problem(problem_name)
            if hasattr(problem_instance, "get_setup"):
                setup_spec = problem_instance.get_setup()
                if setup_spec:
                    function_name = setup_spec.get("function")
                    args = setup_spec.get("args", {})
                else:
                    return {
                        "status": "success",
                        "message": f"Problem '{problem_name}' has no setup",
                    }
            else:
                return {"status": "success", "message": f"Problem '{problem_name}' has no setup"}
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to lookup problem '{problem_name}': {str(e)}",
            }

    if not function_name:
        return {
            "status": "error",
            "message": "No setup function specified (need 'function' or 'name')",
        }

    # Execute setup
    try:
        environment_context = get_environment_context(service_manager)
        try:
            setup_spec = {"function": function_name, "args": args}
            setup_func = SetupRegistry.create_setup(setup_spec, environment_context)
            result = await setup_func()
            await ctx.info(f"Setup completed: {result}")
            return result
        finally:
            await environment_context.close()
    except Exception as e:
        await ctx.error(f"Setup failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Setup error: {str(e)}",
            "function": function_name,
            "args": args,
        }


async def evaluate_tool(
    function: str, args: dict, name: str, ctx: Context, service_manager: ServiceManager
) -> dict:
    """Evaluate the environment based on configuration.

    Args:
        function: Evaluator function name (e.g. 'todo_completed')
        args: Arguments for the evaluator function
        name: Problem name to lookup evaluation from problem registry
        ctx: FastMCP context
        service_manager: Service manager instance

    Returns:
        Evaluation result dictionary with standardized reward format
    """
    function_name = function
    problem_name = name
    args = args or {}

    await ctx.info(f"Evaluation - function: {function_name}, problem: {problem_name}")

    # Problem registry lookup - get evaluation spec from class-based problem
    if problem_name and not function_name:
        try:
            problem_instance = ProblemRegistry.create_problem(problem_name)

            if hasattr(problem_instance, "get_evaluation"):
                eval_spec = problem_instance.get_evaluation()
                if eval_spec:
                    function_name = eval_spec.get("function")
                    args = eval_spec.get("args", {})
                else:
                    return {
                        "reward": 1.0,
                        "done": True,
                        "info": {
                            "success": True,
                            "message": f"Problem '{problem_name}' has no evaluation (setup-only)",
                            "problem": problem_name,
                        },
                    }
            else:
                return {
                    "reward": 1.0,
                    "done": True,
                    "info": {
                        "success": True,
                        "message": f"Problem '{problem_name}' has no evaluation method",
                        "problem": problem_name,
                    },
                }
        except Exception as e:
            return {
                "reward": 0.0,
                "done": True,
                "info": {
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to lookup problem '{problem_name}'",
                    "problem": problem_name,
                },
            }

    if not function_name:
        return {
            "reward": 0.0,
            "done": True,
            "info": {
                "success": False,
                "message": "No evaluation function specified (need 'function' or 'name')",
            },
        }

    # Execute direct evaluation
    try:
        environment_context = get_environment_context(service_manager)
        try:
            evaluator_spec = {"function": function_name, "args": args}
            evaluator = EvaluatorRegistry.create_evaluator(evaluator_spec, environment_context)
            result = await evaluator()
            await ctx.info(f"Evaluation completed")
            return result
        finally:
            await environment_context.close()
    except Exception as e:
        await ctx.error(f"Evaluation failed: {str(e)}")
        return {
            "reward": 0.0,
            "done": True,
            "info": {
                "success": False,
                "message": f"Evaluation error: {str(e)}",
                "function": function_name,
                "args": args,
            },
        }
