"""MCP tools for setup and evaluation in remote browser environment."""

import logging
from typing import Any
from mcp.server.fastmcp import Context
from hud_controller.providers import BrowserProvider
from .playwright_with_memory import PlaywrightToolWithMemory
from .evaluators import EvaluatorRegistry, RemoteBrowserContext
from .setup import SetupRegistry
from .problems import ProblemRegistry

logger = logging.getLogger(__name__)


def get_environment_context(browser_provider, playwright_tool):
    """Get the environment context with current browser provider and playwright tool.

    Args:
        browser_provider: The browser provider instance
        playwright_tool: The PlaywrightToolWithMemory instance with CDP connection

    Returns:
        RemoteBrowserContext instance
    """
    return RemoteBrowserContext(browser_provider, playwright_tool)


async def setup_tool(
    function: str | None = None,
    args: Any = None,
    name: str | None = None,
    ctx: Context | None = None,
    browser_provider: BrowserProvider | None = None,
    playwright_tool: PlaywrightToolWithMemory | None = None,
) -> dict:
    """Setup the remote browser environment based on configuration.

    Args:
        function: Setup function name
        args: Arguments for the setup function
        name: Problem name to lookup setup from problem registry
        ctx: FastMCP context
        browser_provider: Browser provider instance
        playwright_tool: PlaywrightToolWithMemory instance

    Returns:
        Setup result dictionary
    """
    function_name = function
    problem_name = name

    # Debug: log the type and value of args
    import json

    if ctx:
        await ctx.info(f"Debug - args type: {type(args)}, args value: {args}")

    # Handle case where args might be a JSON string
    if isinstance(args, str):
        try:
            args = json.loads(args)
            if ctx:
                await ctx.info(f"Parsed args from string: {args}")
        except json.JSONDecodeError:
            if ctx:
                await ctx.error(f"Failed to parse args as JSON: {args}")

    args = args or {}

    if ctx:
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
        environment_context = get_environment_context(browser_provider, playwright_tool)
        try:
            setup_spec = {"function": function_name, "args": args}
            setup_func = SetupRegistry.create_setup(setup_spec, environment_context)
            result = await setup_func()
            if ctx:
                await ctx.info(f"Setup completed: {result}")
            return result
        finally:
            await environment_context.close()
    except Exception as e:
        if ctx:
            await ctx.error(f"Setup failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Setup error: {str(e)}",
            "function": function_name,
            "args": args,
        }


async def evaluate_tool(
    function: str | None = None,
    args: Any = None,
    name: str | None = None,
    ctx: Context | None = None,
    browser_provider: BrowserProvider | None = None,
    playwright_tool: PlaywrightToolWithMemory | None = None,
) -> dict:
    """Evaluate the remote browser environment based on configuration.

    Args:
        function: Evaluator function name
        args: Arguments for the evaluator function
        name: Problem name to lookup evaluation from problem registry
        ctx: FastMCP context
        browser_provider: Browser provider instance
        playwright_tool: PlaywrightToolWithMemory instance

    Returns:
        Evaluation result dictionary with standardized reward format
    """
    function_name = function
    problem_name = name
    args = args or {}

    if ctx:
        await ctx.info(f"Evaluation - function: {function_name}, problem: {problem_name}")

    # Problem registry lookup
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
        environment_context = get_environment_context(browser_provider, playwright_tool)
        try:
            evaluator_spec = {"function": function_name, "args": args}
            evaluator = EvaluatorRegistry.create_evaluator(evaluator_spec, environment_context)
            result = await evaluator()
            if ctx:
                await ctx.info(f"Evaluation completed")
            return result
        finally:
            await environment_context.close()
    except Exception as e:
        if ctx:
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
