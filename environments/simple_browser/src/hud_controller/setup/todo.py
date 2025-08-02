"""Todo app setup tools."""

import logging
from typing import Dict, Any, List
from .registry import setup

logger = logging.getLogger(__name__)


@setup("todo_seed", app="todo", description="Seed database with default test todos")
class TodoSeedSetup:
    """Setup tool that seeds the database with default test data."""

    async def __call__(self, context, num_items: int = 5) -> Dict[str, Any]:
        """Seed the database with test todo items.

        Args:
            context: BrowserEnvironmentContext
            num_items: Number of test items to create (default: 5)

        Returns:
            Setup result with seeded items info
        """
        try:
            # Call the app's seed API
            result = await context.call_app_api("todo", "/api/eval/seed", method="POST")

            return {
                "status": "success",
                "message": f"Seeded database with test data",
                "items_added": result.get("items_added", num_items),
                "setup": "todo_seed",
            }
        except Exception as e:
            logger.error(f"TodoSeedSetup failed: {e}")
            return {
                "status": "error",
                "message": f"Failed to seed database: {str(e)}",
                "setup": "todo_seed",
            }


@setup("todo_reset", app="todo", description="Reset database to empty state")
class TodoResetSetup:
    """Setup tool that resets the database to empty state."""

    async def __call__(self, context) -> Dict[str, Any]:
        """Reset the database to empty state.

        Args:
            context: BrowserEnvironmentContext

        Returns:
            Setup result with reset confirmation
        """
        try:
            # Call the app's reset API
            result = await context.call_app_api("todo", "/api/eval/reset", method="DELETE")

            return {
                "status": "success",
                "message": "Database reset to empty state",
                "setup": "todo_reset",
            }
        except Exception as e:
            logger.error(f"TodoResetSetup failed: {e}")
            return {
                "status": "error",
                "message": f"Failed to reset database: {str(e)}",
                "setup": "todo_reset",
            }


@setup("todo_custom_seed", app="todo", description="Seed database with custom todo items")
class TodoCustomSeedSetup:
    """Setup tool that seeds the database with custom todo items."""

    async def __call__(self, context, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Seed the database with custom todo items.

        Args:
            context: BrowserEnvironmentContext
            items: List of todo items to create, each with title, description, completed

        Returns:
            Setup result with created items info
        """
        try:
            # First reset to ensure clean state
            await context.call_app_api("todo", "/api/eval/reset", method="DELETE")

            # Add each custom item via the API
            created_items = []
            for item in items:
                create_result = await context.call_app_api(
                    "todo",
                    "/api/items",
                    method="POST",
                    data={
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "completed": item.get("completed", False),
                    },
                )
                created_items.append(create_result)

            return {
                "status": "success",
                "message": f"Created {len(created_items)} custom todo items",
                "items_created": len(created_items),
                "items": created_items,
                "setup": "todo_custom_seed",
            }
        except Exception as e:
            logger.error(f"TodoCustomSeedSetup failed: {e}")
            return {
                "status": "error",
                "message": f"Failed to create custom items: {str(e)}",
                "setup": "todo_custom_seed",
            }


@setup("todo_composite_setup", app="todo", description="Composite setup using multiple sub-setups")
class TodoCompositeSetup:
    """Setup tool that demonstrates composable setup using other setup tools."""

    async def __call__(
        self, context, preset: str = "basic", extra_items: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run composite setup with multiple phases.

        Args:
            context: BrowserEnvironmentContext
            preset: Preset type ('basic', 'full', 'custom')
            extra_items: Additional items to add after preset

        Returns:
            Setup result with composite operation details
        """
        try:
            results = []

            # Phase 1: Reset
            reset_result = await context.execute_setup({"function": "todo_reset", "args": {}})
            results.append({"phase": "reset", "result": reset_result})

            # Phase 2: Apply preset
            if preset == "basic":
                seed_result = await context.execute_setup(
                    {"function": "todo_seed", "args": {"num_items": 3}}
                )
                results.append({"phase": "basic_seed", "result": seed_result})
            elif preset == "full":
                seed_result = await context.execute_setup(
                    {"function": "todo_seed", "args": {"num_items": 5}}
                )
                results.append({"phase": "full_seed", "result": seed_result})

            # Phase 3: Add extra items if provided
            if extra_items:
                custom_result = await context.execute_setup(
                    {"function": "todo_custom_seed", "args": {"items": extra_items}}
                )
                results.append({"phase": "extra_items", "result": custom_result})

            return {
                "status": "success",
                "message": f"Composite setup completed with preset '{preset}'",
                "preset": preset,
                "phases_completed": len(results),
                "phase_results": results,
                "setup": "todo_composite_setup",
            }
        except Exception as e:
            logger.error(f"TodoCompositeSetup failed: {e}")
            return {
                "status": "error",
                "message": f"Composite setup failed: {str(e)}",
                "setup": "todo_composite_setup",
            }
