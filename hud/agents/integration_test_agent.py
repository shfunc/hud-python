from hud.agents.base import MCPAgent, find_reward
from hud.types import Task, Trace
from typing import Any


class IntegrationTestRunner(MCPAgent):
    def __init__(self, **kwargs: Any) -> None:
        kwargs["auto_trace"] = False
        super().__init__(**kwargs)
        self.metadata = {}

    async def run(self, task: Task, max_steps: int = 10) -> Trace:  # noqa: ARG002
        try:
            # Initialize using base to set up client and telemetry correctly
            await self.initialize(task)

            # Validate task shape
            if not getattr(task, "integration_test_tool", None):
                raise ValueError("--integration-test requires task.integration_test_tool (single call)")
            
            if task.setup_tool:
                _ = await self.call_tools(task.setup_tool)

            _ = await self.call_tools(task.integration_test_tool)
            if task.evaluate_tool:
                evaluate_result = await self.call_tools(task.evaluate_tool)

            reward = float(find_reward(evaluate_result[0])) if evaluate_result else 0.0

            return Trace(done=True, reward=reward, info={})
        finally:
            # Ensure resources are cleaned up so the CLI can exit cleanly
            await self._cleanup()

    # Stub implementations to satisfy abstract base class; not used in --integration-test path
    async def get_system_messages(self) -> list[Any]:
        return []

    async def get_response(self, messages: list[Any]):  # noqa: ARG002
        raise NotImplementedError("IntegrationTestRunner does not implement agent loop")

    async def format_blocks(self, blocks: list[Any]) -> list[Any]:  # noqa: ARG002
        return []

    async def format_tool_results(self, tool_calls: list[Any], tool_results: list[Any]) -> list[Any]:  # noqa: ARG002
        return []
