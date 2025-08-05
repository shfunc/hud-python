from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field

from hud.types import CustomGym, Gym, MetadataKeys, SensitiveData
from hud.utils.common import FunctionConfigs
from hud.utils.deprecation import deprecated

if TYPE_CHECKING:
    from hud.agent import Agent


@deprecated(
    reason="Task class is being replaced by TaskConfig for better MCP integration",
    replacement="hud.datasets.TaskConfig",
    version="0.3.0",
    removal_version="0.4.0",
)
class Task(BaseModel):
    """A task that can be executed and evaluated.

    A Task represents a specific activity to be performed in an environment.
    It contains the prompt describing the task and configurations for
    setting up and evaluating the environment.

    The setup and evaluate configurations can be in several formats:
    - String (function name): "chrome.maximize"
    - Tuple (function with args): ("chrome.activate_tab", 5)
    - Dict: {"function": "chrome.navigate", "args": ["https://example.com"]}
    - List of the above: ["chrome.maximize", {"function": "chrome.navigate", "args": ["https://example.com"]}]

    Attributes:
        id: The remote task ID (optional if local-only)
        prompt: The task prompt or instruction
        system_prompt: The system prompt for the evalset (optional)
        setup: Environment setup configuration (optional)
        evaluate: Configuration for evaluating responses
        metadata: Additional task metadata
        sensitive_data: Sensitive data such as API keys, passwords, etc.
        choices: Multiple choice answer list (for Inspect compatibility)
        target: Ideal target output (for Inspect compatibility)
        files: Files that go along with the task (for Inspect compatibility)
        gym: Environment specification
    """

    id: str | None = None  # Remote task ID (optional if local-only)

    prompt: str  # Task prompt or instruction
    system_prompt: str | None = None  # System prompt for the evalset (optional)

    gym: Gym | None = None  # Environment specification

    # Setup and evaluate configurations for the environment (environment specific)
    setup: FunctionConfigs | None = None
    evaluate: FunctionConfigs | None = None

    # Overflow configuration for environments that don't conform to the standard
    config: dict[str, Any] | None = None

    # Sensitive data such as API keys, passwords, etc.
    sensitive_data: SensitiveData = Field(default_factory=dict)

    # Metadata for the task evaluation, information about the agent (see MetadataKeys)
    metadata: dict[MetadataKeys, Any] = Field(default_factory=dict)

    # Description of the task, for extra information about its purpose and context
    description: str | None = None

    # Gold file url for the task
    gold_file_url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        return cls(**data)

    @classmethod
    def from_serialized(cls, data: dict[str, Any]) -> Task:
        gym_data = data.get("gym")
        parsed_gym: Gym | None = gym_data

        parsed_setup = [(param, entry) for param, entry in data.get("setup", [])]
        parsed_evaluate = [(param, entry) for param, entry in data.get("evaluate", [])]

        # Convert dict gym data to CustomGym if needed
        if (
            isinstance(gym_data, dict)
            and gym_data.get("type") == "public"
            and gym_data.get("location") in ("local", "remote")
            and gym_data.get("image_or_build_context") is not None
        ):
            parsed_gym = CustomGym(
                type=cast("Literal['public']", gym_data["type"]),
                location=cast("Literal['local', 'remote']", gym_data["location"]),
                image_or_build_context=Path(gym_data["image_or_build_context"]),
            )

        return cls(
            id=data.get("id"),
            prompt=data.get("prompt", ""),
            system_prompt=data.get("system_prompt"),
            setup=parsed_setup,
            evaluate=parsed_evaluate,
            gym=parsed_gym,
            config=data.get("config"),
            description=data.get("description"),
            sensitive_data=data.get("sensitive_data", {}),
            metadata=data.get("metadata", {}),
            gold_file_url=data.get("gold_file_url"),
        )

    async def fit(self, agent: Agent | type[Agent]) -> None:
        if isinstance(agent, type):
            agent = agent()

        if self.gym is None:
            return
        self.gym = agent.transfer_gyms.get(self.gym, self.gym)

    def serialize(self) -> dict[str, Any]:
        if isinstance(self.setup, list):
            parsed_setup = [[param, entry] for param, entry in self.setup]
        else:
            parsed_setup = self.setup
        if isinstance(self.evaluate, list):
            parsed_evaluate = [[param, entry] for param, entry in self.evaluate]
        else:
            parsed_evaluate = self.evaluate

        if isinstance(self.gym, CustomGym):
            parsed_gym = self.gym.model_dump()
            parsed_gym["image_or_build_context"] = str(parsed_gym["image_or_build_context"])
        else:  # is ServerGym
            parsed_gym = self.gym

        return {
            "id": self.id,
            "prompt": self.prompt,
            "config": self.config,
            "description": self.description,
            "setup": parsed_setup,
            "evaluate": parsed_evaluate,
            "gym": parsed_gym,
            "sensitive_data": self.sensitive_data,
            "metadata": self.metadata,
            "gold_file_url": self.gold_file_url,
        }
