"""Shared types for RL training."""

from pydantic.dataclasses import dataclass
from pydantic import Field, ConfigDict
from typing import Any
from hud.types import Trace
import math

try:
    import torch
except ImportError:
    raise ImportError("uv tool install hud-python[rl] to use this module")

class TrainingSample(Trace):
    """A single training sample for GRPO."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Tokenized inputs to the model (model.forward(*inputs))
    # This includes the input tokens, logit mask, etc.
    inputs: dict[str, torch.Tensor] = Field(default_factory=dict)
    old_logprobs: torch.Tensor | None = Field(default=None)
    ref_logprobs: torch.Tensor | None = Field(default=None)

    # Weighted advantage of group calculation
    advantage: float = Field(default=0.0)

@dataclass
class Metric:
    """A tuple for metrics."""
    name: str = Field(default="")
    mean: float = Field(default=0.0)
    std: float = Field(default=0.0)
    values: list[float] = Field(default_factory=list)

    def update(self, value: float | int | torch.Tensor) -> None:
        """Update metric."""
        self.values.append(value.item() if isinstance(value, torch.Tensor) else value)
        mean_val = sum(self.values) / len(self.values)
        self.mean = mean_val.item() if isinstance(mean_val, torch.Tensor) else float(mean_val)
        variance = sum((x - self.mean) ** 2 for x in self.values) / len(self.values)
        variance_val = variance.item() if isinstance(variance, torch.Tensor) else float(variance)
        self.std = math.sqrt(variance_val)

@dataclass
class TrainingMetrics:
    """Metrics for GRPO training (per training step)."""
    grad_norm: Metric = Field(default=Metric())
    loss: Metric = Field(default=Metric())
    kl: Metric = Field(default=Metric())
    reward: Metric = Field(default=Metric())
    advantage: Metric = Field(default=Metric())
    policy_ratio: Metric = Field(default=Metric())
    tokens: Metric = Field(default=Metric())

    def update(self, metrics: dict[str, Any]) -> None:
        """Update metrics."""
        for key, value in metrics.items():
            if key in self.__dataclass_fields__:
                getattr(self, key).update(value)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        final_metrics = {}
        for key in self.__dataclass_fields__:
            final_metrics[f"{key}_mean"] = getattr(self, key).mean
            final_metrics[f"{key}_std"] = getattr(self, key).std
        return final_metrics
