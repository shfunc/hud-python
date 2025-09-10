"""Shared types for RL training."""

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except ImportError:
    raise ImportError("uv tool install hud-python[rl] to use this module")

@dataclass
class Episode:
    """A complete episode/trajectory."""
    task_id: str
    terminal_reward: float
    conversation_history: list[dict[str, Any]]  # Full OpenAI format conversation
    tool_spec: list[dict[str, Any]]            # Tool specification
    info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_steps(self) -> int:
        return len(self.conversation_history)
    
    @property
    def success(self) -> bool:
        return self.terminal_reward > 0



@dataclass
class TrainingSample:
    """A single training sample for GRPO."""
    inputs: dict[str, torch.Tensor]     # Tokenized prompt
    advantage: float                     # Weighted advantage
    old_logprobs: torch.Tensor | None          # Log probs under old policy
    ref_logprobs: torch.Tensor | None          # Log probs under reference policy
    weight: float = 1.0                  # Turn weight

    # def select(self, indices: list[int]) -> "TrainingSample":
    #     return TrainingSample(
    #         inputs={k: v[indices] for k, v in self.inputs.items()},
    #         advantage=self.advantage[indices],
    #         old_logprobs=self.old_logprobs[indices],
    #         ref_logprobs=self.ref_logprobs[indices],
    #         weight=self.weight[indices],
    #     )

@dataclass
class Batch:
    """A batch of training samples."""
    samples: list[TrainingSample]
    episodes: list[Episode]
    
    @property
    def size(self) -> int:
        return len(self.samples)
    
    @property
    def rewards(self) -> list[float]:
        return [ep.terminal_reward for ep in self.episodes]