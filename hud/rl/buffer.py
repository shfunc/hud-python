"""Replay buffer for storing and sampling episodes."""

import logging
import random
from typing import List
from collections import deque

from .types import Episode

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Simple replay buffer for episodes."""
    
    def __init__(self, max_size: int = 1000, success_buffer_size: int = 64):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.success_buffer = deque(maxlen=success_buffer_size)
    
    def add(self, episodes: list[Episode]):
        """Add episodes to buffer."""
        for ep in episodes:
            self.buffer.append(ep)
            
            # Keep successful episodes in separate buffer
            if ep.success:
                self.success_buffer.append(ep)
    
    def sample(self, batch_size: int) -> list[Episode]:
        """Sample a batch of episodes."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        return random.sample(self.buffer, batch_size)
    
    def sample_success(self) -> Episode | None:
        """Sample a successful episode if available."""
        if self.success_buffer:
            return random.choice(self.success_buffer)
        return None
    
    def get_latest(self, n: int) -> list[Episode]:
        """Get the n most recent episodes."""
        return list(self.buffer)[-n:]
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.success_buffer.clear()
    
    @property
    def size(self) -> int:
        return len(self.buffer)
    
    @property
    def num_successes(self) -> int:
        return len(self.success_buffer)
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        if not self.buffer:
            return {
                "size": 0,
                "num_successes": 0,
                "avg_reward": 0.0,
                "avg_steps": 0.0,
            }
        
        rewards = [ep.terminal_reward for ep in self.buffer]
        steps = [ep.num_steps for ep in self.buffer]
        
        return {
            "size": len(self.buffer),
            "num_successes": self.num_successes,
            "avg_reward": sum(rewards) / len(rewards),
            "avg_steps": sum(steps) / len(steps),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
        }


class GroupedReplayBuffer(ReplayBuffer):
    """Replay buffer with episode grouping for variance reduction."""
    
    def __init__(self, max_size: int = 1000, success_buffer_size: int = 64, group_size: int = 6):
        super().__init__(max_size, success_buffer_size)
        self.group_size = group_size
        self.groups = {}
    
    def add(self, episodes: list[Episode]):
        """Add episodes and group them."""
        super().add(episodes)
        
        # Group episodes by task characteristics
        for ep in episodes:
            key = self._get_group_key(ep)
            if key not in self.groups:
                self.groups[key] = []
            self.groups[key].append(ep)
            
            # Keep groups bounded
            if len(self.groups[key]) > self.group_size * 10:
                self.groups[key] = self.groups[key][-self.group_size * 10:]
    
    def _get_group_key(self, episode: Episode) -> str:
        """Get grouping key for an episode."""
        # Group by task_id first (directly from episode)
        if episode.task_id and episode.task_id != "unknown":
            return episode.task_id
            
        # Then check metadata if available
        if "id" in episode.metadata:
            return episode.metadata["id"]
        if "task_id" in episode.metadata:
            return episode.metadata["task_id"]
        if "task_type" in episode.metadata:
            return episode.metadata["task_type"]
        if "difficulty" in episode.metadata:
            return f"difficulty_{episode.metadata['difficulty']}"
        
        # Default: single group
        return "all"
    
    def sample_groups(self) -> dict:
        """Return all complete groups available for GRPO training."""
        result = {}
        group_counter = 0
        
        for key, episodes in self.groups.items():
            # Return all complete groups as they are
            for i in range(0, len(episodes), self.group_size):
                if i + self.group_size <= len(episodes):
                    # Take exactly group_size episodes
                    group = episodes[i:i + self.group_size]
                    result[f"group_{group_counter}"] = group
                    group_counter += 1
        
        return result
