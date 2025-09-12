"""Replay buffer for storing and sampling episodes."""

import logging
import random
from typing import Callable, TypeVar, Generic, Any
from collections import deque
from hud.utils.design import HUDDesign
from hud.types import Task, Trace
from hud.rl.config import Config

logger = logging.getLogger(__name__)
design = HUDDesign(logger=logger)

T = TypeVar("T")


class Buffer(Generic[T]):
    """Simple buffer for a list of tasks, traces or episodes."""
    
    def __init__(self, max_size: int = 10000) -> None:
        self.max_size = max_size
        self.buffer: deque[T] = deque(maxlen=max_size)
    
    def add(self, items: list[T] | T, shuffle: bool = False) -> None:
        """Add items to buffer."""
        if isinstance(items, list):
            for item in items:
                self.buffer.append(item)
        else:
            self.buffer.append(items)
        if shuffle:
            random.shuffle(self.buffer)

    def add_fill(self, items: list[T] | T, target_size: int, shuffle: bool = False) -> None:
        """Add items to buffer until the buffer is at least the target size."""
        while len(self.buffer) < target_size:
            self.add(items, shuffle)

    def get(self, n: int = 0) -> list[T]:
        """Get items from the buffer."""
        if n == 0:
            return list(self.buffer)
        if n > len(self.buffer):
            raise ValueError("Not enough items in buffer")
        return list(self.buffer)[:n]

    def consume(self, n: int = 0) -> list[T]:
        """Consume items from the buffer."""
        if n == 0:
            return list(self.buffer)
        if n > len(self.buffer):
            raise ValueError("Not enough items in buffer")

        return [self.buffer.pop() for _ in range(n)]

    def get_filtered(self, n: int = 0, filter_fn: Callable[[T], bool] | None = None, consume: bool = False) -> list[T]:
        """Filter the buffer by a filter function."""
        filtered = [item for item in self.buffer if filter_fn(item)] if filter_fn else list(self.buffer)
        if n == 0:
            return filtered
        return self.consume(n) if consume else self.get(n)
    
    def sample(
        self, batch_size: int, n: int = 0, filter_fn: Callable[[T], bool] | None = None, consume: bool = False
    ) -> list[T]:
        """Sample a batch of items with optional filtering."""
        items = self.get_filtered(n, filter_fn, consume)
        
        if len(items) < batch_size:
            design.warning(f"Buffer has {len(items)} items, requested {batch_size}")
            return items
        
        return random.sample(items, batch_size)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Use len() directly on Buffer instances."""
        return len(self.buffer)


class DatasetBuffer(Buffer[Task]):
    """
    Buffer for a dataset.
    Loads in individual tasks that will be trained for a specified number of training steps.
    """
    
    def __init__(
        self,
        dataset: list[Task] | Task,
        config: Config,
    ) -> None:
        self.config = config

        self.group_size = config.training.group_size
        self.batch_size = config.training.batch_size
        self.training_steps = config.training.training_steps

        if self.group_size > self.batch_size:
            raise ValueError(f"Group size is greater than batch size, {self.group_size} > {self.batch_size}")

        if self.batch_size % self.group_size != 0:
            raise ValueError(f"A batch cannot have irregular groups, {self.group_size} % {self.batch_size} != 0")

        if self.group_size % config.training.mini_batch_size != 0:
            raise ValueError(f"Group size is not a multiple of mini batch size, {self.group_size} % {config.training.mini_batch_size} != 0")

        self.groups_per_batch = self.batch_size // self.group_size
        self.number_of_tasks = self.training_steps * self.groups_per_batch

        super().__init__(self.number_of_tasks)

        tasks = self._validate_tasks(dataset)
        if config.training.shuffle_dataset:
            random.shuffle(tasks)
        if len(tasks) > self.number_of_tasks:
            leftovers = len(tasks) - self.number_of_tasks
            design.warning(f"Training steps ({self.training_steps}) will lead to {leftovers} tasks not being trained") # noqa: E501
            tasks = tasks[:self.number_of_tasks]
        
        # Check if the dataset is imbalanced
        self.dataset_size = len(tasks)
        if self.training_steps % self.dataset_size != 0:
            leftovers = self.number_of_tasks % self.dataset_size
            design.warning(f"Dataset imbalanced ({leftovers} tasks will be trained 1 more time)")
            design.warning(f"This is because the number of training steps ({self.training_steps}) is not a multiple of the dataset size ({self.dataset_size})")
        
        self.add_fill(tasks, self.number_of_tasks, config.training.shuffle_dataset)

    
    def _validate_tasks(self, tasks: list[Task]) -> list[Task]:
        """Validate that all tasks are proper HUD Task objects."""
        if not tasks:
            raise ValueError("No tasks provided to DatasetBuffer")
        
        validated_tasks = []
        for i, task in enumerate(tasks):
            if not isinstance(task, Task):
                raise TypeError(f"Task at index {i} is not a HUD Task object, got {type(task)}")
            validated_tasks.append(task)
        
        return validated_tasks

    @property
    def info(self) -> dict[str, int]:
        """Get the info of the buffer."""
        return {
            "total_items": len(self),
            "total_traces": self.number_of_tasks * self.group_size,
            "total_batches": self.training_steps,
            "task_repeats": self.number_of_tasks // self.dataset_size,
            "dataset_size": self.dataset_size,
            "group_size": self.group_size,
            "batch_size": self.batch_size,
        }

    def get_tasks(self, consume: bool = True) -> list[Task]:
        """Get tasks for a batch."""
        tasks = self.consume(self.groups_per_batch) if consume else self.get(self.groups_per_batch)
        return tasks * self.group_size


class ReplayBuffer(Buffer[Trace]):
    """Buffer for traces."""
    
    def __init__(self, config: Config) -> None:
        self.config = config

        self.buffer_steps = config.training.buffer_steps
        self.select_strategy = config.training.select_strategy
        self.group_size = config.training.group_size
        self.batch_size = config.training.batch_size

        buffer_size = self.buffer_steps * self.batch_size
        
        super().__init__(buffer_size)

    def sample_traces(self) -> list[Trace]:
        """Sample traces for a batch."""
        if self.select_strategy == "recent":
            return self.get(self.batch_size)
        elif self.select_strategy == "random":
            return self.sample(self.batch_size)
        elif self.select_strategy == "variance":
            return self._sample_high_variance_traces()
        else:
            raise ValueError(f"Invalid select strategy: {self.select_strategy}")
    
    def _sample_high_variance_traces(self) -> list[Trace]:
        """Sample traces from tasks with high reward variance."""
        from collections import defaultdict
        
        # Step 1: Get recent traces (buffer_steps determines how many recent batches to consider)
        recent_traces = self.get(self.batch_size)
        
        # Step 2: Get unique task IDs from recent traces
        recent_task_ids = set()
        for trace in recent_traces:
            if hasattr(trace, "task") and hasattr(trace.task, "id"):
                recent_task_ids.add(trace.task.id)
        
        if not recent_task_ids:
            return self.get(self.batch_size)
        
        # Step 3: Group ALL traces by task ID (only for tasks seen recently)
        task_groups: dict[str, list[Trace]] = defaultdict(list)
        for trace in self.buffer:
            if hasattr(trace, "task") and hasattr(trace.task, "id") and trace.task.id in recent_task_ids:
                task_groups[trace.task.id].append(trace)
        
        # Step 4: For each task, find the subset with highest variance
        all_selected_traces: list[Trace] = []
        
        for traces in task_groups.values():
            if len(traces) < 2:
                all_selected_traces.extend(traces)
                continue
            
            best_subset = self._find_max_variance_subset(traces, self.group_size)
            all_selected_traces.extend(best_subset)
        
        if len(all_selected_traces) < self.batch_size:
            design.warning(f"Not enough traces to sample ({len(all_selected_traces)}/{self.batch_size})")
            return self.get(self.batch_size)
        
        return all_selected_traces
    
    def _find_max_variance_subset(self, traces: list[Trace], target_size: int) -> list[Trace]:
        """Find subset of traces with maximum variance in rewards."""
        if len(traces) <= target_size:
            return traces

        sorted_traces = sorted(traces, key=lambda t: t.reward)
        
        # Pick traces at regular intervals to maximize spread
        indices = []
        step = len(sorted_traces) / target_size
        for i in range(target_size):
            idx = int(i * step)
            indices.append(min(idx, len(sorted_traces) - 1))
        
        # Remove duplicates while preserving order
        indices = list(dict.fromkeys(indices))
        
        return [sorted_traces[i] for i in indices]
