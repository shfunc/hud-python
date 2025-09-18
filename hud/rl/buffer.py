"""Replay buffer for storing and sampling episodes."""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import TYPE_CHECKING, Generic, TypeVar

from hud.types import Task, Trace
from hud.utils.hud_console import HUDConsole

logger = logging.getLogger(__name__)
hud_console = HUDConsole(logger=logger)

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Callable

    from hud.rl.config import Config


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
        return list(self.buffer)[-n:]

    def consume(self, n: int = 0) -> list[T]:
        """Consume items from the buffer."""
        if n == 0:
            return list(self.buffer)
        if n > len(self.buffer):
            raise ValueError("Not enough items in buffer")

        return [self.buffer.pop() for _ in range(n)]

    def get_filtered(
        self, n: int = 0, filter_fn: Callable[[T], bool] | None = None, consume: bool = False
    ) -> list[T]:
        """Filter the buffer by a filter function."""
        filtered = (
            [item for item in self.buffer if filter_fn(item)] if filter_fn else list(self.buffer)
        )
        if n == 0:
            return filtered
        return self.consume(n) if consume else self.get(n)

    def sample(
        self,
        batch_size: int,
        n: int = 0,
        filter_fn: Callable[[T], bool] | None = None,
        consume: bool = False,
    ) -> list[T]:
        """Sample a batch of items with optional filtering."""
        items = self.get_filtered(n, filter_fn, consume)

        if len(items) < batch_size:
            hud_console.warning(f"Buffer has {len(items)} items, requested {batch_size}")
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
            raise ValueError(
                f"Group size is greater than batch size, {self.group_size} > {self.batch_size}"
            )

        if self.batch_size % self.group_size != 0:
            raise ValueError(
                f"A batch cannot have irregular groups, {self.group_size} % {self.batch_size} != 0"
            )

        if self.group_size % config.training.mini_batch_size != 0:
            raise ValueError(
                f"Group size is not a multiple of mini batch size, {self.group_size} % {config.training.mini_batch_size} != 0"  # noqa: E501
            )

        self.groups_per_batch = self.batch_size // self.group_size
        self.number_of_tasks = self.training_steps * self.groups_per_batch

        super().__init__(self.number_of_tasks)

        dataset = dataset if isinstance(dataset, list) else [dataset]
        tasks = self._validate_tasks(dataset)
        if config.training.shuffle_dataset:
            random.shuffle(tasks)
        if len(tasks) > self.number_of_tasks:
            leftovers = len(tasks) - self.number_of_tasks
            hud_console.warning(
                f"Training steps ({self.training_steps}) will lead to {leftovers} tasks not being trained"  # noqa: E501
            )
            tasks = tasks[: self.number_of_tasks]

        # Check if the dataset is imbalanced
        self.dataset_size = len(tasks)
        if self.training_steps % self.dataset_size != 0:
            leftovers = self.number_of_tasks % self.dataset_size
            hud_console.warning(
                f"Dataset imbalanced ({leftovers} tasks will be trained 1 more time)"
            )
            hud_console.warning(
                f"This is because the number of training steps ({self.training_steps}) is not a multiple of the dataset size ({self.dataset_size})"  # noqa: E501
            )

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
    def info(self) -> dict[str, int | float | str]:
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
        # Create groups where each group contains group_size copies of the same task
        result = []
        for task in tasks:
            result.extend([task] * self.group_size)
        return result


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
        from collections import Counter, defaultdict, deque

        # Expect recent window to already be grouped by task id

        # Build recent window and earlier lookup (short form)
        buf_list = list(self.buffer)
        if len(buf_list) < self.batch_size:
            hud_console.warning(
                f"[group-sampler] Buffer has only {len(buf_list)} traces, need {self.batch_size}"
            )
            while len(buf_list) < self.batch_size:
                take = min(len(buf_list) or 1, self.batch_size - len(buf_list))
                buf_list.extend(buf_list[:take])
        recent_traces = buf_list[-self.batch_size :]
        hud_console.info(
            f"[group-sampler] recent-window histogram: {Counter(getattr(t.task, 'id', 'NA') for t in recent_traces)}"  # noqa: E501
        )

        hud_console.info(
            f"[group-sampler] Building earlier traces lookup, buffer size: {len(buf_list)}"
        )
        earlier_traces_by_task: dict[str, deque[Trace]] = defaultdict(deque)
        for tr in buf_list[: -self.batch_size]:
            earlier_traces_by_task[getattr(tr.task, "id", "NA")].append(tr)

        # Chunk from the most-recent end
        final_traces: list[Trace] = []
        groups_per_batch = self.batch_size // self.group_size
        hud_console.info(f"[group-sampler] Processing {groups_per_batch} groups")
        for g_idx in range(groups_per_batch):
            start = g_idx * self.group_size
            end = start + self.group_size
            group = recent_traces[start:end]

            # Assert homogeneity: every trace in a group must share the same task id
            cnt = Counter(getattr(t.task, "id", "NA") for t in group)
            if len(cnt) != 1:
                raise RuntimeError(f"Group {g_idx} is not homogeneous: {dict(cnt)}")
            target_tid = next(iter(cnt.keys()))

            # Build homogeneous group of target_tid, filling from earlier traces to increase spread
            homogeneous: list[Trace] = [
                t for t in group if getattr(t.task, "id", "NA") == target_tid
            ]
            needed = self.group_size - len(homogeneous)

            # Greedy fill: choose earlier traces (same task-id) farthest from current mean reward
            def current_mean(homogeneous: list[Trace]) -> float:
                if not homogeneous:
                    return 0.0
                vals = [float(getattr(t, "reward", 0.0) or 0.0) for t in homogeneous]
                return sum(vals) / len(vals)

            while needed > 0:
                pool = earlier_traces_by_task.get(target_tid, deque())
                if pool:
                    mu = current_mean(homogeneous)
                    # pick element farthest from current mean
                    best_i = None
                    best_dist = -1.0
                    for i, tr in enumerate(list(pool)):
                        r = float(getattr(tr, "reward", 0.0) or 0.0)
                        dist = abs(r - mu)
                        if dist > best_dist:
                            best_dist = dist
                            best_i = i
                    # pop selected
                    chosen = list(pool)[best_i]  # type: ignore[index]
                    # remove from deque efficiently by rotating
                    left = list(pool)
                    if best_i is not None:
                        left.pop(best_i)  # O(n) but pool is small in practice
                        earlier_traces_by_task[target_tid] = deque(left)
                        homogeneous.append(chosen)
                else:
                    # duplicate extreme within current homogeneous set
                    if not homogeneous:
                        raise RuntimeError(f"Group {g_idx} has no traces for target {target_tid}")
                    mu = current_mean(homogeneous)
                    extreme = max(
                        homogeneous, key=lambda t: abs(float(getattr(t, "reward", 0.0) or 0.0) - mu)
                    )
                    homogeneous.append(extreme)
                needed -= 1

            # Replacement step: swap in earlier traces to increase reward spread
            pool = earlier_traces_by_task.get(target_tid, deque())
            if pool:
                # Log pool stats
                pool_vals = [float(getattr(tr, "reward", 0.0) or 0.0) for tr in list(pool)]
                if pool_vals:
                    pool_mean = sum(pool_vals) / len(pool_vals)
                    pool_var = sum((v - pool_mean) * (v - pool_mean) for v in pool_vals) / len(
                        pool_vals
                    )
                    hud_console.info(
                        f"[group-sampler] Group {g_idx}: earlier-pool size={len(pool_vals)} mean={pool_mean:.4f} std={(pool_var**0.5):.4f}"  # noqa: E501
                    )

                # Decide how many to replace (up to 1/4 of group, at least 1)
                replace_k = max(1, self.group_size // 4)
                replace_k = min(replace_k, len(pool), self.group_size)

                if replace_k > 0:
                    mu = current_mean(homogeneous)
                    # Select replacement candidates from pool farthest from current mean
                    pool_list = list(pool)
                    pool_indices = list(range(len(pool_list)))
                    pool_indices.sort(
                        key=lambda i: abs(
                            (float(getattr(pool_list[i], "reward", 0.0) or 0.0)) - mu
                        ),
                        reverse=True,
                    )
                    chosen_pool_idx = set(pool_indices[:replace_k])
                    replacements = [pool_list[i] for i in pool_indices[:replace_k]]

                    # Remove chosen from pool deque
                    remaining = [tr for i, tr in enumerate(pool_list) if i not in chosen_pool_idx]
                    earlier_traces_by_task[target_tid] = deque(remaining)

                    # Select current group positions closest to mean to replace
                    group_indices = list(range(len(homogeneous)))
                    group_indices.sort(
                        key=lambda i: abs(
                            (float(getattr(homogeneous[i], "reward", 0.0) or 0.0)) - mu
                        )
                    )
                    target_positions = group_indices[:replace_k]

                    for pos, new_tr in zip(target_positions, replacements, strict=False):
                        homogeneous[pos] = new_tr

            # Validate homogeneity
            if any(getattr(t.task, "id", "NA") != target_tid for t in homogeneous):
                raise RuntimeError(f"Group {g_idx} is not homogeneous after sampling")
            final_traces.extend(homogeneous)

        for i in range(0, len(final_traces), self.group_size):
            block = final_traces[i : i + self.group_size]
            if len({getattr(t.task, "id", "NA") for t in block}) != 1:
                raise RuntimeError(f"Homogeneity validation failed for block starting at index {i}")

        hud_console.info(
            f"[group-sampler] final histogram: {Counter(getattr(t.task, 'id', 'NA') for t in final_traces)}"  # noqa: E501
        )
        return final_traces

        # --------------------------------------------------------------------
