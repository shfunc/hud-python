"""Job context manager for grouping related traces."""

from __future__ import annotations

import logging
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from hud.telemetry.exporter import JobStatus, submit_to_worker_loop, update_job_status

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Self

logger = logging.getLogger("hud.telemetry")

# Context variables for current job
current_job_id: ContextVar[str | None] = ContextVar("current_job_id", default=None)
current_job_name: ContextVar[str | None] = ContextVar("current_job_name", default=None)


class JobContext:
    """Context manager for grouping traces under a job."""

    def __init__(
        self, name: str, taskset_name: str | None = None, metadata: dict[str, Any] | None = None
    ) -> None:
        self.id = str(uuid.uuid4())
        self.name = name
        self.metadata = metadata or {}
        self.taskset_name: str | None = taskset_name

    def __enter__(self) -> Self:
        # Auto-detect dataset
        if self.taskset_name is None:
            self._detect_dataset()

        # Set context variables
        current_job_id.set(self.id)
        current_job_name.set(self.name)

        # Send initial status
        job_metadata = {**self.metadata}
        coro = update_job_status(
            self.id, JobStatus.RUNNING, metadata=job_metadata, taskset_name=self.taskset_name
        )
        submit_to_worker_loop(coro)

        logger.info("Started job %s (ID: %s)", self.name, self.id)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        # Determine final status
        if exc_type is not None:
            # Job failed with exception
            error_msg = f"{exc_type.__name__}: {exc_val}"
            coro = update_job_status(
                self.id, JobStatus.ERROR, error_message=error_msg, taskset_name=self.taskset_name
            )
        else:
            # Job completed successfully
            coro = update_job_status(self.id, JobStatus.COMPLETED, taskset_name=self.taskset_name)

        submit_to_worker_loop(coro)

        # Clear context
        current_job_id.set(None)
        current_job_name.set(None)

        status = "failed" if exc_type else "completed"
        logger.info("Job %s %s", self.name, status)

    def _detect_dataset(self) -> None:
        """Auto-detect HuggingFace dataset in parent scope."""
        try:
            # Check frames 2 and 3 (with statement and parent scope)
            for frame_depth in [2, 3]:
                try:
                    frame = sys._getframe(frame_depth)

                    # Search for Dataset objects
                    for var_value in frame.f_locals.values():
                        if hasattr(var_value, "info") and hasattr(var_value.info, "builder_name"):
                            self.taskset_name = var_value.info.builder_name
                            logger.debug(
                                "Auto-detected dataset at frame %d: %s",
                                frame_depth,
                                self.taskset_name,
                            )
                            return
                        elif hasattr(var_value, "builder_name"):
                            # Older dataset format
                            self.taskset_name = var_value.builder_name
                            logger.debug(
                                "Auto-detected dataset at frame %d: %s",
                                frame_depth,
                                self.taskset_name,
                            )
                            return
                except ValueError:
                    # Frame doesn't exist
                    continue
        except Exception as e:
            logger.debug("Dataset auto-detection failed: %s", e)


@contextmanager
def job(
    name: str, taskset_name: str | None = None, metadata: dict[str, Any] | None = None
) -> Generator[JobContext, None, None]:
    """
    Create a job context for grouping related traces.

    Args:
        name: Name for the job
        metadata: Optional metadata to include with the job

    Example:
        with hud.job("evaluation_run") as job:
            for task in tasks:
                with hud.trace(f"task_{task.id}"):
                    # Trace automatically includes job_id
                    result = await agent.run(task)
    """
    with JobContext(name, taskset_name, metadata) as ctx:
        yield ctx


def get_current_job_id() -> str | None:
    """Get the current job ID if inside a job context."""
    return current_job_id.get()


def get_current_job_name() -> str | None:
    """Get the current job name if inside a job context."""
    return current_job_name.get()
