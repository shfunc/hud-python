"""User-facing trace context manager for HUD telemetry.

This module provides the simple trace() API that users interact with.
The actual OpenTelemetry implementation is in hud.otel.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any

from hud.otel import configure_telemetry
from hud.otel import trace as OtelTrace

__all__ = ["trace"]


@contextmanager
def trace(
    task_run_id: str | None = None,
    *,
    root: bool = True,
    name: str = "hud.task",
    attrs: dict[str, Any] | None = None,
    job_id: str | None = None,
):
    """Start a HUD trace context.

    If ``task_run_id`` is not provided, a new one is generated.

    Args:
        task_run_id: Unique identifier for this task run. Auto-generated if not provided.
        root: Whether this is a root trace (updates task status)
        name: Span name for OpenTelemetry
        attrs: Additional attributes to attach to the trace
        job_id: Optional job ID to associate with this trace

    Yields:
        str: The task run ID

    Usage:
        import hud

        with hud.trace() as task_run_id:
            # Your code here
            print(f"Running task: {task_run_id}")

        # Or with job_id:
        with hud.trace(job_id="job-123") as task_run_id:
            pass

        # Or with custom ID:
        with hud.trace("my-specific-task-id") as task_run_id:
            pass
    """
    # Ensure telemetry is configured
    configure_telemetry()

    # Auto-generate a task_run_id if missing
    if not task_run_id:
        task_run_id = f"auto-{uuid.uuid4().hex[:12]}"

    # Delegate to OpenTelemetry implementation
    with OtelTrace(
        task_run_id, is_root=root, span_name=name, attributes=attrs or {}, job_id=job_id
    ) as run_id:
        yield run_id
