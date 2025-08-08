from __future__ import annotations

"""Public façade for HUD tracing.

This module provides the simple, user-friendly API for tracing.
"""

from contextlib import contextmanager
from typing import Any
import uuid

from hud.otel import trace as OtelTrace, span_context, configure_telemetry
from hud.otel.collector import get_trace as get_collected_trace
from hud.types import Trace

__all__ = [
    "trace",
    "span_context", 
    "get_trace",
]

# Rename 'start' to 'trace' for cleaner API
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
    
    Usage:
        import hud.trace
        
        with hud.trace.trace() as task_run_id:
            # Your code here
            print(f"Running task: {task_run_id}")
            
        # Or with job_id:
        with hud.trace.trace(job_id="job-123") as task_run_id:
            pass
    """
    # Ensure provider / instrumentation active
    configure_telemetry()

    # Auto-generate a task_run_id if missing
    if not task_run_id:
        task_run_id = f"auto-{uuid.uuid4().hex[:12]}"

    with OtelTrace(
        task_run_id, 
        is_root=root, 
        span_name=name, 
        attributes=attrs or {},
        job_id=job_id
    ) as run_id:
        yield run_id


def get_trace(task_run_id: str) -> Trace | None:
    """Retrieve the collected trace for a task run.
    
    Returns None if trace collection was disabled or the trace doesn't exist.
    
    Usage:
        import hud
        
        # Run agent
        with hud.trace() as task_run_id:
            agent = MyAgent()
            result = await agent.run("solve task")
            
        # Get the trace
        trace = hud.trace.get_trace(task_run_id)
        if trace:
            print(f"Collected {len(trace.trace)} steps")
    """
    return get_collected_trace(task_run_id)