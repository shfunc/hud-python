from __future__ import annotations

import logging
from typing import Any

from opentelemetry import baggage
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

# Removed import of get_current_task_run_id - now using baggage directly

logger = logging.getLogger(__name__)


class HudEnrichmentProcessor(SpanProcessor):
    """Span processor that enriches every span with HUD-specific context.

    • Adds ``hud.task_run_id`` attribute if available.
    • Adds ``hud.job_id`` attribute if available in baggage.
    """

    def __init__(self) -> None:
        # No state, everything comes from context vars
        super().__init__()

    # --- callback hooks -------------------------------------------------
    def on_start(self, span: Span, parent_context: Any) -> None:  # type: ignore[override]
        try:
            # Get task_run_id from baggage in parent context
            run_id = baggage.get_baggage("hud.task_run_id", context=parent_context)
            if run_id and span.is_recording():
                span.set_attribute("hud.task_run_id", str(run_id))

            # Get job_id from baggage if available
            job_id = baggage.get_baggage("hud.job_id", context=parent_context)
            if job_id and span.is_recording():
                span.set_attribute("hud.job_id", str(job_id))

        except Exception as exc:  # defensive; never fail the tracer
            logger.debug("HudEnrichmentProcessor.on_start error: %s", exc, exc_info=False)

    def on_end(self, span: ReadableSpan) -> None:
        # Nothing to do enrichment is on_start only
        pass

    # Required to fully implement abstract base, but we don't batch spans
    def shutdown(self) -> None:  # type: ignore[override]
        pass

    def force_flush(self, timeout_millis: int | None = None) -> bool:  # type: ignore[override]
        return True
