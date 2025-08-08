from __future__ import annotations

"""Custom OpenTelemetry exporter that sends spans to the existing HUD telemetry
HTTP endpoint (/v2/task_runs/<id>/telemetry-upload).

The exporter groups spans by ``hud.task_run_id`` baggage / attribute so we keep
exactly the same semantics the old async worker in ``hud.telemetry.exporter``
implemented.

This exporter is *synchronous* (derives from :class:`SpanExporter`).  We rely on
``hud.server.make_request_sync`` which already contains retry & auth logic.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from hud.server import make_request_sync
from hud.settings import settings

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_ns_to_iso(ts_ns: int) -> str:
    """Convert a ``Span`` timestamp (nanoseconds) to ISO-8601 string."""
    # OpenTelemetry times are epoch nanoseconds
    dt = datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")





def _span_to_dict(span: ReadableSpan) -> Dict[str, Any]:
    """Convert an OpenTelemetry span to a dict using typed models."""
    from .hud_models import HudSpan, extract_span_attributes
    
    attrs = dict(span.attributes or {})
    
    # Extract method name from span name if not in attributes
    method_name = attrs.get("semconv_ai.mcp.method_name")
    if not method_name and span.name.endswith(".mcp"):
        method_name = span.name[:-4]  # Remove .mcp suffix
    
    # Create typed attributes
    typed_attrs = extract_span_attributes(attrs, method_name, span.name)
    
    # Override span kind from the span itself
    typed_attrs.span_kind = span.kind.name
    
    # Build typed span
    typed_span = HudSpan(
        name=span.name,
        trace_id=format(span.context.trace_id, "032x"),
        span_id=format(span.context.span_id, "016x"),
        parent_span_id=format(span.parent.span_id, "016x") if span.parent and span.parent.span_id else None,
        start_time=_ts_ns_to_iso(span.start_time),
        end_time=_ts_ns_to_iso(span.end_time),
        status_code=span.status.status_code.name if span.status else "UNSET",
        status_message=span.status.description if span.status else None,
        attributes=typed_attrs,
        exceptions=None,
    )
    
    # Add error information if present
    if span.events:
        exceptions = []
        for event in span.events:
            if event.name == "exception":
                exceptions.append({
                    "timestamp": _ts_ns_to_iso(event.timestamp),
                    "attributes": dict(event.attributes or {})
                })
        if exceptions:
            typed_span.exceptions = exceptions
    
    # Convert to dict for export
    return typed_span.model_dump(mode="json", by_alias=True, exclude_none=True)


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class HudSpanExporter(SpanExporter):
    """Exporter that forwards spans to HUD backend using existing endpoint."""

    def __init__(self, *, base_url: str, api_key: str):
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:  # type: ignore[override]
        if not spans:
            return SpanExportResult.SUCCESS

        # Group spans by hud.task_run_id attribute
        grouped: Dict[str, List[ReadableSpan]] = defaultdict(list)
        for span in spans:
            run_id = span.attributes.get("hud.task_run_id")
            if not run_id:
                # Skip spans that are outside HUD traces
                continue
            grouped[str(run_id)].append(span)

        # Send each group synchronously (retry inside make_request_sync)
        for run_id, span_batch in grouped.items():
            try:
                url = f"{self._base_url}/v2/task_runs/{run_id}/telemetry-upload"
                telemetry_spans = [_span_to_dict(s) for s in span_batch]
                payload = {
                    "metadata": {},  # reserved – can be filled later
                    "telemetry": telemetry_spans,
                }
                

                
                logger.debug("HUD exporter sending %d spans to %s", len(span_batch), url)
                make_request_sync(
                    method="POST",
                    url=url,
                    json=payload,
                    api_key=self._api_key,
                )
            except Exception as exc:
                logger.exception("HUD exporter failed to send spans for task %s: %s", run_id, exc)
                # If *any* group fails we return FAILURE so the OTEL SDK can retry
                return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # type: ignore[override]
        # Nothing to cleanup – httpx handled inside make_request_sync
        pass

    def force_flush(self, timeout_millis: int | None = None) -> bool:  # type: ignore[override]
        # Synchronous export – nothing buffered here
        return True

