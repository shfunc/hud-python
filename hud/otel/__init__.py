from __future__ import annotations

"""HUD OpenTelemetry integration package.

This module re-exports the public helpers so callers can simply::

    import hud.otel
    hud.otel.configure_telemetry()

…but in practice we encourage using ``hud.trace`` which wraps everything
for user-facing code.

Using with Alternative OpenTelemetry Backends
----------------------------------------------

By default, traces are sent to HUD's backend. To also send traces to standard
OpenTelemetry collectors (Jaeger, Zipkin, Grafana Tempo, etc.), configure OTLP export:

1. Install the OTLP exporter::

    pip install opentelemetry-exporter-otlp-proto-grpc

2. Configure programmatically when starting your trace::

    from hud.otel import configure_telemetry
    
    # Export to both HUD and a local Jaeger instance
    configure_telemetry(
        enable_otlp=True,
        otlp_endpoint="localhost:4317",  # Default OTLP gRPC port
        # otlp_headers={"authorization": "Bearer <token>"}  # If auth needed
    )
    
    # Then use traces normally
    import hud
    with hud.trace() as task_run_id:
        # Your code here
        pass

3. Common OTLP destinations:

   **Local Jaeger**::
   
       docker run -d -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
       configure_telemetry(enable_otlp=True)  # Uses localhost:4317 by default
       # View traces at http://localhost:16686
   
   **Grafana Cloud**::
   
       configure_telemetry(
           enable_otlp=True,
           otlp_endpoint="tempo-us-central1.grafana.net:443",
           otlp_headers={"authorization": "Bearer <your-token>"}
       )
   
   **Datadog**::
   
       configure_telemetry(
           enable_otlp=True,
           otlp_endpoint="https://trace.agent.datadoghq.com:4317",
           otlp_headers={"DD-API-KEY": "<your-api-key>"}
       )

Note: Traces will be sent to BOTH HUD and your OTLP endpoint. To disable HUD export,
set the environment variable HUD_TELEMETRY_ENABLED=false.
"""

from .config import configure_telemetry, shutdown_telemetry  # noqa: F401
from .processors import HudEnrichmentProcessor  # noqa: F401
from .exporters import HudSpanExporter  # noqa: F401
from .trace import trace  # noqa: F401
from .context import span_context, get_current_task_run_id, is_root_trace  # noqa: F401

__all__ = [
    "configure_telemetry",
    "shutdown_telemetry",
    "trace",
    "span_context",
    "get_current_task_run_id",
    "is_root_trace",
]