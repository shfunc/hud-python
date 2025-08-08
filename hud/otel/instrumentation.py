"""Automatic instrumentation for HUD agents and MCP.

This module provides functions to automatically instrument agent classes
and MCP communication without requiring decorators or code changes.
"""

from __future__ import annotations

import functools
import json
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from hud.agent import MCPAgent


def _wrap_get_model_response(original_method):
    """Wrap get_model_response to emit telemetry."""

    @functools.wraps(original_method)
    async def wrapper(self, messages: list[Any], **kwargs):
        # Extract metadata
        provider = self.__class__.__name__.replace("Agent", "").lower()
        model_name = getattr(self, "model", getattr(self, "model_name", "unknown"))

        # Message metadata
        metadata = {
            "message_count": len(messages),
            "has_system": any(m.get("role") == "system" for m in messages if isinstance(m, dict)),
            "last_role": messages[-1].get("role")
            if messages and isinstance(messages[-1], dict)
            else None,
        }

        tracer = trace.get_tracer("hud-sdk")
        with tracer.start_as_current_span(
            f"agent.{provider}.get_model_response",
            kind=SpanKind.CLIENT,
            attributes={
                "agent.provider": provider,
                "agent.model": model_name,
                "hud.span_type": "agent",
                "category": "agent",
            },
        ) as span:
            try:
                # Set request metadata
                span.set_attribute("agent_request", json.dumps(metadata))

                # Call original method
                response = await original_method(self, messages, **kwargs)

                # Set response data
                response_dict = response.model_dump(mode="json", exclude_none=True)
                span.set_attribute("agent_response", json.dumps(response_dict))

                # Set status
                if response.isError:
                    span.set_status(Status(StatusCode.ERROR, "Model returned error"))
                else:
                    span.set_status(Status(StatusCode.OK))

                return response

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return wrapper


def _wrap_execute_tools(original_method):
    """Wrap execute_tools to emit telemetry."""

    @functools.wraps(original_method)
    async def wrapper(self, *, tool_calls, **kwargs):
        provider = self.__class__.__name__.replace("Agent", "").lower()

        tracer = trace.get_tracer("hud-sdk")
        with tracer.start_as_current_span(
            f"agent.{provider}.execute_tools",
            kind=SpanKind.INTERNAL,
            attributes={
                "agent.provider": provider,
                "tool_count": len(tool_calls),
                "hud.span_type": "agent",
                "category": "agent",
            },
        ) as span:
            try:
                result = await original_method(self, tool_calls=tool_calls, **kwargs)

                # Count successes/failures
                errors = sum(1 for r in result if r.isError)
                if errors > 0:
                    span.set_attribute("tool_errors", errors)

                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return wrapper


def instrument_agent_class(agent_class: type[MCPAgent]) -> None:
    """Automatically instrument an agent class.

    This wraps key methods to emit telemetry spans without
    requiring decorators in the agent code.
    """
    # Check if already instrumented
    if hasattr(agent_class.get_model_response, "_telemetry_wrapped"):
        return

    # Wrap get_model_response
    if hasattr(agent_class, "get_model_response"):
        wrapped = _wrap_get_model_response(agent_class.get_model_response)
        wrapped._telemetry_wrapped = True
        agent_class.get_model_response = wrapped

    # Wrap execute_tools
    if hasattr(agent_class, "execute_tools"):
        wrapped = _wrap_execute_tools(agent_class.execute_tools)
        wrapped._telemetry_wrapped = True
        agent_class.execute_tools = wrapped


def auto_instrument_agents() -> None:
    """Find and instrument all MCPAgent subclasses.

    Call this after all agent classes are defined but before
    creating instances.
    """

    # Find all subclasses of MCPAgent
    def find_subclasses(cls):
        all_subclasses = []
        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(find_subclasses(subclass))
        return all_subclasses

    # Instrument each one
    for agent_cls in find_subclasses(MCPAgent):
        try:
            instrument_agent_class(agent_cls)
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(f"Failed to instrument {agent_cls.__name__}: {e}")


def install_mcp_instrumentation(provider) -> None:
    """Enable community MCP OpenTelemetry instrumentation if present.

    Args:
        provider: The TracerProvider to use for instrumentation
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        from opentelemetry.instrumentation.mcp.instrumentation import (  # type: ignore
            McpInstrumentor,
        )

        McpInstrumentor().instrument(tracer_provider=provider)
        logger.debug("MCP instrumentation installed")
    except ImportError:
        logger.debug("opentelemetry-instrumentation-mcp not available – skipping")
    except Exception as exc:
        logger.warning("Failed to install MCP instrumentation: %s", exc)
