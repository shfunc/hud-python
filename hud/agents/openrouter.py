"""OpenRouter agent facade plus shared tooling helpers."""

from __future__ import annotations

import base64
import json
import re
import uuid
from importlib import import_module
import importlib.util
from pathlib import Path
from io import BytesIO
from typing import Any, Dict, Type
from abc import abstractmethod

import mcp.types as types
from PIL import Image

import litellm

from hud.agents.base import MCPAgent
from hud.tools.computer.settings import computer_settings
from hud.types import MCPToolCall, MCPToolResult, AgentResponse
from hud import instrument
import logging
logger = logging.getLogger(__name__)

from hud.settings import settings
import os

# Shared helper utilities for computer-use adapters
def _random_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"

def _make_reasoning_item(reasoning: str) -> dict[str, Any]:
    return {
        "id": _random_id(),
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": reasoning}],
    }

def _make_output_text_item(content: str) -> dict[str, Any]:
    return {
        "id": _random_id(),
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    }

def _make_computer_call_item(action: dict[str, Any], call_id: str | None = None) -> dict[str, Any]:
    call_id = call_id or _random_id()
    return {
        "id": _random_id(),
        "call_id": call_id,
        "type": "computer_call",
        "status": "completed",
        "pending_safety_checks": [],
        "action": action,
    }

def _make_click_item(x: int, y: int, button: str = "left", call_id: str | None = None) -> dict[str, Any]:
    return _make_computer_call_item({"type": "click", "x": x, "y": y, "button": button}, call_id)

def _make_double_click_item(x: int, y: int, call_id: str | None = None) -> dict[str, Any]:
    return _make_computer_call_item({"type": "double_click", "x": x, "y": y}, call_id)

def _make_drag_item(path: list[dict[str, int]], call_id: str | None = None) -> dict[str, Any]:
    return _make_computer_call_item({"type": "drag", "path": path}, call_id)

def _make_keypress_item(keys: list[str], call_id: str | None = None) -> dict[str, Any]:
    return _make_computer_call_item({"type": "keypress", "keys": keys}, call_id)

def _make_type_item(text: str, call_id: str | None = None) -> dict[str, Any]:
    return _make_computer_call_item({"type": "type", "text": text}, call_id)

def _make_scroll_item(
    x: int,
    y: int,
    scroll_x: int,
    scroll_y: int,
    call_id: str | None = None,
) -> dict[str, Any]:
    action = {"type": "scroll", "x": x, "y": y, "scroll_x": scroll_x, "scroll_y": scroll_y}
    return _make_computer_call_item(action, call_id)

def _make_wait_item(call_id: str | None = None) -> dict[str, Any]:
    return _make_computer_call_item({"type": "wait"}, call_id)

def _make_screenshot_item(call_id: str) -> dict[str, Any]:
    return _make_computer_call_item({"type": "screenshot"}, call_id)

def _make_failed_tool_call_items(
    tool_name: str,
    tool_kwargs: dict[str, Any],
    error_message: str,
    call_id: str,
) -> list[dict[str, Any]]:
    call = _make_computer_call_item({"type": tool_name, **tool_kwargs}, call_id)
    call["status"] = "failed"
    failure_text = _make_output_text_item(f"Tool {tool_name} failed: {error_message}")
    failure_text["role"] = "assistant"
    return [call, failure_text]

def _coerce_to_pixel_coordinates(
    x_val: Any,
    y_val: Any,
    *,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    try:
        x_float = float(x_val)
        y_float = float(y_val)
    except (TypeError, ValueError):
        return None

    def clamp(value: int, maximum: int) -> int:
        return max(0, min(maximum - 1, value))

    abs_x = abs(x_float)
    abs_y = abs(y_float)
    if abs_x <= 1.0 and abs_y <= 1.0:
        px = int(x_float * width)
        py = int(y_float * height)
    elif abs_x <= 999.0 and abs_y <= 999.0:
        px = int((x_float / 999.0) * width)
        py = int((y_float / 999.0) * height)
    else:
        px = int(x_float)
        py = int(y_float)

    return clamp(px, width), clamp(py, height)

def _parse_coordinate_box(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None

    if isinstance(value, str):
        stripped = value.strip()
        try:
            loaded = json.loads(stripped)
        except Exception:
            matches = re.findall(r"-?\d+(?:\.\d+)?", stripped)
            if len(matches) >= 2:
                return float(matches[0]), float(matches[1])
        else:
            if isinstance(loaded, (list, tuple)) and len(loaded) >= 2:
                try:
                    return float(loaded[0]), float(loaded[1])
                except (TypeError, ValueError):
                    return None
    return None

def _coerce_box_to_pixels(
    box: Any,
    *,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    coords = _parse_coordinate_box(box)
    if not coords:
        return None
    return _coerce_to_pixel_coordinates(coords[0], coords[1], width=width, height=height)

def _parse_json_action_string(action_text: str) -> dict[str, Any] | None:
    candidate = action_text.strip()
    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None

    attempts = [candidate]
    if "\\" in candidate:
        try:
            attempts.append(candidate.encode("utf-8").decode("unicode_escape"))
        except Exception:
            pass
        attempts.append(candidate.replace("\\\"", '"'))

    for attempt in attempts:
        try:
            return json.loads(attempt)
        except Exception:
            continue

    return None

def _convert_json_action_to_items(
    json_action: dict[str, Any],
    call_id: str,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    
    action_type = str(json_action.get("type", json_action.get("action_type", ""))).lower()
    if not action_type:
        return items

    if action_type in {"type", "text", "input_text"}:
        text_value = json_action.get("content") or json_action.get("text") or ""
        if text_value:
            items.append(_make_type_item(str(text_value), call_id=call_id))
    elif action_type in {"click", "left_click", "right_click"}:
        # Handle both "start_box" and the new "start_x"/"start_y" format
        start_box = json_action.get("start_box") or json_action.get("startBox")
        coords = _coerce_box_to_pixels(start_box, width=image_width, height=image_height)
        if not coords:
             coords = _coerce_to_pixel_coordinates(
                json_action.get("start_x") or json_action.get("x"),
                json_action.get("start_y") or json_action.get("y"),
                width=image_width,
                height=image_height,
            )
        if coords:
            button = str(json_action.get("button", "left") or "left").lower()
            items.append(_make_click_item(coords[0], coords[1], button=button, call_id=call_id))
    elif action_type in {"double_click", "left_double_click"}:
        start_box = json_action.get("start_box") or json_action.get("startBox")
        coords = _coerce_box_to_pixels(start_box, width=image_width, height=image_height)
        if not coords:
            coords = _coerce_to_pixel_coordinates(
                json_action.get("start_x") or json_action.get("x"),
                json_action.get("start_y") or json_action.get("y"),
                width=image_width,
                height=image_height,
            )
        if coords:
            items.append(_make_double_click_item(coords[0], coords[1], call_id=call_id))
    elif action_type in {"drag", "left_drag"}:
        start_box = json_action.get("start_box") or json_action.get("startBox")
        end_box = json_action.get("end_box") or json_action.get("endBox")
        start_coords = _coerce_box_to_pixels(start_box, width=image_width, height=image_height)
        end_coords = _coerce_box_to_pixels(end_box, width=image_width, height=image_height)
        if not start_coords:
            start_coords = _coerce_to_pixel_coordinates(
                json_action.get("start_x") or json_action.get("x"),
                json_action.get("start_y") or json_action.get("y"),
                width=image_width,
                height=image_height,
            )
        if not end_coords:
            end_coords = _coerce_to_pixel_coordinates(
                json_action.get("end_x"),
                json_action.get("end_y"),
                width=image_width,
                height=image_height,
            )
        if start_coords and end_coords:
            path = [
                {"x": start_coords[0], "y": start_coords[1]},
                {"x": end_coords[0], "y": end_coords[1]},
            ]
            items.append(_make_drag_item(path, call_id=call_id))
    elif action_type == "scroll":
        start_box = json_action.get("start_box") or json_action.get("startBox")
        coords = _coerce_box_to_pixels(start_box, width=image_width, height=image_height)
        if not coords:
            coords = _coerce_to_pixel_coordinates(
                json_action.get("start_x") or json_action.get("x"),
                json_action.get("start_y") or json_action.get("y"),
                width=image_width,
                height=image_height,
            )
        direction = str(json_action.get("direction", "")).lower()
        step = int(json_action.get("step", 5) or 5)
        if coords:
            scroll_x = 0
            scroll_y = 0
            if direction == "up":
                scroll_y = -abs(step)
            elif direction == "down":
                scroll_y = abs(step)
            elif direction == "left":
                scroll_x = -abs(step)
            elif direction == "right":
                scroll_x = abs(step)
            items.append(
                _make_scroll_item(coords[0], coords[1], scroll_x, scroll_y, call_id=call_id)
            )
    # hover/move dropped in minimal action surface
    elif action_type in {"keypress", "key", "key_press"}:
        keys = json_action.get("keys")
        key_list: list[str] = []
        if isinstance(keys, str):
            key_list = [segment.strip() for segment in keys.split("+") if segment.strip()]
        elif isinstance(keys, list):
            key_list = [str(segment).strip() for segment in keys if str(segment).strip()]
        if key_list:
            items.append(_make_keypress_item(key_list, call_id=call_id))
    elif action_type == "wait":
        items.append(_make_wait_item(call_id=call_id))

    return items


def _decode_image_dimensions(image_b64: str) -> tuple[int, int]:
    try:
        data = base64.b64decode(image_b64)
        with Image.open(BytesIO(data)) as img:
            return img.size
    except Exception:  # pragma: no cover - defensive fallback
        return computer_settings.OPENAI_COMPUTER_WIDTH, computer_settings.OPENAI_COMPUTER_HEIGHT


def _extract_user_instruction(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("type") == "message" and message.get("role") == "user":
            content = message.get("content") or []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in {"text", "input_text"}:
                        text = block.get("text")
                        if isinstance(text, str) and text.strip():
                            return text.strip()
    return ""


def get_last_image_from_messages(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        msg_type = message.get("type")
        if msg_type == "computer_call_output":
            output = message.get("output") or {}
            if isinstance(output, dict):
                image_url = output.get("image_url")
                if isinstance(image_url, str) and image_url.startswith("data:image/"):
                    return image_url.split(",", 1)[1]
        if msg_type == "message" and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                for item in reversed(content):
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url_obj = item.get("image_url")
                        if isinstance(url_obj, dict):
                            url = url_obj.get("url")
                            if isinstance(url, str) and url.startswith("data:image/"):
                                return url.split(",", 1)[1]
    return None

class OpenRouterBaseAgent(MCPAgent):
    """Base class for OpenRouter vision-language agents with shared formatting logic."""

    def __init__(self, completion_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.completion_kwargs = completion_kwargs or {}

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[dict[str, Any]]:
        """Format MCP content blocks into message items."""
        content_items: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for block in blocks:
            if isinstance(block, types.TextContent):
                if block.text:
                    text_parts.append(block.text)
            elif isinstance(block, types.ImageContent):
                content_items.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{getattr(block, 'mimeType', 'image/png')};base64,{block.data}",
                                },
                            }
                        ],
                    }
                )

        if text_parts:
            content_items.insert(
                0,
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "\n".join(text_parts)}],
                },
            )

        return content_items

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[dict[str, Any]]:
        """Format tool execution results into message items."""
        import mcp.types as types  # noqa: PLC0415

        rendered: list[dict[str, Any]] = []
        for call, result in zip(tool_calls, tool_results, strict=False):
            call_args = call.arguments or {}
            if result.isError:
                error_text = "".join(
                    c.text for c in result.content if isinstance(c, types.TextContent)
                )
                rendered.extend(
                    _make_failed_tool_call_items(
                        tool_name=call_args.get("type", call.name),
                        tool_kwargs=call_args,
                        error_message=error_text or "Unknown error",
                        call_id=call.id,
                    )
                )
                continue

            screenshot_found = False
            for content in result.content:
                if isinstance(content, types.ImageContent):
                    rendered.append(
                        {
                            "type": "computer_call_output",
                            "call_id": call.id,
                            "output": {
                                "type": "input_image",
                                "image_url": f"data:{content.mimeType};base64,{content.data}",
                            },
                        }
                    )
                    screenshot_found = True
                    break

            text_parts = [
                c.text for c in result.content if isinstance(c, types.TextContent) and c.text
            ]
            if text_parts:
                rendered.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "\n".join(text_parts)}],
                    }
                )

            if not screenshot_found and not text_parts:
                rendered.append(
                    {
                        "type": "computer_call_output",
                        "call_id": call.id,
                        "output": {"type": "input_text", "text": "Tool executed"},
                    }
                )

        return rendered

    @abstractmethod
    async def build_prompt(self, messages: list[dict[str, Any]], instruction: str, screenshot_b64: str) -> list[dict[str, Any]]:
        """Subclass hook to build model-specific prompt/messages."""
        pass

    @abstractmethod
    async def parse_response(self, response: Any, messages: list[dict[str, Any]], screenshot_b64: str) -> AgentResponse:
        """Subclass hook to parse model response into AgentResponse."""
        pass

    @instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    
    async def get_response(self, messages: list[dict[str, Any]]) -> AgentResponse:
        instruction = _extract_user_instruction(messages)
        
        screenshot_b64 = get_last_image_from_messages(messages)
        if not screenshot_b64:
            call_id = _random_id()
            messages.append(_make_screenshot_item(call_id))
            return AgentResponse(
                content="capturing initial screenshot",
                tool_calls=[MCPToolCall(id=call_id, name="openai_computer", arguments={"type": "screenshot"})],
                done=False,
            )
        
        litellm_messages = await self.build_prompt(messages, instruction, screenshot_b64)
        
        api_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": litellm_messages,
        }
        if "openrouter" in self.model_name.lower():
            api_kwargs["api_key"] = settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        api_kwargs.update(self.completion_kwargs)
        
        try:
            response = await litellm.acompletion(**api_kwargs)
        except Exception as exc:
            logger.exception(f"{self.__class__.__name__} completion failed: %s", exc)
            return AgentResponse(
                content=f"{self.__class__.__name__} request failed: {exc}",
                tool_calls=[],
                done=True,
                isError=True,
            )
        
        return await self.parse_response(response, messages, screenshot_b64)


# Adapter dispatch
_ADAPTER_REGISTRY: Dict[str, str] = {
    "z-ai/glm-4.5v": "hud.agents.openrouter.models.glm45v.glm45v:Glm45vAgent",
    "huggingface/bytedance-seed/ui-tars-1.5-7b": "hud.agents.openrouter.models.uitars.uitars:UITarsAgent",
}

def _load_adapter(path: str) -> Type[MCPAgent]:
    module_name, class_name = path.split(":", 1)
    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        here = Path(__file__).resolve()
        # e.g., models/glm45v/glm45v.py
        parts = module_name.split(".models.")
        if len(parts) == 2:
            rel = parts[1].replace(".", "/") + ".py"
            candidate = here.with_name("openrouter") / "models" / Path(rel)
            if candidate.exists():
                spec = importlib.util.spec_from_file_location("hud.agents._adapter", str(candidate))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return getattr(mod, class_name)
        raise
    return getattr(module, class_name)

class OpenRouterAgent:
    """Dispatch wrapper that selects the correct OpenRouter adapter by model."""

    def __init__(self, *, model_name: str = "z-ai/glm-4.5v", **kwargs: Any) -> None:
        normalized = self._normalize_model_name(model_name)
        try:
            adapter_path = _ADAPTER_REGISTRY[normalized]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported OpenRouter model: {model_name}") from exc

        adapter_cls = _load_adapter(adapter_path)
        canonical_model = f"openrouter/{normalized}"
        self.model_name = canonical_model
        self._adapter = adapter_cls(model_name=canonical_model, **kwargs)

    @staticmethod
    def _normalize_model_name(raw_model: str | None) -> str:
        if not raw_model:
            raise ValueError("Model name must be provided for OpenRouterAgent")
        key = raw_model.strip()
        if key.startswith("openrouter/"):
            key = key[len("openrouter/") :]
        key = key.lower()
        if key in _ADAPTER_REGISTRY:
            return key
        raise ValueError(f"Unknown OpenRouter model: {raw_model}")

    def __getattr__(self, item: str) -> Any: 
        return getattr(self._adapter, item)

    def __dir__(self) -> list[str]: 
        base_dir = set(super().__dir__())
        base_dir.update(self.__dict__.keys())
        base_dir.update(dir(self._adapter))
        return sorted(base_dir)

__all__ = [
    "OpenRouterAgent",
    "OpenRouterBaseAgent",
    "_random_id",
    "_make_reasoning_item",
    "_make_output_text_item",
    "_make_computer_call_item",
    "_make_click_item",
    "_make_double_click_item",
    "_make_drag_item",
    "_make_keypress_item",
    "_make_type_item",
    "_make_scroll_item",
    "_make_screenshot_item",
    "_make_failed_tool_call_items",
    "_coerce_to_pixel_coordinates",
    "_parse_coordinate_box",
    "_coerce_box_to_pixels",
    "_parse_json_action_string",
    "_convert_json_action_to_items",
    "_decode_image_dimensions",
    "_extract_user_instruction",
    "get_last_image_from_messages",
]
