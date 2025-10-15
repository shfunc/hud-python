"""UITARS adapter rebuilt using official parser utilities"""

from __future__ import annotations

import ast
import base64
import logging
import math
import os
import re
from io import BytesIO
from typing import Any, ClassVar

from PIL import Image

import litellm
import mcp.types as types

from hud import instrument
from hud.agents.base import MCPAgent
from hud.agents.openrouter import (
    _convert_json_action_to_items,
    _decode_image_dimensions,
    _extract_user_instruction,
    _make_failed_tool_call_items,
    _make_screenshot_item,
    _random_id,
    get_last_image_from_messages,
)
from hud.tools.computer.settings import computer_settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult

logger = logging.getLogger(__name__)

# Constants from the official UITARS parser
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
def _resolve_provider_model_name(model_name: str) -> str:
    key = (model_name or "").strip()
    if key.startswith("openrouter/"):
        key = key[len("openrouter/") :]
    lowered = key.lower()
    if lowered in {"huggingface/bytedance-seed/ui-tars-1.5-7b", "bytedance-seed/ui-tars-1.5-7b"}:
        return "ByteDance-Seed/UI-TARS-1.5-7B"
    return key

COMPUTER_USE_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

def convert_point_to_coordinates(text: str, is_answer: bool = False) -> str:
    pattern = r"<point>(\d+)\s+(\d+)</point>"

    def replace_match(match: re.Match[str]) -> str:
        x1, y1 = map(int, match.groups())
        x = (x1 + x1) // 2
        y = (y1 + y1) // 2
        return f"({x},{y})"

    text = re.sub(r"\[EOS\]", "", text)
    return re.sub(pattern, replace_match, text).strip()


def parse_action(action_str: str) -> dict[str, Any] | None:
    try:
        node = ast.parse(action_str, mode="eval")
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")
        call = node.body
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        kwargs: dict[str, Any] = {}
        for kw in call.keywords:
            key = kw.arg
            if key is None:
                # Skip unpacked kwargs like **extra
                continue
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # compatibility
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {"function": func_name, "args": kwargs}
    except Exception as exc:
        logger.debug("Failed to parse action '%s': %s", action_str, exc)
        return None

def escape_single_quotes(text: str) -> str:
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(int(height / beta), factor)
        w_bar = floor_by_factor(int(width / beta), factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)
    return h_bar, w_bar


def _preprocess_text_for_parsing(text: str) -> str:
    if "<point>" in text:
        text = convert_point_to_coordinates(text)
    if "start_point=" in text:
        text = text.replace("start_point=", "start_box=")
    if "end_point=" in text:
        text = text.replace("end_point=", "end_box=")
    if "point=" in text:
        text = text.replace("point=", "start_box=")
    return text


def parse_action_to_structure_output(
    text: str,
    factor: int,
    origin_resized_height: int,
    origin_resized_width: int,
    model_type: str = "qwen25vl",
    max_pixels: int = MAX_PIXELS,
    min_pixels: int = MIN_PIXELS,
) -> list[dict[str, Any]]:
    text = _preprocess_text_for_parsing(text.strip())

    # Thought/Action extraction
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action: |$)"
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action: |$)"
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"

    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()

    if "Action:" not in text:
        return []
    action_str_full = text.split("Action: ")[-1]

    # Split multiple actions if present (rare; we expect exactly one)
    raw_actions: list[str] = []
    for seg in action_str_full.split(")\n\n"):
        act = seg.strip()
        if not act:
            continue
        if not act.endswith(")"):
            act += ")"
        # Handle type(content='...') with quotes inside
        if "type(content" in act:
            def _unbox(m: re.Match[str]) -> str:
                return m.group(1)
            pat = r"type\(content='(.*?)'\)"
            if re.search(pat, act):
                inner = re.sub(pat, _unbox, act)
                inner = escape_single_quotes(inner)
                act = "type(content='" + inner + "')"
        raw_actions.append(act)

    parsed_actions = [parse_action(a.replace("\n", "\\n").lstrip()) for a in raw_actions]

    actions: list[dict[str, Any]] = []
    for action_instance, raw_str in zip(parsed_actions, raw_actions):
        if not action_instance:
            raise ValueError(f"Action can't parse: {raw_str}")
        action_type = action_instance["function"]
        params = action_instance["args"]

        action_inputs: dict[str, Any] = {}
        for param_name, param in params.items():
            if param == "":
                continue
            if isinstance(param, str):
                param = param.lstrip()
            action_inputs[param_name.strip()] = param

            if "start_box" in param_name or "end_box" in param_name:
                ori_box = str(param)
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                # qwen25vl branch -> absolute pixel coords relative to processed dims -> normalize to 0..1f
                if model_type == "qwen25vl":
                    float_numbers: list[float] = []
                    for idx, num in enumerate(numbers):
                        val = float(num)
                        if (idx + 1) % 2 == 0:
                            float_numbers.append(val / float(origin_resized_height or 1))
                        else:
                            float_numbers.append(val / float(origin_resized_width or 1))
                else:
                    # Otherwise assume factor-based normalization (e.g., 1000)
                    float_numbers = [float(num) / float(factor or 1) for num in numbers]

                if len(float_numbers) == 2:
                    float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                action_inputs[param_name.strip()] = str(float_numbers)

        actions.append(
            {
                "reflection": reflection,
                "thought": thought,
                "action_type": action_type,
                "action_inputs": action_inputs,
                "text": text,
            }
        )
    return actions

def _pil_to_data_uri(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def _resize_for_model(img: Image.Image) -> tuple[Image.Image, int, int]:
    w, h = img.size
    new_h, new_w = smart_resize(h, w)
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img, new_w, new_h

def _format_action_to_doubao_string(action: dict[str, Any], width: int, height: int) -> str | None:
    a_type = (action.get("type") or "").lower()
    
    if a_type in {"click", "left_click"}:
        x = action.get("x", 0)
        y = action.get("y", 0)
        return f"click(start_box='({x},{y})')"
    elif a_type == "double_click":
        x = action.get("x", 0)
        y = action.get("y", 0)
        return f"left_double(start_box='({x},{y})')"
    elif a_type == "drag":
        path = action.get("path", [])
        if len(path) >= 2:
            sx, sy = path[0].get("x", 0), path[0].get("y", 0)
            ex, ey = path[-1].get("x", 0), path[-1].get("y", 0)
            return f"drag(start_point='({sx},{sy})', end_point='({ex},{ey})')"
    elif a_type == "keypress":
        keys = " ".join(action.get("keys", []))
        return f"hotkey(key='{keys}')"
    elif a_type == "type":
        content = action.get("text", "")
        return f"type(content='{content}')"
    elif a_type == "scroll":
        x = action.get("x", 0)
        y = action.get("y", 0)
        direction = action.get("scroll_y", 0)
        dir_str = "down" if direction > 0 else "up"
        return f"scroll(point='({x},{y})', direction='{dir_str}')"
    return None 

def _parse_to_json_action(actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not actions:
        return None
    act = actions[0]
    a_type = (act.get("action_type") or "").lower()
    inputs = act.get("action_inputs") or {}

    def _coerce_box(value: Any) -> list[float] | None:
        try:
            if isinstance(value, str):
                nums = re.findall(r"-?\d+(?:\.\d+)?", value)
                if len(nums) >= 2:
                    return [float(nums[0]), float(nums[1])]
            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                return [float(value[0]), float(value[1])]
        except Exception:
            return None
        return None

    if a_type in {"click", "left_single"}:
        sb = _coerce_box(inputs.get("start_box"))
        if sb:
            return {"type": "click", "button": "left", "start_box": sb}
    if a_type in {"left_double", "double_click"}:
        sb = _coerce_box(inputs.get("start_box"))
        if sb:
            return {"type": "double_click", "start_box": sb}
    if a_type in {"right_single", "right_click"}:
        sb = _coerce_box(inputs.get("start_box"))
        if sb:
            return {"type": "click", "button": "right", "start_box": sb}
    if a_type in {"drag", "select", "left_drag"}:
        s = _coerce_box(inputs.get("start_box"))
        e = _coerce_box(inputs.get("end_box"))
        if s and e:
            return {"type": "drag", "start_box": s, "end_box": e}
    if a_type in {"hotkey", "key", "keydown", "keypress"}:
        key_str = inputs.get("key") or inputs.get("hotkey") or inputs.get("keys") or ""
        key_str = str(key_str)
        # Normalize arrow aliases and spacing
        key_str = key_str.replace("arrowleft", "left").replace("arrowright", "right").replace("arrowup", "up").replace("arrowdown", "down")
        keys = [seg for seg in re.split(r"[+\s]+", key_str.strip()) if seg]
        if keys:
            return {"type": "keypress", "keys": keys}
    if a_type == "type":
        content = inputs.get("content", "")
        return {"type": "type", "content": str(content)}
    if a_type == "scroll":
        sb = _coerce_box(inputs.get("start_box"))
        direction = str(inputs.get("direction") or "down").lower()
        if sb:
            return {"type": "scroll", "start_box": sb, "direction": direction}
    if a_type == "wait":
        return {"type": "wait"}
    if a_type == "finished":
        return {"type": "finished", "content": str(inputs.get("content") or "")}
    return None

class UITarsAgent(MCPAgent):
    """UITARS computer-use agent (Doubao-style prompts + official parser)."""

    metadata: ClassVar[dict[str, Any]] = {
        "display_width": computer_settings.OPENAI_COMPUTER_WIDTH,
        "display_height": computer_settings.OPENAI_COMPUTER_HEIGHT,
    }
    required_tools: ClassVar[list[str]] = ["openai_computer"]

    def __init__(self, *, model_name: str = "ByteDance-Seed/UI-TARS-1.5-7B", completion_kwargs: dict[str, Any] | None = None, **agent_kwargs: Any) -> None:
        super().__init__(**agent_kwargs)
        self.model_name = model_name
        self._base_completion_kwargs = dict(completion_kwargs or {})
        # Allow configuring a Hugging Face endpoint via environment
        env_base = os.getenv("HF_ENDPOINT_BASE_URL")
        env_token = os.getenv("HF_ENDPOINT_TOKEN") or os.getenv("HF_API_KEY")
        env_provider = os.getenv("HF_ENDPOINT_PROVIDER")
        if env_base and "api_base" not in self._base_completion_kwargs:
            self._base_completion_kwargs["api_base"] = env_base
        if env_provider:
            self._base_completion_kwargs.setdefault("custom_llm_provider", str(env_provider))
        if env_token and "api_key" not in self._base_completion_kwargs:
            self._base_completion_kwargs["api_key"] = env_token

        # If HF endpoint is configured and provider not set, default to huggingface
        if os.getenv("HF_ENDPOINT_BASE_URL") and "custom_llm_provider" not in self._base_completion_kwargs:
            self._base_completion_kwargs["custom_llm_provider"] = "huggingface"

        self._provider_model = _resolve_provider_model_name(self.model_name)

    async def get_system_messages(self) -> list[Any]:
        return []

    @instrument(span_type="agent", record_args=False)
    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[dict[str, Any]]:
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

    def _tool_call(self, item: dict[str, Any]) -> MCPToolCall:
        call_id = item.get("call_id") or _random_id()
        action = item.get("action") or {}
        return MCPToolCall(id=call_id, name="openai_computer", arguments=action)

    @instrument(span_type="agent", record_args=False)
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

        # Decode original image dims and make a processed copy for the model
        try:
            data = base64.b64decode(screenshot_b64.split(",", 1)[1] if screenshot_b64.startswith("data:image") else screenshot_b64)
            img = Image.open(BytesIO(data))
            orig_w, orig_h = img.size
        except Exception:
            orig_w, orig_h = _decode_image_dimensions(screenshot_b64)
            img = None

        proc_w, proc_h = orig_w, orig_h
        proc_uri = f"data:image/png;base64,{screenshot_b64}"
        if img is not None:
            img, proc_w, proc_h = _resize_for_model(img)
            proc_uri = _pil_to_data_uri(img)

        # Build messages with history: system prompt + previous turns + current screenshot
        system_prompt = COMPUTER_USE_DOUBAO.format(language="English", instruction=instruction or "")
        
        litellm_messages: list[dict[str, Any]] = []
        
        # Add history of previous actions and screenshots
        for msg in messages[:-1]:  # Skip the current screenshot
            if not isinstance(msg, dict):
                continue
            msg_type = msg.get("type")
            
            if msg_type == "computer_call_output":
                output = msg.get("output") or {}
                if isinstance(output, dict) and output.get("type") == "input_image":
                    image_url = output.get("image_url")
                    if image_url:
                        litellm_messages.append({
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": image_url}}]
                        })
            
            elif msg_type == "computer_call":
                action = msg.get("action") or {}
                action_str = _format_action_to_doubao_string(action, proc_w, proc_h)
                if action_str:
                    litellm_messages.append({
                        "role": "assistant",
                        "content": f"Thought: Executing action.\nAction: {action_str}"
                    })
        
        litellm_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": {"url": proc_uri}},
            ],
        })

        api_kwargs: dict[str, Any] = {
            "model": self._provider_model,
            "messages": litellm_messages,
            "temperature": 0.1,
            "max_tokens": 256,
        }
        api_kwargs.update(self._base_completion_kwargs)

        try:
            response = await litellm.acompletion(**api_kwargs)
        except Exception as exc:  # pragma: no cover - network errors
            logger.exception("uitars completion failed: %s", exc)
            return AgentResponse(content=f"UITARS request failed: {exc}", tool_calls=[], done=True, isError=True)

        content = (getattr(response.choices[0], "message", None) or {}).get("content", "")
        
        if content:
            print(f"\n{'='*60}\nUITars output:\n{content}\n{'='*60}\n")

        actions = parse_action_to_structure_output(
            content or "",
            factor=1000,
            origin_resized_height=proc_h,
            origin_resized_width=proc_w,
            model_type="qwen25vl",
        )
        json_action = _parse_to_json_action(actions)

        items: list[dict[str, Any]] = []
        if json_action:
            call_id = _random_id()
            # Feed original dimensions so normalized coords map to real pixels
            items.extend(
                _convert_json_action_to_items(
                    json_action,
                    call_id=call_id,
                    image_width=orig_w,
                    image_height=orig_h,
                )
            )
            
            # Auto-wait after clicking taskbar/launcher icons (left edge, x < 50)
            if items and json_action.get("type") in {"click", "left_click"}:
                x_coord = json_action.get("x", 0)
                if x_coord < 50:  # Likely a launcher icon
                    logger.info("Detected launcher click at x=%d, adding auto-wait", x_coord)
                    items.append({
                        "type": "computer_call",
                        "action": {"type": "screenshot"},
                        "computer_call_id": _random_id(),
                    })
            
            if not items:
                items.append(_make_screenshot_item(call_id))
        else:
            call_id = _random_id()
            items.append(_make_screenshot_item(call_id))

        tool_calls = [self._tool_call(i) for i in items if i.get("type") == "computer_call"]
        return AgentResponse(content=None, tool_calls=tool_calls, done=not tool_calls, raw=response)

    @instrument(span_type="agent", record_args=False)
    async def format_tool_results(self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        for call, result in zip(tool_calls, tool_results, strict=False):
            call_args = call.arguments or {}
            if result.isError:
                error_text = "".join(c.text for c in result.content if isinstance(c, types.TextContent))
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
                            "output": {"type": "input_image", "image_url": f"data:{content.mimeType};base64,{content.data}"},
                        }
                    )
                    screenshot_found = True
                    break

            text_parts = [c.text for c in result.content if isinstance(c, types.TextContent) and c.text]
            if text_parts:
                rendered.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "\n".join(text_parts)}]})

            if not screenshot_found and not text_parts:
                rendered.append({"type": "computer_call_output", "call_id": call.id, "output": {"type": "input_text", "text": "Tool executed"}})

        return rendered


__all__ = ["UITarsAgent"]
