"""Utility functions for RL training."""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers.utils.chat_template_utils import render_jinja_template

from hud.datasets import Task
from hud.types import Trace
from hud.utils.hud_console import HUDConsole

from .config import Config
from .types import TrainingSample

logger = logging.getLogger(__name__)
hud_console = HUDConsole(logger)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_chat_template(path: str) -> str:
    """Load chat template from file."""
    with open(path) as f:
        return f.read()

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def get_memory_usage() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def get_gpu_utilization() -> float:
    """Get current GPU utilization percentage (0-100)."""
    if not torch.cuda.is_available():
        return 0.0
    
    try:
        import nvidia_ml_py as nvml
        nvml.nvmlInit()
        device_id = torch.cuda.current_device()
        handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        # Fallback: estimate based on memory usage
        # This is less accurate but works without nvidia-ml-py
        return min(100.0, (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100)


def aggregate_metrics_across_ranks(metrics: Any, metrics_to_aggregate: list[str] | None = None) -> None:
    """Aggregate metrics across all ranks for proper distributed statistics.
    
    Args:
        metrics: TrainingMetrics object to update in-place
        metrics_to_aggregate: List of metric names to aggregate. If None, aggregates all numeric metrics.
    
    This function:
    1. Gathers metric values from all ranks
    2. Computes proper mean/std across all GPUs
    3. Updates the metrics object in-place (only on rank 0)
    """
    from hud.rl.distributed import get_local_rank, get_world_size, is_main_process
    
    if get_world_size() <= 1:
        return  # Nothing to aggregate in single GPU mode
    
    # Default metrics that typically vary across GPUs
    if metrics_to_aggregate is None:
        metrics_to_aggregate = ["training_time", "samples_per_second", "gpu_util", "gpu_memory", "grad_norm"]
    
    # Collect current values from this rank
    local_values = {}
    for metric_name in metrics_to_aggregate:
        if hasattr(metrics, metric_name):
            metric_obj = getattr(metrics, metric_name)
            # Get the last value if available, otherwise 0
            local_values[metric_name] = metric_obj.values[-1] if metric_obj.values else 0.0
    
    # Convert to tensor for distributed gathering
    values_tensor = torch.tensor(
        list(local_values.values()),
        device=f"cuda:{get_local_rank()}",
        dtype=torch.float32
    )
    
    # Gather from all ranks using NCCL-supported all_gather
    world_size = get_world_size()
    gather_list = [torch.zeros_like(values_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gather_list, values_tensor)

    # Update metrics on main process only
    if is_main_process():
        # Reshape: [num_gpus, num_metrics]
        all_values = torch.stack(gather_list).cpu().numpy()
        
        # Update each metric with aggregated values
        for i, metric_name in enumerate(local_values.keys()):
            metric_obj = getattr(metrics, metric_name)
            gpu_values = all_values[:, i].tolist()
            
            # Replace single value with all GPU values
            metric_obj.values = gpu_values
            metric_obj.mean = float(np.mean(gpu_values))
            metric_obj.std = float(np.std(gpu_values))


def b64_to_pil(b64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def build_assistant_masks(
    input_ids: list[list[int]],
    tokenizer: Any,
) -> list[list[int]]:
    """
    Build assistant masks from token IDs by finding assistant turns.
    
    Args:
        input_ids: List of token sequences
        tokenizer: Tokenizer to decode tokens and get special token IDs
        verbose: Whether to print verbose information
        
    Returns:
        List of binary masks indicating assistant tokens
    """
    id_im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    id_im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    id_assistant = tokenizer.convert_tokens_to_ids("assistant")

    assistant_masks: list[list[int]] = []

    for seq in input_ids:
        mask = [0] * len(seq)
        i_tok = 0
        assistant_turn_count = 0
        
        while i_tok < len(seq):
            # Detect start of assistant turn
            if (
                seq[i_tok] == id_im_start
                and i_tok + 1 < len(seq)
                and seq[i_tok + 1] == id_assistant
            ):
                assistant_turn_count += 1
                
                # Skip '<|im_start|>', 'assistant' and possible newline token
                i_tok += 2
                # Check for newline after 'assistant'
                if i_tok < len(seq) and tokenizer.decode([seq[i_tok]]) == "\n":
                    i_tok += 1
                
                # Skip leading spaces after assistant\n
                while i_tok < len(seq) and tokenizer.decode([seq[i_tok]]).strip() == "":
                    i_tok += 1
                
                assistant_content_start = i_tok
                
                # Mark tokens until we hit <|im_end|>
                content_end = i_tok
                while i_tok < len(seq) and seq[i_tok] != id_im_end:
                    content_end = i_tok + 1  # Track last non-<|im_end|> position
                    mask[i_tok] = 1
                    i_tok += 1
                
                # Remove trailing spaces from the mask
                while content_end > assistant_content_start:
                    if mask[content_end - 1] == 1 and tokenizer.decode([seq[content_end - 1]]).strip() == "":
                        mask[content_end - 1] = 0
                        content_end -= 1
                    else:
                        break
                
                # Skip the <|im_end|> token
                i_tok += 1
            else:
                i_tok += 1

        assistant_masks.append(mask)
        
    return assistant_masks


def prepare_conversation_history(
    conversation_history: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Sanitize conversation history to avoid vLLM errors."""
    sanitized_messages = []
    images = []
    for m in conversation_history:
        if "tool_calls" in m:
            m = {
                "role": m["role"],
                "content": m.get("content", ""),
                "tool_calls": [
                    tc.model_dump() if not isinstance(tc, dict) else tc for tc in m.get("tool_calls", [])
                ],
            }
        elif m.get("role") == "user":
            user_content = m.get("content", [])
            for c in user_content:
                if isinstance(c, dict) and c.get("type") == "image_url":
                    image_url = c.get("image_url", {})
                    url = image_url.get("url", "")
                    if url.startswith("data:image"):
                        data = url.split(",", 1)[1] if "," in url else url
                        images.append(b64_to_pil(data))
                    elif isinstance(data, (bytes, bytearray)):
                        images.append(Image.open(io.BytesIO(data)).convert("RGB"))
                    c = {"type": "image"}
            m["content"] = user_content
        sanitized_messages.append(m)
    return sanitized_messages, images

def prepare_inputs(
    trace: Trace,
    processor: Any
) -> dict[str, torch.Tensor]:
    """
    Prepare inputs from a trace.
    
    Args:
        trace: Trace to process
        processor: Model processor
    
    Returns:
        Inputs for the model
    """
    # Skip error traces or traces with no messages
    if trace.isError or len(trace.messages) == 0:
        return {}

    # Get images for current turn
    conversation, images = prepare_conversation_history(trace.messages)

    # Get absolute path to chat template
    chat_template_path = Path(__file__).parent / "chat_template.jinja"
    
    text_list, _ = render_jinja_template(
        conversations=[conversation],
        chat_template=load_chat_template(str(chat_template_path)),
        tools=trace.info["tool_spec"] if trace.info["tool_spec"] else None, # mcp_tools
        return_assistant_tokens_mask=True,
        **processor.tokenizer.special_tokens_map,
    )
    inputs = processor(
        images=images if len(images) > 0 else None,
        text=text_list,
        return_offsets_mapping=False,  # we no longer need char offsets
    )

    input_ids_list = inputs["input_ids"]
    assistant_masks = build_assistant_masks(input_ids_list, processor.tokenizer)
    inputs.convert_to_tensors(tensor_type="pt")
    mask_tensor = torch.tensor(assistant_masks, dtype=torch.long)
    if mask_tensor.dim() == 1:
        mask_tensor = mask_tensor.unsqueeze(0)

    inputs["logits_to_keep"] = mask_tensor[:, 1:].bool()

    # Log amount of assistant tokens, and the first 10 tokens that are non 0, decoded
    hud_console.info(f"Amount of assistant tokens: {mask_tensor.sum()}")
    hud_console.info(f"Decoded assistant tokens: {processor.tokenizer.decode(mask_tensor[0].nonzero())}")

    return inputs


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy

def preprocess_advantages(group: list[Trace], group_size: int, config: Config) -> list[TrainingSample]:
    """Preprocess a group of traces."""
    if config.training.batch_level == "group":
        groups = [group[i:i+group_size] for i in range(0, len(group), group_size)]
    elif config.training.batch_level == "batch":
        groups = [group]
    else:
        raise ValueError(f"Invalid batch level: {config.training.batch_level}")

    all_samples = []
    for group in groups:
        rewards = np.array([trace.reward for trace in group])
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        # Calculate advantages
        samples = [TrainingSample(**trace.model_dump()) for trace in group]
        for sample, reward in zip(samples, rewards, strict=True):
            if sample.isError:
                sample.advantage = torch.Tensor(0.0)
                continue
            # No std (DR-GRPO)
            if config.training.no_std:
                advantage_value = (reward - mean_reward)
            else:
                # Avoid division by zero
                if std_reward < 1e-6:
                    advantage_value = torch.Tensor(0.0)
                else:
                    advantage_value = ((reward - mean_reward) / std_reward)
            # Leave one out RLOO/LOOP
            if config.training.leave_one_out:
                advantage_value = advantage_value * len(group) / (len(group) - 1)
            sample.advantage = torch.Tensor(advantage_value)
        all_samples.extend(samples)

    return all_samples

def batch_training_samples(samples: list[TrainingSample]) -> TrainingSample:
    """Create batched model inputs from a list of TrainingSample.

    Pads token sequences to the maximum length in the list and zero-pads
    images to the maximum H/W when present. Returns a dictionary of batched
    tensors suitable for a single forward pass. Keeps assistant_masks for
    masked scoring.
    """
    if not samples:
        return {}

    import torch.nn.functional as F
    new_sample = TrainingSample()

    input_keys_to_expand = ["input_ids", "attention_mask", "logits_to_keep", "pixel_values", "image_grid_thw"]
    updated_inputs = {k: [] for k in input_keys_to_expand}

    for s in samples:
        for k in input_keys_to_expand:
            val = s.inputs[k]
            if val is not None:
                if val.dim() >= 2 and val.size(0) == 1:
                    val = val[0]
                updated_inputs[k].append(val)


    # Pad 1D sequences to max length
    max_len = max(t.size(-1) for t in updated_inputs["input_ids"])
    def pad_1d(x: torch.Tensor, pad_to: int, pad_value: int) -> torch.Tensor:
        pad = pad_to - x.size(-1)
        return F.pad(x, (0, pad), value=pad_value) if pad > 0 else x

    for k in input_keys_to_expand:
        updated_inputs[k] = torch.stack([pad_1d(x, max_len, 0) for x in updated_inputs[k]], dim=0)

    new_sample.inputs = updated_inputs

    # Add the logprobs and advantages
    new_sample.old_logprobs = torch.stack([s.old_logprobs for s in samples], dim=0)
    new_sample.ref_logprobs = torch.stack([s.ref_logprobs for s in samples], dim=0)
    new_sample.advantage = torch.stack([s.advantage for s in samples], dim=0)

    return new_sample
