"""Utility functions for RL training."""

from __future__ import annotations

import base64
import io
import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from transformers.utils.chat_template_utils import render_jinja_template

from hud.utils.hud_console import HUDConsole

from .types import TrainingSample

if TYPE_CHECKING:
    from hud.types import Trace

    from .config import Config

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
        import nvidia_ml_py as nvml  # type: ignore

        nvml.nvmlInit()
        device_id = torch.cuda.current_device()
        handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        # Fallback: estimate based on memory usage
        # This is less accurate but works without nvidia-ml-py
        return min(100.0, (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100)


def aggregate_metrics_across_ranks(
    metrics: Any, metrics_to_aggregate: list[str] | None = None
) -> None:
    """Aggregate metrics across all ranks for proper distributed statistics.

    Args:
        metrics: TrainingMetrics object to update in-place
        metrics_to_aggregate: List of metric names to aggregate. If None, aggregates all numeric metrics.

    This function:
    1. Gathers metric values from all ranks
    2. Computes proper mean/std across all GPUs
    3. Updates the metrics object in-place (only on rank 0)
    """  # noqa: E501
    from hud.rl.distributed import get_local_rank, get_world_size, is_main_process

    if get_world_size() <= 1:
        return  # Nothing to aggregate in single GPU mode

    # Default metrics that typically vary across GPUs
    if metrics_to_aggregate is None:
        metrics_to_aggregate = [
            "training_time",
            "samples_per_second",
            "gpu_util",
            "gpu_memory",
            "grad_norm",
            # Include core training scalars
            "loss",
            "kl",
            "entropy",
            "tokens",
            "policy_ratio",
        ]

    # Collect current values from this rank
    local_values = {}
    for metric_name in metrics_to_aggregate:
        if hasattr(metrics, metric_name):
            metric_obj = getattr(metrics, metric_name)
            # Get the last value if available, otherwise 0
            local_values[metric_name] = metric_obj.values[-1] if metric_obj.values else 0.0

    # Convert to tensor for distributed gathering
    values_tensor = torch.tensor(
        list(local_values.values()), device=f"cuda:{get_local_rank()}", dtype=torch.float32
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

            # Replace last value with cross-rank mean for reporting
            if len(metric_obj.values) == 0:
                metric_obj.values.append(0.0)
            metric_obj.values[-1] = float(sum(gpu_values) / len(gpu_values))
            # Recompute mean/std across history using updated last value
            metric_obj.mean = float(sum(metric_obj.values) / len(metric_obj.values))
            variance = sum((x - metric_obj.mean) ** 2 for x in metric_obj.values) / len(
                metric_obj.values
            )
            metric_obj.std = float(variance**0.5)


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
                    if (
                        mask[content_end - 1] == 1
                        and tokenizer.decode([seq[content_end - 1]]).strip() == ""
                    ):
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
    conversation_history: list[dict[str, Any]],
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
                    tc.model_dump() if not isinstance(tc, dict) else tc
                    for tc in m.get("tool_calls", [])
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
                    elif isinstance(data, bytes | bytearray):
                        images.append(Image.open(io.BytesIO(data)).convert("RGB"))
                    c = {"type": "image"}
            m["content"] = user_content
        sanitized_messages.append(m)
    return sanitized_messages, images


def prepare_inputs(trace: Trace, processor: Any) -> dict[str, torch.Tensor]:
    """
    Prepare inputs from a trace.

    Args:
        trace: Trace to process
        processor: Model processor

    Returns:
        Inputs for the model
    """
    if len(trace.messages) == 0:
        return {}

    # Get images for current turn
    conversation, images = prepare_conversation_history(trace.messages)

    # Get absolute path to chat template
    chat_template_path = Path(__file__).parent / "chat_template.jinja"

    # For VL models, processor has a tokenizer attribute; for text models, processor IS tokenizer
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    text_list, _ = render_jinja_template(
        conversations=[conversation],
        chat_template=load_chat_template(str(chat_template_path)),
        tools=trace.info["tool_spec"] if trace.info["tool_spec"] else None,  # mcp_tools
        return_assistant_tokens_mask=True,
        **tokenizer.special_tokens_map,
    )
    # For text models, don't pass images parameter
    if hasattr(processor, "tokenizer"):
        # VL model - processor accepts images
        inputs = processor(
            images=images if len(images) > 0 else None,
            text=text_list,
            return_offsets_mapping=False,  # we no longer need char offsets
        )
    else:
        # Text model - processor is tokenizer, doesn't accept images
        inputs = processor(
            text=text_list,
            return_offsets_mapping=False,  # we no longer need char offsets
        )

    assistant_masks = build_assistant_masks(inputs["input_ids"], tokenizer)
    mask_tensor = torch.tensor(assistant_masks, dtype=torch.long)

    # Ensure mask_tensor is 2D before slicing
    if mask_tensor.dim() == 1:
        mask_tensor = mask_tensor.unsqueeze(0)

    # Slice to align with targets [B, T-1]
    inputs["assistant_mask"] = mask_tensor[:, 1:].bool()

    # Log amount of assistant tokens, and the first 10 tokens that are non 0, decoded
    # assistant_batches = render_assistant_tokens(mask_tensor, inputs['input_ids'], processor)
    inputs.convert_to_tensors(tensor_type="pt")

    return inputs


def render_assistant_tokens(
    mask_tensor: torch.Tensor, input_ids: torch.Tensor, processor: Any
) -> list[str]:
    """Render assistant tokens as a list of continuous batches."""
    # Get the mask as a 1D tensor
    mask_1d = mask_tensor[0]

    # Find continuous sequences of non-zero values
    batches = []
    start_idx = None

    for i in range(len(mask_1d)):
        if mask_1d[i] != 0 and start_idx is None:
            # Start of a new batch
            start_idx = i
        elif mask_1d[i] == 0 and start_idx is not None:
            # End of current batch
            # Extract and decode the tokens in this batch
            batch_token_ids = input_ids[0][start_idx:i].tolist()
            decoded_batch = processor.decode(batch_token_ids)
            batches.append(decoded_batch)
            start_idx = None

    # Handle case where the last batch extends to the end
    if start_idx is not None:
        batch_token_ids = input_ids[0][start_idx:].tolist()
        decoded_batch = processor.decode(batch_token_ids)
        batches.append(decoded_batch)

    return batches


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Calculate entropy from logits in a memory-efficient way."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)
    return entropy


def preprocess_advantages(group: list[Trace], config: Config) -> list[TrainingSample]:
    """Preprocess a group of traces."""
    group_size = config.training.group_size
    if config.training.batch_level == "group":
        groups = [group[i : i + group_size] for i in range(0, len(group), group_size)]
    elif config.training.batch_level == "batch":
        groups = [group]
    else:
        raise ValueError(f"Invalid batch level: {config.training.batch_level}")

    all_samples = []
    for i, group in enumerate(groups):
        rewards = np.array([trace.reward for trace in group])
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        # Calculate advantages
        samples = [TrainingSample(**trace.model_dump()) for trace in group]
        for sample, reward in zip(samples, rewards, strict=True):
            if sample.isError:
                sample.advantage = torch.Tensor(np.array([0.0]))
                continue
            # No std (non-baseline GRPO)
            if config.training.no_std:
                advantage_value = reward - mean_reward
            else:
                # Avoid division by zero
                if std_reward < 1e-6:
                    advantage_value = torch.Tensor(np.array([0.0]))
                else:
                    advantage_value = (reward - mean_reward) / std_reward
            # Leave one out RLOO/LOOP
            if config.training.leave_one_out:
                advantage_value = advantage_value * len(group) / (len(group) - 1)
            sample.advantage = torch.Tensor(np.array([advantage_value]))
        hud_console.info_log(
            f"Advantages for group {i} [{mean_reward:.4f} ± {std_reward:.4f}]:"
            f"{[round(sample.advantage.item(), 4) for sample in samples if sample.advantage is not None]}"  # noqa: E501
        )

        all_samples.extend(samples)

    return all_samples


def batch_training_samples(samples: list[TrainingSample]) -> list[TrainingSample]:
    """Create batched model inputs from a list of TrainingSample.

    Pads token sequences to the maximum length in the list and zero-pads
    images to the maximum H/W when present. Returns a dictionary of batched
    tensors suitable for a single forward pass. Keeps assistant_masks for
    masked scoring.
    """
    if not samples:
        hud_console.warning("No samples to batch.")
        return []

    for s in samples:
        if (
            "assistant_mask" not in s.inputs
            or s.inputs["assistant_mask"].sum() == 0
            or s.advantage == 0.0
        ) and len(samples) > 1:
            hud_console.info("Removing sample with zero advantage.")
            samples.remove(s)

    if len(samples) == 1:
        return samples

    import torch.nn.functional as F

    new_samples = [TrainingSample()]

    input_keys_to_expand = ["input_ids", "attention_mask", "assistant_mask"]
    input_keys_to_cat = ["pixel_values", "image_grid_thw"]
    updated_inputs: dict[str, list[torch.Tensor]] = {
        k: [] for k in input_keys_to_expand + input_keys_to_cat
    }

    # Sanity check dimensions
    for s in samples:
        for k in input_keys_to_expand + input_keys_to_cat:
            val = s.inputs.get(k)
            if val is not None:
                if k in input_keys_to_expand:
                    if val.dim() == 2 and val.size(0) == 1:
                        val = val[0]
                    elif val.dim() != 1:
                        raise ValueError(f"{k} has unexpected dimensions: {val.shape}")
                updated_inputs[k].append(val)

    # Pad 1D sequences to max length
    max_len = max(t.size(-1) for t in updated_inputs["input_ids"])

    def pad_1d(x: torch.Tensor, pad_to: int, pad_value: int) -> torch.Tensor:
        pad = pad_to - x.size(-1)
        return F.pad(x, (0, pad), value=pad_value) if pad > 0 else x

    stacked_inputs: dict[str, torch.Tensor] = {}
    # These are 1D sequences that need padding
    for k in input_keys_to_expand:
        if updated_inputs[k]:
            # assistant_mask is T-1, others are T
            if k == "assistant_mask":
                stacked_inputs[k] = torch.stack(
                    [pad_1d(x, max_len - 1, 0) for x in updated_inputs[k]], dim=0
                )
            else:
                stacked_inputs[k] = torch.stack(
                    [pad_1d(x, max_len, 0) for x in updated_inputs[k]], dim=0
                )

    for k in input_keys_to_cat:
        if updated_inputs[k]:
            # pixel_values and image_grid_thw are concatenated across all images from all samples
            # Shape of pixel_values: (sum of all patches from all images, feature_dim)
            # Shape of image_grid_thw: (sum of all images, 3)
            stacked_inputs[k] = torch.cat(updated_inputs[k], dim=0)
        else:
            stacked_inputs.pop(k)

    new_samples[0].inputs = stacked_inputs

    # Pad logprobs to max length before stacking
    # old_logprobs and ref_logprobs have shape [seq_len] or [1, seq_len] after gathering
    def pad_logprobs(logprobs: torch.Tensor | None, max_len: int) -> torch.Tensor:
        # Always work with 1D tensor, squeeze batch dim if present
        if logprobs is None:
            return torch.tensor([float("-inf")], dtype=torch.float32)
        if logprobs.dim() == 2 and logprobs.size(0) == 1:
            logprobs = logprobs.squeeze(0)
        elif logprobs.dim() != 1:
            raise ValueError(
                f"Expected logprobs to have 1 or 2 dimensions, got {logprobs.dim()} with shape {logprobs.shape}"  # noqa: E501
            )

        # Now logprobs is [seq_len]
        seq_len = logprobs.size(0) if logprobs is not None else 0
        if seq_len < max_len:
            pad_size = max_len - seq_len
            # Pad with -inf (log of 0 probability) along sequence dimension
            return F.pad(logprobs, (0, pad_size), value=float("-inf"))
        return logprobs

    # Stack padded logprobs (these are T-1 length)
    old_logprobs_list = [pad_logprobs(s.old_logprobs, max_len - 1) for s in samples]
    ref_logprobs_list = [pad_logprobs(s.ref_logprobs, max_len - 1) for s in samples]

    new_samples[0].old_logprobs = torch.stack(old_logprobs_list, dim=0)
    new_samples[0].ref_logprobs = torch.stack(ref_logprobs_list, dim=0)

    # Stack advantages, checking for None values
    advantages = [s.advantage for s in samples]
    if any(adv is None for adv in advantages):
        raise ValueError(
            "Some samples have None advantages. Make sure advantages are computed before batching."
        )
    new_samples[0].advantage = torch.stack(advantages, dim=0)  # type: ignore

    return new_samples
