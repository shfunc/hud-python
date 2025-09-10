"""Utility functions for RL training."""

import os
import io
import base64
import random
import json
from typing import Any, List, Union
from pathlib import Path
from PIL import Image
import torch
from transformers.utils.chat_template_utils import render_jinja_template
import bisect
from hud.datasets import Task
from .types import Episode, TrainingSample


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_chat_template(path: str):
    """Load chat template from file."""
    with open(path, "r") as f:
        return f.read()

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_tasks(tasks_input: Union[str, List[dict]], system_prompt: str = None) -> List[Task]:
    """Load tasks from various sources.
    
    Args:
        tasks_input: Either:
            - Path to a JSONL file
            - HuggingFace dataset name (format: "username/dataset" or "username/dataset:split")
            - List of task dictionaries
        system_prompt: Default system prompt to use if not specified in task
    
    Returns:
        List of validated HUD Task objects
    """
    tasks = []
    
    if isinstance(tasks_input, list):
        # Direct list of task dicts
        print(f"Loading {len(tasks_input)} tasks from provided list")
        for item in tasks_input:
            task = Task(
                id=item.get("id"),
                prompt=item["prompt"],
                mcp_config=item["mcp_config"],
                setup_tool=item.get("setup_tool"),
                evaluate_tool=item.get("evaluate_tool"),
                system_prompt=item.get("system_prompt", system_prompt),
                metadata=item.get("metadata", {})
            )
            tasks.append(task)
    
    elif isinstance(tasks_input, str):
        # Check if it's a file path
        if Path(tasks_input).exists():
            print(f"Loading tasks from file: {tasks_input}")
            with open(tasks_input) as f:
                for line in f:
                    item = json.loads(line.strip())
                    task = Task(
                        id=item.get("id"),
                        prompt=item["prompt"],
                        mcp_config=item["mcp_config"],
                        setup_tool=item.get("setup_tool"),
                        evaluate_tool=item.get("evaluate_tool"),
                        system_prompt=item.get("system_prompt", system_prompt),
                        metadata=item.get("metadata", {})
                    )
                    tasks.append(task)
        
        # Check if it's a HuggingFace dataset
        elif "/" in tasks_input:
            print(f"Loading tasks from HuggingFace dataset: {tasks_input}")
            try:
                from datasets import load_dataset
                
                # Parse dataset name and optional split
                if ":" in tasks_input:
                    dataset_name, split = tasks_input.split(":", 1)
                else:
                    dataset_name = tasks_input
                    split = "train"  # Default split
                
                dataset = load_dataset(dataset_name, split=split)
                
                # Convert dataset rows to Task objects
                for item in dataset:
                    # Handle different possible field names in HF datasets
                    task_id = item.get("id") or item.get("task_id") or None
                    prompt = item.get("prompt") or item.get("instruction") or item.get("question")
                    mcp_config = item.get("mcp_config") or {"local": {"command": "echo", "args": ["No MCP config provided"]}}
                    
                    task = Task(
                        id=task_id,
                        prompt=prompt,
                        mcp_config=mcp_config,
                        setup_tool=item.get("setup_tool"),
                        evaluate_tool=item.get("evaluate_tool"),
                        system_prompt=item.get("system_prompt", system_prompt),
                        metadata=item.get("metadata", {})
                    )
                    tasks.append(task)
                    
            except ImportError:
                raise ImportError("Please install 'datasets' package to load from HuggingFace: pip install datasets")
            except Exception as e:
                raise ValueError(f"Failed to load HuggingFace dataset '{tasks_input}': {e}")
        
        else:
            raise ValueError(f"Invalid tasks input: '{tasks_input}' is neither a file path nor a HuggingFace dataset")
    
    else:
        raise TypeError(f"tasks_input must be str or list, got {type(tasks_input)}")
    
    print(f"Loaded {len(tasks)} tasks")
    return tasks


def b64_to_pil(b64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def build_assistant_masks(
    input_ids: list[list[int]], 
    tokenizer: Any, 
    debug: bool = False
) -> list[list[int]]:
    """
    Build assistant masks from token IDs by finding assistant turns.
    
    Args:
        input_ids: List of token sequences
        tokenizer: Tokenizer to decode tokens and get special token IDs
        debug: Whether to print debug information
        
    Returns:
        List of binary masks indicating assistant tokens
    """
    id_im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    id_im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    id_assistant = tokenizer.convert_tokens_to_ids("assistant")

    assistant_masks: list[list[int]] = []

    for seq_idx, seq in enumerate(input_ids):
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
                turn_start = i_tok
                
                # Skip '<|im_start|>', 'assistant' and possible newline token
                i_tok += 2
                # Check for newline after 'assistant'
                if i_tok < len(seq) and tokenizer.decode([seq[i_tok]]) == '\n':
                    i_tok += 1
                
                # Skip leading spaces after assistant\n
                while i_tok < len(seq) and tokenizer.decode([seq[i_tok]]).strip() == '':
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
                    if mask[content_end - 1] == 1 and tokenizer.decode([seq[content_end - 1]]).strip() == '':
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


def prepare_conversation_history(conversation_history: list[dict[str, Any]]) -> (list[dict[str, Any]], list[Image.Image]):
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

def prepare_training_samples(
    episode: Episode,
    processor,
    learner,
    config
) -> list[TrainingSample]:
    """
    Prepare training samples from an episode.
    
    Args:
        episode: Episode to process
        processor: Model processor
        learner: Learner instance (for computing logprobs)
    
    Returns:
        List of training samples
    """
    if len(episode.conversation_history) == 0:
        return []
    
    samples = []

    # Get images for current turn
    conversation, images = prepare_conversation_history(episode.conversation_history)

    # Get absolute path to chat template
    chat_template_path = Path(__file__).parent / "chat_template.jinja"
    
    text_list, generation_indices = render_jinja_template(
        conversations=[conversation],
        chat_template=load_chat_template(str(chat_template_path)),
        tools=episode.tool_spec if episode.tool_spec else None, # mcp_tools
        return_assistant_tokens_mask=True,
        **processor.tokenizer.special_tokens_map,
    )
    inputs = processor(
        images=images if len(images) > 0 else None,
        text=text_list,
        return_offsets_mapping=False,  # we no longer need char offsets
    )

    # Build assistant masks from token IDs
    input_ids = inputs["input_ids"]  # list of lists (length 1 batch)
    assistant_masks = build_assistant_masks(input_ids, processor.tokenizer, config.debug)
    inputs["assistant_masks"] = assistant_masks
    inputs.convert_to_tensors(tensor_type="pt")

    mask_tensor = inputs["assistant_masks"]  # shape [B, T]
    if mask_tensor.dim() == 1:
        mask_tensor = mask_tensor.unsqueeze(0)

    # logits_to_keep are positions where previous token (label axis) is assistant
    logits_to_keep = (mask_tensor[0, 1:] == 1).nonzero(as_tuple=True)[0]
    inputs["logits_to_keep"] = logits_to_keep
    inputs = {k: v.to(learner.device) for k, v in inputs.items()}
        
    samples.append(TrainingSample(
        inputs=inputs,
        advantage=0.0,  # Will be set during training
        old_logprobs=None, # Will be set during training
        ref_logprobs=None, # Will be set during training
    ))
    
    return samples


# def concat_training_samples(
#     samples: list[TrainingSample],
# ) -> TrainingSample:
#     """Concatenate training samples from a list of episodes."""
#     sample = TrainingSample(
#         inputs={},
#         advantage=torch.tensor([]),
#         old_logprobs=torch.tensor([]),
#         ref_logprobs=torch.tensor([]),
#         weight=torch.tensor([]),
#     )
#     # apply padding to all inputs and make it so that the length of the inputs is the same for all samples
#     max_length = max(len(s.inputs["input_ids"][0]) for s in samples)
#     for s in samples:
#         for k in ["input_ids", "assistant_masks", "attention_mask"]:
#             s.inputs[k] = torch.nn.functional.pad(s.inputs[k], (0, max_length - s.inputs[k].shape[-1]))
    
#     sample.inputs = {k: torch.stack([s.inputs[k] for s in samples]) for k in samples[0].inputs}
#     sample.advantage = torch.stack([s.advantage for s in samples])
#     sample.old_logprobs = torch.stack([s.old_logprobs for s in samples])
#     sample.ref_logprobs = torch.stack([s.ref_logprobs for s in samples])
#     sample.weight = torch.stack([s.weight for s in samples])
#     return sample


def compute_format_penalty(episode: Episode, penalty: float = -1.0) -> float:
    """
    Compute format penalty for episodes with errors.
    
    Args:
        episode: Episode to check
        penalty: Penalty value for errors
    
    Returns:
        Total format penalty
    """
    total_penalty = 0.0
    
    # Check for errors in episode info
    if "error" in episode.info:
        total_penalty += penalty
    
    # Check conversation history for tool errors
    for msg in episode.conversation_history:
        if msg.get("role") == "tool" and "error" in msg.get("content", "").lower():
            total_penalty += penalty * 0.5
    
    return total_penalty