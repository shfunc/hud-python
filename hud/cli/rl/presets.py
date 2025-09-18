"""Training configuration presets for different GPU configurations."""

from __future__ import annotations

from typing import Any


def get_training_presets(gpu_memory_gb: float) -> list[dict[str, Any]]:
    """Get training configuration presets based on GPU memory."""
    # Time estimates based on provided benchmarks
    if gpu_memory_gb >= 40:  # A100 40GB or better
        presets = [
            {
                "name": "More Steps",
                "max_steps_per_episode": 12,
                "mini_batch_size": 1,
                "group_size": 4,
                "batch_size": 8,
                "max_new_tokens": 256,
                "tasks_per_hour": 847,
                "steps_per_hour": 424,
                "lr": 3e-5,
                "epochs": 2,
            },
            {
                "name": "Balanced (Recommended)",
                "max_steps_per_episode": 5,
                "mini_batch_size": 1,
                "group_size": 6,
                "batch_size": 12,
                "max_new_tokens": 1024,
                "tasks_per_hour": 738,
                "steps_per_hour": 415,
                "lr": 3e-5,
                "epochs": 2,
            },
            {
                "name": "Low Variance",
                "max_steps_per_episode": 3,
                "mini_batch_size": 2,
                "group_size": 8,
                "batch_size": 16,
                "max_new_tokens": 512,
                "tasks_per_hour": 900,
                "steps_per_hour": 450,
                "lr": 3e-5,
                "epochs": 2,
            },
        ]
    elif gpu_memory_gb >= 24:  # RTX 4090, A10, etc
        presets = [
            {
                "name": "Balanced (Recommended)",
                "max_steps_per_episode": 4,
                "mini_batch_size": 1,
                "group_size": 4,
                "batch_size": 16,
                "lr": 1e-4,
                "epochs": 2,
            },
            {
                "name": "Low Variance",
                "max_steps_per_episode": 3,
                "mini_batch_size": 2,
                "group_size": 4,
                "batch_size": 16,
                "lr": 5e-5,
                "epochs": 2,
            },
        ]
    else:  # Smaller GPUs
        presets = [
            {
                "name": "Test",
                "max_steps_per_episode": 5,
                "mini_batch_size": 1,
                "group_size": 4,
                "batch_size": 8,
                "lr": 1e-4,
                "epochs": 1,
            },
        ]

    return presets


def estimate_memory_usage(
    mini_batch_size: int, max_steps: int, max_new_tokens: int, max_pixels: int
) -> float:
    """Calculate estimated GPU memory usage using the formula from train.py."""
    INITIAL_MEMORY = 8.0
    SCALING_FACTOR = 4 / (28 * 28 * 256 * 1024)
    token_estimate = mini_batch_size * max_steps * max_new_tokens
    image_estimate = max_pixels
    total_memory = INITIAL_MEMORY + SCALING_FACTOR * token_estimate * image_estimate
    return total_memory
