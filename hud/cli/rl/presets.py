"""Training configuration presets for different GPU configurations."""

from typing import List, Dict, Any


def get_training_presets(gpu_memory_gb: float) -> List[Dict[str, Any]]:
    """Get training configuration presets based on GPU memory."""
    # Time estimates based on provided benchmarks
    SEC_INIT = 20
    SEC_STEP = 2
    SEC_START = 10
    SEC_GRAD = 7
    
    if gpu_memory_gb >= 40:  # A100 40GB or better
        presets = [
            {
                "name": "More Steps",
                "max_steps_per_episode": 12,
                "mini_batch_size": 1,
                "group_size": 4,
                "episodes_per_batch": 32,
                "tasks_per_hour": 847,
                "steps_per_hour": 424,
                "lr": 1e-5,
                "epochs": 2,
            },
            {
                "name": "Balanced (Recommended)",
                "max_steps_per_episode": 6,
                "mini_batch_size": 2,
                "group_size": 6,
                "episodes_per_batch": 16,
                "tasks_per_hour": 738,
                "steps_per_hour": 415,
                "lr": 2e-5,
                "epochs": 2,
            },
            {
                "name": "Low Variance",
                "max_steps_per_episode": 3,
                "mini_batch_size": 4,
                "group_size": 8,
                "episodes_per_batch": 32,
                "tasks_per_hour": 900,
                "steps_per_hour": 450,
                "lr": 1e-4,
                "epochs": 3,
            },
        ]
    elif gpu_memory_gb >= 24:  # RTX 4090, A10, etc
        presets = [
            {
                "name": "Test",
                "max_steps_per_episode": 10,
                "mini_batch_size": 1,
                "group_size": 4,
                "episodes_per_batch": 16,
                "lr": 1e-4,
                "epochs": 2,
            },
            {
                "name": "Balanced (Recommended)",
                "max_steps_per_episode": 5,
                "mini_batch_size": 2,
                "group_size": 4,
                "episodes_per_batch": 16,
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
                "group_size": 2,
                "episodes_per_batch": 8,
                "lr": 1e-4,
                "epochs": 1,
            },
        ]
    
    return presets


def estimate_memory_usage(mini_batch_size: int, max_steps: int, max_pixels: int) -> float:
    """Calculate estimated GPU memory usage using the formula from train.py."""
    INITIAL_MEMORY = 8.0
    SCALING_FACTOR = 5.5
    constant = mini_batch_size * max_steps
    quadratic = (max_pixels / (28 * 28 * 256)) ** 2
    return INITIAL_MEMORY + SCALING_FACTOR * constant * quadratic
