"""GPU detection and validation utilities for RL training."""

from __future__ import annotations

import subprocess
from typing import Any


def detect_cuda_devices() -> dict[str, Any]:
    """Detect available CUDA devices and their properties."""
    try:
        # Check if CUDA is available
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )

        if result.returncode != 0:
            return {"available": False, "error": "nvidia-smi command failed"}

        devices = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split(", ")
            if len(parts) >= 3:
                devices.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_gb": float(parts[2]) / 1024,  # Convert MB to GB
                    }
                )

        return {"available": True, "devices": devices}

    except FileNotFoundError:
        return {
            "available": False,
            "error": "nvidia-smi not found - CUDA drivers may not be installed",
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def select_gpu_for_vllm(devices: list[dict[str, Any]]) -> int:
    """Select the best GPU for vLLM server (typically GPU 1 if available)."""
    if len(devices) > 1:
        # Prefer GPU 1 for vLLM to leave GPU 0 for other processes
        return 1
    return 0


def validate_gpu_memory(gpu_memory_gb: float, model_size: str = "3B") -> bool:
    """Validate if GPU has sufficient memory for the model."""
    min_memory_requirements = {
        "3B": 12.0,  # Minimum for Qwen 2.5 VL 3B
        "7B": 24.0,
        "14B": 40.0,
    }

    min_required = min_memory_requirements.get(model_size, 12.0)
    return gpu_memory_gb >= min_required
