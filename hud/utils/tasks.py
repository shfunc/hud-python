from __future__ import annotations

import json
from pathlib import Path

from hud.types import Task
from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


def load_tasks(tasks_input: str | list[dict]) -> list[Task]:
    """Load tasks from various sources.

    Args:
        tasks_input: Either:
            - Path to a JSON file (array of tasks)
            - Path to a JSONL file (one task per line)
            - HuggingFace dataset name (format: "username/dataset" or "username/dataset:split")
            - List of task dictionaries
        system_prompt: Default system prompt to use if not specified in task

    Returns:
        List of validated HUD Task objects
    """
    tasks = []

    if isinstance(tasks_input, list):
        # Direct list of task dicts
        hud_console.info(f"Loading {len(tasks_input)} tasks from provided list")
        for item in tasks_input:
            task = Task(**item)
            tasks.append(task)

    elif isinstance(tasks_input, str):
        # Check if it's a file path
        if Path(tasks_input).exists():
            file_path = Path(tasks_input)
            hud_console.info(f"Loading tasks from file: {tasks_input}")

            with open(file_path) as f:
                # Handle JSON files (array of tasks)
                if file_path.suffix.lower() == ".json":
                    data = json.load(f)
                    if not isinstance(data, list):
                        raise ValueError(
                            f"JSON file must contain an array of tasks, got {type(data)}"
                        )

                    for item in data:
                        task = Task(**item)
                        tasks.append(task)

                # Handle JSONL files (one task per line)
                else:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            item = json.loads(line)

                            # Handle case where line contains an array of tasks
                            if isinstance(item, list):
                                for task_item in item:
                                    task = Task(**task_item)
                                    tasks.append(task)
                            # Handle normal case where line contains a single task object
                            elif isinstance(item, dict):
                                task = Task(**item)
                                tasks.append(task)
                            else:
                                raise ValueError(
                                    f"Invalid JSONL format: expected dict or list of dicts, got {type(item)}"  # noqa: E501
                                )

        # Check if it's a HuggingFace dataset
        elif "/" in tasks_input:
            hud_console.info(f"Loading tasks from HuggingFace dataset: {tasks_input}")
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
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"Invalid HuggingFace dataset: expected dict, got {type(item)}"
                        )
                    if not item["mcp_config"] or not item["prompt"]:
                        raise ValueError(
                            f"Invalid HuggingFace dataset: expected mcp_config and prompt, got {item}"  # noqa: E501
                        )
                    task = Task(**item)
                    tasks.append(task)

            except ImportError as e:
                raise ImportError(
                    "Please install 'datasets' to load from HuggingFace: uv pip install datasets"
                ) from e
            except Exception as e:
                raise ValueError(f"Failed to load HuggingFace dataset '{tasks_input}': {e}") from e

        else:
            raise ValueError(
                f"Invalid tasks input: '{tasks_input}' is neither a file path nor a HuggingFace dataset"  # noqa: E501
            )

    else:
        raise TypeError(f"tasks_input must be str or list, got {type(tasks_input)}")

    hud_console.info(f"Loaded {len(tasks)} tasks")
    return tasks
