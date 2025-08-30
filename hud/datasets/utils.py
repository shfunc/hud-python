"""Dataset utilities for loading, saving, and fetching datasets."""

from __future__ import annotations

import json
import logging
from typing import Any

from datasets import Dataset

from .task import Task

logger = logging.getLogger("hud.datasets")


async def fetch_system_prompt_from_dataset(dataset_id: str) -> str | None:
    """
    Fetch system_prompt.txt from a HuggingFace dataset repository.

    Args:
        dataset_id: HuggingFace dataset identifier (e.g., "hud-evals/SheetBench-50")

    Returns:
        System prompt text if found, None otherwise
    """
    try:
        # Import here to avoid unnecessary dependency
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        # Try to download the system_prompt.txt file
        try:
            file_path = hf_hub_download(
                repo_id=dataset_id, filename="system_prompt.txt", repo_type="dataset"
            )

            # Read and return the content
            with open(file_path, encoding="utf-8") as f:  # noqa: ASYNC230
                content = f.read().strip()
                if content:
                    logger.info(
                        "Loaded system prompt from %s (length: %d chars)", dataset_id, len(content)
                    )
                    return content
                else:
                    logger.warning("System prompt file is empty in %s", dataset_id)
                    return None

        except EntryNotFoundError:
            logger.debug("No system_prompt.txt found in dataset %s", dataset_id)
            return None

    except ImportError:
        logger.warning(
            "huggingface_hub not installed. Install it to fetch system prompts from datasets."
        )
        return None
    except Exception as e:
        logger.error("Error fetching system prompt from %s: %s", dataset_id, e)
        return None


def save_tasks(
    tasks: list[dict[str, Any]], repo_id: str, fields: list[str] | None = None, **kwargs: Any
) -> None:
    """
    Save data to HuggingFace dataset with JSON string serialization.

    Complex fields (dicts, lists) are serialized as JSON strings to maintain clean schema
    and avoid null value pollution in HuggingFace datasets.

    Args:
        tasks: List of dictionaries to save
        repo_id: HuggingFace repository ID (e.g., "hud-evals/my-tasks")
        fields: Optional list of fields to save. If None, saves all fields from each dict.
        **kwargs: Additional arguments passed to dataset.push_to_hub()
    """
    # Safety check: Ensure we're not saving Task objects (which have resolved env vars)
    if tasks and isinstance(tasks[0], Task):
        raise ValueError(
            "save_tasks expects dictionaries, not Task objects. "
            "Task objects have resolved environment variables which would expose secrets. "
            "Please pass raw dictionaries with template strings like '${HUD_API_KEY}' preserved."
        )

    # Convert to rows with JSON string fields
    data = []
    for i, tc_dict in enumerate(tasks):
        # Additional safety check for each item
        if isinstance(tc_dict, Task):
            raise ValueError(
                f"Item {i} is a Task object, not a dictionary. "
                "This would expose resolved environment variables. "
                "Please convert to dictionary format with template strings preserved."
            )

        row = {}

        # Determine which fields to process
        fields_to_process = fields if fields is not None else list(tc_dict.keys())

        for field in fields_to_process:
            if field in tc_dict:
                value = tc_dict[field]
                # Serialize complex types as JSON strings
                if isinstance(value, (dict | list)):
                    row[field] = json.dumps(value)
                elif isinstance(value, (str | int | float | bool | type(None))):
                    row[field] = value if value is not None else ""
                else:
                    # For other types, convert to string
                    row[field] = str(value)

        data.append(row)

    # Create and push dataset
    dataset = Dataset.from_list(data)
    dataset.push_to_hub(repo_id, **kwargs)
