#!/usr/bin/env python3
"""
Convert TaskConfig JSON files to Hugging Face datasets.

This script reads a JSON file containing task configurations and pushes them
to Hugging Face Hub as a dataset with proper JSON string serialization.

Usage:
    python taskconfig_to_hf.py path/to/taskconfigs.json --repo-id username/dataset-name
    python taskconfig_to_hf.py path/to/taskconfigs.json --repo-id username/dataset-name --private
"""

import argparse
import json
import sys
from pathlib import Path

from hud.datasets import save_taskconfigs


def load_taskconfigs(file_path: str) -> list[dict]:
    """Load task configurations from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Expected a list of task configurations")
    
    return data


def preview_taskconfigs(taskconfigs: list[dict]) -> None:
    """Preview the first few task configurations."""
    print("\nPreview (first 3 tasks):")
    for i, tc_dict in enumerate(taskconfigs[:3]):
        print(f"\nTask {i + 1}:")
        print(f"  ID: {tc_dict.get('id', 'N/A')}")
        print(f"  Prompt: {tc_dict['prompt'][:100]}...")
        if 'metadata' in tc_dict:
            print(f"  Metadata: {tc_dict['metadata']}")
        if 'setup_tool' in tc_dict:
            print(f"  Setup Tool: {tc_dict['setup_tool'].get('name', 'N/A')}")
        if 'evaluate_tool' in tc_dict:
            print(f"  Evaluate Tool: {tc_dict['evaluate_tool'].get('name', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TaskConfig JSON files to Hugging Face datasets"
    )
    parser.add_argument(
        "file",
        help="Path to the JSON file containing task configurations"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repository ID (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private on Hugging Face Hub"
    )
    parser.add_argument(
        "--token",
        help="Hugging Face API token (defaults to HF_TOKEN env var)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the dataset without pushing to Hub"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Loading task configurations from: {args.file}")
        taskconfigs = load_taskconfigs(args.file)
        print(f"Loaded {len(taskconfigs)} task configurations")
        
        if args.preview:
            preview_taskconfigs(taskconfigs)
        else:
            print(f"\nPushing to Hugging Face Hub: {args.repo_id}")
            
            kwargs = {}
            if args.private:
                kwargs["private"] = True
            if args.token:
                kwargs["token"] = args.token
            
            save_taskconfigs(taskconfigs, args.repo_id, **kwargs)
            
            hub_url = f"https://huggingface.co/datasets/{args.repo_id}"
            print(f"Dataset successfully pushed to: {hub_url}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()