"""Get command for downloading HuggingFace datasets."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def get_command(
    dataset_name: str = typer.Argument(
        ...,
        help="HuggingFace dataset name (e.g., 'hud-evals/browser-2048-tasks')"
    ),
    split: str = typer.Option(
        "train",
        "--split",
        "-s",
        help="Dataset split to download (train/test/validation)"
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output filename (defaults to dataset_name.jsonl)"
    ),
    limit: int = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit number of examples to download"
    ),
):
    """Download a HuggingFace dataset and save it as JSONL."""
    console.print(f"\n[cyan]📥 Downloading dataset: {dataset_name}[/cyan]")
    
    # Import datasets library
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]Error: datasets library not installed[/red]")
        console.print("[yellow]Install with: pip install datasets[/yellow]")
        raise typer.Exit(1)
    
    # Determine output filename
    if output is None:
        # Convert dataset name to filename (e.g., "hud-evals/browser-2048" -> "browser-2048.jsonl")
        dataset_filename = dataset_name.split("/")[-1] + ".jsonl"
        output = Path(dataset_filename)
    
    # Download dataset with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Loading {dataset_name}...", total=None)
        
        try:
            dataset = load_dataset(dataset_name, split=split)
            progress.update(task, completed=100)
        except ValueError as e:
            if "Unknown split" in str(e):
                console.print(f"[red]Error: Split '{split}' not found in dataset[/red]")
                console.print("[yellow]Common splits: train, test, validation[/yellow]")
            else:
                console.print(f"[red]Error loading dataset: {e}[/red]")
            raise typer.Exit(1)
        except FileNotFoundError:
            console.print(f"[red]Error: Dataset '{dataset_name}' not found[/red]")
            console.print("[yellow]Check the dataset name on HuggingFace Hub[/yellow]")
            raise typer.Exit(1)
        except Exception as e:
            if "authentication" in str(e).lower() or "401" in str(e):
                console.print("[red]Error: Dataset requires authentication[/red]")
                console.print("[yellow]Login with: huggingface-cli login[/yellow]")
            else:
                console.print(f"[red]Error loading dataset: {e}[/red]")
            raise typer.Exit(1)
    
    # Apply limit if specified
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        console.print(f"[yellow]Limited to {len(dataset)} examples[/yellow]")
    
    # Save as JSONL
    console.print(f"[cyan]Writing to {output}...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        transient=True,
    ) as progress:
        task = progress.add_task("Saving...", total=len(dataset))
        
        with open(output, "w") as f:
            for i, example in enumerate(dataset):
                # Convert to dict if needed
                if hasattr(example, "to_dict"):
                    example = example.to_dict()
                
                # Write as JSON line
                f.write(json.dumps(example) + "\n")
                progress.update(task, advance=1)
    
    # Show summary
    console.print(f"\n[green]✅ Downloaded {len(dataset)} examples to {output}[/green]")
    
    # Show sample of fields
    if len(dataset) > 0:
        first_example = dataset[0]
        if hasattr(first_example, "to_dict"):
            first_example = first_example.to_dict()
        
        console.print("\n[yellow]Dataset fields:[/yellow]")
        for field in first_example.keys():
            console.print(f"  • {field}")
        
        # Show example if small enough
        if len(json.dumps(first_example)) < 500:
            console.print("\n[yellow]First example:[/yellow]")
            console.print(json.dumps(first_example, indent=2))
    
    # Show next steps
    console.print("\n[dim]Next steps:[/dim]")
    console.print(f"[dim]• View the file: cat {output} | head[/dim]")
    console.print(f"[dim]• Use for training: hud rl {output}[/dim]")
    console.print(f"[dim]• Use for evaluation: hud eval {output}[/dim]")


# Export the command
__all__ = ["get_command"]
