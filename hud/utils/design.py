"""HUD Design System - Consistent styling utilities for CLI output.

This module provides a unified design system for HUD CLI commands,
ensuring consistent colors, formatting, and visual hierarchy across
all commands.

Color Palette:
- Gold (#c0960c): Primary brand color for headers and important elements
- Black: Standard text and underlined links
- Red: Errors and failures
- Green: Success messages
- Dim/Gray: Secondary information
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text


# HUD Brand Colors
GOLD = "rgb(192,150,12)"  # #c0960c
RED = "red"
GREEN = "green"
DIM = "dim"


class HUDDesign:
    """Design system for HUD CLI output."""
    
    def __init__(self, console: Console | None = None, stderr: bool = False):
        """Initialize the design system.
        
        Args:
            console: Rich console instance. Creates new one if not provided.
            stderr: If True, output to stderr instead of stdout.
        """
        self.console = console or Console(stderr=stderr)
    
    def header(self, title: str, icon: str = "🚀") -> None:
        """Print a header panel with gold border.
        
        Args:
            title: The title text
            icon: Optional emoji icon
        """
        self.console.print(
            Panel.fit(
                f"{icon} [bold]{title}[/bold]",
                border_style=GOLD
            )
        )
    
    def section_title(self, title: str) -> None:
        """Print a section title in gold.
        
        Args:
            title: The section title
        """
        self.console.print(f"\n[bold {GOLD}]{title}[/bold {GOLD}]")
    
    def success(self, message: str) -> None:
        """Print a success message.
        
        Args:
            message: The success message
        """
        self.console.print(f"[{GREEN} not bold]✅ {message}[/{GREEN} not bold]")
    
    def error(self, message: str) -> None:
        """Print an error message.
        
        Args:
            message: The error message
        """
        self.console.print(f"[{RED} not bold]❌ {message}[/{RED} not bold]")
    
    def warning(self, message: str) -> None:
        """Print a warning message.
        
        Args:
            message: The warning message
        """
        self.console.print(f"[yellow]⚠️  {message}[/yellow]")
    
    def info(self, message: str) -> None:
        """Print an info message.
        
        Args:
            message: The info message
        """
        self.console.print(f"[default not bold]{message}[/default not bold]")
    
    def dim_info(self, label: str, value: str) -> None:
        """Print dimmed info with a label.
        
        Args:
            label: The label text
            value: The value text
        """
        self.console.print(f"[{DIM}]{label}[/{DIM}] [default]{value}[/default]")
    
    def link(self, url: str) -> None:
        """Print an underlined link.
        
        Args:
            url: The URL to display
        """
        self.console.print(f"[default not bold underline]{url}[/default not bold underline]")
    
    def json_config(self, json_str: str) -> None:
        """Print JSON configuration with light theme.
        
        Args:
            json_str: JSON string to display
        """
        # Just print the JSON as plain text to avoid any syntax coloring
        self.console.print(f"[default not bold]{json_str}[/default not bold]")
    
    def key_value_table(self, data: dict[str, str], show_header: bool = False) -> None:
        """Print a key-value table.
        
        Args:
            data: Dictionary of key-value pairs
            show_header: Whether to show table header
        """
        table = Table(show_header=show_header, box=None, padding=(0, 1))
        table.add_column("Key", style=DIM, no_wrap=True)
        table.add_column("Value")
        
        for key, value in data.items():
            table.add_row(key, value)
        
        self.console.print(table)
    
    def progress_message(self, message: str) -> None:
        """Print a progress message.
        
        Args:
            message: The progress message
        """
        self.console.print(f"[{DIM} not bold]{message}[/{DIM} not bold]")
    
    def phase(self, phase_num: int, title: str) -> None:
        """Print a phase header (for debug command).
        
        Args:
            phase_num: Phase number
            title: Phase title
        """
        self.console.print(f"\n{'=' * 80}", style=GOLD)
        self.console.print(f"[bold {GOLD}]PHASE {phase_num}: {title}[/bold {GOLD}]")
        self.console.print(f"{'=' * 80}", style=GOLD)
    
    def command(self, cmd: list[str]) -> None:
        """Print a command being executed.
        
        Args:
            cmd: Command parts as list
        """
        self.console.print(f"[bold]$ {' '.join(cmd)}[/bold]")
    
    def hint(self, hint: str) -> None:
        """Print a hint message.
        
        Args:
            hint: The hint text
        """
        self.console.print(f"\n[yellow]💡 Hint: {hint}[/yellow]")


# Global design instance for convenience
design = HUDDesign()
