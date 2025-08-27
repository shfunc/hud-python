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

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# HUD Brand Colors
GOLD = "rgb(192,150,12)"  # #c0960c
RED = "red"
GREEN = "green"
DIM = "dim"


class HUDDesign:
    """Design system for HUD CLI output."""

    def __init__(self) -> None:
        """Initialize the design system."""
        self._stdout_console = Console(stderr=False)
        self._stderr_console = Console(stderr=True)

    def header(self, title: str, icon: str = "🚀", stderr: bool = True) -> None:
        """Print a header panel with gold border.

        Args:
            title: The title text
            icon: Optional emoji icon
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(Panel.fit(f"{icon} [bold]{title}[/bold]", border_style=GOLD))

    def section_title(self, title: str, stderr: bool = True) -> None:
        """Print a section title in gold.

        Args:
            title: The section title
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"\n[bold {GOLD}]{title}[/bold {GOLD}]")

    def success(self, message: str, stderr: bool = True) -> None:
        """Print a success message.

        Args:
            message: The success message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{GREEN} not bold]✅ {message}[/{GREEN} not bold]")

    def error(self, message: str, stderr: bool = True) -> None:
        """Print an error message.

        Args:
            message: The error message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{RED} not bold]❌ {message}[/{RED} not bold]")

    def warning(self, message: str, stderr: bool = True) -> None:
        """Print a warning message.

        Args:
            message: The warning message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[yellow]⚠️  {message}[/yellow]")

    def info(self, message: str, stderr: bool = True) -> None:
        """Print an info message.

        Args:
            message: The info message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[default not bold]{message}[/default not bold]")

    def print(self, message: str, stderr: bool = True) -> None:
        """Print a message.

        Args:
            message: The message to print
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(message)

    def dim_info(self, label: str, value: str, stderr: bool = True) -> None:
        """Print dimmed info with a label.

        Args:
            label: The label text
            value: The value text
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{DIM}]{label}[/{DIM}] [default]{value}[/default]")

    def link(self, url: str, stderr: bool = True) -> None:
        """Print an underlined link.

        Args:
            url: The URL to display
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[default not bold underline]{url}[/default not bold underline]")

    def json_config(self, json_str: str, stderr: bool = True) -> None:
        """Print JSON configuration with light theme.

        Args:
            json_str: JSON string to display
            stderr: If True, output to stderr (default), otherwise stdout
        """
        # Just print the JSON as plain text to avoid any syntax coloring
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[default not bold]{json_str}[/default not bold]")

    def key_value_table(
        self, data: dict[str, str], show_header: bool = False, stderr: bool = True
    ) -> None:
        """Print a key-value table.

        Args:
            data: Dictionary of key-value pairs
            show_header: Whether to show table header
            stderr: If True, output to stderr (default), otherwise stdout
        """
        table = Table(show_header=show_header, box=None, padding=(0, 1))
        table.add_column("Key", style=DIM, no_wrap=True)
        table.add_column("Value")

        for key, value in data.items():
            table.add_row(key, value)

        console = self._stderr_console if stderr else self._stdout_console
        console.print(table)

    def progress_message(self, message: str, stderr: bool = True) -> None:
        """Print a progress message.

        Args:
            message: The progress message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{DIM} not bold]{message}[/{DIM} not bold]")

    def phase(self, phase_num: int, title: str, stderr: bool = True) -> None:
        """Print a phase header (for debug command).

        Args:
            phase_num: Phase number
            title: Phase title
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"\n{'=' * 80}", style=GOLD)
        console.print(f"[bold {GOLD}]PHASE {phase_num}: {title}[/bold {GOLD}]")
        console.print(f"{'=' * 80}", style=GOLD)

    def command(self, cmd: list[str], stderr: bool = True) -> None:
        """Print a command being executed.

        Args:
            cmd: Command parts as list
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[bold]$ {' '.join(cmd)}[/bold]")

    def hint(self, hint: str, stderr: bool = True) -> None:
        """Print a hint message.

        Args:
            hint: The hint text
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"\n[yellow]💡 Hint: {hint}[/yellow]")

    def status_item(
        self,
        label: str,
        value: str,
        status: str = "success",
        primary: bool = False,
        stderr: bool = True,
    ) -> None:
        """Print a status item with indicator.

        Args:
            label: The label text
            value: The value text
            status: Status type - "success" (✓), "error" (✗), "warning" (⚠), "info" (•)
            primary: If True, highlight the value as primary
            stderr: If True, output to stderr (default), otherwise stdout
        """
        indicators = {
            "success": f"[{GREEN} not bold]✓[/{GREEN} not bold]",
            "error": f"[{RED} not bold]✗[/{RED} not bold]",
            "warning": "[yellow]⚠[/yellow]",
            "info": f"[{DIM}]•[/{DIM}]",
        }

        indicator = indicators.get(status, indicators["info"])
        console = self._stderr_console if stderr else self._stdout_console

        if primary:
            console.print(f"{indicator} {label}: [bold cyan]{value}[/bold cyan]")
        else:
            console.print(f"{indicator} {label}: {value}")

    def command_example(
        self, command: str, description: str | None = None, stderr: bool = True
    ) -> None:
        """Print a command example with cyan highlighting.

        Args:
            command: The command to show
            description: Optional description after the command
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        if description:
            console.print(f"  [cyan]{command}[/cyan]  # {description}")
        else:
            console.print(f"  [cyan]{command}[/cyan]")

    @property
    def console(self) -> Console:
        """Get the stderr console for direct access when needed."""
        return self._stderr_console


# Global design instance for convenience
design = HUDDesign()
