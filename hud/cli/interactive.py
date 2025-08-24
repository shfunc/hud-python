"""Interactive mode for testing MCP environments."""

from __future__ import annotations

import json
from typing import Any

import questionary
from mcp.types import TextContent
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.tree import Tree

from hud.clients import MCPClient
from hud.utils.design import HUDDesign

console = Console()


class InteractiveMCPTester:
    """Interactive MCP environment tester."""

    def __init__(self, server_url: str, verbose: bool = False) -> None:
        """Initialize the interactive tester.

        Args:
            server_url: URL of the MCP server (e.g., http://localhost:8765/mcp)
            verbose: Enable verbose output
        """
        self.server_url = server_url
        self.verbose = verbose
        self.client: MCPClient | None = None
        self.tools: list[Any] = []
        self.design = HUDDesign()

    async def connect(self) -> bool:
        """Connect to the MCP server."""
        try:
            # Create MCP config for HTTP transport
            config = {"server": {"url": self.server_url}}

            self.client = MCPClient(
                mcp_config=config,
                verbose=self.verbose,
                auto_trace=False,  # Disable telemetry for interactive testing
            )
            await self.client.initialize()

            # Fetch available tools
            self.tools = await self.client.list_tools()

            return True
        except Exception as e:
            self.design.error(f"Failed to connect: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self.client:
            await self.client.shutdown()
            self.client = None

    def display_tools(self) -> None:
        """Display available tools in a nice format."""
        if not self.tools:
            console.print("[yellow]No tools available[/yellow]")
            return

        # Group tools by hub
        regular_tools = []
        hub_tools = {}

        for tool in self.tools:
            if "/" in tool.name:
                hub, name = tool.name.split("/", 1)
                if hub not in hub_tools:
                    hub_tools[hub] = []
                hub_tools[hub].append(tool)
            else:
                regular_tools.append(tool)

        # Display tools tree
        tree = Tree("🔧 Available Tools")

        if regular_tools:
            regular_node = tree.add("[cyan]Regular Tools[/cyan]")
            for i, tool in enumerate(regular_tools, 1):
                tool_node = regular_node.add(f"{i}. [white]{tool.name}[/white]")
                if tool.description:
                    tool_node.add(f"[dim]{tool.description}[/dim]")

        # Add hub tools
        tool_index = len(regular_tools) + 1
        for hub_name, tools in hub_tools.items():
            hub_node = tree.add(f"[yellow]{hub_name} Hub[/yellow]")
            for tool in tools:
                tool_node = hub_node.add(f"{tool_index}. [white]{tool.name}[/white]")
                if tool.description:
                    tool_node.add(f"[dim]{tool.description}[/dim]")
                tool_index += 1

        console.print(tree)

    async def select_tool(self) -> Any | None:
        """Let user select a tool."""
        if not self.tools:
            return None

        # Build choices list
        choices = []
        tool_map = {}

        for _, tool in enumerate(self.tools):
            # Create display name
            if "/" in tool.name:
                hub, name = tool.name.split("/", 1)
                display = f"[{hub}] {name}"
            else:
                display = tool.name

            # Add description if available
            if tool.description:
                display += f" - {tool.description}"

            choices.append(display)
            tool_map[display] = tool

        # Add quit option
        choices.append("❌ Quit")

        # Show selection menu with arrow keys
        console.print("\n[cyan]Select a tool (use arrow keys):[/cyan]")

        try:
            # Use questionary's async select with custom styling
            selected = await questionary.select(
                "",
                choices=choices,
                style=questionary.Style(
                    [
                        ("question", ""),
                        ("pointer", "fg:#ff9d00 bold"),
                        ("highlighted", "fg:#ff9d00 bold"),
                        ("selected", "fg:#cc5454"),
                        ("separator", "fg:#6c6c6c"),
                        ("instruction", "fg:#858585 italic"),
                    ]
                ),
            ).unsafe_ask_async()

            if selected is None:
                console.print("[yellow]No selection made (ESC or Ctrl+C pressed)[/yellow]")
                return None

            if selected == "❌ Quit":
                return None

            return tool_map[selected]

        except KeyboardInterrupt:
            console.print("[yellow]Interrupted by user[/yellow]")
            return None
        except Exception as e:
            console.print(f"[red]Error in tool selection: {e}[/red]")
            return None

    async def get_tool_arguments(self, tool: Any) -> dict[str, Any] | None:
        """Prompt user for tool arguments."""
        if not hasattr(tool, "inputSchema") or not tool.inputSchema:
            return {}

        schema = tool.inputSchema

        # Show schema
        console.print("\n[yellow]Tool Parameters:[/yellow]")
        schema_str = json.dumps(schema, indent=2)
        syntax = Syntax(schema_str, "json", theme="monokai", line_numbers=False)
        console.print(Panel(syntax, title=f"{tool.name} Schema", border_style="dim"))

        # Handle different schema types
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            if not properties:
                return {}

            # Prompt for each property
            args = {}
            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get("type", "string")
                description = prop_schema.get("description", "")
                is_required = prop_name in required

                # Build prompt
                prompt = f"{prop_name}"
                if description:
                    prompt += f" ({description})"
                if not is_required:
                    prompt += " [optional]"

                # Get value based on type
                if prop_type == "boolean":
                    if is_required:
                        value = await questionary.confirm(prompt).unsafe_ask_async()
                    else:
                        # For optional booleans, offer a choice
                        choice = await questionary.select(
                            prompt, choices=["true", "false", "skip (leave unset)"]
                        ).unsafe_ask_async()
                        if choice == "skip (leave unset)":
                            continue
                        value = choice == "true"
                elif prop_type == "number" or prop_type == "integer":
                    value_str = await questionary.text(
                        prompt,
                        default="",
                        validate=lambda text, pt=prop_type, req=is_required: True
                        if not text and not req
                        else (
                            text.replace("-", "").replace(".", "").isdigit()
                            if pt == "number"
                            else text.replace("-", "").isdigit()
                        )
                        or f"Please enter a valid {pt}",
                    ).unsafe_ask_async()
                    if not value_str and not is_required:
                        continue
                    value = int(value_str) if prop_type == "integer" else float(value_str)
                elif prop_type == "array":
                    value_str = await questionary.text(
                        prompt + " (comma-separated)", default=""
                    ).unsafe_ask_async()
                    if not value_str and not is_required:
                        continue
                    value = [v.strip() for v in value_str.split(",")]
                else:  # string or unknown
                    value = await questionary.text(prompt, default="").unsafe_ask_async()
                    if not value and not is_required:
                        continue

                args[prop_name] = value

            return args
        else:
            # For non-object schemas, just get a single value
            console.print("[yellow]Enter value (or press Enter to skip):[/yellow]")
            value = Prompt.ask("Value", default="")
            return {"value": value} if value else {}

    async def call_tool(self, tool: Any, arguments: dict[str, Any]) -> None:
        """Call a tool and display results."""
        if not self.client:
            return

        try:
            # Show what we're calling
            console.print(f"\n[cyan]Calling {tool.name}...[/cyan]")
            if arguments:
                console.print(f"[dim]Arguments: {json.dumps(arguments, indent=2)}[/dim]")

            # Make the call
            result = await self.client.call_tool(name=tool.name, arguments=arguments)

            # Display results
            console.print("\n[green]✓ Tool executed successfully[/green]")

            if result.isError:
                console.print("[red]Error result:[/red]")

            # Display content blocks
            for content in result.content:
                if isinstance(content, TextContent):
                    console.print(
                        Panel(
                            content.text,
                            title="Result",
                            border_style="green" if not result.isError else "red",
                        )
                    )
                else:
                    # Handle other content types
                    console.print(json.dumps(content, indent=2))

        except Exception as e:
            console.print(f"[red]✗ Tool execution failed: {e}[/red]")

    async def run(self) -> None:
        """Run the interactive testing loop."""
        self.design.header("Interactive MCP Tester")

        # Connect to server
        console.print(f"[cyan]Connecting to {self.server_url}...[/cyan]")
        if not await self.connect():
            return

        console.print("[green]✓ Connected successfully[/green]")
        console.print(f"[dim]Found {len(self.tools)} tools[/dim]\n")

        try:
            while True:
                # Select tool
                tool = await self.select_tool()
                if not tool:
                    break

                # Get arguments
                console.print(f"\n[cyan]Selected: {tool.name}[/cyan]")
                arguments = await self.get_tool_arguments(tool)
                if arguments is None:
                    console.print("[yellow]Skipping tool call[/yellow]")
                    continue

                # Call tool
                await self.call_tool(tool, arguments)

                # Just add a separator and continue to tool selection
                console.print("\n" + "─" * 50)

        finally:
            # Disconnect
            console.print("\n[cyan]Disconnecting...[/cyan]")
            await self.disconnect()

            # Show next steps tutorial
            self.design.section_title("Next Steps")
            self.design.info("🏗️  Ready to test with real agents? Run:")
            self.design.info("    [cyan]hud build[/cyan]")
            self.design.info("")
            self.design.info("This will:")
            self.design.info("  1. Build your environment image")
            self.design.info("  2. Generate a hud.lock.yaml file")
            self.design.info("  3. Prepare it for testing with agents")
            self.design.info("")
            self.design.info("Then you can:")
            self.design.info("  • Test locally: [cyan]hud run <image>[/cyan]")
            self.design.info("  • Push to registry: [cyan]hud push --image <registry/name>[/cyan]")
            self.design.info("  • Use with agents via the lock file")

            console.print("\n[dim]Happy testing! 🎉[/dim]")


async def run_interactive_mode(server_url: str, verbose: bool = False) -> None:
    """Run interactive MCP testing mode.

    Args:
        server_url: URL of the MCP server
        verbose: Enable verbose output
    """
    tester = InteractiveMCPTester(server_url, verbose)
    await tester.run()
