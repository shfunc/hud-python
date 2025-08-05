#!/usr/bin/env python3
"""
MCP Resources Example for HUD Browser Environment

This example demonstrates how to discover and use MCP resources to explore
the evaluation system, setup tools, problems, and telemetry data.

Resources provide a way to discover what capabilities are available in an
environment without having to call tools directly.

Usage:
    python examples/environments/resources_example.py
"""

import asyncio
import json
import sys
from typing import Dict, Any, List
from mcp_use import MCPClient


class HudResourcesDemo:
    """Demo class for exploring HUD Browser MCP resources."""

    def __init__(self):
        self.client = None
        self.session = None

    async def setup_connection(self):
        """Setup connection to hud-browser environment."""
        print("🚀 HUD Browser MCP Resources Demo")
        print("=" * 50)

        # Configure client for Docker stdio transport
        config = {
            "mcp_config": {
                "browser": {
                    "command": "docker",
                    "args": [
                        "run",
                        "--rm",
                        "-i",
                        "-p",
                        "8080:8080",  # VNC port
                        "-e",
                        "LAUNCH_APPS=todo",  # Launch todo app
                        "hud-browser",
                    ],
                }
            }
        }

        print("📡 Connecting to hud-browser environment...")
        self.client = MCPClient.from_dict(config)
        self.session = await self.client.create_session("browser")
        print("✅ Connected successfully!")

    async def discover_resources(self):
        """Discover all available MCP resources."""
        print("\n🔍 Discovering Available Resources")
        print("-" * 30)

        resources = await self.session.connector.list_resources()

        print(f"Found {len(resources)} MCP resources:")
        for i, resource in enumerate(resources, 1):
            uri_str = str(resource.uri)
            print(f"  {i}. {uri_str}")
            print(f"     📄 {resource.description}")
            if resource.name:
                print(f"     🏷️  Name: {resource.name}")

        return resources

    async def explore_evaluators_registry(self):
        """Explore the evaluators registry to see available evaluation functions."""
        print("\n📊 Exploring Evaluators Registry")
        print("-" * 30)

        try:
            result = await self.session.connector.read_resource("evaluators://registry")
            data = json.loads(result.contents[0].text)

            evaluators = data.get("evaluators", [])
            print(f"Found {len(evaluators)} evaluators:")

            for evaluator in evaluators:
                name = evaluator.get("name", "Unknown")
                description = evaluator.get("description", "No description")
                app = evaluator.get("app", "unknown")
                print(f"  🎯 {name} ({app})")
                print(f"     {description}")

            # Show example of how to use an evaluator
            if evaluators:
                example_evaluator = evaluators[0]["name"]
                print(f"\n💡 Example: To use the '{example_evaluator}' evaluator:")
                print(f'   await session.connector.call_tool("evaluate", {{')
                print(
                    f'       "config": {{"function": "{example_evaluator}", "args": {{"expected_count": 2}}}}'
                )
                print(f"   }})")

        except Exception as e:
            print(f"❌ Error reading evaluators registry: {e}")

    async def explore_setup_registry(self):
        """Explore the setup tools registry."""
        print("\n🔧 Exploring Setup Tools Registry")
        print("-" * 30)

        try:
            result = await self.session.connector.read_resource("setup://registry")
            data = json.loads(result.contents[0].text)

            setup_tools = data.get("setup_tools", {})
            print(f"Found {len(setup_tools)} setup tools:")

            for name, tool_info in setup_tools.items():
                description = tool_info.get("description", "No description")
                app = tool_info.get("app", "unknown")
                print(f"  🛠️  {name} ({app})")
                print(f"     {description}")

            # Show example usage
            if setup_tools:
                example_setup = list(setup_tools.keys())[0]
                print(f"\n💡 Example: To use the '{example_setup}' setup tool:")
                print(f'   await session.connector.call_tool("setup", {{')
                print(
                    f'       "config": {{"function": "{example_setup}", "args": {{"num_items": 5}}}}'
                )
                print(f"   }})")

        except Exception as e:
            print(f"❌ Error reading setup registry: {e}")

    async def explore_problems_registry(self):
        """Explore the problems registry to see available evaluation scenarios."""
        print("\n🎯 Exploring Problems Registry")
        print("-" * 30)

        try:
            result = await self.session.connector.read_resource("problems://registry")
            data = json.loads(result.contents[0].text)

            problems = data.get("problems", {})
            print(f"Found {len(problems)} evaluation problems:")

            # Group problems by difficulty
            by_difficulty = {}
            for name, problem_info in problems.items():
                difficulty = problem_info.get("difficulty", "unknown")
                if difficulty not in by_difficulty:
                    by_difficulty[difficulty] = []
                by_difficulty[difficulty].append((name, problem_info))

            for difficulty in sorted(by_difficulty.keys()):
                print(f"\n  📚 {difficulty.upper()} Problems:")
                for name, info in by_difficulty[difficulty]:
                    description = info.get("description", "No description")
                    task_type = info.get("task_type", "unknown")
                    app = info.get("app", "unknown")
                    print(f"    🧩 {name} ({app}, {task_type})")
                    print(f"       {description}")

            # Show example usage
            if problems:
                example_problem = list(problems.keys())[0]
                print(f"\n💡 Example: To run the '{example_problem}' problem:")
                print(f"   # Setup the problem")
                print(
                    f'   await session.connector.call_tool("setup", {{"config": {{"name": "{example_problem}"}}}})'
                )
                print(f"   # Evaluate the problem")
                print(
                    f'   await session.connector.call_tool("evaluate", {{"config": {{"name": "{example_problem}"}}}})'
                )

        except Exception as e:
            print(f"❌ Error reading problems registry: {e}")

    async def explore_telemetry(self):
        """Explore live telemetry data."""
        print("\n📡 Exploring Live Telemetry")
        print("-" * 30)

        try:
            result = await self.session.connector.read_resource("telemetry://live")
            data = json.loads(result.contents[0].text)

            print("Current system status:")
            for key, value in data.items():
                if key == "services":
                    print(f"  📊 {key}:")
                    for service, status in value.items():
                        status_icon = "✅" if status == "running" else "❌"
                        print(f"      {status_icon} {service}: {status}")
                elif key == "live_url":
                    print(f"  🖥️  {key}: {value}")
                elif key == "timestamp":
                    print(f"  ⏰ {key}: {value}")
                else:
                    print(f"  📋 {key}: {value}")

        except Exception as e:
            print(f"❌ Error reading telemetry: {e}")

    async def demonstrate_practical_usage(self):
        """Demonstrate practical usage of the resource system."""
        print("\n🎮 Practical Usage Demo")
        print("-" * 30)

        print("Let's run a complete evaluation workflow using resources!")

        # 1. Read problems registry to find an easy problem
        try:
            result = await self.session.connector.read_resource("problems://registry")
            problems_data = json.loads(result.contents[0].text)
            problems = problems_data.get("problems", {})

            # Find an easy problem
            easy_problems = [
                name for name, info in problems.items() if info.get("difficulty") == "easy"
            ]

            if not easy_problems:
                print("❌ No easy problems found")
                return

            chosen_problem = easy_problems[0]
            problem_info = problems[chosen_problem]

            print(f"\n🎯 Selected problem: {chosen_problem}")
            print(f"   📄 Description: {problem_info.get('description')}")
            print(f"   📊 Difficulty: {problem_info.get('difficulty')}")
            print(f"   🏷️  Type: {problem_info.get('task_type')}")

            # 2. Run the setup
            print(f"\n🔧 Running setup for {chosen_problem}...")
            setup_result = await self.session.connector.call_tool(
                "setup", {"config": {"name": chosen_problem}}
            )

            if setup_result.isError:
                print(f"❌ Setup failed: {setup_result.content}")
                return
            else:
                print("✅ Setup completed successfully!")

            # 3. Run the evaluation
            print(f"\n📊 Running evaluation for {chosen_problem}...")
            eval_result = await self.session.connector.call_tool(
                "evaluate", {"config": {"name": chosen_problem}}
            )

            if eval_result.isError:
                print(f"❌ Evaluation failed: {eval_result.content}")
                return
            else:
                # Parse the evaluation result
                try:
                    eval_data = json.loads(eval_result.content[0].text)
                    reward = eval_data.get("reward", 0)
                    done = eval_data.get("done", False)
                    info = eval_data.get("info", {})

                    print("✅ Evaluation completed!")
                    print(f"   🎯 Reward: {reward}")
                    print(f"   ✔️  Done: {done}")
                    if info:
                        print(f"   📊 Info: {info}")

                except Exception as e:
                    print(f"✅ Evaluation completed (raw result): {eval_result.content[0].text}")

        except Exception as e:
            print(f"❌ Demo failed: {e}")

    async def cleanup(self):
        """Cleanup connections."""
        if self.client:
            await self.client.close_all_sessions()
            print("\n🧹 Disconnected from environment")


async def main():
    """Main demo function."""
    demo = HudResourcesDemo()

    try:
        # Setup connection
        await demo.setup_connection()

        # Discover resources
        await demo.discover_resources()

        # Explore each registry
        await demo.explore_evaluators_registry()
        await demo.explore_setup_registry()
        await demo.explore_problems_registry()
        await demo.explore_telemetry()

        # Demonstrate practical usage
        await demo.demonstrate_practical_usage()

        print("\n" + "=" * 50)
        print("🎉 Resources Demo Complete!")
        print("\n💡 Key Takeaways:")
        print("   • Resources provide discoverability without tool calls")
        print("   • Registries contain metadata about available capabilities")
        print("   • Problems combine setup and evaluation in reusable scenarios")
        print("   • Telemetry provides live system status and VNC access")
        print("\n🖥️  Manual exploration:")
        print("   • VNC Viewer: http://localhost:8080/vnc.html")
        print("   • Todo App: http://localhost:3000")

    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return False
    finally:
        await demo.cleanup()

    return True


if __name__ == "__main__":
    print("🔧 HUD Browser MCP Resources Example")
    print("This example requires the hud-browser Docker image to be built.")
    print("Make sure you have run: docker build -t hud-browser .")
    print("\nPress Enter to continue or Ctrl+C to cancel...")

    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)

    success = asyncio.run(main())
    sys.exit(0 if success else 1)
