#!/usr/bin/env python3
"""
Browser Environment Agent Loop Example

This example demonstrates a complete agent lifecycle for browser-based tasks:
- Configurable for different apps (2048, todo)
- Task definition using problem-based setup and evaluation
- Agent initialization with browser environment
- Agent execution loop
- Problem-based evaluation
- Proper cleanup

Usage:
    python 02_browser_agent_loop.py          # defaults to 2048
    python 02_browser_agent_loop.py --app 2048
    python 02_browser_agent_loop.py --app todo
"""

import asyncio
import argparse
import json
import hud
from hud.datasets import Task
from hud.agents import ClaudeAgent
from hud.clients import MCPClient


# App-specific configurations
APP_CONFIGS = {
    "2048": {
        "prompt": """Play the 2048 game and try to reach the 512 tile.
        
        Strategy tips:
        - Keep your highest tiles in a corner
        - Build tiles in descending order
        - Avoid random moves
        - Use arrow keys or swipe to move tiles
        
        Make strategic moves to maximize your score and reach the target.""",
        "setup_name": "game_2048_board",
        "setup_args": {"board_size": 4, "target_tile": 512},
        "evaluate_name": "game_2048_max_number",
        "evaluate_args": {"target": 512},
        "allowed_tools": ["anthropic_computer", "playwright", "launch_app"],
        "max_steps": 50,
        "check_interval": 5,  # Check progress every N moves
        # Display helpers
        "progress_formatter": lambda info: f"🎯 Highest tile: {info.get('highest_tile', 0)}",
        "final_formatter": lambda info: f"🏆 Highest tile: {info.get('highest_tile', 'unknown')}\n   🎯 Target tile: 512",
    },
    "todo": {
        "prompt": """Complete the following tasks in the todo app:
        
        1. You'll see 5 items with 2 already completed
        2. Mark 2 more items as completed (for a total of 4 completed)
        3. Use the filter features to view completed/active items
        4. Test the search functionality
        
        Be systematic and interact with the app's features.""",
        "setup_name": "todo_seed",  # Seed with 5 test items (2 pre-completed)
        "setup_args": {"num_items": 5},
        "evaluate_name": "todo_completed",  # Check completed count
        "evaluate_args": {"expected_count": 4},  # 2 pre-completed + 2 more = 4
        "allowed_tools": ["anthropic_computer", "playwright", "launch_app"],
        "max_steps": 30,
        "check_interval": 10,  # Check progress every N actions
        # Display helpers
        "progress_formatter": lambda info: f"✅ Completed: {info.get('completed_count', 0)} items",
        "final_formatter": lambda info: f"✅ Items completed: {info.get('completed_count', 0)}\n   📝 Total items: {info.get('total_count', 0)}",
    },
}


async def main():
    parser = argparse.ArgumentParser(description="Browser Environment Agent Loop")
    parser.add_argument(
        "--app",
        type=str,
        default="2048",
        choices=list(APP_CONFIGS.keys()),
        help="Which app to run (2048 or todo)",
    )
    parser.add_argument(
        "--model", type=str, default="claude-3-7-sonnet-20250219", help="Model to use for the agent"
    )

    args = parser.parse_args()
    app_name = args.app
    config = APP_CONFIGS[app_name]

    print(f"🎮 Browser Agent Loop Example - {app_name.upper()}")
    print("=" * 50)
    print(f"Running agent for {app_name} app with {args.model}")
    print("=" * 50)

    # MCP configuration for browser environment
    mcp_config = {
        "local": {
            "command": "docker",
            "args": [
                "run",
                "--rm",
                "-i",
                "-p",
                "8080:8080",  # noVNC web interface
                "hud-browser",
            ],
        }
    }

    # Define task using problem-based approach
    task_dict = {
        "prompt": config["prompt"],
        "mcp_config": mcp_config,
        # Use problem-based setup
        "setup_tool": {
            "name": "setup",
            "arguments": {"name": config["setup_name"], "arguments": config["setup_args"]},
        },
        # Use problem-based evaluation
        "evaluate_tool": {
            "name": "evaluate",
            "arguments": {"name": config["evaluate_name"], "arguments": config["evaluate_args"]},
        },
    }
    task = Task(**task_dict)

    # Create MCP client with resolved config
    client = MCPClient(mcp_config=task.mcp_config)

    # Create agent with browser automation capabilities
    agent = ClaudeAgent(
        mcp_client=client,
        model=args.model,
        allowed_tools=["anthropic_computer"],
        initial_screenshot=True,  # Capture initial state
    )

    try:
        # Phase 1: Initialize agent with task context
        print(f"\n🎮 Initializing {app_name} agent...")
        await agent.initialize(task)

        # List available tools to debug
        print("\n📋 Available tools:")
        tools = await client.list_tools()
        for tool in tools:
            print(f"  - {tool.name}")

        # Phase 2: Launch the app
        print(f"\n🚀 Launching {app_name} app...")
        try:
            launch_result = await client.call_tool("launch_app", {"app_name": app_name})
            print(f"Launch result: {launch_result}")
            await asyncio.sleep(2)  # Wait for page to load
        except Exception as e:
            print(f"Failed to launch app: {e}")

        # Phase 3: Setup the task
        print(f"\n🎯 Setting up {app_name}...")
        if task.setup_tool:
            try:
                # Call the setup hub
                setup_result = await client.call_tool(
                    "setup", {"name": config["setup_name"], "arguments": config["setup_args"]}
                )
                print(f"Setup result: {setup_result}")
            except Exception as e:
                print(f"Failed to setup: {e}")
        print("✅ Task initialized")

        # Phase 4: Run agent loop
        print(f"\n🤖 Starting task: {config['prompt'][:50]}...")
        messages = await agent.create_initial_messages(task.prompt, None)

        done = False
        steps = 0
        max_steps = config["max_steps"]
        best_reward = 0.0
        consecutive_no_progress = 0

        while not done and steps < max_steps:
            # Get model response (agent decides next action)
            response = await agent.get_model_response(messages)
            print(f"\n   Step {steps + 1}:")

            if response.content:
                # Show agent's thinking
                thinking = response.content[:100]
                print(f"   💭 {thinking}...")

            if response.tool_calls:
                # Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call.name

                    # Show what the agent is doing
                    if "computer" in tool_name:
                        action = tool_call.arguments.get("action", "")
                        if "key" in action:
                            key = tool_call.arguments.get("text", "")
                            print(f"   ⌨️ Pressing: {key}")
                        elif "click" in action:
                            print(f"   🖱️ Clicking")
                        elif "type" in action:
                            text = tool_call.arguments.get("text", "")[:30]
                            print(f"   ⌨️ Typing: {text}...")
                        else:
                            print(f"   🎯 Action: {action}")
                    else:
                        print(f"   🔧 Tool: {tool_name}")

                    result = await agent.call_tool(tool_call)
                    tool_results.append(result)

                # Format results back into messages
                messages.extend(await agent.format_tool_results(response.tool_calls, tool_results))

                # Periodically check progress using problem evaluation
                if steps % config["check_interval"] == 0:
                    print("   📊 Checking progress...")
                    if task.evaluate_tool:
                        try:
                            eval_result = await client.call_tool(
                                "evaluate",
                                {
                                    "name": config["evaluate_name"],
                                    "arguments": config["evaluate_args"],
                                },
                            )
                        except Exception as e:
                            print(f"   Failed to evaluate: {e}")
                            eval_result = None

                        if eval_result and not eval_result.isError:
                            eval_content = eval_result.content[0] if eval_result.content else {}

                            # Parse the JSON result
                            if hasattr(eval_content, "text"):
                                import json

                                try:
                                    result_data = json.loads(eval_content.text)
                                    reward = result_data.get("reward", 0.0)
                                    done_flag = result_data.get("done", False)
                                    info = result_data.get("info", {})

                                    print(f"      📈 Reward: {reward:.4f}")

                                    # App-specific progress display
                                    if "progress_formatter" in config:
                                        print(f"      {config['progress_formatter'](info)}")

                                    # Check if we're making progress
                                    if reward > best_reward:
                                        best_reward = reward
                                        consecutive_no_progress = 0
                                    else:
                                        consecutive_no_progress += 1

                                    # Check if target reached
                                    if done_flag or reward >= 1.0:
                                        print(f"   🎉 Target reached!")
                                        done = True

                                except json.JSONDecodeError:
                                    print(f"      ⚠️ Could not parse evaluation result")
                                    pass

                    # Stop if no progress for too long
                    if consecutive_no_progress > 3:
                        print(
                            f"   ⚠️ No progress for {consecutive_no_progress * config['check_interval']} steps"
                        )
                        # Agent might want to try different strategy
                        messages.append(
                            {
                                "role": "user",
                                "content": "You haven't made progress recently. Try a different strategy.",
                            }
                        )
                        consecutive_no_progress = 0

            else:
                # No more tool calls, agent thinks it's done
                done = True

            steps += 1

        # Phase 5: Final evaluation
        print("\n📊 Final evaluation...")
        if task.evaluate_tool:
            try:
                final_eval = await client.call_tool(
                    "evaluate",
                    {"name": config["evaluate_name"], "arguments": config["evaluate_args"]},
                )
            except Exception as e:
                print(f"Failed to evaluate: {e}")
                final_eval = None

            if final_eval and final_eval.isError:
                print(f"❌ Evaluation failed: {final_eval.content}")
            elif final_eval:
                # Extract final stats from JSON result
                eval_data = final_eval.content[0] if final_eval.content else {}

                # Parse the JSON result
                if hasattr(eval_data, "text"):
                    import json

                    try:
                        result_data = json.loads(eval_data.text)
                        final_reward = result_data.get("reward", best_reward)
                        info = result_data.get("info", {})

                        print(f"✅ Task complete!")
                        print(f"   📈 Final reward: {final_reward:.4f}")

                        # App-specific final stats
                        if "final_formatter" in config:
                            print(f"   {config['final_formatter'](info)}")

                    except json.JSONDecodeError:
                        print(f"✅ Task complete!")
                        print(f"   📈 Best reward: {best_reward:.4f}")
                else:
                    print(f"✅ Task complete!")
                    print(f"   📈 Best reward: {best_reward:.4f}")

        # Summary
        print("\n📈 Summary:")
        print(f"   App: {app_name}")
        print(f"   Total steps: {steps}")
        print(f"   Best reward: {best_reward:.4f}")
        # Check task completion based on best reward
        task_completed = done and best_reward >= 1.0
        print(f"   Task completed: {task_completed}")

    except Exception as e:
        print(f"   Error occurred: {e}")

    finally:
        # Phase 6: Cleanup
        print("\n🧹 Cleaning up...")
        await client.close()

    print(f"\n✨ {app_name} agent loop complete!")


if __name__ == "__main__":
    asyncio.run(main())
