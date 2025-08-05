import asyncio

from datasets import Dataset
import hud

from hud.datasets import TaskConfig
from hud.mcp.claude import ClaudeMCPAgent
from hud.mcp.client import MCPClient


async def run_eval():
    # Define task configurations similar to gmail_local.py
    task_configs = [
        {
            "prompt": "Create a new todo item with the title 'Buy groceries' and the description 'Buy groceries for the week.'",
            "mcp_config": {
                "gmail": {
                    "command": "docker",
                    "args": [
                        "run",
                        "-i",  # interactive mode for stdio MCP server
                        "--rm",  # remove the container after it exits
                        "-p",
                        "8080:8080",  # map port 8080 to the host for noVNC
                        "-e",
                        "LAUNCH_APPS=todo",
                        "simple-browser-test",  # use hud browser image with gmail app
                    ],
                }
            },
            "setup_tool": {
                "name": "setup",
                "arguments": {
                    "name": "todo_basic_usage",
                },
            },
            "evaluate_tool": {
                "name": "evaluate",
                "arguments": {
                    "name": "todo_basic_usage",
                },
            },
            "metadata": {"name": "forward-series-b-deck-to-billgates"},
        }
    ]

    # Create HuggingFace dataset
    dataset = Dataset.from_list(task_configs)
    # Give it a name for auto-detection
    dataset.info.builder_name = "gmail_tasks"

    print(f"Created dataset with {len(dataset)} tasks")
    print(f"Dataset columns: {dataset.column_names}")
    print(f"\nFirst task: {dataset[0]}")

    # Convert to TaskConfig
    task = TaskConfig(**dataset[0])
    print(task)

    # Create MCP client outside trace to ensure cleanup
    client = MCPClient(mcp_config=task.mcp_config)
    agent = ClaudeMCPAgent(
        mcp_client=client,
        allowed_tools=["anthropic_computer"],
        initial_screenshot=True,
    )

    with hud.trace("test-claude"):  # Trace the agent execution
        try:
            # Initialize the agent with the task
            await agent.initialize(task)

            # Setup the task
            await agent.call_tool(task.setup_tool)

            # Create initial messages
            messages = await agent.create_initial_messages(task.prompt, None)

            done = False
            # Run the agent
            max_steps = 45
            for step in range(max_steps):
                print(f"Step {step}/{max_steps}")

                # Get agent response
                response = await agent.get_model_response(messages)
                print(
                    f"Agent response: {response.content[:200] if response.content else 'No content'}..."
                )  # Truncated

                # Check if agent wants to use tools
                tool_results = []
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        print(f"Using tool: {tool_call.name} with args {tool_call.arguments}")
                        tool_result = await agent.call_tool(tool_call)
                        tool_results.append(tool_result)
                else:
                    print("No tool calls, done")
                    done = True

                # Format tool results for the model for the next step
                tool_messages = await agent.format_tool_results(response.tool_calls, tool_results)
                messages.extend(tool_messages)
                if done:
                    break

            # Evaluate the task
            eval_result = await agent.call_tool(task.evaluate_tool)
            print(f"Evaluation result: {eval_result}")
        finally:
            # Always close the MCP client
            await client.close()


if __name__ == "__main__":
    # Run the evaluation
    try:
        asyncio.run(run_eval())
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        raise
