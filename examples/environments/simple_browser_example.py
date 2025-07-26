#!/usr/bin/env python3
"""
Simple Browser Environment Example

Setup: cd environments/simple_browser && docker-compose up -d
Access: http://localhost:8080/vnc.html (browser view)
"""

import asyncio
from hud.mcp_agent import ClaudeMCPAgent
from mcp_use import MCPClient

async def main():
    # Connect to MCP server for setup
    config = {"mcpServers": {"browser": {"url": "http://localhost:8041/mcp"}}}
    client = MCPClient.from_dict(config)
    
    # Create a session for the browser server
    session = await client.create_session("browser")
    
    # Setup environment ourselves using the session's connector
    print("Setting up todo app...")
    apps = await session.connector.call_tool("list_apps", {})
    print(f"Available apps: {apps}")
    
    result = await session.connector.call_tool("setup_websites", {
        "app_name": "todo", 
        "frontend_port": 3000,
        "backend_port": 5000
    })
    print(f"Setup result: {result}")
    
    # Wait for app to start
    await asyncio.sleep(5)
    
    # Create agent with task tools only
    agent = ClaudeMCPAgent(
        client=client,
        model="claude-sonnet-4-20250514",
        allowed_tools=["computer_anthropic", "api_request", "query_database"]
    )
    
    # Agent tasks
    await agent.run("Take a screenshot and navigate to http://localhost:3000")
    await agent.run("Create test data using the API, then interact with the todo interface")
    
    # Clean up sessions
    await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(main()) 