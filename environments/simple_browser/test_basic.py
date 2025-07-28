#!/usr/bin/env python3
"""
Basic test script for hud-browser - quick validation.

Usage:
    python test_basic.py
"""

import asyncio
import json
import sys
from mcp_use import MCPClient

async def basic_test():
    """Run basic MCP protocol test."""
    print("🚀 Basic HUD Browser Test")
    print("-" * 30)
    
    # Configure client
    config = {
        "mcpServers": {
            "browser": {
                "command": "docker",
                "args": ["run", "--rm", "-i", "-p", "8080:8080", "-e", "LAUNCH_APPS=todo", "hud-browser"],
            }
        }
    }
    
    client = MCPClient.from_dict(config)
    
    try:
        print("📡 Connecting to hud-browser...")
        session = await client.create_session("browser")
        print("✅ Connected!")
        
        # Test tools
        print("\n🛠️ Listing tools...")
        tools = await session.list_tools()
        tool_names = [tool.name for tool in tools.tools]
        print(f"Found tools: {', '.join(tool_names)}")
        
        # Test resources
        print("\n📋 Listing resources...")
        resources = await session.list_resources()
        resource_uris = [resource.uri for resource in resources.resources]
        print(f"Found {len(resource_uris)} resources")
        
        # Test computer tool (screenshot)
        print("\n📸 Taking screenshot...")
        result = await session.call_tool("computer", {"action": "screenshot"})
        print("✅ Screenshot successful!")
        
        # Test setup
        print("\n🔧 Running setup...")
        result = await session.call_tool("setup", {"config": {"function": "todo_seed", "args": {"num_items": 3}}})
        print("✅ Setup completed!")
        
        # Test evaluate
        print("\n📊 Running evaluation...")
        result = await session.call_tool("evaluate", {"config": {"function": "todo_completed", "args": {"expected_count": 1}}})
        print("✅ Evaluation completed!")
        
        # Test telemetry
        print("\n📡 Getting telemetry...")
        result = await session.read_resource("telemetry://live")
        data = json.loads(result.contents[0].text)
        print(f"VNC URL: {data.get('live_url', 'Not found')}")
        
        print("\n🎉 All basic tests PASSED!")
        print("💡 Manual checks:")
        print("   - VNC Viewer: http://localhost:8080/vnc.html")
        print("   - Todo App: http://localhost:3000")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        await client.close_all_sessions()
    
    return True

if __name__ == "__main__":
    success = asyncio.run(basic_test())
    sys.exit(0 if success else 1) 