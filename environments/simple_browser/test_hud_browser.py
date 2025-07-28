#!/usr/bin/env python3
"""
Comprehensive test suite for hud-browser MCP environment.

This script tests:
- MCP protocol basics (initialize, tools, resources)
- Core functionality (computer, playwright, apps)
- Evaluation system (setup, evaluate, problems)
- Error handling and edge cases
- VNC access and telemetry

Usage:
    python test_hud_browser.py [--stdio | --http]
"""

import asyncio
import json
import sys
import time
import argparse
from typing import Dict, Any, List
import logging
from mcp_use import MCPClient
from hud.mcp_agent import ClaudeMCPAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class HudBrowserTester:
    def __init__(self, use_stdio: bool = True):
        self.use_stdio = use_stdio
        self.client = None
        self.session = None
        self.agent = None
        self.test_results = []
        
    async def setup_client(self):
        """Setup MCP client based on transport method."""
        if self.use_stdio:
            config = {
                "mcpServers": {
                    "browser": {
                        "command": "docker",
                        "args": [
                            "run", "--rm", "-i",
                            "-p", "8080:8080",  # VNC port
                            "-e", "LAUNCH_APPS=todo",
                            "-e", "BROWSER_URL=http://localhost:3000",
                            "hud-browser"
                        ],
                    }
                }
            }
        else:
            config = {
                "mcpServers": {
                    "browser": {
                        "url": "http://localhost:8041/mcp"
                    }
                }
            }
        
        self.client = MCPClient.from_dict(config)
        self.session = await self.client.create_session("browser")
        
        # Create agent with comprehensive tool access
        self.agent = ClaudeMCPAgent(
            client=self.client,
            model="claude-sonnet-4-20250514",
            allowed_tools=["computer", "playwright", "launch_app", "setup", "evaluate", "api_request"],
        )
        
    async def teardown_client(self):
        """Cleanup MCP client."""
        if self.client:
            await self.client.close_all_sessions()
    
    def record_test(self, name: str, success: bool, details: str = ""):
        """Record test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append({
            "name": name,
            "success": success,
            "details": details
        })
        logger.info(f"{status}: {name} - {details}")
    
    async def test_mcp_protocol_basics(self):
        """Test basic MCP protocol functionality."""
        logger.info("🔍 Testing MCP Protocol Basics...")
        
        try:
            # Test tools/list
            tools = await self.session.list_tools()
            expected_tools = ["computer", "playwright", "launch_app", "setup", "evaluate", "api_request", "query_database"]
            
            tool_names = [tool.name for tool in tools.tools]
            missing_tools = [tool for tool in expected_tools if tool not in tool_names]
            
            if missing_tools:
                self.record_test("Tools Discovery", False, f"Missing tools: {missing_tools}")
            else:
                self.record_test("Tools Discovery", True, f"Found {len(tool_names)} tools")
            
            # Test resources/list
            resources = await self.session.list_resources()
            resource_uris = [resource.uri for resource in resources.resources]
            
            expected_resources = [
                "evaluators://registry",
                "setup://registry", 
                "problems://registry",
                "telemetry://live"
            ]
            
            missing_resources = [res for res in expected_resources if not any(res in uri for uri in resource_uris)]
            
            if missing_resources:
                self.record_test("Resources Discovery", False, f"Missing resources: {missing_resources}")
            else:
                self.record_test("Resources Discovery", True, f"Found {len(resource_uris)} resources")
                
        except Exception as e:
            self.record_test("MCP Protocol", False, f"Error: {str(e)}")
    
    async def test_core_tools(self):
        """Test core MCP tools."""
        logger.info("🛠️ Testing Core Tools...")
        
        # Test computer tool
        try:
            result = await self.session.call_tool("computer", {"action": "screenshot"})
            self.record_test("Computer Tool", True, "Screenshot captured")
        except Exception as e:
            self.record_test("Computer Tool", False, f"Error: {str(e)}")
        
        # Test launch_app tool
        try:
            result = await self.session.call_tool("launch_app", {"app_name": "todo"})
            if "todo" in str(result.content).lower():
                self.record_test("Launch App Tool", True, "Todo app launched")
            else:
                self.record_test("Launch App Tool", False, f"Unexpected result: {result.content}")
        except Exception as e:
            self.record_test("Launch App Tool", False, f"Error: {str(e)}")
        
        # Test API request tool
        try:
            # Wait a moment for todo app to be ready
            await asyncio.sleep(3)
            result = await self.session.call_tool("api_request", {
                "url": "http://localhost:5000/api/eval/health",
                "method": "GET"
            })
            
            response_data = result.content[0].text if result.content else ""
            if "healthy" in response_data or "200" in response_data:
                self.record_test("API Request Tool", True, "Todo health check successful")
            else:
                self.record_test("API Request Tool", False, f"Health check failed: {response_data}")
        except Exception as e:
            self.record_test("API Request Tool", False, f"Error: {str(e)}")
    
    async def test_evaluation_system(self):
        """Test the evaluation system thoroughly."""
        logger.info("📊 Testing Evaluation System...")
        
        # Test setup tool with problem name
        try:
            result = await self.session.call_tool("setup", {"config": {"name": "todo_basic_usage"}})
            response_text = str(result.content)
            
            if "success" in response_text.lower():
                self.record_test("Setup Tool (Problem)", True, "Basic usage problem setup completed")
            else:
                self.record_test("Setup Tool (Problem)", False, f"Setup failed: {response_text}")
        except Exception as e:
            self.record_test("Setup Tool (Problem)", False, f"Error: {str(e)}")
        
        # Test setup tool with direct function
        try:
            result = await self.session.call_tool("setup", {
                "config": {
                    "function": "todo_seed",
                    "args": {"num_items": 3}
                }
            })
            response_text = str(result.content)
            
            if "success" in response_text.lower():
                self.record_test("Setup Tool (Direct)", True, "Direct function setup completed")
            else:
                self.record_test("Setup Tool (Direct)", False, f"Setup failed: {response_text}")
        except Exception as e:
            self.record_test("Setup Tool (Direct)", False, f"Error: {str(e)}")
        
        # Test evaluate tool with problem name
        try:
            result = await self.session.call_tool("evaluate", {"config": {"name": "todo_basic_usage"}})
            response_text = str(result.content)
            
            if "reward" in response_text.lower() and "done" in response_text.lower():
                self.record_test("Evaluate Tool (Problem)", True, "Basic usage problem evaluation completed")
            else:
                self.record_test("Evaluate Tool (Problem)", False, f"Evaluation failed: {response_text}")
        except Exception as e:
            self.record_test("Evaluate Tool (Problem)", False, f"Error: {str(e)}")
        
        # Test evaluate tool with direct function
        try:
            result = await self.session.call_tool("evaluate", {
                "config": {
                    "function": "todo_completed",
                    "args": {"expected_count": 2}
                }
            })
            response_text = str(result.content)
            
            if "reward" in response_text.lower():
                self.record_test("Evaluate Tool (Direct)", True, "Direct function evaluation completed")
            else:
                self.record_test("Evaluate Tool (Direct)", False, f"Evaluation failed: {response_text}")
        except Exception as e:
            self.record_test("Evaluate Tool (Direct)", False, f"Error: {str(e)}")
    
    async def test_mcp_resources(self):
        """Test MCP resources for registry discovery."""
        logger.info("📋 Testing MCP Resources...")
        
        # Test evaluators registry
        try:
            result = await self.session.read_resource("evaluators://registry")
            data = json.loads(result.contents[0].text)
            
            if "evaluators" in data and len(data["evaluators"]) > 0:
                self.record_test("Evaluators Registry", True, f"Found {len(data['evaluators'])} evaluators")
            else:
                self.record_test("Evaluators Registry", False, "No evaluators found")
        except Exception as e:
            self.record_test("Evaluators Registry", False, f"Error: {str(e)}")
        
        # Test todo-specific evaluators
        try:
            result = await self.session.read_resource("evaluators://todo")
            data = json.loads(result.contents[0].text)
            
            if "evaluators" in data and len(data["evaluators"]) > 0:
                self.record_test("Todo Evaluators", True, f"Found {len(data['evaluators'])} todo evaluators")
            else:
                self.record_test("Todo Evaluators", False, "No todo evaluators found")
        except Exception as e:
            self.record_test("Todo Evaluators", False, f"Error: {str(e)}")
        
        # Test problems registry
        try:
            result = await self.session.read_resource("problems://registry")
            data = json.loads(result.contents[0].text)
            
            if "problems" in data and len(data["problems"]) > 0:
                self.record_test("Problems Registry", True, f"Found {len(data['problems'])} problems")
            else:
                self.record_test("Problems Registry", False, "No problems found")
        except Exception as e:
            self.record_test("Problems Registry", False, f"Error: {str(e)}")
        
        # Test telemetry resource
        try:
            result = await self.session.read_resource("telemetry://live")
            data = json.loads(result.contents[0].text)
            
            if "live_url" in data and "vnc" in data["live_url"]:
                self.record_test("Telemetry Resource", True, f"VNC URL: {data['live_url']}")
            else:
                self.record_test("Telemetry Resource", False, "No VNC URL found")
        except Exception as e:
            self.record_test("Telemetry Resource", False, f"Error: {str(e)}")
    
    async def test_agent_integration(self):
        """Test high-level agent integration."""
        logger.info("🤖 Testing Agent Integration...")
        
        try:
            # Test agent can perform setup task
            result = await self.agent.run("Set up the todo app with test data for evaluation")
            
            if result and "setup" in result.lower():
                self.record_test("Agent Setup Task", True, "Agent completed setup task")
            else:
                self.record_test("Agent Setup Task", False, f"Agent failed: {result}")
        except Exception as e:
            self.record_test("Agent Setup Task", False, f"Error: {str(e)}")
        
        try:
            # Test agent can perform evaluation task
            result = await self.agent.run("Evaluate if at least 2 todos are completed")
            
            if result and ("evaluation" in result.lower() or "completed" in result.lower()):
                self.record_test("Agent Evaluation Task", True, "Agent completed evaluation task")
            else:
                self.record_test("Agent Evaluation Task", False, f"Agent failed: {result}")
        except Exception as e:
            self.record_test("Agent Evaluation Task", False, f"Error: {str(e)}")
    
    async def test_error_handling(self):
        """Test error handling and edge cases."""
        logger.info("⚠️ Testing Error Handling...")
        
        # Test invalid tool call
        try:
            result = await self.session.call_tool("nonexistent_tool", {})
            self.record_test("Invalid Tool", False, "Should have raised an error")
        except Exception as e:
            self.record_test("Invalid Tool", True, "Correctly rejected invalid tool")
        
        # Test invalid setup configuration
        try:
            result = await self.session.call_tool("setup", {"config": {"invalid": "config"}})
            response_text = str(result.content)
            
            if "error" in response_text.lower():
                self.record_test("Invalid Setup Config", True, "Correctly handled invalid config")
            else:
                self.record_test("Invalid Setup Config", False, "Should have returned error")
        except Exception as e:
            self.record_test("Invalid Setup Config", True, f"Correctly raised error: {str(e)}")
        
        # Test invalid resource
        try:
            result = await self.session.read_resource("nonexistent://resource")
            self.record_test("Invalid Resource", False, "Should have raised an error")
        except Exception as e:
            self.record_test("Invalid Resource", True, "Correctly rejected invalid resource")
    
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "="*60)
        print("🧪 HUD BROWSER TEST RESULTS SUMMARY")
        print("="*60)
        
        passed = sum(1 for test in self.test_results if test["success"])
        total = len(self.test_results)
        
        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests PASSED! The hud-browser image is working correctly.")
        else:
            print("⚠️ Some tests FAILED. Check the details below:")
            
        print("\nDetailed Results:")
        print("-" * 40)
        
        for test in self.test_results:
            status = "✅" if test["success"] else "❌"
            print(f"{status} {test['name']}: {test['details']}")
        
        print("\n" + "="*60)
        
        if self.use_stdio:
            print("💡 Manual checks:")
            print("   - VNC Viewer: http://localhost:8080/vnc.html")
            print("   - Todo App: http://localhost:3000 (if launched)")
            print("   - Check container logs for any errors")
    
    async def run_all_tests(self):
        """Run the complete test suite."""
        print("🚀 Starting HUD Browser Comprehensive Test Suite")
        print(f"Transport: {'stdio' if self.use_stdio else 'HTTP'}")
        print("-" * 60)
        
        try:
            await self.setup_client()
            
            # Run test suites
            await self.test_mcp_protocol_basics()
            await self.test_core_tools()
            await self.test_evaluation_system()
            await self.test_mcp_resources()
            await self.test_agent_integration()
            await self.test_error_handling()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            self.record_test("Test Suite", False, f"Critical error: {str(e)}")
            
        finally:
            await self.teardown_client()
            self.print_summary()


async def main():
    parser = argparse.ArgumentParser(description="Test hud-browser MCP environment")
    parser.add_argument(
        "--transport", 
        choices=["stdio", "http"], 
        default="stdio",
        help="Transport method (default: stdio)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    use_stdio = args.transport == "stdio"
    
    if not use_stdio:
        print("⚠️ HTTP transport requires docker-compose to be running:")
        print("   cd environments/simple_browser && docker-compose up -d")
        print()
    
    tester = HudBrowserTester(use_stdio=use_stdio)
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main()) 