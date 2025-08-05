#!/usr/bin/env python3
"""
Simple Docker MCP Server Debugger

Usage: python docker_debug.py <docker-image>
Example: python docker_debug.py hudpython/gmail-clone:latest
"""

import asyncio
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional

# ANSI color codes for better visual clarity
# Enable ANSI colors on Windows
if sys.platform == "win32":
    import os

    os.system("")  # Enable ANSI escape sequences on Windows


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    GOLD = "\033[33m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


# Configure logging to stderr with minimal format
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def timestamp():
    """Get minimal timestamp HH:MM:SS"""
    return datetime.now().strftime("%H:%M:%S")


def log_phase(phase_num: int, title: str):
    """Log a phase header with nice formatting"""
    logger.info(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.ENDC}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}PHASE {phase_num}: {title}{Colors.ENDC}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.ENDC}\n")


def log_command(cmd: list):
    """Log the command being executed"""
    logger.info(f"{Colors.BOLD}$ {' '.join(cmd)}{Colors.ENDC}")


def log_success(message: str):
    """Log a success message"""
    logger.info(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")


def log_error(message: str):
    """Log an error message"""
    logger.info(f"{Colors.RED}❌ {message}{Colors.ENDC}")


def log_info(message: str):
    """Log an info message"""
    logger.info(f"[{timestamp()}] {message}")


def log_stdio(message: str):
    """Log STDIO communication in gold"""
    logger.info(f"{Colors.GOLD}[STDIO] {message}{Colors.ENDC}")


def log_stderr(message: str):
    """Log STDERR output in gray"""
    logger.info(f"{Colors.GRAY}[STDERR] {message}{Colors.ENDC}")


# Hint registry with patterns and priorities (higher number = higher priority)
HINT_REGISTRY = [
    {
        "patterns": [r"Can't connect to display", r"X11", r"DISPLAY.*not set", r"Xlib.*error"],
        "priority": 10,
        "hint": """GUI environment needs X11. Common fixes:
   - Start Xvfb before importing GUI libraries in your entrypoint
   - Use a base image with X11 pre-configured (e.g., hudpython/novnc-base)
   - Delay GUI imports until after X11 is running""",
    },
    {
        "patterns": [r"ModuleNotFoundError", r"ImportError", r"No module named"],
        "priority": 9,
        "hint": """Missing Python dependencies. Check:
   - Is pyproject.toml complete with all dependencies?
   - Did 'uv pip install' run successfully in Dockerfile?
   - Recommendation: Use 'uv' for faster, more reliable installs""",
    },
    {
        "patterns": [r"json\.decoder\.JSONDecodeError", r"Expecting value.*line.*column"],
        "priority": 8,
        "hint": """Invalid JSON-RPC communication. Check:
   - MCP server is using proper JSON-RPC format
   - No debug prints are corrupting stdout
   - Character encoding is UTF-8""",
    },
    {
        "patterns": [r"Permission denied", r"EACCES", r"Operation not permitted"],
        "priority": 7,
        "hint": """Container permission issues. Try:
   - Running non-root user in Dockerfile
   - Setting proper file permissions
   - Using --privileged flag if absolutely needed""",
    },
    {
        "patterns": [r"Cannot allocate memory", r"killed", r"OOMKilled"],
        "priority": 6,
        "hint": """Container resource limits. Consider:
   - Increasing Docker memory limits
   - Optimizing memory usage in your code
   - Checking for memory leaks""",
    },
    {
        "patterns": [r"bind.*address already in use", r"EADDRINUSE", r"port.*already allocated"],
        "priority": 5,
        "hint": """Port conflict detected. Options:
   - Use a different port
   - Check if another container is running
   - Ensure proper cleanup in previous runs""",
    },
    {
        "patterns": [r"FileNotFoundError", r"No such file or directory"],
        "priority": 4,
        "hint": """File or directory missing. Check:
   - All required files are COPYed in Dockerfile
   - Working directory is set correctly
   - File paths are correct for the container environment""",
    },
    {
        "patterns": [r"AttributeError", r"NameError", r"TypeError"],
        "priority": 3,
        "hint": """Python runtime error. Debug with:
   - Run: docker run --rm <image> python -c "import your_module"
   - Check for missing environment variables
   - Verify all dependencies are installed""",
    },
    {
        "patterns": [r"Traceback.*most recent call last", r"Exception"],
        "priority": 2,
        "hint": """Server crashed during startup. Common causes:
   - Missing environment variables
   - Import errors in your module
   - Initialization code failing""",
    },
    {
        "patterns": [r"timeout", r"timed out"],
        "priority": 1,
        "hint": """Server taking too long to start. Consider:
   - Using @mcp_initialize_wrapper() for heavy initialization
   - Moving slow operations to setup() tool instead
   - Checking for deadlocks or infinite loops""",
    },
    {
        "patterns": [r"psutil not installed", r"No module named 'psutil'"],
        "priority": 8,
        "hint": """psutil module required for resource monitoring. Install with:
   - pip install psutil
   - Or add to pyproject.toml dependencies
   - Resource monitoring will be skipped without it""",
    },
]


def analyze_error_for_hints(error_text: str) -> Optional[str]:
    """Analyze error text and return the highest priority matching hint."""
    if not error_text:
        return None

    matches = []
    for hint_data in HINT_REGISTRY:
        for pattern in hint_data["patterns"]:
            if re.search(pattern, error_text, re.IGNORECASE):
                matches.append((hint_data["priority"], hint_data["hint"]))
                break  # Only need one pattern match per hint

    if matches:
        # Sort by priority (highest first) and return the top hint
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[0][1]

    return None


def log_hint(hint: str):
    """Log a hint message"""
    logger.info(f"\n{Colors.YELLOW}💡 Hint: {hint}{Colors.ENDC}")


def show_progress_bar(completed: int, total: int):
    """Show a visual progress bar of phases completed"""
    filled = "█" * completed
    empty = "░" * (total - completed)
    percentage = (completed / total) * 100

    logger.info(
        f"\n{Colors.BOLD}Progress: [{filled}{empty}] {completed}/{total} phases ({percentage:.0f}%){Colors.ENDC}"
    )

    if completed == 0:
        logger.info(f"{Colors.RED}Failed at Phase 1 - Docker container startup{Colors.ENDC}")
    elif completed == 1:
        logger.info(f"{Colors.YELLOW}Failed at Phase 2 - MCP initialization{Colors.ENDC}")
    elif completed == 2:
        logger.info(f"{Colors.YELLOW}Failed at Phase 3 - Tool discovery{Colors.ENDC}")
    elif completed == 3:
        logger.info(f"{Colors.YELLOW}Failed at Phase 4 - Remote deployment readiness{Colors.ENDC}")
    elif completed == 4:
        logger.info(
            f"{Colors.YELLOW}Failed at Phase 5 - Concurrent clients & resources{Colors.ENDC}"
        )
    elif completed == 5:
        logger.info(f"{Colors.GREEN}All phases completed successfully!{Colors.ENDC}")


async def debug_mcp_docker(image: str) -> None:
    """Debug a Docker MCP server step by step."""

    logger.info(f"\n{Colors.BOLD}🔍 Docker MCP Server Debugger{Colors.ENDC}")
    logger.info(f"{Colors.GRAY}Image: {image}{Colors.ENDC}")

    # Show extra docker args if provided
    extra_args = getattr(__builtins__, "_docker_extra_args", [])
    if extra_args:
        logger.info(f"{Colors.GRAY}Extra args: {' '.join(extra_args)}{Colors.ENDC}")

    logger.info(f"{Colors.GRAY}Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")

    # Explain color coding
    logger.info(f"\n{Colors.BOLD}Color Key:{Colors.ENDC}")
    logger.info(f"  {Colors.BOLD}■{Colors.ENDC} Commands (bold)")
    logger.info(f"  {Colors.GOLD}■{Colors.ENDC} STDIO (MCP protocol)")
    logger.info(f"  {Colors.GRAY}■{Colors.ENDC} STDERR (container logs)")
    logger.info(f"  {Colors.GREEN}■{Colors.ENDC} Success messages")
    logger.info(f"  {Colors.RED}■{Colors.ENDC} Error messages")
    logger.info(f"  ■ Info messages")

    # Track progress
    phases_completed = 0
    total_phases = 5
    start_time = time.time()

    # Phase 1: Basic Docker Test
    log_phase(1, "Basic Docker Container Test")

    try:
        # Get extra docker args if provided
        extra_args = getattr(__builtins__, "_docker_extra_args", [])
        cmd = ["docker", "run", "--rm"] + extra_args + [image, "echo", "Container OK"]
        log_command(cmd)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            log_success("Docker container starts successfully")
            phases_completed = 1
        else:
            log_error(f"Docker container failed with exit code {result.returncode}")
            if result.stderr:
                logger.info(f"{Colors.RED}Error output: {result.stderr}{Colors.ENDC}")
                # Analyze error for hints
                hint = analyze_error_for_hints(result.stderr)
                if hint:
                    log_hint(hint)
            show_progress_bar(phases_completed, total_phases)
            return
    except Exception as e:
        log_error(f"Docker test failed: {e}")
        show_progress_bar(phases_completed, total_phases)
        return

    # Phase 2: MCP Initialize Test
    log_phase(2, "MCP Server Initialize Test")

    log_info("STDIO is used for MCP protocol, STDERR for container logs")

    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"roots": {"listChanged": True}},
            "clientInfo": {"name": "DebugClient", "version": "1.0.0"},
        },
    }

    try:
        # Get extra docker args if provided
        extra_args = getattr(__builtins__, "_docker_extra_args", [])
        cmd = ["docker", "run", "--rm", "-i"] + extra_args + [image]
        log_command(cmd)

        log_stdio(f"Sending: {json.dumps(init_request)}")

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Send initialize
        proc.stdin.write(json.dumps(init_request) + "\n")
        proc.stdin.flush()

        # Collect stderr in background
        import threading

        stderr_lines = []

        def read_stderr():
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    log_stderr(line)
                    stderr_lines.append(line)

        stderr_thread = threading.Thread(target=read_stderr)
        stderr_thread.daemon = True
        stderr_thread.start()

        # Wait for response
        response = None
        start = time.time()
        while time.time() - start < 15:
            line = proc.stdout.readline()
            if line:
                try:
                    response = json.loads(line)
                    if response.get("id") == 1:
                        log_stdio(f"Received: {json.dumps(response)}")
                        break
                except:
                    continue

        # Give stderr thread time to collect output
        time.sleep(0.5)

        if response and "result" in response:
            log_success("MCP server initialized successfully")
            server_info = response["result"].get("serverInfo", {})
            log_info(
                f"Server: {server_info.get('name', 'Unknown')} v{server_info.get('version', '?')}"
            )

            # Show capabilities
            caps = response["result"].get("capabilities", {})
            if caps:
                log_info(f"Capabilities: {', '.join(caps.keys())}")
            phases_completed = 2
        else:
            log_error("No valid MCP response received")

            # Analyze stderr for hints
            if stderr_lines:
                all_stderr = "\n".join(stderr_lines)
                hint = analyze_error_for_hints(all_stderr)
                if hint:
                    log_hint(hint)
            else:
                # No stderr output, likely stdout pollution
                log_hint("""MCP requires clean stdout. Ensure:
   - All print() statements use file=sys.stderr
   - Logging is configured to use stderr
   - No libraries are printing to stdout""")

            show_progress_bar(phases_completed, total_phases)
            proc.terminate()
            proc.wait(timeout=5)
            return

        proc.terminate()
        proc.wait(timeout=5)

    except Exception as e:
        log_error(f"MCP test failed: {e}")
        # Try to analyze the exception for hints
        hint = analyze_error_for_hints(str(e))
        if hint:
            log_hint(hint)
        show_progress_bar(phases_completed, total_phases)
        return

    # Phase 3: Tool Discovery
    log_phase(3, "MCP Tool Discovery Test")

    try:
        from hud.mcp import MCPClient

        # Get extra docker args if provided
        extra_args = getattr(__builtins__, "_docker_extra_args", [])
        mcp_config = {
            "test": {"command": "docker", "args": ["run", "--rm", "-i"] + extra_args + [image]}
        }

        cmd = ["docker"] + mcp_config["test"]["args"]
        log_command(cmd)

        log_info("Creating MCP client via hud...")
        client = MCPClient(mcp_config=mcp_config, verbose=False)

        await client.initialize()

        # Wait for initialization
        log_info("Waiting for server initialization...")
        await asyncio.sleep(5)

        # Get tools
        tools = client.get_available_tools()

        if tools:
            log_success(f"Found {len(tools)} tools")

            # Check for lifecycle tools
            tool_names = [t.name for t in tools]
            has_setup = "setup" in tool_names
            has_evaluate = "evaluate" in tool_names

            log_info(
                f"Lifecycle tools: setup={'✅' if has_setup else '❌'}, evaluate={'✅' if has_evaluate else '❌'}"
            )

            # Check for interaction tools
            interaction_tools = [
                name
                for name in tool_names
                if name in ["computer", "playwright", "click", "type", "interact"]
            ]
            if interaction_tools:
                log_info(f"Interaction tools: {', '.join(interaction_tools)}")
            else:
                log_info("Interaction tools: None found")

            # List all tools
            log_info(f"All tools: {', '.join(tool_names)}")

            # Try to list resources
            try:
                session = client._sessions.get("test")
                if session and hasattr(session, "list_resources"):
                    resources = await session.list_resources()
                    if resources:
                        log_info(
                            f"Found {len(resources)} resources: {', '.join(r.uri for r in resources[:3])}..."
                        )
            except:
                pass

            # Check if we have the minimum required tools
            if has_setup and has_evaluate:
                phases_completed = 3
            else:
                log_error("Missing required lifecycle tools (setup/evaluate)")
                log_hint("""Lifecycle tools missing. Ensure:
   - @mcp.tool() decorator is used on setup/evaluate functions
   - Tools are registered before mcp.run()
   - No import errors preventing tool registration""")
                await client.close()
                show_progress_bar(phases_completed, total_phases)
                return

        else:
            log_error("No tools found")
            log_hint("""Lifecycle tools missing. Ensure:
   - @mcp.tool() decorator is used on setup/evaluate functions
   - Tools are registered before mcp.run()
   - No import errors preventing tool registration""")
            await client.close()
            show_progress_bar(phases_completed, total_phases)
            return

        # Keep client open for Phase 4
        # await client.close()

    except Exception as e:
        log_error(f"Tool discovery failed: {e}")
        if "verbose" in str(e).lower():
            # If error is about verbose mode, show simpler error
            logger.info(f"{Colors.GRAY}Error details hidden (verbose mode issue){Colors.ENDC}")
        else:
            import traceback

            error_details = traceback.format_exc()
            logger.info(f"{Colors.RED}{error_details}{Colors.ENDC}")

            # Analyze error for hints
            hint = analyze_error_for_hints(error_details)
            if hint:
                log_hint(hint)
        show_progress_bar(phases_completed, total_phases)
        return

    # Phase 4: Remote Deployment Readiness
    log_phase(4, "Remote Deployment Readiness")

    try:
        log_info("Testing setup and evaluate tools...")

        # Test setup tool
        setup_success = False
        if "setup" in [t.name for t in tools]:
            try:
                log_info("Calling setup tool (no params to test existence)...")
                setup_result = await client.call_tool("setup", {})

                # Even if it errors, if we get a response it means the tool exists
                if hasattr(setup_result, "isError") and setup_result.isError:
                    log_info(
                        f"Setup tool exists but returned error (expected): {setup_result.content[0].text if setup_result.content else 'Unknown error'}"
                    )
                    setup_success = True  # Tool exists, that's what we're checking
                elif isinstance(setup_result, dict) and "status" in setup_result:
                    log_success(f"Setup tool returned: {setup_result}")
                    setup_success = setup_result.get("status") == "success"
                else:
                    log_success(f"Setup tool exists and returned: {type(setup_result)}")
                    setup_success = True
            except Exception as e:
                log_error(f"Setup tool failed: {e}")

        # Test evaluate tool
        evaluate_success = False
        if "evaluate" in [t.name for t in tools]:
            try:
                log_info("Calling evaluate tool (no params to test existence)...")
                eval_result = await client.call_tool("evaluate", {})

                # Even if it errors, if we get a response it means the tool exists
                if hasattr(eval_result, "isError") and eval_result.isError:
                    log_info(
                        f"Evaluate tool exists but returned error (expected): {eval_result.content[0].text if eval_result.content else 'Unknown error'}"
                    )
                    evaluate_success = True  # Tool exists, that's what we're checking
                elif (
                    isinstance(eval_result, dict)
                    and "reward" in eval_result
                    and "done" in eval_result
                ):
                    log_success(
                        f"Evaluate tool returned: reward={eval_result['reward']}, done={eval_result['done']}"
                    )
                    evaluate_success = True
                else:
                    log_success(f"Evaluate tool exists and returned: {type(eval_result)}")
                    evaluate_success = True
            except Exception as e:
                log_error(f"Evaluate tool failed: {e}")

        # Check resources
        log_info("Checking MCP resources...")
        resources_found = []
        try:
            session = client._sessions.get("test")
            if session and hasattr(session, "connector"):
                resources = await session.connector.list_resources()
                for res in resources.resources:
                    resources_found.append(res.uri)
                    if "telemetry://live" in res.uri:
                        log_info(f"Found telemetry resource: {res.uri}")
                    elif "registry" in res.uri:
                        log_info(f"Found registry resource: {res.uri}")
                if not resources_found:
                    log_info("No resources exposed by this environment")
            else:
                log_info("Session connector not available for resource listing")
        except Exception as e:
            log_info(f"Resource check skipped: {e}")

        # Performance check
        log_info("Checking initialization performance...")
        init_time = time.time() - start_time
        log_info(f"Total initialization time: {init_time:.2f}s")

        if init_time > 30:
            log_error("Initialization took >30s - may be too slow for remote deployment")
            log_hint("""Consider optimizing startup time:
   - Use @mcp_initialize_wrapper() for heavy initialization
   - Move slow operations to setup() tool
   - Pre-build/cache dependencies in Docker image""")

        # Overall phase 4 success check
        if setup_success or evaluate_success:
            phases_completed = 4
            log_success("Remote deployment readiness checks passed")
        else:
            log_error("Missing or failing lifecycle tools")
            await client.close()
            show_progress_bar(phases_completed, total_phases)
            return

        # Close client from Phase 3/4
        await client.close()

    except Exception as e:
        log_error(f"Phase 4 failed: {e}")
        try:
            await client.close()
        except:
            pass
        show_progress_bar(phases_completed, total_phases)
        return

    # Phase 5: Concurrent Clients & Resource Testing
    log_phase(5, "Concurrent Clients & Resource Testing")

    concurrent_clients = []
    try:
        import psutil

        # Get baseline resource usage
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        baseline_cpu = process.cpu_percent(interval=0.1)

        log_info(f"Baseline: Memory={baseline_memory:.1f}MB, CPU={baseline_cpu:.1f}%")

        # Get extra docker args if provided
        extra_args = getattr(__builtins__, "_docker_extra_args", [])

        # Create multiple concurrent clients
        log_info("Creating 3 concurrent MCP clients...")

        for i in range(3):
            client_config = {
                f"test_concurrent_{i}": {
                    "command": "docker",
                    "args": ["run", "--rm", "-i"] + extra_args + [image],
                }
            }

            concurrent_client = MCPClient(mcp_config=client_config, verbose=False)
            await concurrent_client.initialize()
            concurrent_clients.append(concurrent_client)
            log_info(f"Client {i + 1} connected")

        # Test concurrent tool calls
        log_info("Testing concurrent tool calls...")
        tasks = []
        for i, client in enumerate(concurrent_clients):
            if "setup" in [t.name for t in client.get_available_tools()]:
                task = client.call_tool("setup", {"config": {"client_id": i}})
                tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            log_info(f"Concurrent calls: {success_count}/{len(tasks)} succeeded")

        # Check resource usage under load
        await asyncio.sleep(1)  # Let things settle
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        current_cpu = process.cpu_percent(interval=0.1)
        memory_growth = current_memory - baseline_memory

        log_info(
            f"Under load: Memory={current_memory:.1f}MB (+{memory_growth:.1f}MB), CPU={current_cpu:.1f}%"
        )

        if memory_growth > 500:  # More than 500MB growth
            log_error(f"Excessive memory growth: {memory_growth:.1f}MB")
            log_hint("Check for memory leaks in your MCP server")

        # Test clean shutdown
        log_info("Testing clean shutdown of all clients...")
        for i, client in enumerate(concurrent_clients):
            try:
                await client.close()
                log_info(f"Client {i + 1} disconnected")
            except Exception as e:
                log_info(f"Client {i + 1} close error: {e}")

        # Small delay to allow cleanup
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

        try:
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_freed = current_memory - final_memory
            log_info(f"After cleanup: Memory={final_memory:.1f}MB (freed {memory_freed:.1f}MB)")
        except:
            pass

        phases_completed = 5
        log_success("Concurrent client testing completed")

    except ImportError:
        log_error("psutil not installed - skipping resource monitoring")
        log_info("Install with: pip install psutil")

        # Still test basic concurrent connections
        try:
            # Get extra docker args if provided
            extra_args = getattr(__builtins__, "_docker_extra_args", [])

            for i in range(3):
                client_config = {
                    f"test_concurrent_{i}": {
                        "command": "docker",
                        "args": ["run", "--rm", "-i"] + extra_args + [image],
                    }
                }

                concurrent_client = MCPClient(mcp_config=client_config, verbose=False)
                await concurrent_client.initialize()
                concurrent_clients.append(concurrent_client)

            log_success(f"Created {len(concurrent_clients)} concurrent clients")

            for client in concurrent_clients:
                await client.close()

            phases_completed = 5

        except Exception as e:
            log_error(f"Concurrent client test failed: {e}")

    except Exception as e:
        log_error(f"Phase 5 failed: {e}")

    finally:
        # Ensure all clients are closed
        if concurrent_clients:
            log_info("Final cleanup of any remaining clients...")
            for client in concurrent_clients:
                try:
                    await client.close()
                except:
                    pass
            # Small delay for cleanup
            try:
                await asyncio.sleep(0.2)
            except:
                pass

    # All phases completed
    show_progress_bar(phases_completed, total_phases)


if __name__ == "__main__":
    import warnings
    import gc

    if len(sys.argv) < 2:
        print("Usage: python docker_debug.py <docker-image> [docker-args...]")
        print("Example: python docker_debug.py hudpython/gmail-clone:latest")
        print(
            "Example: python docker_debug.py my-env:latest -e BROWSER_PROVIDER=browserbase -e API_KEY=xxx"
        )
        sys.exit(1)

    docker_image = sys.argv[1]
    docker_extra_args = sys.argv[2:] if len(sys.argv) > 2 else []

    # Suppress cleanup warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Store extra args globally so they can be used in docker commands
    import builtins

    setattr(builtins, "_docker_extra_args", docker_extra_args)

    asyncio.run(debug_mcp_docker(docker_image))

    # Force cleanup to avoid warnings
    gc.collect()
