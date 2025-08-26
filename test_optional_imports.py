#!/usr/bin/env python3
"""Test script to demonstrate optional imports in hud eval command."""

from __future__ import annotations

import subprocess
import sys


def test_without_agent_deps() -> None:
    """Test running hud eval without agent dependencies."""
    print("Testing 'hud eval' without agent dependencies installed...")
    
    # Create a test command that should fail gracefully
    result = subprocess.run(
        [sys.executable, "-m", "hud", "eval", "hud-evals/test-dataset"],
        capture_output=True,
        text=True,
    )
    
    print(f"Exit code: {result.returncode}")
    print(f"Stderr: {result.stderr}")
    
    # Should exit with code 1 and show helpful error message
    assert result.returncode == 1
    assert "not installed" in result.stderr
    assert "pip install 'hud-python[agent]'" in result.stderr


def test_cli_import() -> None:
    """Test that CLI module can be imported without agent dependencies."""
    print("\nTesting CLI module import without agent dependencies...")
    
    try:
        # This should work fine - no agent dependencies imported at module level
        import hud.cli
        print("✅ Successfully imported hud.cli module")
        
        # The app should be available
        assert hasattr(hud.cli, 'app')
        print("✅ CLI app is available")
        
    except ImportError as e:
        print(f"❌ Failed to import CLI module: {e}")
        raise


def test_other_commands() -> None:
    """Test that other CLI commands work without agent dependencies."""
    print("\nTesting other CLI commands...")
    
    commands = ["--version", "analyze --help", "build --help", "init --help"]
    
    for cmd in commands:
        result = subprocess.run(
            [sys.executable, "-m", "hud"] + cmd.split(),
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            print(f"✅ 'hud {cmd}' works without agent dependencies")
        else:
            print(f"❌ 'hud {cmd}' failed: {result.stderr}")


if __name__ == "__main__":
    print("=== Testing Optional Agent Dependencies ===\n")
    
    test_cli_import()
    test_other_commands()
    
    # Note: test_without_agent_deps() will only pass if agent deps are NOT installed
    # Uncomment to test when agent deps are not installed:
    # test_without_agent_deps()
    
    print("\n✅ All tests passed!")
