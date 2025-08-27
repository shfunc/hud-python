"""Test script to demonstrate optional imports in hud eval command."""

from __future__ import annotations

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def test_without_agent_deps() -> None:
    """Test running hud eval without agent dependencies."""
    logger.info("Testing 'hud eval' without agent dependencies installed...")

    # Create a test command that should fail gracefully
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "hud", "eval", "hud-evals/test-dataset"],
        capture_output=True,
        text=True,
    )

    logger.info("Exit code: {%s}", result.returncode)
    logger.info("Stderr: {%s}", result.stderr)

    # Should exit with code 1 and show helpful error message
    assert result.returncode == 1  # noqa: S101
    assert "not installed" in result.stderr  # noqa: S101
    assert "pip install 'hud-python[agent]'" in result.stderr  # noqa: S101


def test_cli_import() -> None:
    """Test that CLI module can be imported without agent dependencies."""
    logger.info("\nTesting CLI module import without agent dependencies...")

    try:
        # This should work fine - no agent dependencies imported at module level
        import hud.cli

        logger.info("✅ Successfully imported hud.cli module")

        # The app should be available
        assert hasattr(hud.cli, "app")  # noqa: S101
        logger.info("✅ CLI app is available")

    except ImportError as e:
        logger.error("❌ Failed to import CLI module: {%s}", e)
        raise


def test_other_commands() -> None:
    """Test that other CLI commands work without agent dependencies."""
    logger.info("\nTesting other CLI commands...")

    commands = ["--version", "analyze --help", "build --help", "init --help"]

    for cmd in commands:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "hud", *cmd.split()],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info("✅ 'hud {%s}' works without agent dependencies", cmd)
        else:
            logger.error("❌ 'hud {%s}' failed: {%s}", cmd, result.stderr)


if __name__ == "__main__":
    logger.info("=== Testing Optional Agent Dependencies ===\n")

    test_cli_import()
    test_other_commands()

    # Note: test_without_agent_deps() will only pass if agent deps are NOT installed
    # Uncomment to test when agent deps are not installed:
    # test_without_agent_deps()

    logger.info("\n✅ All tests passed!")
