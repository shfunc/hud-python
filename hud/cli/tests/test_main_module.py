"""Tests for hud.cli.__main__ module."""

from __future__ import annotations

import subprocess
import sys


class TestMainModule:
    """Tests for the CLI __main__ module."""

    def test_main_module_imports_correctly(self):
        """Test that __main__.py imports correctly."""
        # Simply importing the module should work without errors
        import hud.cli.__main__

        # Verify the module has the expected attributes
        assert hasattr(hud.cli.__main__, "main")

    def test_main_module_executes(self):
        """Test that running the module as main executes correctly."""
        # Use subprocess to run the module as __main__ and check it doesn't crash
        # We expect it to show help/error since we're not providing arguments
        result = subprocess.run(
            [sys.executable, "-m", "hud.cli"], capture_output=True, text=True, timeout=10
        )

        # Should exit with an error code but not crash
        # (The actual main function will show help or error for missing args)
        assert result.returncode != 0  # CLI should exit with error for no args
