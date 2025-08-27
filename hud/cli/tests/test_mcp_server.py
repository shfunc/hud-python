"""Tests for hud.cli.dev module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from hud.cli.dev import (
    create_proxy_server,
    get_docker_cmd,
    get_image_name,
    inject_supervisor,
    run_mcp_dev_server,
    update_pyproject_toml,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestCreateMCPServer:
    """Test MCP server creation."""

    def test_create_mcp_server(self) -> None:
        """Test that MCP server is created with correct configuration."""
        mcp = create_proxy_server(".", "test-image:latest")
        assert mcp._mcp_server.name == "HUD Dev Proxy - test-image:latest"
        # Proxy server doesn't define its own tools, it forwards to Docker containers


class TestDockerUtils:
    """Test Docker utility functions."""

    def test_get_docker_cmd(self) -> None:
        """Test extracting CMD from Docker image."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '["python", "-m", "server"]'
            mock_run.return_value = mock_result

            cmd = get_docker_cmd("test-image:latest")
            assert cmd is None

    def test_get_docker_cmd_failure(self) -> None:
        """Test handling when Docker inspect fails."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            # check=True causes CalledProcessError on non-zero return
            mock_run.side_effect = subprocess.CalledProcessError(1, "docker inspect")

            cmd = get_docker_cmd("test-image:latest")
            assert cmd is None

    def test_inject_supervisor(self) -> None:
        """Test supervisor injection into Docker CMD."""
        original_cmd = ["python", "-m", "server"]
        modified = inject_supervisor(original_cmd)

        assert modified[0] == "sh"
        assert modified[1] == "-c"
        assert "watchfiles" in modified[2]
        assert "python -m server" in modified[2]


class TestImageResolution:
    """Test image name resolution."""

    def test_get_image_name_override(self) -> None:
        """Test image name with override."""
        name, source = get_image_name(".", "custom-image:v1")
        assert name == "custom-image:v1"
        assert source == "override"

    def test_get_image_name_from_pyproject(self, tmp_path: Path) -> None:
        """Test image name from pyproject.toml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.hud]
image = "my-project:latest"
""")

        name, source = get_image_name(str(tmp_path))
        assert name == "my-project:latest"
        assert source == "cache"

    def test_get_image_name_auto_generate(self, tmp_path: Path) -> None:
        """Test auto-generated image name."""
        test_dir = tmp_path / "my_test_project"
        test_dir.mkdir()

        name, source = get_image_name(str(test_dir))
        assert name == "hud-my-test-project:dev"
        assert source == "auto"

    def test_update_pyproject_toml(self, tmp_path: Path) -> None:
        """Test updating pyproject.toml with image name."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[project]
name = "test"
""")

        update_pyproject_toml(str(tmp_path), "new-image:v1", silent=True)

        content = pyproject.read_text()
        assert "[tool.hud]" in content
        assert 'image = "new-image:v1"' in content


class TestRunMCPDevServer:
    """Test the main server runner."""

    def test_run_dev_server_image_not_found(self) -> None:
        """Test handling when Docker image doesn't exist."""
        import click

        with (
            patch("hud.cli.dev.image_exists", return_value=False),
            patch("click.confirm", return_value=False),
            pytest.raises(click.Abort),
        ):
            run_mcp_dev_server(
                directory=".",
                image="missing:latest",
                build=False,
                no_cache=False,
                transport="http",
                port=8765,
                no_reload=False,
                verbose=False,
                inspector=False,
                no_logs=False,
                docker_args=[],
                interactive=False,
            )
