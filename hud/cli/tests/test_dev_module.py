from __future__ import annotations

import sys

import pytest

from hud.cli.dev import run_mcp_dev_server

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Prefers Linux")


def test_run_mcp_dev_server_auto_detect_failure(monkeypatch):
    # Import the dev module file (not the dev function from hud.cli)
    import hud.cli.dev as dev_module

    # Patch at module level where they're defined
    monkeypatch.setattr(dev_module, "should_use_docker_mode", lambda cwd: False)
    monkeypatch.setattr(dev_module, "auto_detect_module", lambda: (None, None))

    # Patch sys.exit to raise SystemExit we can catch
    def _exit(code=0):
        raise SystemExit(code)

    monkeypatch.setattr("sys.exit", _exit)

    with pytest.raises(SystemExit) as exc:
        run_mcp_dev_server(
            module=None,
            stdio=False,
            port=8765,
            verbose=False,
            inspector=False,
            interactive=False,
            watch=None,
            docker=False,
            docker_args=[],
        )
    assert exc.value.code == 1
