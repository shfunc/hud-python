"""Test utils package imports."""

from __future__ import annotations


def test_utils_imports():
    """Test that utils package can be imported."""
    import hud.utils

    # Check that the module exists
    assert hud.utils is not None

    # Try importing submodules
    from hud.utils import agent, common, config, misc, progress, telemetry

    assert agent is not None
    assert common is not None
    assert config is not None
    assert misc is not None
    assert progress is not None
    assert telemetry is not None
