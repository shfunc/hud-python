from __future__ import annotations

from hud.evaluators.inspect import inspect_evaluate


def test_inspect_evaluate_basic():
    """Test basic functionality of inspect_evaluate."""
    result = inspect_evaluate("Test response", "Test answer")

    assert result.score == 0.0
    assert result.reason == "Inspect evaluation not implemented"
    assert result.mode == "inspect"
