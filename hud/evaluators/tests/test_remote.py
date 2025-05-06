from __future__ import annotations

import pytest

from hud.evaluators.base import EvaluationResult
from hud.evaluators.remote import _remote_eval_call, remote_evaluate


@pytest.mark.asyncio
async def test_remote_eval_call_success(mocker):
    mock_response = {
        "score": 0.85,
        "reason": "Good match",
        "details": {"relevance": 0.9, "correctness": 0.8},
    }
    mock_make_request = mocker.patch(
        "hud.evaluators.remote.make_request", return_value=mock_response
    )

    result = await _remote_eval_call(
        response="test response", answer="test answer", eval_type="match"
    )

    assert result == mock_response
    mock_make_request.assert_called_once()
    call_args = mock_make_request.call_args[1]
    assert call_args["method"] == "POST"
    assert "evaluations/evaluate" in call_args["url"]
    assert call_args["json"]["response"] == "test response"
    assert call_args["json"]["answer"] == "test answer"
    assert call_args["json"]["type"] == "match"


@pytest.mark.asyncio
async def test_remote_eval_call_with_config(mocker):
    mock_response = {"score": 0.75, "reason": "Good", "details": {}}
    mock_make_request = mocker.patch(
        "hud.evaluators.remote.make_request", return_value=mock_response
    )

    config = {"threshold": 0.8, "strict": True}
    result = await _remote_eval_call(
        response="test response", answer="test answer", eval_type="judge", config=config
    )

    assert result == mock_response
    mock_make_request.assert_called_once()
    call_args = mock_make_request.call_args[1]
    assert call_args["json"]["config"] == config


@pytest.mark.asyncio
async def test_remote_eval_call_failure(mocker):
    mocker.patch("hud.evaluators.remote.make_request", side_effect=Exception("API error"))

    result = await _remote_eval_call(
        response="test response", answer="test answer", eval_type="match"
    )

    assert result["score"] == -1.0
    assert "Remote evaluation failed" in result["reason"]
    assert "API error" in result["reason"]
    assert result["details"] == {}


def test_remote_evaluate(mocker):
    mock_result = {"score": 0.9, "reason": "Excellent match", "details": {"similarity": 0.95}}

    async def mock_remote_call(*args, **kwargs):
        return mock_result

    mocker.patch("hud.evaluators.remote._remote_eval_call", side_effect=mock_remote_call)

    result = remote_evaluate(
        response="test response", answer="test answer", eval_type="custom_eval"
    )

    assert isinstance(result, EvaluationResult)
    assert result.score == 0.9
    assert result.reason == "Excellent match"
    assert result.mode == "custom_eval"
    assert result.criteria_scores == {"similarity": 0.95}


def test_remote_evaluate_missing_fields(mocker):
    mock_result = {"score": 0.8}  # Missing reason and details

    async def mock_remote_call(*args, **kwargs):
        return mock_result

    mocker.patch("hud.evaluators.remote._remote_eval_call", side_effect=mock_remote_call)

    result = remote_evaluate(response="test response", answer="test answer")

    assert result.score == 0.8
    assert result.reason == "Remote evaluation completed"
    assert result.mode == "default"
    assert result.criteria_scores == {}
