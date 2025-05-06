from __future__ import annotations

import base64

import pytest

from hud.evaluators.base import EvaluationResult
from hud.evaluators.judge import (
    _call_eval_endpoint,
    _evaluate_with_llm,
    _is_base64_image,
    _process_input,
    judge,
)


class _MockLLM:
    """Mock LLM for testing."""

    def __init__(self, response_text):
        self.response_text = response_text

    async def ainvoke(self, _prompt: str) -> str:
        return self.response_text


@pytest.mark.parametrize(
    "input_data, expected_result",
    [
        ("Hello world", "Hello world"),
        (123, "123"),
        (["Hello", "world"], ["Hello", "world"]),
        ({"key": "value"}, {"key": "value"}),
        (b"Hello world", base64.b64encode(b"Hello world").decode("utf-8")),
    ],
)
def test_process_input(input_data, expected_result):
    """Test processing various input types."""
    result = _process_input(input_data)
    assert result == expected_result


@pytest.mark.parametrize(
    "input_data, expected_result",
    [
        ("not an image", False),
        ("data:image/png;base64,abc123", True),
        (b"not an image", False),
        (123, False),
    ],
)
def test_is_base64_image(input_data, expected_result):
    """Test base64 image detection."""
    assert _is_base64_image(input_data) == expected_result


def test_is_base64_image_with_signatures(mocker):
    """Test base64 image detection with common image signatures."""
    # Mock base64.b64decode to return different image signatures
    mock_b64decode = mocker.patch("base64.b64decode")

    # Test JPEG signature
    mock_b64decode.return_value = b"\xff\xd8\xff" + b"some data"
    assert _is_base64_image("not_really_base64_but_mocked") is True

    # Test PNG signature
    mock_b64decode.return_value = b"\x89PNG\r\n\x1a\n" + b"some data"
    assert _is_base64_image("not_really_base64_but_mocked") is True

    # Test GIF signature
    mock_b64decode.return_value = b"GIF8" + b"some data"
    assert _is_base64_image("not_really_base64_but_mocked") is True

    # Test RIFF signature (WebP)
    mock_b64decode.return_value = b"RIFF" + b"some data"
    assert _is_base64_image("not_really_base64_but_mocked") is True


@pytest.mark.asyncio
async def test_call_eval_endpoint_success(mocker):
    """Test successful remote evaluation call."""
    mock_response = {
        "score": 0.8,
        "reason": "Good response",
        "criteria_scores": {"relevance": 0.9, "accuracy": 0.7},
    }
    mock_make_request = mocker.patch(
        "hud.evaluators.judge.make_request", return_value=mock_response
    )
    result = await _call_eval_endpoint("test response", "test answer", [], "LLM")
    assert result == mock_response
    mock_make_request.assert_called_once()


@pytest.mark.asyncio
async def test_call_eval_endpoint_failure(mocker):
    """Test remote evaluation call failure."""
    mocker.patch("hud.evaluators.judge.make_request", side_effect=Exception("API error"))
    result = await _call_eval_endpoint("test response", "test answer", [], "LLM")
    assert result["score"] == -1.0
    assert "Remote evaluation failed" in result["reason"]
    assert result["criteria_scores"] == {}


def test_judge_without_llm(mocker):
    """Test judge function without custom LLM."""
    mock_result = {
        "score": 0.9,
        "reason": "Good answer",
        "criteria_scores": {"relevance": 1.0},
    }

    async def mock_endpoint(*args, **kwargs):
        return mock_result

    mocker.patch("hud.evaluators.judge._call_eval_endpoint", mock_endpoint)
    result = judge("test response", "test answer")

    assert result.score == 0.9
    assert result.reason == "Good answer"
    assert result.mode == "LLM"
    assert result.criteria_scores == {"relevance": 1.0}


def test_judge_with_image_answer(mocker):
    """Test judge function with an image as the answer."""
    mock_result = {
        "score": 0.85,
        "reason": "Good image analysis",
        "criteria_scores": {"visual_accuracy": 0.85},
    }

    async def mock_endpoint(*args, **kwargs):
        return mock_result

    mocker.patch("hud.evaluators.judge._call_eval_endpoint", mock_endpoint)

    # Create a mock image
    image_data = b"fake_image_data"
    base64_image = base64.b64encode(image_data).decode("utf-8")
    image_uri = f"data:image/jpeg;base64,{base64_image}"

    result = judge("description of image", image_uri)

    assert result.score == 0.85
    assert result.reason == "Good image analysis"
    assert result.mode == "VLM"  # Should use VLM mode for images
    assert result.criteria_scores == {"visual_accuracy": 0.85}


def test_judge_with_llm(mocker):
    """Test judge function with custom LLM."""
    mock_llm = _MockLLM('{"score": 0.75, "reason": "Pretty good"}')
    mock_result = EvaluationResult(score=0.75, reason="Pretty good", mode="custom_llm")
    mocker.patch("hud.evaluators.judge._evaluate_with_llm", return_value=mock_result)
    result = judge("test response", "test answer", llm=mock_llm)
    assert result.score == 0.75
    assert result.reason == "Pretty good"
    assert result.mode == "custom_llm"


def test_evaluate_with_llm_valid_json():
    """Test _evaluate_with_llm with valid JSON response."""
    llm = _MockLLM('{"score": 0.85, "reason": "The response is accurate and well-structured."}')
    result = _evaluate_with_llm("test response", "test answer", llm)

    assert result.score == 0.85
    assert result.reason == "The response is accurate and well-structured."
    assert result.mode == "custom_llm"


def test_evaluate_with_llm_json_in_text():
    """Test _evaluate_with_llm with JSON embedded in text."""
    llm_response = """
    I've evaluated the response and here's my assessment:
    
    {"score": 0.7, "reason": "Good but could be more detailed"}
    
    I hope this helps!
    """
    llm = _MockLLM(llm_response)
    result = _evaluate_with_llm("test response", "test answer", llm)

    assert result.score == 0.7
    assert result.reason == "Good but could be more detailed"
    assert result.mode == "custom_llm"


def test_evaluate_with_llm_invalid_json():
    """Test _evaluate_with_llm with invalid JSON response."""
    llm = _MockLLM("This is not a JSON response")
    result = _evaluate_with_llm("test response", "test answer", llm)

    assert result.score == 0.5  # Default score for unparseable responses
    assert "Unable to parse LLM response as JSON" in result.reason
    assert result.mode == "custom_llm"


def test_evaluate_with_llm_exception(mocker):
    """Test _evaluate_with_llm when an exception occurs."""
    # Mock the LLM to raise an exception
    failing_llm = _MockLLM("doesn't matter")
    mocker.patch.object(failing_llm, "ainvoke", side_effect=Exception("LLM API error"))

    result = _evaluate_with_llm("test response", "test answer", failing_llm)

    assert result.score == 0.0  # Zero score for errors
    assert "LLM evaluation error: LLM API error" in result.reason
    assert result.mode == "custom_llm"


def test_evaluate_with_llm_with_criteria():
    """Test _evaluate_with_llm with evaluation criteria."""
    llm = _MockLLM('{"score": 0.9, "reason": "Excellent match on all criteria"}')

    # Test with string criteria
    string_criteria = ["Accuracy", "Relevance", "Completeness"]
    result = _evaluate_with_llm("test response", "test answer", llm, criteria=string_criteria)

    assert result.score == 0.9
    assert result.reason == "Excellent match on all criteria"

    # Test with dict criteria
    dict_criteria = [
        {"description": "Factual accuracy", "weight": 0.6},
        {"description": "Grammar and spelling", "weight": 0.4},
    ]
    result = _evaluate_with_llm("test response", "test answer", llm, criteria=dict_criteria)

    assert result.score == 0.9
    assert result.reason == "Excellent match on all criteria"
