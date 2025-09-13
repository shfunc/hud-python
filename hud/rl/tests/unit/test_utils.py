"""Tests for utility functions."""
from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from hud.rl.types import Episode
from hud.rl.utils import b64_to_pil, blocks_to_images, turn_weights


def test_turn_weights_last():
    """Test last turn weighting."""
    weights = turn_weights(5, scheme="last")
    assert len(weights) == 5
    assert weights == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert sum(weights) == pytest.approx(1.0)


def test_turn_weights_last_k():
    """Test last-k turn weighting."""
    weights = turn_weights(5, scheme="last_k", last_k=3, gamma=0.9)
    assert len(weights) == 5
    assert weights[0] == 0.0  # First turns have no weight
    assert weights[1] == 0.0
    assert weights[-1] > weights[-2]  # More recent turns have higher weight
    assert sum(weights) == pytest.approx(1.0)


def test_turn_weights_all_discounted():
    """Test all turns discounted weighting."""
    weights = turn_weights(4, scheme="all_discounted", gamma=0.9)
    assert len(weights) == 4
    assert all(w > 0 for w in weights)
    assert weights[-1] > weights[0]  # More recent turns weighted higher
    assert sum(weights) == pytest.approx(1.0)


def test_turn_weights_empty():
    """Test weights for empty episode."""
    weights = turn_weights(0)
    assert weights == []


def test_turn_weights_single():
    """Test weights for single turn."""
    weights = turn_weights(1, scheme="last")
    assert weights == [1.0]


def test_b64_to_pil():
    """Test base64 to PIL conversion."""
    # Create a small test image
    img = Image.new("RGB", (10, 10), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Convert back
    result = b64_to_pil(b64_str)
    assert isinstance(result, Image.Image)
    assert result.size == (10, 10)


def test_blocks_to_images():
    """Test converting content blocks to images."""
    # Create test image data
    img = Image.new("RGB", (5, 5), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    blocks = [
        {"type": "text", "text": "Some text"},  # Should be ignored
        {"type": "image", "data": b64_str},
        {"type": "image", "bytes": buffer.getvalue()},
    ]
    
    images = blocks_to_images(blocks)
    assert len(images) == 2
    assert all(isinstance(img, Image.Image) for img in images)


def test_compute_format_penalty():
    """Test format penalty computation."""
    # Episode with no errors
    episode = Episode(
        task_id="test",
        terminal_reward=1.0,
        conversation_history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        turns=[],
        info={}
    )
    
    penalty = compute_format_penalty(episode, penalty=-1.0)
    assert penalty == 0.0
    
    # Episode with error in info
    episode.info["error"] = "Something went wrong"
    penalty = compute_format_penalty(episode, penalty=-1.0)
    assert penalty == -1.0
    
    # Episode with tool error
    episode.conversation_history.append({
        "role": "tool",
        "content": "Error: Tool failed"
    })
    penalty = compute_format_penalty(episode, penalty=-1.0)
    assert penalty == -1.5  # -1.0 for info error, -0.5 for tool error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
