"""Integration tests for the training pipeline."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from hud.rl.config import Config
from hud.rl.actor import Actor
from hud.rl.learner import GRPOLearner
from hud.rl.buffer import GroupedReplayBuffer
from hud.rl.types import Episode, Turn, Batch, TrainingSample
import torch


@pytest.mark.asyncio
async def test_actor_collect_episodes(test_config, fixtures_dir, mock_openai_client):
    """Test actor can collect episodes."""
    test_config.actor.tasks_file = str(fixtures_dir / "mock_tasks.jsonl")
    
    with patch('src.actor.AsyncOpenAI', return_value=mock_openai_client):
        with patch('src.actor.GenericOpenAIChatAgent') as MockAgent:
            # Mock the agent
            mock_agent = AsyncMock()
            mock_result = Mock()
            mock_result.reward = 1.0
            mock_result.info = {}
            mock_agent.run.return_value = mock_result
            mock_agent.conversation_history = [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"}
            ]
            MockAgent.return_value = mock_agent
            
            actor = Actor(test_config)
            episodes = await actor.collect(2)
            
            assert len(episodes) == 2
            assert all(isinstance(ep, Episode) for ep in episodes)


def test_buffer_operations(mock_episode):
    """Test buffer add and sample operations."""
    buffer = GroupedReplayBuffer(group_size=3)
    
    # Add episodes
    episodes = [mock_episode] * 5
    buffer.add(episodes)
    
    assert buffer.size == 5
    assert buffer.num_successes == 5  # All have reward > 0
    
    # Sample groups
    groups = buffer.sample_groups()
    
    # Should have at least one group
    for key, group in groups.items():
        assert len(group) == 3


def test_config_serialization():
    """Test config can be serialized and deserialized."""
    config = Config()
    
    # To dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "model" in config_dict
    assert "training" in config_dict
    assert "actor" in config_dict
    
    # From dict
    config2 = Config.from_dict(config_dict)
    assert config2.model.base_model == config.model.base_model
    assert config2.training.lr == config.training.lr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_learner_initialization(test_config):
    """Test learner can be initialized (requires GPU)."""
    with patch('src.learner.AutoProcessor.from_pretrained') as mock_proc:
        with patch('src.learner.Qwen2_5_VLForConditionalGeneration.from_pretrained') as mock_model:
            with patch('src.learner.get_peft_model') as mock_peft:
                with patch('src.learner.torch.optim.AdamW') as mock_optimizer:
                    # Mock the model loading
                    mock_model_instance = Mock()
                    mock_model_instance.modules.return_value = []
                    mock_model_instance.parameters.return_value = []
                    mock_model.return_value = mock_model_instance
                    mock_proc.return_value = Mock()
                    mock_peft.return_value = mock_model_instance
                    mock_optimizer.return_value = Mock()
                    
                    learner = GRPOLearner(test_config)
                    
                    assert learner.policy is not None
                    assert learner.ref is not None
                    assert learner.optimizer is not None


def test_training_sample_creation():
    """Test creating training samples."""
    # Create a sample
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    completion_ids = torch.tensor([[4, 5, 6]])
    
    sample = TrainingSample(
        inputs=inputs,
        completion_ids=completion_ids,
        advantage=0.5,
        old_logprobs=torch.tensor([0.1, 0.2, 0.3]),
        ref_logprobs=torch.tensor([0.15, 0.25, 0.35]),
        weight=1.0
    )
    
    assert sample.advantage == 0.5
    assert sample.weight == 1.0
    assert sample.completion_ids.shape == (1, 3)


def test_batch_creation(mock_episode):
    """Test batch creation from episodes and samples."""
    episodes = [mock_episode] * 3
    samples = []  # Would be created from episodes
    
    batch = Batch(samples=samples, episodes=episodes)
    
    assert batch.size == 0  # No samples
    assert len(batch.rewards) == 3
    assert batch.rewards[0] == 1.0




if __name__ == "__main__":
    pytest.main([__file__, "-v"])