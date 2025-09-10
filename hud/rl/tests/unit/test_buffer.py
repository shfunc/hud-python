"""Tests for replay buffer."""

import pytest
from hud.rl.buffer import ReplayBuffer, GroupedReplayBuffer
from hud.rl.types import Episode, Turn


def create_test_episode(task_id: str, reward: float, num_turns: int = 3) -> Episode:
    """Create a test episode."""
    turns = []
    for i in range(num_turns):
        turns.append(Turn(
            history_msgs=[{"role": "user", "content": f"Turn {i}"}],
            obs_blocks=[],
            assistant_text=f"Response {i}"
        ))
    
    return Episode(
        task_id=task_id,
        terminal_reward=reward,
        conversation_history=[],
        turns=turns,
        metadata={"test": True}
    )


def test_replay_buffer_add_and_sample():
    """Test adding and sampling from buffer."""
    buffer = ReplayBuffer(max_size=10)
    
    # Add episodes
    episodes = [
        create_test_episode("task1", 1.0),
        create_test_episode("task2", 0.0),
        create_test_episode("task3", 1.0),
    ]
    buffer.add(episodes)
    
    assert buffer.size == 3
    assert buffer.num_successes == 2
    
    # Sample
    sampled = buffer.sample(2)
    assert len(sampled) == 2
    assert all(isinstance(ep, Episode) for ep in sampled)


def test_replay_buffer_success_sampling():
    """Test sampling successful episodes."""
    buffer = ReplayBuffer()
    
    # Add mix of success and failure
    episodes = [
        create_test_episode("fail1", 0.0),
        create_test_episode("success1", 1.0),
        create_test_episode("fail2", -1.0),
        create_test_episode("success2", 2.0),
    ]
    buffer.add(episodes)
    
    # Sample success
    success = buffer.sample_success()
    assert success is not None
    assert success.success
    assert success.terminal_reward > 0


def test_replay_buffer_stats():
    """Test buffer statistics."""
    buffer = ReplayBuffer()
    
    # Empty buffer stats
    stats = buffer.get_stats()
    assert stats["size"] == 0
    
    # Add episodes
    episodes = [
        create_test_episode("task1", 1.0, num_turns=2),
        create_test_episode("task2", -0.5, num_turns=4),
        create_test_episode("task3", 0.5, num_turns=3),
    ]
    buffer.add(episodes)
    
    stats = buffer.get_stats()
    assert stats["size"] == 3
    assert stats["num_successes"] == 2
    assert stats["avg_reward"] == pytest.approx(0.333, rel=0.01)
    assert stats["avg_steps"] == 3.0


def test_grouped_replay_buffer():
    """Test grouped replay buffer."""
    buffer = GroupedReplayBuffer(group_size=3)
    
    # Add episodes with metadata for grouping
    episodes = []
    for i in range(6):
        ep = create_test_episode(f"task{i}", float(i % 2))
        ep.metadata["task_type"] = "type_a" if i < 3 else "type_b"
        episodes.append(ep)
    
    buffer.add(episodes)
    
    # Sample groups
    groups = buffer.sample_groups()
    
    # Should have groups if enough episodes
    assert len(groups) > 0
    for key, group in groups.items():
        assert len(group) == buffer.group_size


def test_buffer_max_size():
    """Test buffer respects max size."""
    buffer = ReplayBuffer(max_size=3)
    
    # Add more than max size
    episodes = [create_test_episode(f"task{i}", float(i)) for i in range(5)]
    buffer.add(episodes)
    
    # Should only keep last 3
    assert buffer.size == 3
    latest = buffer.get_latest(3)
    assert latest[-1].task_id == "task4"  # Most recent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])