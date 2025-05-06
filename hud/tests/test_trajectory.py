from __future__ import annotations

import pytest
from IPython.display import HTML, Markdown

from hud.trajectory import Trajectory, TrajectoryStep


@pytest.fixture
def sample_trajectory():
    return Trajectory(
        id="traj-789",
        trajectory=[
            TrajectoryStep(
                observation_url="https://example.com/img1.png",
                observation_text="First observation",
                actions=[{"type": "click", "target": "button1"}],
                start_timestamp="2023-01-01T12:00:00Z",
                end_timestamp="2023-01-01T12:01:00Z",
            ),
            TrajectoryStep(
                observation_url=None,
                observation_text="Second observation",
                actions=[{"type": "type", "text": "Hello"}],
                start_timestamp="2023-01-01T12:01:00Z",
                end_timestamp="2023-01-01T12:02:30Z",
            ),
            TrajectoryStep(
                observation_url="https://example.com/img3.png",
                observation_text=None,
                actions=[{"type": "move", "coordinates": [10, 20]}],
                start_timestamp="2023-01-01T12:02:30Z",
                end_timestamp="2023-01-01T12:03:15Z",
            ),
            TrajectoryStep(
                observation_url=None,
                observation_text=None,
                actions=[{"type": "wait"}],
                start_timestamp="2023-01-01T12:03:15Z",
                end_timestamp="2023-01-01T12:04:00Z",
            ),
        ],
    )


def test_trajectory_display_with_observation_url(mocker, sample_trajectory):
    mock_display = mocker.patch("hud.trajectory.display")
    mock_print = mocker.patch("hud.trajectory.print")

    sample_trajectory.display()

    assert mock_display.call_count > 0

    assert mock_print.call_count > 0

    markdown_calls = sum(
        1 for args in mock_display.call_args_list if isinstance(args[0][0], Markdown)
    )
    html_calls = sum(1 for args in mock_display.call_args_list if isinstance(args[0][0], HTML))

    assert markdown_calls > 0
    assert html_calls > 0

    # Verify all steps had their actions printed
    action_prints = sum(1 for args in mock_print.call_args_list if "Actions:" in str(args[0][0]))
    assert action_prints == 4  # One for each step


def test_trajectory_display_with_observation_text(mocker, sample_trajectory):
    mocker.patch(
        "hud.trajectory.display",
    )
    mock_print = mocker.patch("hud.trajectory.print")

    sample_trajectory.display()

    # Count the number of observation text prints
    text_prints = 0
    first_found = False
    second_found = False

    for args in mock_print.call_args_list:
        arg_str = str(args[0][0])
        if "Observation Text:" in arg_str:
            text_prints += 1
            if "First observation" in arg_str:
                first_found = True
            if "Second observation" in arg_str:
                second_found = True

    assert text_prints >= 2  # Steps 1 and 2 have observation text
    assert first_found
    assert second_found


def test_trajectory_display_no_observations(mocker, sample_trajectory):
    mocker.patch("hud.trajectory.display")
    mock_print = mocker.patch("hud.trajectory.print")

    sample_trajectory.display()

    # Verify message for step with no observations
    no_obs_messages = 0
    for args in mock_print.call_args_list:
        if "No visual or text observation provided" in str(args[0][0]):
            no_obs_messages += 1

    assert no_obs_messages >= 1  # Step 4 has no observations


def test_trajectory_display_duration_calculation(mocker, sample_trajectory):
    mocker.patch("hud.trajectory.display")
    mock_print = mocker.patch("hud.trajectory.print")

    sample_trajectory.display()

    # Verify duration calculations
    duration_prints = 0
    total_duration_prints = 0

    for args in mock_print.call_args_list:
        arg_str = str(args[0][0])
        if "Step Duration:" in arg_str:
            duration_prints += 1
        if "Total Duration:" in arg_str:
            total_duration_prints += 1

    assert duration_prints == 4  # One for each step
    assert total_duration_prints == 4  # One for each step


def test_trajectory_display_invalid_timestamp(mocker):
    mocker.patch("hud.trajectory.display")
    mock_print = mocker.patch("hud.trajectory.print")

    trajectory = Trajectory(
        id="traj-error",
        trajectory=[
            TrajectoryStep(
                actions=[{"type": "click"}],
                start_timestamp="2023-01-01T12:00:00Z",
                end_timestamp="2023-01-01T12:01:00Z",
            ),
            TrajectoryStep(
                actions=[{"type": "move"}],
                start_timestamp="invalid-timestamp",  # Invalid timestamp
                end_timestamp="2023-01-01T12:02:00Z",
            ),
        ],
    )

    trajectory.display()

    # Should have handled the invalid timestamp
    error_messages = 0
    for args in mock_print.call_args_list:
        if "Error parsing timestamps" in str(args[0][0]):
            error_messages += 1

    assert error_messages >= 1

    assert error_messages >= 1


def test_trajectory_empty(mocker):
    mocker.patch("hud.trajectory.display")
    mocker.patch("hud.trajectory.print")

    trajectory = Trajectory(id="traj-empty", trajectory=[])

    # Empty trajectory should raise IndexError when we try to display it
    # because it tries to access trajectory[0]
    with pytest.raises(IndexError):
        trajectory.display()
