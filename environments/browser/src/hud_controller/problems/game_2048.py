"""2048 game problems."""

from typing import Dict, Any
from .registry import problem


# === BASE PROBLEM CLASSES (for inheritance) ===


class Base2048Setup:
    """Base class for problems with standard 2048 setup."""

    def get_setup(self) -> Dict[str, Any]:
        """Default setup for 2048 game."""
        return {"name": "game_2048_board", "arguments": {"board_size": 4, "target_tile": 2048}}


class Base2048NearWinSetup:
    """Base class for problems that start near winning."""

    def get_setup(self) -> Dict[str, Any]:
        """Setup board near winning state."""
        return {"name": "game_2048_near_win", "arguments": {"target_tile": 2048}}


class Base2048Evaluation:
    """Base class for standard 2048 evaluation."""

    def get_evaluation(self) -> Dict[str, Any]:
        """Default evaluation for reaching 2048."""
        return {"name": "game_2048_max_number", "arguments": {"target": 2048}}


# === CONCRETE PROBLEM CLASSES ===


@problem(
    "2048_reach_128",
    app="2048",
    description="Reach the 128 tile",
    difficulty="easy",
    task_type="game",
)
class Game2048Reach128(Base2048Setup):
    """Reach the 128 tile in 2048."""

    def get_setup(self) -> Dict[str, Any]:
        return {"name": "game_2048_board", "arguments": {"board_size": 4, "target_tile": 128}}

    def get_evaluation(self) -> Dict[str, Any]:
        return {"name": "game_2048_max_number", "arguments": {"target": 128}}


@problem(
    "2048_reach_256",
    app="2048",
    description="Reach the 256 tile",
    difficulty="easy",
    task_type="game",
)
class Game2048Reach256(Base2048Setup):
    """Reach the 256 tile in 2048."""

    def get_setup(self) -> Dict[str, Any]:
        return {"name": "game_2048_board", "arguments": {"board_size": 4, "target_tile": 256}}

    def get_evaluation(self) -> Dict[str, Any]:
        return {"name": "game_2048_max_number", "arguments": {"target": 256}}


@problem(
    "2048_reach_512",
    app="2048",
    description="Reach the 512 tile",
    difficulty="medium",
    task_type="game",
)
class Game2048Reach512(Base2048Setup):
    """Reach the 512 tile in 2048."""

    def get_setup(self) -> Dict[str, Any]:
        return {"name": "game_2048_board", "arguments": {"board_size": 4, "target_tile": 512}}

    def get_evaluation(self) -> Dict[str, Any]:
        return {"name": "game_2048_max_number", "arguments": {"target": 512}}


@problem(
    "2048_reach_1024",
    app="2048",
    description="Reach the 1024 tile",
    difficulty="medium",
    task_type="game",
)
class Game2048Reach1024(Base2048Setup):
    """Reach the 1024 tile in 2048."""

    def get_setup(self) -> Dict[str, Any]:
        return {"name": "game_2048_board", "arguments": {"board_size": 4, "target_tile": 1024}}

    def get_evaluation(self) -> Dict[str, Any]:
        return {"name": "game_2048_max_number", "arguments": {"target": 1024}}


@problem(
    "2048_reach_2048",
    app="2048",
    description="Reach the 2048 tile",
    difficulty="hard",
    task_type="game",
)
class Game2048Reach2048(Base2048Setup, Base2048Evaluation):
    """Reach the 2048 tile in 2048 - the ultimate goal."""

    pass  # Uses defaults from base classes


@problem(
    "2048_efficiency_easy",
    app="2048",
    description="Achieve 50 points per move efficiency",
    difficulty="easy",
    task_type="performance",
)
class Game2048EfficiencyEasy(Base2048Setup):
    """Test efficiency with low requirement."""

    def get_evaluation(self) -> Dict[str, Any]:
        return {"name": "game_2048_efficiency", "arguments": {"min_ratio": 50.0}}


@problem(
    "2048_efficiency_medium",
    app="2048",
    description="Achieve 100 points per move efficiency",
    difficulty="medium",
    task_type="performance",
)
class Game2048EfficiencyMedium(Base2048Setup):
    """Test efficiency with medium requirement."""

    def get_evaluation(self) -> Dict[str, Any]:
        return {"name": "game_2048_efficiency", "arguments": {"min_ratio": 100.0}}


@problem(
    "2048_efficiency_hard",
    app="2048",
    description="Achieve 150 points per move efficiency",
    difficulty="hard",
    task_type="performance",
)
class Game2048EfficiencyHard(Base2048Setup):
    """Test efficiency with high requirement."""

    def get_evaluation(self) -> Dict[str, Any]:
        return {"name": "game_2048_efficiency", "arguments": {"min_ratio": 150.0}}


@problem(
    "2048_score_5000",
    app="2048",
    description="Reach a score of 5000",
    difficulty="easy",
    task_type="score",
)
class Game2048Score5000(Base2048Setup):
    """Reach a score of 5000."""

    def get_evaluation(self) -> Dict[str, Any]:
        return {"name": "game_2048_score_reached", "arguments": {"target_score": 5000}}


@problem(
    "2048_score_10000",
    app="2048",
    description="Reach a score of 10000",
    difficulty="medium",
    task_type="score",
)
class Game2048Score10000(Base2048Setup):
    """Reach a score of 10000."""

    def get_evaluation(self) -> Dict[str, Any]:
        return {"name": "game_2048_score_reached", "arguments": {"target_score": 10000}}


@problem(
    "2048_score_20000",
    app="2048",
    description="Reach a score of 20000",
    difficulty="hard",
    task_type="score",
)
class Game2048Score20000(Base2048Setup):
    """Reach a score of 20000."""

    def get_evaluation(self) -> Dict[str, Any]:
        return {"name": "game_2048_score_reached", "arguments": {"target_score": 20000}}


@problem(
    "2048_near_win_test",
    app="2048",
    description="Test completing a near-win board",
    difficulty="easy",
    task_type="test",
)
class Game2048NearWinTest(Base2048NearWinSetup):
    """Start with a board near winning and complete it."""

    def get_evaluation(self) -> Dict[str, Any]:
        return {"name": "game_2048_game_won", "arguments": {}}


@problem(
    "2048_small_board",
    app="2048",
    description="Play on a 3x3 board and reach 256",
    difficulty="easy",
    task_type="variant",
)
class Game2048SmallBoard:
    """Play on a smaller 3x3 board."""

    def get_setup(self) -> Dict[str, Any]:
        return {"name": "game_2048_board", "arguments": {"board_size": 3, "target_tile": 256}}

    def get_evaluation(self) -> Dict[str, Any]:
        return {"name": "game_2048_max_number", "arguments": {"target": 256}}


@problem(
    "2048_large_board",
    app="2048",
    description="Play on a 5x5 board and reach 512",
    difficulty="medium",
    task_type="variant",
)
class Game2048LargeBoard:
    """Play on a larger 5x5 board."""

    def get_setup(self) -> Dict[str, Any]:
        return {"name": "game_2048_board", "arguments": {"board_size": 5, "target_tile": 512}}

    def get_evaluation(self) -> Dict[str, Any]:
        return {"name": "game_2048_max_number", "arguments": {"target": 512}}


@problem(
    "2048_setup_only",
    app="2048",
    description="Setup-only problem for testing",
    difficulty="easy",
    task_type="setup_test",
)
class Game2048SetupOnly(Base2048Setup):
    """Problem that only does setup, no evaluation - useful for preparation."""

    # No get_evaluation method - this problem only does setup


@problem(
    "2048_test_seed",
    app="2048",
    description="Test with seeded board configuration",
    difficulty="easy",
    task_type="test",
)
class Game2048TestSeed:
    """Test with a seeded board configuration."""

    def get_setup(self) -> Dict[str, Any]:
        return {"name": "game_2048_test_seed", "arguments": {}}

    def get_evaluation(self) -> Dict[str, Any]:
        return {"name": "game_2048_max_number", "arguments": {"target": 512}}
