"""
Persistent context manager for 2048 game state.

This module provides file-based persistence for game state that survives
hot-reloads during development with `hud dev`.
"""

import json
import os
from pathlib import Path
import logging
from typing import Optional
import numpy as np

from .game import Game2048

logger = logging.getLogger(__name__)

# State file location - outside /app/src so it's not watched by watchfiles
STATE_FILE = Path("/tmp/text_2048_state.json")


class PersistentGame2048:
    """Wrapper around Game2048 that persists state to disk."""
    
    def __init__(self):
        self.game = Game2048()
        self.load_state()
    
    def load_state(self) -> bool:
        """Load game state from disk if it exists."""
        if not STATE_FILE.exists():
            logger.info("No saved state found, starting fresh game")
            return False
            
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            
            # Restore game state
            self.game.size = len(state['board'])
            self.game.board = np.array(state['board'], dtype=int) # type: ignore
            self.game.score = state['score']
            self.game.moves_made = state['moves']
            self.game.game_over = state['game_over']
            
            logger.info(f"Loaded game state: Score={self.game.score}, Moves={self.game.moves_made}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def save_state(self):
        """Save current game state to disk."""
        try:
            state = self.game.get_state()
            
            # Ensure directory exists
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)
                
            logger.debug(f"Saved game state: Score={self.game.score}")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def move(self, direction: str) -> bool:
        """Make a move and save state."""
        result = self.game.move(direction)
        if result:
            self.save_state()
        return result
    
    def reset(self, size: int = 4):
        """Reset game and clear saved state."""
        self.game.reset(size)
        self.save_state()
        logger.info("Game reset and state cleared")
    
    def clear_state(self):
        """Remove the saved state file."""
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            logger.info("Cleared saved state")
    
    # Delegate all other methods to the wrapped game
    def __getattr__(self, name):
        return getattr(self.game, name)


# Singleton instance
_persistent_game: Optional[PersistentGame2048] = None


def get_game() -> PersistentGame2048:
    """Get or create the persistent game instance."""
    global _persistent_game
    if _persistent_game is None:
        _persistent_game = PersistentGame2048()
    return _persistent_game
