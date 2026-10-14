"""Built-in game adapters."""

from gamebench.games.base import GameAdapter, GameSpec, GameState, ObservationSpec
from gamebench.games.connect4 import ConnectFourAdapter
from gamebench.games.othello import OthelloAdapter

__all__ = [
    "ConnectFourAdapter",
    "GameAdapter",
    "GameSpec",
    "GameState",
    "ObservationSpec",
    "OthelloAdapter",
]
