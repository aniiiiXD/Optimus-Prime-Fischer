"""CPU-first, pluggable infrastructure for game-playing experiments."""

from gamebench.registry import make_agent, make_game

__all__ = ["make_agent", "make_game"]
__version__ = "0.1.0"
