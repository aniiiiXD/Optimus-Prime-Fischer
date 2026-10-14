"""Common interface for every Tic-Tac-Toe player."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Agent(ABC):
    """Something that chooses one legal square from a board position."""

    name: str

    @abstractmethod
    def select_move(self, board: Sequence[str], current_player: str) -> int:
        """Return a human-numbered square from 1 through 9."""
