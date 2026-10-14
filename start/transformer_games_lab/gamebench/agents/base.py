from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Any

from gamebench.games.base import GameState


class Agent(ABC):
    """A policy that selects one action from a cloned game state."""

    name = "agent"

    @abstractmethod
    def select_action(self, state: GameState, rng: random.Random) -> int:
        pass

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__}
