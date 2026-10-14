"""Random legal-move baseline."""

from __future__ import annotations

from collections.abc import Sequence
import random

from tic_tac_toe_lab.tic_tac_toe import available_moves

from .base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def select_move(self, board: Sequence[str], current_player: str) -> int:
        del current_player
        legal = available_moves(list(board))
        if not legal:
            raise ValueError("cannot choose a move from a terminal board")
        return self.rng.choice(legal)
