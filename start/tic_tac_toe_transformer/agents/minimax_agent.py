"""Perfect-play Minimax baseline."""

from __future__ import annotations

from collections.abc import Sequence

from tic_tac_toe_lab.tic_tac_toe import best_move

from .base import Agent


class MinimaxAgent(Agent):
    name = "minimax"

    def select_move(self, board: Sequence[str], current_player: str) -> int:
        return best_move(list(board), ai_mark=current_player)
