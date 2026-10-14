"""Move-selection wrapper around a trained Transformer checkpoint."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from tic_tac_toe_lab.tic_tac_toe import available_moves

from ..input import tokenize_position
from ..model import TicTacToePolicy
from ..policy import policy_probabilities
from .base import Agent


class TransformerAgent(Agent):
    name = "transformer"

    def __init__(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model = TicTacToePolicy()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.checkpoint_path = checkpoint_path
        self.checkpoint_metadata = {
            key: value for key, value in checkpoint.items() if key != "model_state_dict"
        }

    @torch.inference_mode()
    def probabilities(
        self, board: Sequence[str], current_player: str
    ) -> torch.Tensor:
        legal_moves = available_moves(list(board))
        if not legal_moves:
            raise ValueError("cannot choose a move from a terminal board")
        tokens = torch.tensor(
            [tokenize_position(list(board), current_player)], dtype=torch.long
        )
        legal_mask = torch.tensor(
            [[square in legal_moves for square in range(1, 10)]], dtype=torch.bool
        )
        return policy_probabilities(self.model(tokens), legal_mask)[0]

    def select_move(self, board: Sequence[str], current_player: str) -> int:
        return int(self.probabilities(board, current_player).argmax().item()) + 1

    def top_moves(
        self, board: Sequence[str], current_player: str, count: int = 3
    ) -> list[tuple[int, float]]:
        probabilities = self.probabilities(board, current_player)
        legal_count = len(available_moves(list(board)))
        indices = torch.argsort(probabilities, descending=True)[: min(count, legal_count)]
        return [
            (int(index.item()) + 1, float(probabilities[index].item()))
            for index in indices
        ]
