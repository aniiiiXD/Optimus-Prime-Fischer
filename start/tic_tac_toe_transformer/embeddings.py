"""Learned token and 2D board-position embeddings for Tic-Tac-Toe."""

from __future__ import annotations

import torch
from torch import nn

from .input import (
    BOARD_SIZE,
    PAD,
    SEQUENCE_LENGTH,
    VOCABULARY_SIZE,
)


class TicTacToeEmbeddings(nn.Module):
    """Turn token IDs into vectors with explicit row/column information.

    Input shape:  ``(batch, 10)`` — 9 row-major squares, then the turn token.
    Output shape: ``(batch, 10, d_model)``.

    For a board square (r, c), its vector is
    ``token_embedding + row_embedding[r] + column_embedding[c]``.
    The final turn token receives its token embedding and a learned turn-slot
    vector. Token ID 0 is padding; ordinary boards use EMPTY=1, X=2, O=3.
    """

    def __init__(self, d_model: int = 32) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")

        self.d_model = d_model
        self.token_embedding = nn.Embedding(
            num_embeddings=VOCABULARY_SIZE,
            embedding_dim=d_model,
            padding_idx=PAD,
        )
        self.row_embedding = nn.Embedding(3, d_model)
        self.column_embedding = nn.Embedding(3, d_model)
        self.turn_slot_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

        # The first nine sequence positions map to the 3x3 board in row-major order.
        square_ids = torch.arange(BOARD_SIZE)
        self.register_buffer("row_ids", square_ids // 3, persistent=False)
        self.register_buffer("column_ids", square_ids % 3, persistent=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed a batch of encoded positions."""
        if token_ids.ndim != 2 or token_ids.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(
                f"token_ids must have shape (batch, {SEQUENCE_LENGTH})"
            )
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long dtype")

        cell_tokens = token_ids[:, :BOARD_SIZE]
        turn_tokens = token_ids[:, BOARD_SIZE:]

        cells = (
            self.token_embedding(cell_tokens)
            + self.row_embedding(self.row_ids).unsqueeze(0)
            + self.column_embedding(self.column_ids).unsqueeze(0)
        )
        turn = self.token_embedding(turn_tokens) + self.turn_slot_embedding
        return torch.cat((cells, turn), dim=1)


__all__ = ["TicTacToeEmbeddings"]
