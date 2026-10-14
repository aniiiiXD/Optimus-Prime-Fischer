"""Score Tic-Tac-Toe moves from the Transformer square representations."""

from __future__ import annotations

import torch
from torch import nn

from .config import MODEL_CONFIG, ModelConfig
from .input import BOARD_SIZE


class MovePolicyHead(nn.Module):
    """Apply one shared learned scorer to each of the 9 contextual square rows."""

    def __init__(self, config: ModelConfig = MODEL_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.score_square = nn.Linear(config.d_model, 1)

    def forward(self, contextual_tokens: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized move score per square, shape ``(B, 9)``."""
        if (
            contextual_tokens.ndim != 3
            or contextual_tokens.shape[1] != self.config.sequence_length
            or contextual_tokens.shape[2] != self.config.d_model
        ):
            raise ValueError(
                "contextual_tokens must have shape "
                f"(batch, {self.config.sequence_length}, {self.config.d_model})"
            )

        square_vectors = contextual_tokens[:, :BOARD_SIZE, :]
        return self.score_square(square_vectors).squeeze(-1)


def mask_illegal_moves(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Set illegal move scores to negative infinity before softmax/argmax.

    ``legal_mask`` is supplied by the game rules and must be boolean, with
    ``True`` for a legal square. The function supports batched ``(B, 9)``
    logits or a single unbatched ``(9,)`` row.
    """
    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask must have the same shape")
    if logits.shape[-1] != BOARD_SIZE:
        raise ValueError(f"the final dimension must contain {BOARD_SIZE} moves")
    if legal_mask.dtype != torch.bool:
        raise TypeError("legal_mask must have torch.bool dtype")
    if not torch.all(legal_mask.any(dim=-1)):
        raise ValueError("each position must have at least one legal move")

    return logits.masked_fill(~legal_mask, -torch.inf)


def policy_probabilities(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Return probabilities over legal moves; illegal moves get probability 0."""
    return torch.softmax(mask_illegal_moves(logits, legal_mask), dim=-1)


def choose_move(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Choose the highest-scoring legal square (zero-based index 0–8)."""
    return mask_illegal_moves(logits, legal_mask).argmax(dim=-1)


__all__ = [
    "MovePolicyHead",
    "mask_illegal_moves",
    "policy_probabilities",
    "choose_move",
]
