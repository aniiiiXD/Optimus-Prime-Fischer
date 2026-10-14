"""Compose the Tic-Tac-Toe embeddings, Transformer, and move policy."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import TicTacToeTransformerBlocks
from .config import MODEL_CONFIG, ModelConfig
from .embeddings import TicTacToeEmbeddings
from .policy import MovePolicyHead


class TicTacToePolicy(nn.Module):
    """Map a batch of 10-token positions to nine square scores."""

    def __init__(self, config: ModelConfig = MODEL_CONFIG) -> None:
        super().__init__()
        self.embeddings = TicTacToeEmbeddings(d_model=config.d_model)
        self.transformer = TicTacToeTransformerBlocks(config)
        self.policy_head = MovePolicyHead(config)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return unnormalized move logits with shape ``(batch, 9)``."""
        embedded = self.embeddings(token_ids)
        contextual = self.transformer(embedded)
        return self.policy_head(contextual)


__all__ = ["TicTacToePolicy"]
