"""Pre-norm Transformer blocks for the Tic-Tac-Toe board representation."""

from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadSelfAttention
from .config import MODEL_CONFIG, ModelConfig


class TransformerBlock(nn.Module):
    """One attention + MLP processing step, preserving the input shape."""

    def __init__(self, config: ModelConfig = MODEL_CONFIG) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
        )
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_hidden),
            nn.GELU(),
            nn.Linear(config.mlp_hidden, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Update token representations through attention and the MLP."""
        x = x + self.attention(self.attention_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TicTacToeTransformerBlocks(nn.Module):
    """Stack the configured number of blocks over already embedded tokens."""

    def __init__(self, config: ModelConfig = MODEL_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_blocks)
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, embedded_tokens: torch.Tensor) -> torch.Tensor:
        """Return contextualized tokens with the same shape as the input."""
        if embedded_tokens.ndim != 3 or embedded_tokens.shape[-1] != self.config.d_model:
            raise ValueError(
                "embedded_tokens must have shape "
                f"(batch, tokens, {self.config.d_model})"
            )
        x = embedded_tokens
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


__all__ = ["TransformerBlock", "TicTacToeTransformerBlocks"]
