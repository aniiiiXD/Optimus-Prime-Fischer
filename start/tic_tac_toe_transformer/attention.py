"""Educational multi-head self-attention over the complete board input."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import MODEL_CONFIG


class MultiHeadSelfAttention(nn.Module):
    """One attention layer that lets all board tokens exchange information.

    Input and output shape: ``(batch, tokens, d_model)``.
    Attention weights, when requested, have shape
    ``(batch, heads, query_tokens, key_tokens)``.

    There is deliberately no causal mask: all squares and the turn token
    describe the current position and are visible at the same time.
    """

    def __init__(
        self,
        d_model: int = MODEL_CONFIG.d_model,
        n_heads: int = MODEL_CONFIG.n_heads,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # One projection creates Q, K, and V for every token.
        self.qkv_projection = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply scaled dot-product attention to a batch of token vectors."""
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"x must have shape (batch, tokens, {self.d_model})"
            )

        batch_size, sequence_length, _ = x.shape

        # (B, T, 3D) -> three tensors shaped (B, heads, T, head_dim).
        qkv = self.qkv_projection(x)
        qkv = qkv.reshape(
            batch_size, sequence_length, 3, self.n_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(dim=0)

        # Each query scores every key, then uses those scores to mix values.
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(weights, values)

        # Merge the heads back to width d_model and combine their outputs.
        mixed = mixed.transpose(1, 2).contiguous()
        mixed = mixed.reshape(batch_size, sequence_length, self.d_model)
        output = self.output_projection(mixed)

        if return_weights:
            return output, weights
        return output


__all__ = ["MultiHeadSelfAttention"]
