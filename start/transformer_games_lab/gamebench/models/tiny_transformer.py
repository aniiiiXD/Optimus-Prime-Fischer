"""A small causal policy Transformer shared by every discrete-action game."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TinyTransformerConfig:
    vocab_size: int
    action_size: int
    context_length: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if min(self.vocab_size, self.action_size, self.context_length, self.d_model, self.n_layers) <= 0:
            raise ValueError("model dimensions must be positive")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class TinyGameTransformer(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.policy_head = nn.Linear(config.d_model, config.action_size)
        self.apply(self._initialize)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one action-logit vector per sequence.

        `tokens` is right-padded with token 0. `lengths` identifies the final
        real position; omit it only for unpadded benchmark batches.
        """
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")
        batch, time = tokens.shape
        if time > self.config.context_length:
            raise ValueError(f"sequence length {time} exceeds context {self.config.context_length}")
        positions = torch.arange(time, device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal_mask = torch.triu(torch.ones(time, time, dtype=torch.bool, device=tokens.device), diagonal=1)
        padding_mask = tokens.eq(0)
        hidden = self.blocks(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        hidden = self.final_norm(hidden)
        if lengths is None:
            final = hidden[:, -1]
        else:
            final = hidden[torch.arange(batch, device=tokens.device), lengths - 1]
        return self.policy_head(final)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
