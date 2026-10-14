"""Configuration for the first CPU-sized Connect Four Transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_blocks: int = 2
    mlp_hidden: int = 128
    dropout: float = 0.1
    board_tokens: int = 42
    action_count: int = 7

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.n_blocks <= 0 or self.mlp_hidden <= 0:
            raise ValueError("n_blocks and mlp_hidden must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


DEFAULT_CONFIG = ModelConfig()
