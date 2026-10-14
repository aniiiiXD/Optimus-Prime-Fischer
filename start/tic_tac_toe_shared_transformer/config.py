"""Configuration for the candidate-action Transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tic_tac_toe_transformer.config import ModelConfig


@dataclass(frozen=True)
class SharedTransformerConfig:
    """Small CPU-first model, matched to the classical board encoder."""

    d_model: int = 32
    n_heads: int = 4
    n_blocks: int = 2
    mlp_hidden: int = 64
    action_count: int = 9
    outcome_count: int = 3

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.action_count <= 0 or self.outcome_count != 3:
            raise ValueError("this experiment expects 9 actions and 3 outcomes")

    def state_encoder_config(self) -> ModelConfig:
        return ModelConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            mlp_hidden=self.mlp_hidden,
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


DEFAULT_CONFIG = SharedTransformerConfig()
