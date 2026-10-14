"""Small, inspectable first-pass Transformer dimensions for Tic-Tac-Toe."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 6
    sequence_length: int = 10
    d_model: int = 32
    n_heads: int = 4
    n_blocks: int = 2
    mlp_hidden: int = 64

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


MODEL_CONFIG = ModelConfig()
