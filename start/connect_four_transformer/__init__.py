"""CPU-first Connect Four search, data, and Transformer package."""

from .config import ModelConfig
from .model import ConnectFourTransformer
from .search import AlphaBetaTeacher

__all__ = ["AlphaBetaTeacher", "ConnectFourTransformer", "ModelConfig"]
