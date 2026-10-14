"""Candidate-action and multi-task Transformer experiments for Tic-Tac-Toe."""

from .config import SharedTransformerConfig
from .model import SharedActionTransformer

__all__ = ["SharedActionTransformer", "SharedTransformerConfig"]
