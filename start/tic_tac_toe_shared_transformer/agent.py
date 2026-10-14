"""Playable agent and inspectable decisions for shared Transformer checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from tic_tac_toe_lab.tic_tac_toe import available_moves
from tic_tac_toe_transformer.agents.base import Agent
from tic_tac_toe_transformer.input import tokenize_position

from .config import SharedTransformerConfig
from .model import SharedActionTransformer


class SharedTransformerAgent(Agent):
    name = "shared-transformer"

    def __init__(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = SharedTransformerConfig(**checkpoint.get("model_config", {}))
        self.model = SharedActionTransformer(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.name = str(checkpoint.get("architecture", self.name))
        self.checkpoint_path = checkpoint_path
        self.metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}

    @torch.inference_mode()
    def inspect(self, board: Sequence[str], current_player: str) -> dict[str, Any]:
        legal_moves = available_moves(list(board))
        if not legal_moves:
            raise ValueError("cannot inspect a terminal position")
        candidate_ids = torch.tensor(
            [[square - 1 for square in legal_moves]], dtype=torch.long
        )
        candidate_mask = torch.ones_like(candidate_ids, dtype=torch.bool)
        tokens = torch.tensor(
            [tokenize_position(list(board), current_player)], dtype=torch.long
        )
        output = self.model(tokens, candidate_ids, candidate_mask)
        probabilities = torch.softmax(output.policy_logits[0], dim=-1)
        q_probabilities = torch.softmax(output.action_value_logits[0], dim=-1)
        state_probabilities = torch.softmax(output.state_value_logits[0], dim=-1)
        choices = []
        for index, move in enumerate(legal_moves):
            q_class = int(q_probabilities[index].argmax().item())
            choices.append(
                {
                    "move": move,
                    "policy_probability": float(probabilities[index].item()),
                    "predicted_outcome": ("loss", "draw", "win")[q_class],
                    "outcome_probabilities": {
                        "loss": float(q_probabilities[index, 0].item()),
                        "draw": float(q_probabilities[index, 1].item()),
                        "win": float(q_probabilities[index, 2].item()),
                    },
                }
            )
        choices.sort(key=lambda item: item["policy_probability"], reverse=True)
        state_class = int(state_probabilities.argmax().item())
        return {
            "choices": choices,
            "predicted_state_outcome": ("loss", "draw", "win")[state_class],
            "state_outcome_probabilities": {
                "loss": float(state_probabilities[0].item()),
                "draw": float(state_probabilities[1].item()),
                "win": float(state_probabilities[2].item()),
            },
        }

    def select_move(self, board: Sequence[str], current_player: str) -> int:
        return int(self.inspect(board, current_player)["choices"][0]["move"])


__all__ = ["SharedTransformerAgent"]
