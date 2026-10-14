"""Load a checkpoint and combine its move scores with optional one-ply checks."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch

from connect_four_engine import ConnectFour

from .config import ModelConfig
from .data import board_tokens
from .model import ConnectFourTransformer


class TransformerAgent:
    """Transformer policy plus a rules-based safety pass over legal actions."""

    def __init__(self, checkpoint_path: Path, *, tactical_guard: bool = True) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model = ConnectFourTransformer(ModelConfig(**checkpoint["model_config"]))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
        self.tactical_guard = tactical_guard

    @staticmethod
    def _immediate_wins(game: ConnectFour, player: int) -> list[int]:
        """List columns where ``player`` would win on their next drop."""
        wins = []
        for action in game.legal_actions():
            branch = game.copy()
            branch.current_player = player
            if branch.step(action).winner == player:
                wins.append(action)
        return wins

    @classmethod
    def _opponent_wins_after(cls, game: ConnectFour, action: int) -> list[int]:
        """Simulate our drop, then report opponent's immediate winning replies."""
        branch = game.copy()
        actor = branch.current_player
        result = branch.step(action)
        if result.winner == actor or result.is_terminal:
            return []
        return cls._immediate_wins(branch, branch.current_player)

    @torch.inference_mode()
    def inspect(self, game: ConnectFour) -> dict[str, Any]:
        if game.is_terminal:
            raise ValueError("cannot inspect a terminal state")
        decision_started = time.perf_counter_ns()
        legal = list(game.legal_actions())
        count = len(legal)
        candidates = legal + [7] * (7 - count)
        landing = [5 - game.column_heights[action] for action in legal] + [6] * (7 - count)
        mask = [True] * count + [False] * (7 - count)
        started = time.perf_counter_ns()
        output = self.model(
            torch.tensor([board_tokens(game.board, game.current_player)], dtype=torch.long),
            torch.tensor([candidates], dtype=torch.long),
            torch.tensor([landing], dtype=torch.long),
            torch.tensor([mask], dtype=torch.bool),
        )
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        probabilities = torch.softmax(output.policy_logits[0, :count], dim=-1)
        choices = [
            {
                "column": action,
                "probability": float(probabilities[index].item()),
                "q": float(output.action_values[0, index].item()),
            }
            for index, action in enumerate(legal)
        ]
        choices.sort(key=lambda choice: choice["probability"], reverse=True)
        model_top = int(choices[0]["column"])
        tactical_scan_ms = 0.0
        reason = "raw_transformer_argmax"
        if self.tactical_guard:
            scan_started = time.perf_counter_ns()
            actor = game.current_player
            immediate_wins = self._immediate_wins(game, actor)
            opponent_wins_by_action = {
                int(choice["column"]): self._opponent_wins_after(game, int(choice["column"]))
                for choice in choices
            }
            tactical_scan_ms = (time.perf_counter_ns() - scan_started) / 1_000_000
            for choice in choices:
                action = int(choice["column"])
                replies = opponent_wins_by_action[action]
                choice["opponent_winning_replies"] = [reply + 1 for reply in replies]
                choice["safe_next_turn"] = not replies

            if immediate_wins:
                selected = max(
                    (choice for choice in choices if int(choice["column"]) in immediate_wins),
                    key=lambda choice: choice["probability"],
                )
                reason = "take_immediate_win"
            else:
                safe = [choice for choice in choices if choice["safe_next_turn"]]
                if safe:
                    selected = max(safe, key=lambda choice: choice["probability"])
                    reason = "avoid_immediate_loss" if model_top not in {
                        int(choice["column"]) for choice in safe
                    } else "model_pick_is_safe"
                else:
                    # If every move allows a win next turn, prefer the one
                    # that leaves the opponent the fewest winning replies.
                    selected = max(
                        choices,
                        key=lambda choice: (
                            -len(opponent_wins_by_action[int(choice["column"])]),
                            choice["probability"],
                        ),
                    )
                    reason = "minimize_immediate_losses"
            selected_action = int(selected["column"])
        else:
            selected_action = model_top
            for choice in choices:
                choice["opponent_winning_replies"] = []
                choice["safe_next_turn"] = None

        for choice in choices:
            choice["selected"] = int(choice["column"]) == selected_action
        return {
            "choices": choices,
            "model_top_action": model_top,
            "selected_action": selected_action,
            "decision_reason": reason,
            "tactical_guard": self.tactical_guard,
            "tactical_scan_ms": tactical_scan_ms,
            "decision_latency_ms": (time.perf_counter_ns() - decision_started) / 1_000_000,
            "value": float(output.state_value[0].item()),
            "latency_ms": latency_ms,
        }

    def choose(self, game: ConnectFour) -> int:
        return int(self.inspect(game)["selected_action"])


__all__ = ["TransformerAgent"]
