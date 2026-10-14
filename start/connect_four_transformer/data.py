"""Dataset loading, player-relative tokenization, and leakage-safe grouping."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from connect_four_engine import find_winner, mirror_board


DEFAULT_DATASET = Path(__file__).parent / "artifacts" / "data" / "positions.jsonl"


def load_rows(path: Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def board_tokens(board: list[int], player: int) -> list[int]:
    return [0 if cell == 0 else (1 if cell == player else 2) for cell in board]


def canonical_position(row: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    """Identity under left/right reflection, including the player to move."""
    board = tuple(int(cell) for cell in row["board"])
    return min(board, mirror_board(board)), int(row["current_player"])


def tactical_target(row: dict[str, Any]) -> list[int]:
    """Return forced winning moves or a unique immediate block, if present."""
    board = list(row["board"])
    player = int(row["current_player"])
    legal = list(row["legal_actions"])
    landing = list(row["landing_rows"])

    def wins(actor: int) -> list[int]:
        actions = []
        for action, landing_row in zip(legal, landing):
            next_board = board.copy()
            next_board[landing_row * 7 + action] = actor
            if find_winner(next_board) == actor:
                actions.append(action)
        return actions

    own = wins(player)
    if own:
        return own
    opponent = wins(-player)
    return opponent if len(opponent) == 1 else []


class SearchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        indices: Iterable[int],
        *,
        mirror_probability: float = 0.0,
        hard_tactical_targets: bool = False,
        tactical_weight: float = 1.0,
    ) -> None:
        if not 0 <= mirror_probability <= 1:
            raise ValueError("mirror_probability must be in [0, 1]")
        if tactical_weight < 1:
            raise ValueError("tactical_weight must be at least 1")
        self.rows = rows
        self.indices = list(indices)
        self.mirror_probability = mirror_probability
        self.hard_tactical_targets = hard_tactical_targets
        self.tactical_weight = tactical_weight
        self.critical_actions = {
            index: tactical_target(rows[index]) for index in self.indices
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = self.indices[item]
        row = self.rows[index]
        board = list(row["board"])
        legal = list(row["legal_actions"])
        landing_by_action = dict(zip(legal, row["landing_rows"]))
        policy_by_action = list(row["policy_target"])
        q_by_action = list(row["action_values"])
        critical = self.critical_actions.get(index, [])
        if critical and self.hard_tactical_targets:
            policy_by_action = [
                1.0 / len(critical) if action in critical else 0.0
                for action in range(7)
            ]
        if self.mirror_probability and random.random() < self.mirror_probability:
            board = [
                board[row_index * 7 + column]
                for row_index in range(6)
                for column in range(6, -1, -1)
            ]
            legal = sorted(6 - action for action in legal)
            landing_by_action = {
                6 - action: landing_row
                for action, landing_row in landing_by_action.items()
            }
            policy_by_action.reverse()
            q_by_action.reverse()
            critical = [6 - action for action in critical]
        count = len(legal)
        candidates = legal + [7] * (7 - count)
        landing_rows = [landing_by_action[action] for action in legal] + [6] * (7 - count)
        policy = [float(policy_by_action[action]) for action in legal] + [0.0] * (7 - count)
        q_values = [float(q_by_action[action]) for action in legal] + [0.0] * (7 - count)
        row_weight = float(row.get("curriculum_weight", 1.0))
        if row_weight < 1:
            raise ValueError("curriculum_weight must be at least 1")
        return {
            "board_tokens": torch.tensor(
                board_tokens(board, row["current_player"]), dtype=torch.long
            ),
            "candidate_ids": torch.tensor(candidates, dtype=torch.long),
            "landing_rows": torch.tensor(landing_rows, dtype=torch.long),
            "candidate_mask": torch.tensor([True] * count + [False] * (7 - count)),
            "policy_target": torch.tensor(policy, dtype=torch.float32),
            "q_target": torch.tensor(q_values, dtype=torch.float32),
            "state_value_target": torch.tensor(float(row["state_value"]), dtype=torch.float32),
            "policy_weight": torch.tensor(
                max(self.tactical_weight if critical else 1.0, row_weight),
                dtype=torch.float32,
            ),
            "critical_mask": torch.tensor(
                [action in critical for action in legal] + [False] * (7 - count),
                dtype=torch.bool,
            ),
        }


def grouped_split(
    rows: list[dict[str, Any]], *, seed: int = 2026
) -> dict[str, list[int]]:
    """Keep every opening group in one train/validation/test partition."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["opening_group"])].append(index)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda key: len(groups[key]), reverse=True)
    targets = {
        "train": max(1, round(len(rows) * 0.70)),
        "validation": max(1, round(len(rows) * 0.15)),
        "test": max(1, round(len(rows) * 0.15)),
    }
    splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for key in keys:
        destination = min(
            splits,
            key=lambda name: len(splits[name]) / targets[name],
        )
        splits[destination].extend(groups[key])
    for indices in splits.values():
        indices.sort()
    if not all(splits.values()):
        raise ValueError("dataset needs more independent opening groups for a three-way split")
    return splits


__all__ = ["DEFAULT_DATASET", "SearchDataset", "board_tokens", "grouped_split", "load_rows"]
