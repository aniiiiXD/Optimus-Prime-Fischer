"""Exact Minimax targets and symmetry-safe splits for controlled experiments."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from tic_tac_toe_transformer.data.symmetry import canonical_board
from tic_tac_toe_transformer.dataset import minimax_value, other


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "tic_tac_toe_transformer"
    / "artifacts"
    / "minimax_positions.jsonl"
)


def load_rows(path: Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def action_value_classes(board_key: str, player: str) -> list[int]:
    """Return loss/draw/win classes 0/1/2; illegal actions use -100."""
    board = list(board_key)
    targets = [-100] * 9
    for action, cell in enumerate(board):
        if cell != " ":
            continue
        board[action] = player
        targets[action] = -minimax_value("".join(board), other(player)) + 1
        board[action] = " "
    return targets


class DecisionDataset(Dataset[dict[str, torch.Tensor]]):
    """Expose policy, action-value, state-value, and legality targets."""

    def __init__(self, rows: list[dict[str, Any]], indices: Iterable[int]) -> None:
        self.rows = rows
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = self.rows[self.indices[item]]
        return {
            "tokens": torch.tensor(row["tokens"], dtype=torch.long),
            "legal_mask": torch.tensor(row["legal_mask"], dtype=torch.bool),
            "target_policy": torch.tensor(row["target_policy"], dtype=torch.float32),
            "optimal_mask": torch.tensor(
                [move in row["optimal_moves"] for move in range(1, 10)],
                dtype=torch.bool,
            ),
            "action_value_targets": torch.tensor(
                action_value_classes(row["board"], row["current_player"]),
                dtype=torch.long,
            ),
            "state_value_target": torch.tensor(row["minimax_value"] + 1, dtype=torch.long),
        }


def grouped_split(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 2026,
) -> dict[str, list[int]]:
    """Split whole rotation/reflection families to prevent symmetry leakage."""
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be nonnegative and sum below one")

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(canonical_board(row["board"]), row["current_player"])].append(index)
    keys = list(groups)
    random.Random(seed).shuffle(keys)

    targets = {
        "test": round(len(rows) * test_fraction),
        "validation": round(len(rows) * validation_fraction),
    }
    result: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for key in keys:
        if len(result["test"]) < targets["test"]:
            split = "test"
        elif len(result["validation"]) < targets["validation"]:
            split = "validation"
        else:
            split = "train"
        result[split].extend(groups[key])
    for indices in result.values():
        indices.sort()
    return result


def nested_training_subset(
    rows: list[dict[str, Any]], train_indices: list[int], fraction: float, seed: int = 2026
) -> list[int]:
    """Select complete symmetry families; fractions are nested and approximate."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in train_indices:
        row = rows[index]
        groups[(canonical_board(row["board"]), row["current_player"])].append(index)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, round(len(train_indices) * fraction))
    selected: list[int] = []
    for key in keys:
        selected.extend(groups[key])
        if len(selected) >= target:
            break
    return sorted(selected)


def all_candidates(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_ids = torch.arange(9, device=device).unsqueeze(0).expand(batch_size, -1)
    candidate_mask = torch.ones_like(candidate_ids, dtype=torch.bool)
    return candidate_ids, candidate_mask


__all__ = [
    "DEFAULT_DATASET",
    "DecisionDataset",
    "action_value_classes",
    "all_candidates",
    "grouped_split",
    "load_rows",
    "nested_training_subset",
]
