"""Strict teacher-label format and leakage-aware split for Othello."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

from gamebench.games.othello import OthelloState


@dataclass(frozen=True)
class Example:
    board: tuple[int, ...]
    player: int
    legal: tuple[int, ...]
    teacher_scores: tuple[float, ...]
    result: float | None
    group: str
    teacher_id: str


def transform_action(action: int, rotation: int, reflect: bool) -> int:
    """Transform one row-major square under an element of D4; pass is fixed."""
    if action == 64:
        return 64
    if not 0 <= action < 64:
        raise ValueError(f"invalid action: {action}")
    row, col = divmod(action, 8)
    if reflect:
        col = 7 - col
    for _ in range(rotation):
        row, col = col, 7 - row
    return row * 8 + col


def transform_board(board: tuple[int, ...], rotation: int, reflect: bool) -> tuple[int, ...]:
    transformed = [-1] * 64
    for index, cell in enumerate(board):
        transformed[transform_action(index, rotation, reflect)] = cell
    return tuple(transformed)


def canonical_state_key(example: Example) -> tuple[int, tuple[int, ...]]:
    """Eight geometric board symmetries; side-to-move remains distinct."""
    return example.player, min(
        transform_board(example.board, rotation, reflect)
        for reflect in (False, True)
        for rotation in range(4)
    )


def deduplicate_examples(examples: list[Example]) -> tuple[list[Example], int]:
    """Keep one orientation per canonical state before splitting or fitting."""
    seen: set[tuple[int, tuple[int, ...]]] = set()
    unique: list[Example] = []
    for example in examples:
        key = canonical_state_key(example)
        if key not in seen:
            seen.add(key)
            unique.append(example)
    return unique, len(examples) - len(unique)


def load_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                board = tuple(row["board"])
                player = row["player"]
                legal = tuple(row["legal"])
                scores = tuple(float(score) for score in row["teacher_scores"])
                result = None if row.get("result") is None else float(row["result"])
                group = row["group"]
                teacher_id = row["teacher_id"]
                if (
                    len(board) != 64
                    or any(type(cell) is not int or cell not in (-1, 0, 1) for cell in board)
                    or type(player) is not int
                    or player not in (0, 1)
                    or not legal
                    or len(legal) != len(scores)
                    or any(type(action) is not int or not 0 <= action <= 64 for action in legal)
                    or len(set(legal)) != len(legal)
                    or not all(math.isfinite(score) for score in scores)
                    or (result is not None and result not in (-1.0, 0.0, 1.0))
                    or not isinstance(group, str)
                    or not group
                    or not isinstance(teacher_id, str)
                    or not teacher_id
                ):
                    raise ValueError("invalid board, targets, result, group, or teacher ID")
                state = OthelloState(board=list(board), _current_player=player)
                if state.is_terminal() or set(legal) != set(state.legal_actions()):
                    raise ValueError("legal actions do not match our Othello rules")
                examples.append(Example(board, player, legal, scores, result, group, teacher_id))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not examples:
        raise ValueError(f"empty dataset: {path}")
    teacher_ids = {example.teacher_id for example in examples}
    if len(teacher_ids) != 1:
        raise ValueError(f"mixed teacher configurations: {sorted(teacher_ids)}")
    return examples


def split_examples(
    examples: list[Example], validation_fraction: float, seed: int
) -> tuple[list[Example], list[Example], int]:
    """Split by opening lineage; remove canonical states shared across groups."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    groups = sorted({example.group for example in examples})
    if len(groups) < 2:
        raise ValueError("at least two opening groups are required")
    rng = random.Random(seed)
    rng.shuffle(groups)
    count = max(1, min(len(groups) - 1, round(len(groups) * validation_fraction)))
    validation_groups = set(groups[:count])
    training = [example for example in examples if example.group not in validation_groups]
    validation = [example for example in examples if example.group in validation_groups]
    train_keys = {canonical_state_key(example) for example in training}
    clean_validation = [example for example in validation if canonical_state_key(example) not in train_keys]
    removed = len(validation) - len(clean_validation)
    if not clean_validation:
        raise ValueError("validation empty after symmetry/lineage leakage filtering")
    return training, clean_validation, removed
