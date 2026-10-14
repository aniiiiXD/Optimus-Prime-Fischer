"""Measure policy quality over every labelled board position."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from typing import Any

from ..agents.base import Agent
from ..dataset import minimax_value, other
from .metrics import summarize_latencies


def _empty_bucket() -> dict[str, float | int]:
    return {"positions": 0, "optimal": 0, "blunders": 0, "regret_sum": 0.0}


def _finish_bucket(bucket: dict[str, float | int]) -> dict[str, float | int]:
    count = int(bucket["positions"])
    return {
        **bucket,
        "optimal_accuracy": float(bucket["optimal"]) / count if count else 0.0,
        "blunder_rate": float(bucket["blunders"]) / count if count else 0.0,
        "mean_regret": float(bucket["regret_sum"]) / count if count else 0.0,
    }


def evaluate_positions(agent: Agent, dataset_path: Path) -> dict[str, Any]:
    """Evaluate optimality, regret, and failure patterns on labelled positions."""
    with dataset_path.open(encoding="utf-8") as dataset_file:
        examples = [json.loads(line) for line in dataset_file if line.strip()]

    total = _empty_bucket()
    by_ply: dict[int, dict[str, float | int]] = defaultdict(_empty_bucket)
    by_player: dict[str, dict[str, float | int]] = defaultdict(_empty_bucket)
    chosen_squares: Counter[int] = Counter()
    latencies: list[float] = []
    losing_blunders = 0

    for example in examples:
        board = list(example["board"])
        player = example["current_player"]
        started = time.perf_counter()
        move = agent.select_move(board, player)
        latencies.append((time.perf_counter() - started) * 1_000)
        if not example["legal_mask"][move - 1]:
            raise RuntimeError(f"{agent.name} selected illegal square {move}")

        chosen_squares[move] += 1
        board[move - 1] = player
        chosen_value = -minimax_value("".join(board), other(player))
        best_value = int(example["minimax_value"])
        regret = best_value - chosen_value
        optimal = regret == 0
        blunder = regret > 0
        if chosen_value == -1 and best_value >= 0:
            losing_blunders += 1

        ply = sum(cell != " " for cell in example["board"])
        for bucket in (total, by_ply[ply], by_player[player]):
            bucket["positions"] = int(bucket["positions"]) + 1
            bucket["optimal"] = int(bucket["optimal"]) + int(optimal)
            bucket["blunders"] = int(bucket["blunders"]) + int(blunder)
            bucket["regret_sum"] = float(bucket["regret_sum"]) + regret

    return {
        "scope": "all reachable nonterminal positions; diagnostic, not a held-out test set",
        **_finish_bucket(total),
        "losing_blunders": losing_blunders,
        "losing_blunder_rate": losing_blunders / len(examples),
        "by_ply": {str(ply): _finish_bucket(bucket) for ply, bucket in sorted(by_ply.items())},
        "by_player": {
            player: _finish_bucket(bucket) for player, bucket in sorted(by_player.items())
        },
        "chosen_square_counts": {str(square): chosen_squares[square] for square in range(1, 10)},
        "move_latency": summarize_latencies(latencies),
    }
