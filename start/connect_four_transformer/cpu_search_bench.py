"""Reproduce the Python search-hotpath comparison without changing checkpoints.

The reference methods are copies of the pre-optimization search hotpath. They
are only installed inside this standalone benchmark process, never in play.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

import torch

from connect_four_engine import ConnectFour
from .hybrid_search import TransformerGuidedSearchAgent
from .search import CENTER_FIRST


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "connect_four_distillation/artifacts/checkpoints/candidate_pons_phases_v3_5min.pt"
)


def reference_immediate_wins(game: ConnectFour, player: int) -> list[int]:
    wins = []
    original_player = game.current_player
    for action in game.legal_actions():
        game.current_player = player
        result = game.step(action)
        game.undo()
        game.current_player = original_player
        if result.winner == player:
            wins.append(action)
    return wins


def reference_heuristic(game: ConnectFour) -> int:
    board = game.board
    player = game.current_player
    score = 3 * sum(board[row * 7 + 3] == player for row in range(6))
    score -= 3 * sum(board[row * 7 + 3] == -player for row in range(6))
    weights = (0, 1, 8, 70, 1_000)
    for row in range(6):
        for column in range(7):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if not (0 <= row + 3 * dr < 6 and 0 <= column + 3 * dc < 7):
                    continue
                window = [board[(row + i * dr) * 7 + column + i * dc] for i in range(4)]
                own, opponent = window.count(player), window.count(-player)
                if opponent == 0:
                    score += weights[own]
                elif own == 0:
                    score -= weights[opponent]
    return max(-5_000, min(5_000, score))


def reference_ordered_actions(game: ConnectFour) -> list[int]:
    legal = set(game.legal_actions())
    wins, remaining = [], []
    for action in CENTER_FIRST:
        if action not in legal:
            continue
        actor = game.current_player
        result = game.step(action)
        game.undo()
        (wins if result.winner == actor else remaining).append(action)
    return wins + remaining


def positions(seed: int, count: int) -> list[ConnectFour]:
    rng = random.Random(seed)
    games = []
    while len(games) < count:
        game = ConnectFour()
        for _ in range(rng.randint(8, 22)):
            if game.is_terminal:
                break
            game.step(rng.choice(game.legal_actions()))
        if not game.is_terminal:
            games.append(game)
    return games


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def run(
    checkpoint: Path, games: list[ConnectFour],
    *, depth: int, node_limit: int, reference: bool,
) -> dict:
    original = (
        TransformerGuidedSearchAgent.__dict__["_immediate_wins"],
        TransformerGuidedSearchAgent.__dict__["_heuristic"],
        TransformerGuidedSearchAgent.__dict__["_ordered_actions"],
    )
    if reference:
        TransformerGuidedSearchAgent._immediate_wins = staticmethod(reference_immediate_wins)
        TransformerGuidedSearchAgent._heuristic = staticmethod(reference_heuristic)
        TransformerGuidedSearchAgent._ordered_actions = staticmethod(reference_ordered_actions)
    try:
        agent = TransformerGuidedSearchAgent.from_checkpoint(
            checkpoint, max_depth=depth, node_limit=node_limit,
            model_value_weight=0.0, max_model_evaluations=1,
        )
        records = []
        for game in games:
            before = game.snapshot()
            start = time.perf_counter_ns()
            result = agent.evaluate(game)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            if game.snapshot() != before:
                raise AssertionError("search changed its input board")
            records.append({
                "move": result.best_action + 1,
                "reason": result.reason,
                "depth": result.stats.completed_depth,
                "nodes": result.stats.nodes,
                "elapsed_ms": elapsed_ms,
            })
        times = [row["elapsed_ms"] for row in records]
        return {
            "median_ms": statistics.median(times),
            "p95_ms": percentile(times, 0.95),
            "total_nodes": sum(row["nodes"] for row in records),
            "records": records,
        }
    finally:
        (
            TransformerGuidedSearchAgent._immediate_wins,
            TransformerGuidedSearchAgent._heuristic,
            TransformerGuidedSearchAgent._ordered_actions,
        ) = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=951)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--node-limit", type=int, default=5000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.positions, args.depth, args.node_limit) < 1:
        parser.error("positions, depth, and node-limit must be positive")
    torch.set_num_threads(1)
    games = positions(args.seed, args.positions)
    # Execute both variants on the same boards, model weights, and node limits.
    baseline = run(args.checkpoint, games, depth=args.depth, node_limit=args.node_limit, reference=True)
    optimized = run(args.checkpoint, games, depth=args.depth, node_limit=args.node_limit, reference=False)
    same_decisions = all(
        (left["move"], left["reason"], left["depth"], left["nodes"])
        == (right["move"], right["reason"], right["depth"], right["nodes"])
        for left, right in zip(baseline["records"], optimized["records"])
    )
    if not same_decisions:
        raise AssertionError("optimized search changed a move or search trace")
    report = {
        "checkpoint": str(args.checkpoint), "positions": args.positions,
        "seed": args.seed, "depth": args.depth, "node_limit": args.node_limit,
        "threads": 1, "machine": platform.platform(),
        "same_moves_reasons_depths_and_nodes": same_decisions,
        "reference": baseline, "optimized": optimized,
        "median_speedup": baseline["median_ms"] / optimized["median_ms"],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"reference {baseline['median_ms']:.1f} ms median / {baseline['p95_ms']:.1f} ms p95; "
        f"optimized {optimized['median_ms']:.1f} ms median / {optimized['p95_ms']:.1f} ms p95; "
        f"{report['median_speedup']:.2f}x; identical choices/nodes={same_decisions}"
    )


if __name__ == "__main__":
    main()
