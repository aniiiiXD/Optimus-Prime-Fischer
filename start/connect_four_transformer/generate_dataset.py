"""Generate diverse legal positions and label every legal move with CPU search."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from connect_four_engine import ConnectFour, mirror_action, mirror_board
from connect_four_transformer.search import AlphaBetaTeacher, CENTER_FIRST


DEFAULT_OUTPUT = Path(__file__).parent / "artifacts" / "data" / "positions.jsonl"


@dataclass
class Candidate:
    moves: tuple[int, ...]
    board: tuple[int, ...]
    current_player: int
    source_policy: str
    game_id: int
    opening_group: str
    trajectory_result: int = 0


def canonical_identity(board: tuple[int, ...], player: int) -> tuple[tuple[int, ...], int]:
    return min(board, mirror_board(board)), player


def opening_group(moves: list[int], length: int = 6) -> str:
    prefix = tuple(moves[:length])
    mirrored = tuple(mirror_action(move) for move in prefix)
    canonical = min(prefix, mirrored)
    return "-".join(map(str, canonical)) or "empty"


def immediate_wins(game: ConnectFour, player: int) -> list[int]:
    wins = []
    for action in game.legal_actions():
        branch = game.copy()
        branch.current_player = player
        result = branch.step(action)
        if result.winner == player:
            wins.append(action)
    return wins


def choose_move(
    game: ConnectFour,
    mode: str,
    rng: random.Random,
    shallow_teacher: AlphaBetaTeacher,
) -> int:
    legal = list(game.legal_actions())
    if mode == "random":
        return rng.choice(legal)

    wins = immediate_wins(game, game.current_player)
    if wins:
        return min(wins, key=lambda action: CENTER_FIRST.index(action))
    blocks = immediate_wins(game, -game.current_player)
    if blocks:
        return min(blocks, key=lambda action: CENTER_FIRST.index(action))

    if mode == "shallow" and rng.random() >= 0.2:
        return shallow_teacher.evaluate(game, max_depth=2, node_limit=2_000).best_action
    ordered = [action for action in CENTER_FIRST if action in legal]
    if mode in {"center", "tactical", "shallow"}:
        center_weights = (1, 2, 3, 4, 3, 2, 1)
        weights = [center_weights[action] for action in ordered]
        return rng.choices(ordered, weights=weights, k=1)[0]
    if mode == "edge":
        edge_weights = (3, 2, 1, 1, 1, 2, 3)
        return rng.choices(ordered, weights=[edge_weights[action] for action in ordered], k=1)[0]
    if mode == "safe":
        safe = []
        for action in ordered:
            branch = game.copy()
            branch.step(action)
            if not immediate_wins(branch, branch.current_player):
                safe.append(action)
        return rng.choice(safe or ordered)
    if mode == "long":
        non_winning = []
        for action in ordered:
            branch = game.copy()
            if branch.step(action).winner is None:
                non_winning.append(action)
        return rng.choice(non_winning or ordered)
    raise ValueError(f"unknown source mode: {mode}")


def phase(ply: int) -> str:
    if ply <= 8:
        return "opening"
    if ply <= 18:
        return "early_mid"
    if ply <= 30:
        return "late_mid"
    return "late"


def collect_candidates(target: int, seed: int) -> list[Candidate]:
    rng = random.Random(seed)
    shallow_teacher = AlphaBetaTeacher(max_table_entries=100_000)
    modes = ("random", "center", "tactical", "shallow", "long", "edge", "safe")
    mode_weights = (0.18, 0.12, 0.22, 0.28, 0.08, 0.06, 0.06)
    quotas = {
        "opening": round(target * 0.20),
        "early_mid": round(target * 0.30),
        "late_mid": round(target * 0.35),
        "late": target,
    }
    counts = {name: 0 for name in quotas}
    selected: dict[tuple[tuple[int, ...], int], Candidate] = {}
    overflow: list[tuple[tuple[tuple[int, ...], int], Candidate]] = []
    max_games = max(200, target * 20)

    for game_id in range(max_games):
        if len(selected) >= target:
            break
        mode = rng.choices(modes, weights=mode_weights, k=1)[0]
        game = ConnectFour()
        moves: list[int] = []
        game_positions: list[tuple[tuple[tuple[int, ...], int], Candidate, str]] = []
        while not game.is_terminal:
            board = tuple(game.board)
            identity = canonical_identity(board, game.current_player)
            candidate = Candidate(
                moves=tuple(moves),
                board=board,
                current_player=game.current_player,
                source_policy=mode,
                game_id=game_id,
                opening_group=opening_group(moves),
            )
            game_positions.append((identity, candidate, phase(game.move_count)))
            action = choose_move(game, mode, rng, shallow_teacher)
            game.step(action)
            moves.append(action)

        # Limit each trajectory's influence so one long game cannot dominate a
        # split. Keep up to two positions from each game phase.
        final_opening_group = opening_group(moves)
        sampled_positions: list[tuple[tuple[tuple[int, ...], int], Candidate, str]] = []
        for bucket_name in ("opening", "early_mid", "late_mid", "late"):
            bucket_positions = [item for item in game_positions if item[2] == bucket_name]
            rng.shuffle(bucket_positions)
            sampled_positions.extend(bucket_positions[:2])

        for identity, candidate, bucket in sampled_positions:
            if identity in selected:
                continue
            candidate.opening_group = final_opening_group
            if game.winner is None:
                candidate.trajectory_result = 0
            else:
                candidate.trajectory_result = 1 if game.winner == candidate.current_player else -1
            if counts[bucket] < quotas[bucket]:
                selected[identity] = candidate
                counts[bucket] += 1
            else:
                overflow.append((identity, candidate))
            if len(selected) >= target:
                break

    if len(selected) < target:
        rng.shuffle(overflow)
        for identity, candidate in overflow:
            if identity not in selected:
                selected[identity] = candidate
            if len(selected) >= target:
                break
    if len(selected) < target:
        raise RuntimeError(f"collected only {len(selected)} of {target} requested positions")
    candidates = list(selected.values())
    rng.shuffle(candidates)
    return candidates[:target]


def label_candidate(
    candidate: Candidate,
    teacher: AlphaBetaTeacher,
    *,
    depth: int,
    node_limit: int | None,
    temperature: float,
) -> dict[str, Any]:
    game = ConnectFour.from_moves(candidate.moves)
    result = teacher.evaluate(
        game,
        max_depth=depth,
        node_limit=node_limit,
        temperature=temperature,
    )
    legal = list(game.legal_actions())
    landing_rows = [5 - game.column_heights[action] for action in legal]
    policy = [0.0] * 7
    q_values: list[float | None] = [None] * 7
    search_scores: list[int | None] = [None] * 7
    for action in legal:
        policy[action] = result.policy[action]
        q_values[action] = result.action_values[action]
        search_scores[action] = result.action_scores[action]
    canonical_board, _ = canonical_identity(tuple(game.board), game.current_player)
    return {
        "schema_version": 1,
        "board": list(game.board),
        "canonical_board": list(canonical_board),
        "current_player": game.current_player,
        "ply": game.move_count,
        "legal_actions": legal,
        "landing_rows": landing_rows,
        "policy_target": policy,
        "action_values": q_values,
        "search_scores": search_scores,
        "state_value": result.state_value,
        "trajectory_result": candidate.trajectory_result,
        "teacher": "negamax_alpha_beta",
        "teacher_depth_requested": depth,
        "teacher_depth_reached": result.stats.completed_depth,
        "teacher_nodes": result.stats.nodes,
        "teacher_seconds": result.stats.elapsed_seconds,
        "teacher_cache_hits": result.stats.cache_hits,
        "teacher_cutoffs": result.stats.cutoffs,
        "source_policy": candidate.source_policy,
        "game_id": candidate.game_id,
        "opening_group": candidate.opening_group,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--positions", type=int, default=2_000)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--node-limit", type=int, default=20_000)
    parser.add_argument("--temperature", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.positions <= 0 or args.depth <= 0 or args.progress_every <= 0:
        parser.error("positions, depth, and progress-every must be positive")

    candidates = collect_candidates(args.positions, args.seed)
    teacher = AlphaBetaTeacher()
    rows = []
    started = time.perf_counter()
    for index, candidate in enumerate(candidates, start=1):
        rows.append(
            label_candidate(
                candidate,
                teacher,
                depth=args.depth,
                node_limit=args.node_limit,
                temperature=args.temperature,
            )
        )
        if index % args.progress_every == 0 or index == len(candidates):
            elapsed = time.perf_counter() - started
            print(f"labeled {index:>5}/{len(candidates)} | {index / elapsed:,.1f} positions/s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")

    durations = [float(row["teacher_seconds"]) for row in rows]
    nodes = [int(row["teacher_nodes"]) for row in rows]
    phase_counts = {name: sum(phase(int(row["ply"])) == name for row in rows) for name in ("opening", "early_mid", "late_mid", "late")}
    print(f"saved {len(rows):,} positions to {args.output}")
    print(
        f"teacher time median={statistics.median(durations) * 1000:.2f} ms | "
        f"p95={percentile(durations, 0.95) * 1000:.2f} ms | "
        f"mean nodes={statistics.mean(nodes):,.0f}"
    )
    print(f"phase counts: {phase_counts}")


if __name__ == "__main__":
    main()
