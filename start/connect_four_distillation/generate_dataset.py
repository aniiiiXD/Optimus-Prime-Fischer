"""Build a Connect Four distillation set from Pascal Pons's exact CPU solver."""

from __future__ import annotations

import argparse
import atexit
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
import sys
import time
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from connect_four_engine import ConnectFour, mirror_board
from connect_four_transformer.generate_dataset import choose_move, opening_group
from connect_four_transformer.search import AlphaBetaTeacher
from connect_four_distillation.oracle import PonsOracle, SOLVER


ROOT = Path(__file__).parent
DEFAULT_OUTPUT = ROOT / "artifacts" / "data" / "pons_exact_v1.jsonl"
DEFAULT_CACHE = ROOT / "artifacts" / "cache" / "pons_labels.sqlite3"
SOLVER_REVISION = "PascalPons/connect4@d6ba50d8aaf2308c769d9bf2abd42d90f34baf41"
_WORKER_ORACLE: PonsOracle | None = None


def _start_worker(timeout: float) -> None:
    global _WORKER_ORACLE
    _WORKER_ORACLE = PonsOracle(timeout_seconds=timeout)
    _WORKER_ORACLE.__enter__()
    atexit.register(_WORKER_ORACLE.__exit__, None, None, None)


def _query_worker(history: str) -> tuple[list[int] | None, float]:
    if _WORKER_ORACLE is None:
        raise RuntimeError("Pons worker was not initialized")
    started = time.perf_counter()
    try:
        scores = _WORKER_ORACLE.analyze(history)["scores"]
    except TimeoutError:
        return None, time.perf_counter() - started
    return scores, time.perf_counter() - started


def canonical_moves(game: ConnectFour, moves: list[int]) -> tuple[int, ...]:
    """Orient a history so its board matches the canonical left-right board."""
    board = tuple(game.board)
    if mirror_board(board) < board:
        return tuple(6 - action for action in moves)
    return tuple(moves)


def collect_candidates(
    target: int,
    *,
    candidate_pool: int,
    min_ply: int,
    max_ply: int,
    seed: int,
    max_games: int,
) -> list[dict[str, Any]]:
    """Sample unique states from varied legal trajectories, before exact solving."""
    rng = random.Random(seed)
    shallow_teacher = AlphaBetaTeacher(max_table_entries=100_000)
    modes = ("random", "center", "tactical", "shallow", "long", "edge", "safe")
    weights = (0.18, 0.12, 0.22, 0.28, 0.08, 0.06, 0.06)
    candidates: dict[tuple[tuple[int, ...], int], dict[str, Any]] = {}

    for game_id in range(max_games):
        if len(candidates) >= candidate_pool:
            break
        mode = rng.choices(modes, weights=weights, k=1)[0]
        game = ConnectFour()
        moves: list[int] = []
        while not game.is_terminal:
            ply = game.move_count
            if min_ply <= ply <= max_ply:
                board = tuple(game.board)
                canonical_board = min(board, mirror_board(board))
                history = canonical_moves(game, moves)
                key = (canonical_board, game.current_player)
                if key not in candidates:
                    candidates[key] = {
                        "moves": history,
                        "board": canonical_board,
                        "current_player": game.current_player,
                        "ply": ply,
                        "source_policy": mode,
                        "game_id": game_id,
                        "opening_group": opening_group(list(history)),
                    }
            action = choose_move(game, mode, rng, shallow_teacher)
            game.step(action)
            moves.append(action)

    if len(candidates) < target:
        raise RuntimeError(
            f"only sampled {len(candidates)} unique states in plies {min_ply}–{max_ply}; "
            f"need {target}. Increase --candidate-pool or --max-games."
        )
    rows = list(candidates.values())
    rng.shuffle(rows)
    return rows[:candidate_pool]


def open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS labels ("
        "state_key TEXT PRIMARY KEY, scores TEXT NOT NULL, "
        "best_columns TEXT NOT NULL, best_score INTEGER NOT NULL)"
    )
    return connection


def cache_key(candidate: dict[str, Any]) -> str:
    board = ",".join(map(str, candidate["board"]))
    return f"{SOLVER_REVISION}:exact:{candidate['current_player']}:{board}"


def balance_phases(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Interleave opening, middle, and late positions before the solve budget."""
    buckets = [[], [], []]
    for candidate in candidates:
        ply = int(candidate["ply"])
        buckets[0 if ply < 20 else 1 if ply < 30 else 2].append(candidate)
    ordered = []
    while any(buckets):
        for bucket in buckets:
            if bucket:
                ordered.append(bucket.pop())
    return ordered


def row_from_scores(candidate: dict[str, Any], scores: list[int]) -> dict[str, Any]:
    moves = tuple(candidate["moves"])
    game = ConnectFour.from_moves(moves)
    legal = list(game.legal_actions())
    if tuple(game.board) != tuple(candidate["board"]):
        raise ValueError("canonical history did not reproduce the canonical board")
    best_score = max(scores[action] for action in legal)
    best_actions = [action for action in legal if scores[action] == best_score]
    policy = [0.0] * 7
    for action in best_actions:
        policy[action] = 1.0 / len(best_actions)
    # Pons's exact scores encode outcome and distance. A scale of six maps
    # typical Connect Four win/loss distances into the model's tanh Q range.
    q_values: list[float | None] = [None] * 7
    for action in legal:
        q_values[action] = math.tanh(scores[action] / 6.0)
    state_value = math.tanh(best_score / 6.0)
    return {
        "schema_version": 1,
        "board": list(game.board),
        "canonical_board": list(game.board),
        "current_player": game.current_player,
        "ply": game.move_count,
        "legal_actions": legal,
        "landing_rows": [5 - game.column_heights[action] for action in legal],
        "policy_target": policy,
        "action_values": q_values,
        "search_scores": [scores[action] if action in legal else None for action in range(7)],
        "state_value": state_value,
        "trajectory_result": None,
        "teacher": "pons_exact_solver",
        "teacher_version": SOLVER_REVISION,
        "teacher_score_mode": "exact_outcome_and_distance",
        "teacher_best_columns": [action + 1 for action in best_actions],
        "teacher_best_score": best_score,
        "move_history": "".join(str(action + 1) for action in moves),
        "source_policy": candidate["source_policy"],
        "game_id": candidate["game_id"],
        "opening_group": candidate["opening_group"],
    }


def build_dataset(
    *,
    positions: int,
    candidate_pool: int,
    min_ply: int,
    max_ply: int,
    seed: int,
    max_games: int,
    timeout: float,
    output: Path,
    cache_path: Path,
    progress_every: int,
    workers: int = 1,
    phase_balanced: bool = False,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    candidates = collect_candidates(
        positions,
        candidate_pool=candidate_pool,
        min_ply=min_ply,
        max_ply=max_ply,
        seed=seed,
        max_games=max_games,
    )
    if phase_balanced:
        candidates = balance_phases(candidates)
    cache = open_cache(cache_path)
    rows: list[dict[str, Any]] = []
    durations: list[float] = []
    cache_hits = 0
    timeouts = 0
    solver_queries = 0
    uncommitted = 0
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            oracle = stack.enter_context(PonsOracle(timeout_seconds=timeout)) if workers == 1 else None
            executor = (
                stack.enter_context(ProcessPoolExecutor(max_workers=workers, initializer=_start_worker, initargs=(timeout,)))
                if workers > 1 else None
            )
            pending: dict[int, Future[tuple[list[int] | None, float]]] = {}
            prefetched_cache: dict[int, list[int]] = {}
            next_to_submit = 0

            def prefetch(index: int) -> None:
                nonlocal next_to_submit
                if executor is None:
                    return
                while next_to_submit < min(len(candidates), index + 2 * workers):
                    position = next_to_submit
                    next_to_submit += 1
                    candidate = candidates[position]
                    existing = cache.execute(
                        "SELECT scores FROM labels WHERE state_key = ?", (cache_key(candidate),)
                    ).fetchone()
                    if existing:
                        prefetched_cache[position] = json.loads(existing[0])
                    else:
                        history = "".join(str(action + 1) for action in candidate["moves"])
                        pending[position] = executor.submit(_query_worker, history)

            for index, candidate in enumerate(candidates, start=1):
                prefetch(index - 1)
                key = cache_key(candidate)
                if executor is not None:
                    scores = prefetched_cache.pop(index - 1, None)
                    cached = scores is not None
                else:
                    existing = cache.execute(
                        "SELECT scores FROM labels WHERE state_key = ?", (key,)
                    ).fetchone()
                    cached = existing is not None
                    scores = json.loads(existing[0]) if existing else None
                if cached:
                    cache_hits += 1
                else:
                    if executor is not None:
                        scores, elapsed = pending.pop(index - 1).result()
                    else:
                        assert oracle is not None
                        history = "".join(str(action + 1) for action in candidate["moves"])
                        query_started = time.perf_counter()
                        try:
                            scores = oracle.analyze(history)["scores"]
                        except TimeoutError:
                            scores = None
                        elapsed = time.perf_counter() - query_started
                    if scores is None:
                        timeouts += 1
                        if index % progress_every == 0 or index == len(candidates):
                            print(
                                f"queried {index}/{len(candidates)} candidates; "
                                f"kept={len(rows)}, timeouts={timeouts}",
                                flush=True,
                            )
                        continue
                    durations.append(elapsed)
                    solver_queries += 1
                    game = ConnectFour.from_moves(candidate["moves"])
                    legal = game.legal_actions()
                    best_score = max(scores[action] for action in legal)
                    cache.execute(
                        "INSERT OR REPLACE INTO labels "
                        "(state_key, scores, best_columns, best_score) VALUES (?, ?, ?, ?)",
                        (
                            key,
                            json.dumps(scores),
                            json.dumps([action + 1 for action in legal if scores[action] == best_score]),
                            best_score,
                        ),
                    )
                    uncommitted += 1
                    if uncommitted >= 64:
                        cache.commit()
                        uncommitted = 0
                rows.append(row_from_scores(candidate, scores))
                if len(rows) >= positions:
                    break
                if index % progress_every == 0 or index == len(candidates):
                    print(
                        f"queried {index}/{len(candidates)} candidates; "
                        f"kept={len(rows)}, cache_hits={cache_hits}, timeouts={timeouts}",
                        flush=True,
                    )
    finally:
        cache.commit()
        cache.close()

    if len(rows) < positions:
        raise RuntimeError(
            f"labeled only {len(rows)}/{positions} positions; "
            "rerun with a larger candidate pool or later minimum ply"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")

    report = {
        "teacher": SOLVER_REVISION,
        "positions": len(rows),
        "candidate_pool": len(candidates),
        "min_ply": min_ply,
        "max_ply": max_ply,
        "timeout_seconds": timeout,
        "solver_queries": solver_queries,
        "workers": workers,
        "phase_balanced": phase_balanced,
        "cache_hits": cache_hits,
        "timeouts": timeouts,
        "wall_seconds": time.perf_counter() - started,
        "solver_query_median_seconds": statistics.median(durations) if durations else 0.0,
        "solver_query_p95_seconds": sorted(durations)[math.ceil(0.95 * len(durations)) - 1]
        if durations
        else 0.0,
        "source_policies": dict(Counter(row["source_policy"] for row in rows)),
        "ply_buckets": dict(Counter("early_under_20" if row["ply"] < 20 else "middle_20_29" if row["ply"] < 30 else "late_30_plus" for row in rows)),
        "score_distribution": dict(Counter(str(row["teacher_best_score"]) for row in rows)),
        "output": str(output),
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=256)
    parser.add_argument("--candidate-pool", type=int, default=1024)
    parser.add_argument("--min-ply", type=int, default=30)
    parser.add_argument("--max-ply", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2047)
    parser.add_argument("--max-games", type=int, default=20_000)
    parser.add_argument("--timeout", type=float, default=2.0, help="per exact solver query")
    parser.add_argument("--workers", type=int, default=1, help="independent Pons solver processes")
    parser.add_argument("--phase-balanced", action="store_true", help="interleave early/middle/late candidates")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    if not SOLVER.is_file():
        parser.error(f"Pons solver is missing: {SOLVER}; see README for build steps")
    if args.positions <= 0 or args.candidate_pool < args.positions:
        parser.error("require positions > 0 and candidate-pool >= positions")
    if args.min_ply < 1 or args.max_ply < args.min_ply or args.max_ply > 41:
        parser.error("require 1 <= min-ply <= max-ply <= 41")
    if args.timeout <= 0 or args.progress_every <= 0 or args.max_games <= 0 or args.workers <= 0:
        parser.error("timeout, progress-every, max-games, and workers must be positive")
    build_dataset(
        positions=args.positions,
        candidate_pool=args.candidate_pool,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        seed=args.seed,
        max_games=args.max_games,
        timeout=args.timeout,
        output=args.output,
        cache_path=args.cache,
        progress_every=args.progress_every,
        workers=args.workers,
        phase_balanced=args.phase_balanced,
    )


if __name__ == "__main__":
    main()
