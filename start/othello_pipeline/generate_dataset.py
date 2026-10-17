"""Diverse random-position pilot with cached Neural Reversi all-move labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import time

from gamebench.games.othello import OthelloAdapter, OthelloState
from .dataset import transform_action, transform_board
from .teacher import TeacherConfig, label_positions


def _canonical_state(state: OthelloState) -> OthelloState:
    board = min(transform_board(tuple(state.board), rotation, reflect)
                for reflect in (False, True) for rotation in range(4))
    return OthelloState(list(board), state.current_player, [], state.consecutive_passes)


def _opening_group(history: tuple[int, ...], game_index: int) -> str:
    if len(history) < 6:
        return f"short-{game_index}"
    prefix = history[:6]
    canonical = min(tuple(transform_action(action, rotation, reflect) for action in prefix)
                    for reflect in (False, True) for rotation in range(4))
    return "open-" + "-".join(str(action) for action in canonical)


def _phase(ply: int) -> str:
    return "early" if ply <= 20 else "middle" if ply <= 40 else "late"


def sample_positions(
    positions: int, candidate_games: int, min_ply: int, max_ply: int, seed: int
) -> list[tuple[OthelloState, str, int, str]]:
    rng = random.Random(seed)
    pools: dict[str, list[tuple[OthelloState, str, int, str]]] = {phase: [] for phase in ("early", "middle", "late")}
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for game_index in range(candidate_games):
        state = OthelloAdapter().reset()
        while not state.is_terminal():
            ply = len(state.history)
            if min_ply <= ply <= max_ply:
                canonical = _canonical_state(state)
                key = canonical.current_player, tuple(canonical.board)
                if key not in seen:
                    seen.add(key)
                    phase = _phase(ply)
                    pools[phase].append((canonical, _opening_group(state.history, game_index), ply, phase))
            state.apply_action(rng.choice(state.legal_actions()))
    quotas = {"early": (positions + 2) // 3, "middle": (positions + 1) // 3, "late": positions // 3}
    selected: list[tuple[OthelloState, str, int, str]] = []
    for phase, pool in pools.items():
        rng.shuffle(pool)
        selected.extend(pool[: quotas[phase]])
    if len(selected) < positions:
        leftovers = [item for phase, pool in pools.items() for item in pool[quotas[phase]:]]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: positions - len(selected)])
    if len(selected) < positions:
        raise ValueError(f"only {len(selected)} unique states found; increase --candidate-games")
    rng.shuffle(selected)
    return selected[:positions]


def _cache_key(state: OthelloState, teacher_id: str) -> str:
    payload = f"{teacher_id}:{state.current_player}:{','.join(map(str, state.board))}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def cached_labels(
    states: list[OthelloState], config: TeacherConfig, cache_path: Path, batch_size: int
) -> tuple[list[dict[int, float]], int]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS labels (key TEXT PRIMARY KEY, scores_json TEXT NOT NULL)")
        results: list[dict[int, float] | None] = [None] * len(states)
        missing: list[int] = []
        for index, state in enumerate(states):
            row = connection.execute("SELECT scores_json FROM labels WHERE key=?", (_cache_key(state, config.teacher_id),)).fetchone()
            if row is None:
                missing.append(index)
            else:
                results[index] = {int(action): float(score) for action, score in json.loads(row[0]).items()}
        for start in range(0, len(missing), batch_size):
            indices = missing[start : start + batch_size]
            scored = label_positions([states[index] for index in indices], config)
            for index, scores in zip(indices, scored):
                results[index] = scores
                connection.execute(
                    "INSERT OR REPLACE INTO labels(key, scores_json) VALUES (?,?)",
                    (_cache_key(states[index], config.teacher_id), json.dumps(scores, sort_keys=True)),
                )
            connection.commit()
            print(f"teacher labeled {min(start + batch_size, len(missing))}/{len(missing)} cache misses")
        if any(result is None for result in results):
            raise RuntimeError("incomplete teacher labels")
        complete = [result for result in results if result is not None]
        for state, scores in zip(states, complete):
            if set(scores) != set(state.legal_actions()):
                raise ValueError("cached teacher labels do not match current Othello rules")
        return complete, len(states) - len(missing)
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cached, phase-balanced Othello teacher dataset.")
    parser.add_argument("--positions", type=int, default=256)
    parser.add_argument("--candidate-games", type=int, default=120)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=55)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--level", type=int, default=8)
    parser.add_argument("--teacher-threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("othello_pipeline/artifacts/cache/teacher_labels.sqlite"))
    args = parser.parse_args()
    if min(args.positions, args.candidate_games, args.batch_size) <= 0 or not 0 <= args.min_ply <= args.max_ply:
        parser.error("positions, games, and batch size must be positive; check ply range")
    if args.output.exists():
        parser.error(f"output exists (will not overwrite): {args.output}")
    root = Path(__file__).resolve().parent
    config = TeacherConfig(
        root / "vendor/neural-reversi/target/release/cli",
        root / "artifacts/teacher_weights/eval-e6bbc4f6.zst",
        root / "artifacts/teacher_weights/eval_sm-e6bbc4f6.zst",
        level=args.level,
        threads=args.teacher_threads,
        hash_mb=args.hash_mb,
    )
    config.check_files()
    started = time.perf_counter()
    selected = sample_positions(args.positions, args.candidate_games, args.min_ply, args.max_ply, args.seed)
    scores, cache_hits = cached_labels([item[0] for item in selected], config, args.cache, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        for (state, group, ply, phase), score_map in zip(selected, scores):
            legal = list(state.legal_actions())
            row = {
                "board": state.board,
                "player": state.current_player,
                "legal": legal,
                "teacher_scores": [score_map[action] for action in legal],
                "result": None,
                "group": group,
                "teacher_id": config.teacher_id,
                "ply": ply,
                "phase": phase,
            }
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")
    phases = {phase: sum(item[3] == phase for item in selected) for phase in ("early", "middle", "late")}
    print(f"saved {len(selected)} unique states ({phases}), {cache_hits} cache hits, {time.perf_counter() - started:.1f} s: {args.output}")


if __name__ == "__main__":
    main()
