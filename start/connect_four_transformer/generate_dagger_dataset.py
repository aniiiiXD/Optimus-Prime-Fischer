"""Collect agent-visited positions and relabel them with deeper CPU search.

The collector alternates the current Transformer-guided agent against the
depth-3 alpha-beta baseline, with randomized legal openings. Every saved state
is then labeled independently by a depth-5 alpha-beta teacher. This is a small,
reproducible DAgger-style experiment; it is not outcome-only self-play RL.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from connect_four_engine import ConnectFour
from connect_four_transformer.generate_dataset import (
    Candidate,
    canonical_identity,
    label_candidate,
    opening_group,
    phase,
)
from connect_four_transformer.hybrid_search import TransformerGuidedSearchAgent
from connect_four_transformer.search import AlphaBetaTeacher


ROOT = Path(__file__).parent
DEFAULT_INPUT_CHECKPOINT = (
    ROOT.parent
    / "connect_four_distillation"
    / "artifacts"
    / "checkpoints"
    / "candidate_pons_phases_v3_5min.pt"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "data" / "dagger_search_v1_2000.jsonl"


def collect_candidates(
    target: int,
    *,
    checkpoint: Path,
    seed: int,
    max_games: int,
    opening_max: int,
    max_per_phase_per_game: int,
) -> tuple[list[Candidate], dict[str, Any]]:
    rng = random.Random(seed)
    agent = TransformerGuidedSearchAgent.from_checkpoint(
        checkpoint,
        max_depth=5,
        node_limit=5_000,
        model_value_weight=0.0,
        max_model_evaluations=1,
    )
    shallow_teacher = AlphaBetaTeacher(max_table_entries=100_000)
    quotas = {
        "opening": round(target * 0.20),
        "early_mid": round(target * 0.30),
        "late_mid": round(target * 0.35),
        "late": target,
    }
    counts = {name: 0 for name in quotas}
    selected: dict[tuple[tuple[int, ...], int], Candidate] = {}
    overflow: list[tuple[tuple[tuple[int, ...], int], Candidate, str]] = []

    for game_id in range(max_games):
        if len(selected) >= target:
            break
        opening_length = rng.randint(0, opening_max)
        game = ConnectFour()
        moves: list[int] = []
        for _ in range(opening_length):
            if game.is_terminal:
                break
            action = rng.choice(game.legal_actions())
            game.step(action)
            moves.append(action)
        hybrid_player = 1 if game_id % 2 == 0 else -1
        positions: list[tuple[tuple[tuple[int, ...], int], Candidate, str]] = []
        while not game.is_terminal:
            identity = canonical_identity(tuple(game.board), game.current_player)
            candidate = Candidate(
                moves=tuple(moves),
                board=tuple(game.board),
                current_player=game.current_player,
                source_policy="hybrid_search_vs_depth3",
                game_id=game_id,
                opening_group="pending",
            )
            positions.append((identity, candidate, phase(game.move_count)))
            if game.current_player == hybrid_player:
                action = agent.choose(game)
            else:
                action = shallow_teacher.evaluate(
                    game, max_depth=3, node_limit=5_000
                ).best_action
            game.step(action)
            moves.append(action)

        # Group every state from one game by its opening prefix; no game can
        # leak near-identical positions across the later train/val/test split.
        game_group = opening_group(moves, length=6)
        per_game: Counter[str] = Counter()
        rng.shuffle(positions)
        for identity, candidate, bucket in positions:
            if per_game[bucket] >= max_per_phase_per_game:
                continue
            per_game[bucket] += 1
            candidate.opening_group = game_group
            candidate.trajectory_result = (
                0 if game.winner is None
                else 1 if game.winner == candidate.current_player
                else -1
            )
            if identity in selected:
                continue
            if counts[bucket] < quotas[bucket]:
                selected[identity] = candidate
                counts[bucket] += 1
            else:
                overflow.append((identity, candidate, bucket))
            if len(selected) >= target:
                break
        if (game_id + 1) % 100 == 0:
            print(
                f"self-play games={game_id + 1} unique={len(selected)}/{target} "
                f"phases={counts}", flush=True
            )

    if len(selected) < target:
        rng.shuffle(overflow)
        for identity, candidate, bucket in overflow:
            if identity not in selected:
                selected[identity] = candidate
                counts[bucket] += 1
            if len(selected) >= target:
                break
    if len(selected) < target:
        raise RuntimeError(
            f"collected {len(selected)}/{target} unique states in {max_games} games; "
            "increase --max-games or reduce --positions"
        )
    candidates = list(selected.values())[:target]
    rng.shuffle(candidates)
    return candidates, {
        "games_played": game_id + 1,
        "phase_counts": dict(Counter(phase(len(item.moves)) for item in candidates)),
        "unique_positions": len(candidates),
        "hybrid_player_games": (game_id + 1 + 1) // 2,
        "baseline_player_games": (game_id + 1) // 2,
    }


def build_dataset(
    *,
    positions: int,
    checkpoint: Path,
    output: Path,
    seed: int,
    max_games: int,
    opening_max: int,
    max_per_phase_per_game: int,
    teacher_depth: int,
    node_limit: int,
    temperature: float,
    progress_every: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    candidates, collection_stats = collect_candidates(
        positions,
        checkpoint=checkpoint,
        seed=seed,
        max_games=max_games,
        opening_max=opening_max,
        max_per_phase_per_game=max_per_phase_per_game,
    )
    print(f"collected {len(candidates)} unique agent-visited positions", flush=True)

    teacher = AlphaBetaTeacher(max_table_entries=500_000)
    rows = []
    label_seconds = []
    reached_depth = Counter()
    for index, candidate in enumerate(candidates, start=1):
        started = time.perf_counter()
        row = label_candidate(
            candidate,
            teacher,
            depth=teacher_depth,
            node_limit=node_limit,
            temperature=temperature,
        )
        label_seconds.append(time.perf_counter() - started)
        reached_depth[str(row["teacher_depth_reached"])] += 1
        rows.append(row)
        if index % progress_every == 0 or index == len(candidates):
            print(
                f"search labels={index}/{len(candidates)} depth={dict(reached_depth)} "
                f"elapsed={time.perf_counter() - wall_started:.1f}s",
                flush=True,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")
    report = {
        "method": "DAgger-style state collection; AlphaBetaTeacher supervised labels",
        "teacher": "negamax alpha-beta with configured search depth/node cap",
        "input_checkpoint": str(checkpoint),
        "positions": len(rows),
        "seed": seed,
        "opening_max_random_plies": opening_max,
        "max_positions_per_phase_per_game": max_per_phase_per_game,
        "collection": collection_stats,
        "source_policies": dict(Counter(row["source_policy"] for row in rows)),
        "phases": dict(Counter(phase(int(row["ply"])) for row in rows)),
        "trajectory_outcomes": dict(Counter(str(row["trajectory_result"]) for row in rows)),
        "teacher_depth_reached": dict(reached_depth),
        "teacher_depth_requested": teacher_depth,
        "teacher_node_limit": node_limit,
        "label_seconds_median": statistics.median(label_seconds),
        "label_seconds_p95": sorted(label_seconds)[int(0.95 * (len(label_seconds) - 1))],
        "wall_seconds": time.perf_counter() - wall_started,
        "process_cpu_seconds": time.process_time() - cpu_started,
        "output": str(output),
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_INPUT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--positions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--max-games", type=int, default=5_000)
    parser.add_argument("--opening-max", type=int, default=4)
    parser.add_argument("--max-per-phase-per-game", type=int, default=2)
    parser.add_argument("--teacher-depth", type=int, default=5)
    parser.add_argument("--node-limit", type=int, default=20_000)
    parser.add_argument("--temperature", type=float, default=0.18)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if min(args.positions, args.max_games, args.max_per_phase_per_game, args.teacher_depth, args.node_limit, args.progress_every) <= 0:
        parser.error("positions, games, phase cap, teacher settings, and progress must be positive")
    if not 0 <= args.opening_max <= 12 or args.temperature <= 0:
        parser.error("opening-max must be in [0,12] and temperature must be positive")
    build_dataset(
        positions=args.positions,
        checkpoint=args.checkpoint,
        output=args.output,
        seed=args.seed,
        max_games=args.max_games,
        opening_max=args.opening_max,
        max_per_phase_per_game=args.max_per_phase_per_game,
        teacher_depth=args.teacher_depth,
        node_limit=args.node_limit,
        temperature=args.temperature,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
