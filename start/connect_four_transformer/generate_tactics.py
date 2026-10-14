"""Make a small CPU-labelled tactical supplement without touching held-out groups."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from connect_four_engine import ConnectFour

from .data import grouped_split, load_rows
from .generate_dataset import canonical_identity, collect_candidates, immediate_wins, label_candidate
from .search import AlphaBetaTeacher


HERE = Path(__file__).parent
DEFAULT_BASE = HERE / "artifacts" / "data" / "positions_v0_2000.jsonl"
DEFAULT_OUTPUT = HERE / "artifacts" / "data" / "tactics_v1.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-kind", type=int, default=400)
    parser.add_argument("--candidate-pool", type=int, default=6_000)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--node-limit", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=2037)
    args = parser.parse_args()
    if min(args.per_kind, args.candidate_pool, args.depth, args.node_limit) < 1:
        parser.error("counts, search depth and node limit must be positive")

    base = load_rows(args.base)
    split = grouped_split(base)
    heldout_groups = {
        str(base[index]["opening_group"])
        for partition in ("validation", "test")
        for index in split[partition]
    }
    existing = {
        canonical_identity(tuple(row["board"]), int(row["current_player"]))
        for row in base
    }
    candidates = collect_candidates(args.candidate_pool, args.seed)
    teacher = AlphaBetaTeacher()
    counts: Counter[str] = Counter()
    rows = []
    started = time.perf_counter()
    for candidate in candidates:
        if candidate.opening_group in heldout_groups:
            continue
        identity = canonical_identity(candidate.board, candidate.current_player)
        if identity in existing:
            continue
        game = ConnectFour.from_moves(candidate.moves)
        own = immediate_wins(game, game.current_player)
        opponent = immediate_wins(game, -game.current_player) if not own else []
        if own:
            kind, critical = "win_now", own
        elif len(opponent) == 1:
            kind, critical = "forced_block", opponent
        else:
            continue
        if counts[kind] >= args.per_kind:
            continue
        row = label_candidate(
            candidate, teacher, depth=args.depth, node_limit=args.node_limit,
            temperature=0.18,
        )
        row["policy_target"] = [
            1.0 / len(critical) if action in critical else 0.0
            for action in range(7)
        ]
        row["tactical_category"] = kind
        row["target_type"] = "exact_one_ply_tactic"
        row["critical_actions"] = critical
        rows.append(row)
        counts[kind] += 1
        existing.add(identity)
        if all(counts[kind] >= args.per_kind for kind in ("win_now", "forced_block")):
            break
    if not rows:
        raise RuntimeError("no new tactical positions found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} new positions: {dict(counts)}")
    print(f"search and labeling time: {time.perf_counter() - started:.2f} s")
    print(f"excluded {len(heldout_groups)} held-out opening groups and all base board identities")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
