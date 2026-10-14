"""Mine model-specific Connect Four mistakes and label them with search."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
import time
from typing import Any

from connect_four_engine import ConnectFour, mirror_action

from .agent import TransformerAgent
from .data import grouped_split, load_rows
from .generate_dataset import (
    Candidate,
    canonical_identity,
    collect_candidates,
    immediate_wins,
    label_candidate,
)
from .search import AlphaBetaTeacher
from .teacher_cache import TeacherLabelCache


HERE = Path(__file__).parent
DEFAULT_BASE = HERE / "artifacts" / "data" / "positions_v0_2000.jsonl"
DEFAULT_OLD_TACTICS = HERE / "artifacts" / "data" / "tactics_v1.jsonl"
DEFAULT_CHECKPOINT = HERE / "artifacts" / "checkpoints" / "candidate_tactical_v3.pt"
DEFAULT_OUTPUT = HERE / "artifacts" / "data" / "curriculum_v5_unique.jsonl"
DEFAULT_CACHE = HERE / "artifacts" / "cache" / "teacher_labels.sqlite3"
DEFAULT_PREVIOUS_CURRICULUM = HERE / "artifacts" / "data" / "curriculum_v4.jsonl"


def find_fork_sequence(game: ConnectFour) -> bool:
    """Detect a two-ply setup that leaves the mover multiple immediate wins.

    This is a motif label, not a proof that the opponent has no better defense;
    depth-limited search supplies the actual move values.
    """
    actor = game.current_player
    for action in game.legal_actions():
        after_attack = game.copy()
        result = after_attack.step(action)
        if result.is_terminal:
            continue
        for reply in after_attack.legal_actions():
            after_reply = after_attack.copy()
            response = after_reply.step(reply)
            if response.is_terminal:
                continue
            if len(immediate_wins(after_reply, actor)) >= 2:
                return True
    return False


def canonicalize_candidate(candidate: Candidate) -> Candidate:
    """Orient a state and its move history to the canonical board mirror."""
    canonical_board, _ = canonical_identity(candidate.board, candidate.current_player)
    if tuple(candidate.board) == canonical_board:
        return candidate
    return replace(
        candidate,
        moves=tuple(mirror_action(action) for action in candidate.moves),
        board=canonical_board,
    )


def categorize(game: ConnectFour) -> str:
    """Assign one mutually exclusive label, from highest to lowest urgency."""
    own_wins = immediate_wins(game, game.current_player)
    if own_wins:
        return "win_now"
    opponent_wins = immediate_wins(game, -game.current_player)
    if len(opponent_wins) == 1:
        return "forced_block"
    if len(opponent_wins) > 1:
        return "multiple_threats"
    if find_fork_sequence(game):
        return "fork_sequence"

    unsafe = 0
    legal = game.legal_actions()
    for action in legal:
        after_move = game.copy()
        if after_move.step(action).is_terminal:
            continue
        unsafe += bool(immediate_wins(after_move, after_move.current_player))
    if unsafe and unsafe < len(legal):
        return "support_square_risk"
    return "ordinary"


def build_training_pool(
    candidates: list[Candidate],
    *,
    base_rows: list[dict[str, Any]],
    old_tactics: list[dict[str, Any]],
    previously_mined: list[dict[str, Any]] | None = None,
    candidate_limit: int,
    seed: int,
) -> tuple[list[tuple[Candidate, str]], int, int]:
    split = grouped_split(base_rows)
    heldout_groups = {
        str(base_rows[index]["opening_group"])
        for partition in ("validation", "test")
        for index in split[partition]
    }
    known = {
        canonical_identity(tuple(row["board"]), int(row["current_player"]))
        for row in base_rows + old_tactics + (previously_mined or [])
    }
    rng = random.Random(seed)
    rng.shuffle(candidates)
    categorized: list[tuple[Candidate, str]] = []
    for candidate in candidates:
        if candidate.opening_group in heldout_groups:
            continue
        identity = canonical_identity(candidate.board, candidate.current_player)
        if identity in known:
            continue
        known.add(identity)
        categorized.append((candidate, categorize(ConnectFour.from_moves(candidate.moves))))
        if len(categorized) >= candidate_limit:
            break
    return categorized, len(heldout_groups), len(known)


def choose_for_labeling(
    candidates: list[tuple[Candidate, str]], limit: int, seed: int
) -> list[tuple[Candidate, str]]:
    """Label a balanced tactical slice before selecting the hardest examples."""
    quotas = {
        "win_now": 0.15,
        "forced_block": 0.20,
        "multiple_threats": 0.10,
        "fork_sequence": 0.15,
        "support_square_risk": 0.25,
        "ordinary": 0.15,
    }
    buckets: dict[str, list[tuple[Candidate, str]]] = defaultdict(list)
    for item in candidates:
        buckets[item[1]].append(item)
    rng = random.Random(seed + 1)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    chosen: list[tuple[Candidate, str]] = []
    chosen_ids: set[tuple[tuple[int, ...], int]] = set()
    for category, fraction in quotas.items():
        count = round(limit * fraction)
        for item in buckets[category][:count]:
            chosen.append(item)
            chosen_ids.add(canonical_identity(item[0].board, item[0].current_player))

    leftovers = [
        item for bucket in buckets.values() for item in bucket
        if canonical_identity(item[0].board, item[0].current_player) not in chosen_ids
    ]
    rng.shuffle(leftovers)
    chosen.extend(leftovers[: max(0, limit - len(chosen))])
    return chosen[:limit]


def rank_and_keep(
    records: list[dict[str, Any]], *, keep_limit: int, seed: int
) -> list[dict[str, Any]]:
    """Keep a category-balanced mix of model failures and varied replay rows."""
    targets = {
        "win_now": 0.14,
        "forced_block": 0.18,
        "multiple_threats": 0.08,
        "fork_sequence": 0.16,
        "support_square_risk": 0.24,
        "ordinary": 0.20,
    }
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record["tactical_category"]].append(record)
    rng = random.Random(seed + 2)
    selected: list[dict[str, Any]] = []
    selected_ids: set[tuple[tuple[int, ...], int]] = set()
    per_group: Counter[str] = Counter()

    def take(record: dict[str, Any]) -> bool:
        identity = canonical_identity(tuple(record["board"]), int(record["current_player"]))
        group = str(record["opening_group"])
        if identity in selected_ids or per_group[group] >= 4:
            return False
        selected.append(record)
        selected_ids.add(identity)
        per_group[group] += 1
        return True

    for category, fraction in targets.items():
        bucket = buckets[category]
        rng.shuffle(bucket)
        # A 50/50 mix: high-regret states are hard replay; the rest preserve
        # coverage so training does not collapse onto only the model's errors.
        hard = sorted(bucket, key=lambda row: row["model_teacher_regret"], reverse=True)
        target = round(keep_limit * fraction)
        hard_quota = (target + 1) // 2
        for row in hard:
            if len([item for item in selected if item["tactical_category"] == category]) >= hard_quota:
                break
            take(row)
        category_count = sum(item["tactical_category"] == category for item in selected)
        for row in bucket:
            if category_count >= target:
                break
            if take(row):
                category_count += 1

    remaining = sorted(records, key=lambda row: row["model_teacher_regret"], reverse=True)
    for row in remaining:
        if len(selected) >= keep_limit:
            break
        take(row)
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--old-tactics", type=Path, default=DEFAULT_OLD_TACTICS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--exclude", type=Path, action="append", default=[DEFAULT_PREVIOUS_CURRICULUM])
    parser.add_argument("--candidate-pool", type=int, default=8_000)
    parser.add_argument("--label-limit", type=int, default=6_000)
    parser.add_argument("--keep-limit", type=int, default=6_000)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--node-limit", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=2046)
    args = parser.parse_args()
    if min(args.candidate_pool, args.label_limit, args.keep_limit, args.depth, args.node_limit) < 1:
        parser.error("pool sizes, depth, and node limit must be positive")
    if args.keep_limit < args.label_limit:
        parser.error("keep-limit must be at least label-limit so computed labels are not discarded")

    started = time.perf_counter()
    base_rows = load_rows(args.base)
    old_tactics = load_rows(args.old_tactics) if args.old_tactics.exists() else []
    previously_mined = [
        row for path in args.exclude if path.exists() for row in load_rows(path)
    ]
    candidates = collect_candidates(args.candidate_pool, args.seed)
    eligible, heldout_groups, known_count = build_training_pool(
        candidates,
        base_rows=base_rows,
        old_tactics=old_tactics,
        previously_mined=previously_mined,
        candidate_limit=args.candidate_pool,
        seed=args.seed,
    )
    pool_counts = Counter(category for _, category in eligible)
    to_label = choose_for_labeling(eligible, args.label_limit, args.seed)
    agent = TransformerAgent(args.checkpoint, tactical_guard=False)
    teacher = AlphaBetaTeacher()
    records: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    label_started = time.perf_counter()
    with TeacherLabelCache(args.cache) as cache:
        for index, (candidate, category) in enumerate(to_label, start=1):
            candidate = canonicalize_candidate(candidate)
            canonical_board = tuple(candidate.board)

            row = cache.get(
                canonical_board,
                candidate.current_player,
                depth=args.depth,
                node_limit=args.node_limit,
                temperature=0.18,
            )
            if row is None:
                cache_misses += 1
                row = label_candidate(
                    candidate,
                    teacher,
                    depth=args.depth,
                    node_limit=args.node_limit,
                    temperature=0.18,
                )
                cache.put(
                    row,
                    canonical_board,
                    candidate.current_player,
                    depth=args.depth,
                    node_limit=args.node_limit,
                    temperature=0.18,
                )
            else:
                cache_hits += 1
                # These are trajectory metadata, not teacher-label content.
                row["source_policy"] = candidate.source_policy
                row["game_id"] = candidate.game_id
                row["opening_group"] = candidate.opening_group
                row["trajectory_result"] = candidate.trajectory_result

            game = ConnectFour.from_moves(candidate.moves)
            decision = agent.inspect(game)
            model_action = int(decision["model_top_action"])
            best_value = max(float(row["action_values"][action]) for action in row["legal_actions"])
            model_value = float(row["action_values"][model_action])
            regret = max(0.0, best_value - model_value)
            row["tactical_category"] = category
            row["model_v3_top_action"] = model_action
            row["model_teacher_regret"] = regret
            row["curriculum_weight"] = round(1.0 + min(1.5, regret), 3)
            if category in ("win_now", "forced_block"):
                row["target_type"] = "exact_one_ply_tactic"
                row["critical_actions"] = [
                    action for action, value in enumerate(row["policy_target"]) if value > 0
                ]
            records.append(row)
            if index % 250 == 0 or index == len(to_label):
                elapsed = time.perf_counter() - label_started
                print(
                    f"labelled {index:>4}/{len(to_label)} | {index / elapsed:,.1f} positions/s | "
                    f"cache hit {cache_hits}, search {cache_misses}",
                    flush=True,
                )

    # All selected records are unique by canonical board and already category
    # stratified.  Keep them all; the prior 1,400-of-2,400 step threw away 42%.
    kept = records[: args.keep_limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for row in kept:
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")

    chosen_counts = Counter(row["tactical_category"] for row in kept)
    model_error_count = sum(row["model_teacher_regret"] > 1e-6 for row in kept)
    print(f"eligible categorized pool: {len(eligible):,} | categories: {dict(pool_counts)}")
    print(f"search labels: {len(records):,} | curriculum kept: {len(kept):,} | {dict(chosen_counts)}")
    print(f"canonical unique labelled boards: {len({canonical_identity(tuple(row['board']), int(row['current_player'])) for row in kept}):,}")
    print(f"persistent teacher cache: {args.cache} | hits={cache_hits:,}, new labels={cache_misses:,}")
    print(f"v3 top move had teacher regret in {model_error_count}/{len(kept)} kept rows")
    print(f"excluded {heldout_groups} validation/test opening groups; known base/old boards: {known_count:,}")
    print(f"generation wall time: {time.perf_counter() - started:.1f} s")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
