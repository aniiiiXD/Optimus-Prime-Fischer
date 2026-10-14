"""Compare saved Connect Four checkpoints on one frozen suite and same openings."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
from typing import Any

import torch

from connect_four_engine import PLAYER_ONE

from .agent import TransformerAgent
from .arena_cpu import (
    DEFAULT_DATASET,
    TACTICAL_CATEGORIES,
    THREAT_CATEGORIES,
    category_and_critical,
    from_dataset_row,
    immediate_wins,
    percentile,
)
from .data import grouped_split, load_rows
from .tournament import (
    ModelAgent,
    RandomAgent,
    SearchAgent,
    TacticalAgent,
    make_openings,
    matchup,
)


HERE = Path(__file__).parent
DEFAULT_CHECKPOINT_DIR = HERE / "artifacts" / "checkpoints"
DEFAULT_REPORT = HERE / "artifacts" / "reports" / "model_benchmark_v1.json"
DEFAULT_GAMES_PER_SIDE = 30


def percentile_or_none(values: list[float], fraction: float) -> float | None:
    return percentile(values, fraction) if values else None


def summarize_positions(records: list[dict[str, Any]]) -> dict[str, Any]:
    # A position with multiple simultaneous opponent wins is categorized for
    # analysis, but it is not a single-move forced-block test.
    tactical = [record for record in records if record["category"] in TACTICAL_CATEGORIES]
    return {
        "positions": len(records),
        "teacher_best_move_rate": statistics.mean(
            record["teacher_best_move"] for record in records
        ),
        "mean_teacher_regret": statistics.mean(
            record["teacher_regret"] for record in records
        ),
        "critical_positions": len(tactical),
        "unavoidable_loss_positions": sum(
            record["category"] == "unavoidable_loss" for record in records
        ),
        "raw_critical_move_rate": (
            statistics.mean(record["raw_critical_move"] for record in tactical)
            if tactical else None
        ),
        "guarded_critical_move_rate": (
            statistics.mean(record["guarded_critical_move"] for record in tactical)
            if tactical else None
        ),
    }


def weight_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def resolve_checkpoints(specs: list[str] | None, directory: Path) -> list[tuple[str, Path]]:
    if specs:
        result = []
        for spec in specs:
            if "=" in spec:
                name, raw_path = spec.split("=", 1)
                path = Path(raw_path)
            else:
                path = Path(spec)
                name = path.stem
            if not name.strip() or not path.is_file():
                raise FileNotFoundError(f"checkpoint spec is invalid: {spec}")
            result.append((name.strip(), path))
    else:
        result = [(path.stem, path) for path in sorted(directory.glob("*.pt"))]
    if not result:
        raise FileNotFoundError(f"no .pt checkpoints found in {directory}")
    names = [name for name, _ in result]
    if len(names) != len(set(names)):
        raise ValueError("checkpoint display names must be unique")
    return result


def score_heldout(agent: TransformerAgent, rows: list[dict[str, Any]], indices: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    model_latencies: list[float] = []
    decision_latencies: list[float] = []
    for index in indices:
        row = rows[index]
        game = from_dataset_row(row)
        category, critical = category_and_critical(game)
        started = time.perf_counter_ns()
        decision = agent.inspect(game)
        decision_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        model_latencies.append(float(decision["latency_ms"]))

        legal = game.legal_actions()
        teacher_values = row["action_values"]
        best_value = max(float(teacher_values[action]) for action in legal)
        raw_action = int(decision["model_top_action"])
        guarded_action = int(decision["selected_action"])
        raw_value = float(teacher_values[raw_action])
        records.append({
            "dataset_index": index,
            "category": category,
            "player": game.current_player,
            "critical_actions": critical,
            "opponent_threat_columns": immediate_wins(game, -game.current_player)
            if category in THREAT_CATEGORIES else [],
            "raw_action": raw_action,
            "guarded_action": guarded_action,
            "teacher_best_move": abs(raw_value - best_value) <= 1e-6,
            "teacher_regret": max(0.0, best_value - raw_value),
            "raw_critical_move": raw_action in critical if critical else None,
            "guarded_critical_move": guarded_action in critical if critical else None,
            "top3": decision["choices"][:3],
        })

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        categories[record["category"]].append(record)
    summary = summarize_positions(records)
    summary["by_category"] = {
        name: summarize_positions(group) for name, group in sorted(categories.items())
    }
    summary["latency_ms"] = {
        "model_forward_median": statistics.median(model_latencies),
        "model_forward_p95": percentile(model_latencies, 0.95),
        "full_guarded_decision_median": statistics.median(decision_latencies),
        "full_guarded_decision_p95": percentile(decision_latencies, 0.95),
    }
    return summary, records


def build_opponent(name: str, seed: int) -> Any:
    if name == "random":
        return RandomAgent(seed)
    if name == "tactical":
        return TacticalAgent()
    if name == "alpha_beta_depth3":
        return SearchAgent()
    raise ValueError(f"unknown benchmark opponent {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", action="append",
        help="optional PATH or NAME=PATH; repeat to select checkpoints (default: all *.pt)",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--exclude-training-data", type=Path, action="append", default=[],
        help="exclude any benchmark row sharing an opening_group with these datasets; repeat as needed",
    )
    parser.add_argument(
        "--all-positions", action="store_true",
        help="score every remaining row (use with an independent external evaluation dataset)",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--opening-seed", type=int, default=41027)
    parser.add_argument("--games-per-side", type=int, default=DEFAULT_GAMES_PER_SIDE)
    parser.add_argument("--skip-tournament", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.games_per_side) < 1:
        parser.error("threads and games-per-side must be positive")

    torch.set_num_threads(args.threads)
    checkpoints = resolve_checkpoints(args.checkpoint, args.checkpoint_dir)
    rows = load_rows(args.dataset)
    excluded_groups: set[str] = set()
    for training_path in args.exclude_training_data:
        excluded_groups.update(
            str(row.get("opening_group", "")) for row in load_rows(training_path)
        )
    original_position_count = len(rows)
    if excluded_groups:
        rows = [
            row for row in rows
            if str(row.get("opening_group", "")) not in excluded_groups
        ]
    if not rows:
        raise ValueError("no benchmark positions remain after excluding training opening groups")
    test_indices = (
        list(range(len(rows)))
        if args.all_positions
        else grouped_split(rows, seed=args.split_seed)["test"]
    )
    openings = make_openings(args.games_per_side, args.opening_seed)
    model_records: list[dict[str, Any]] = []
    weight_hashes: dict[str, list[str]] = defaultdict(list)
    benchmark_started = time.perf_counter()
    cpu_started = time.process_time()

    for name, checkpoint_path in checkpoints:
        started = time.perf_counter()
        agent = TransformerAgent(checkpoint_path)
        heldout, heldout_records = score_heldout(agent, rows, test_indices)
        model_hash = weight_fingerprint(agent.model)
        weight_hashes[model_hash].append(name)
        metadata = agent.metadata
        parameters = sum(parameter.numel() for parameter in agent.model.parameters())
        entry: dict[str, Any] = {
            "name": name,
            "checkpoint": str(checkpoint_path),
            "weight_sha256": model_hash,
            "parameters": parameters,
            "model_config": metadata.get("model_config"),
            "training_metadata": {
                key: metadata.get(key)
                for key in (
                    "best_epoch", "init_checkpoint", "dataset", "extra_train",
                    "hard_tactical_targets", "tactical_weight", "selection_metric",
                )
                if key in metadata
            },
            "heldout": heldout,
            "heldout_examples": heldout_records,
            "tournament": [],
        }

        if not args.skip_tournament:
            model = ModelAgent(checkpoint_path)
            for opponent_name in ("random", "tactical", "alpha_beta_depth3"):
                for side in (PLAYER_ONE, -PLAYER_ONE):
                    opponent_seed = args.opening_seed + 1
                    opponent = build_opponent(opponent_name, opponent_seed)
                    result = matchup(model, opponent, openings, model_player=side)
                    result["score_rate"] = (
                        float(result["wins"]) + 0.5 * float(result["draws"])
                    ) / float(result["games"])
                    entry["tournament"].append(result)
        entry["benchmark_wall_seconds"] = time.perf_counter() - started
        model_records.append(entry)

        search_result = next(
            (result for result in entry["tournament"]
             if result["opponent"] == "alpha_beta_depth3" and result["model_side"] == "X"),
            None,
        )
        print(
            f"{name:<28} teacher={heldout['teacher_best_move_rate']:.1%} "
            f"regret={heldout['mean_teacher_regret']:.3f} "
            f"critical raw/guard={heldout['raw_critical_move_rate']:.1%}/"
            f"{heldout['guarded_critical_move_rate']:.1%} "
            + (
                f"vs depth3 as X={search_result['wins']}-{search_result['draws']}-{search_result['losses']}"
                if search_result else "tournament skipped"
            ),
            flush=True,
        )

    duplicate_weights = [names for names in weight_hashes.values() if len(names) > 1]
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "protocol": {
            "dataset": str(args.dataset),
            "heldout_split": "all remaining rows" if args.all_positions else "grouped test partition",
            "positions_before_opening_exclusion": original_position_count,
            "positions_excluded_by_training_opening_groups": original_position_count - len(rows),
            "training_data_excluded_by_opening_group": [str(path) for path in args.exclude_training_data],
            "split_seed": args.split_seed,
            "tournament_openings_seed": args.opening_seed,
            "games_per_side": None if args.skip_tournament else args.games_per_side,
            "threads": args.threads,
            "opponents": [] if args.skip_tournament else ["random", "tactical", "alpha_beta_depth3"],
            "same_positions_and_openings_for_each_checkpoint": True,
        },
        "checkpoints": model_records,
        "identical_weight_hash_groups": duplicate_weights,
        "environment": {
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "wall_seconds": time.perf_counter() - benchmark_started,
            "process_cpu_seconds": time.process_time() - cpu_started,
            "peak_rss_mib": peak_rss / (1024 * 1024 if sys.platform == "darwin" else 1024),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"identical checkpoint weight groups: {duplicate_weights or 'none'}")
    print(f"benchmark report: {args.report}")


if __name__ == "__main__":
    main()
