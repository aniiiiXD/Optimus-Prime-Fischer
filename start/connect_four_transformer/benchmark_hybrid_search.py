"""Paired, reproducible strength/speed test for Transformer-guided search.

Each opening is played twice per candidate (both colors) against the same
depth-3 alpha-beta baseline. Reports retain every game so comparisons can be
paired by opening instead of inferred from aggregate W/D/L counts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import time
from typing import Any, Callable

import torch

from connect_four_engine import PLAYER_ONE, ConnectFour

from .agent import TransformerAgent
from .hybrid_search import TransformerGuidedSearchAgent
from .search import AlphaBetaTeacher
from .tournament import make_openings


HERE = Path(__file__).parent
DEFAULT_REPORT = HERE / "artifacts" / "reports" / "hybrid_search_benchmark.json"
DEFAULT_CHECKPOINT = (
    HERE.parent
    / "connect_four_distillation"
    / "artifacts"
    / "checkpoints"
    / "candidate_pons_phases_v3_5min.pt"
)


class Candidate:
    def __init__(
        self,
        name: str,
        choose: Callable[[ConnectFour], tuple[int, dict[str, Any]]],
        reset: Callable[[], None] | None = None,
    ) -> None:
        self.name = name
        self._choose = choose
        self._reset = reset or (lambda: None)

    def choose(self, game: ConnectFour) -> tuple[int, dict[str, Any]]:
        return self._choose(game)

    def reset(self) -> None:
        self._reset()


def build_candidates(checkpoint: Path) -> list[Candidate]:
    plain_agent = TransformerAgent(checkpoint)
    hybrid_d3 = TransformerGuidedSearchAgent.from_checkpoint(
        checkpoint,
        max_depth=3,
        node_limit=1_500,
        model_value_weight=0.35,
        max_model_evaluations=64,
    )
    hybrid_d5 = TransformerGuidedSearchAgent.from_checkpoint(
        checkpoint,
        max_depth=5,
        node_limit=5_000,
        model_value_weight=0.35,
        max_model_evaluations=128,
    )
    hybrid_d5_32 = TransformerGuidedSearchAgent.from_checkpoint(
        checkpoint,
        max_depth=5,
        node_limit=5_000,
        model_value_weight=0.35,
        max_model_evaluations=32,
    )
    hybrid_d5_no_value = TransformerGuidedSearchAgent.from_checkpoint(
        checkpoint,
        max_depth=5,
        node_limit=5_000,
        model_value_weight=0.0,
        max_model_evaluations=128,
    )
    plain_d5 = AlphaBetaTeacher()

    def plain(game: ConnectFour) -> tuple[int, dict[str, Any]]:
        result = plain_agent.inspect(game)
        return int(result["selected_action"]), {
            "model_ms": float(result["latency_ms"]),
            "decision_ms": float(result["decision_latency_ms"]),
        }

    def hybrid(agent: TransformerGuidedSearchAgent, game: ConnectFour) -> tuple[int, dict[str, Any]]:
        result = agent.evaluate(game)
        return result.best_action, {
            "decision_ms": result.stats.elapsed_seconds * 1_000,
            "nodes": result.stats.nodes,
            "completed_depth": result.stats.completed_depth,
            "model_evaluations": result.stats.model_evaluations,
            "model_budget_exhausted": result.stats.model_budget_exhausted,
            "reason": result.reason,
        }

    def alpha_beta_d5(game: ConnectFour) -> tuple[int, dict[str, Any]]:
        result = plain_d5.evaluate(game, max_depth=5, node_limit=5_000)
        return result.best_action, {
            "decision_ms": result.stats.elapsed_seconds * 1_000,
            "nodes": result.stats.nodes,
            "completed_depth": result.stats.completed_depth,
        }

    return [
        Candidate("alphabeta_d5", alpha_beta_d5, reset=plain_d5.clear_cache),
        Candidate("transformer_guarded", plain),
        Candidate("hybrid_d3", lambda game: hybrid(hybrid_d3, game)),
        Candidate("hybrid_d5_value_032", lambda game: hybrid(hybrid_d5_32, game)),
        Candidate("hybrid_d5_value_035", lambda game: hybrid(hybrid_d5, game)),
        Candidate("hybrid_d5_no_model_value", lambda game: hybrid(hybrid_d5_no_value, game)),
    ]


def play_one(
    candidate: Candidate,
    opening: tuple[int, ...],
    candidate_player: int,
    baseline: AlphaBetaTeacher,
) -> dict[str, Any]:
    game = ConnectFour.from_moves(opening)
    candidate.reset()
    baseline.clear_cache()
    candidate_times: list[float] = []
    baseline_times: list[float] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    while not game.is_terminal:
        is_candidate = game.current_player == candidate_player
        if is_candidate:
            started = time.perf_counter_ns()
            action, diagnostics = candidate.choose(game)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            candidate_times.append(elapsed_ms)
            candidate_diagnostics.append(diagnostics)
        else:
            started = time.perf_counter_ns()
            action = baseline.evaluate(game, max_depth=3, node_limit=5_000).best_action
            baseline_times.append((time.perf_counter_ns() - started) / 1_000_000)
        if action not in game.legal_actions():
            raise RuntimeError(f"{candidate.name} or baseline selected illegal action {action}")
        game.step(action)

    result = "draw" if game.winner is None else "win" if game.winner == candidate_player else "loss"
    return {
        "candidate": candidate.name,
        "candidate_side": "X" if candidate_player == PLAYER_ONE else "O",
        "opening_moves": list(opening),
        "result": result,
        "plies": game.move_count,
        "candidate_move_median_ms": statistics.median(candidate_times),
        "candidate_move_p95_ms": percentile(candidate_times, 0.95),
        "baseline_move_median_ms": statistics.median(baseline_times) if baseline_times else None,
        "hybrid_completed_depth_median": median_field(candidate_diagnostics, "completed_depth"),
        "hybrid_nodes_median": median_field(candidate_diagnostics, "nodes"),
        "hybrid_model_evaluations_median": median_field(candidate_diagnostics, "model_evaluations"),
        "hybrid_budget_exhaustion_rate": mean_field(candidate_diagnostics, "model_budget_exhausted"),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def median_field(records: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in records if row.get(field) is not None]
    return statistics.median(values) if values else None


def mean_field(records: list[dict[str, Any]], field: str) -> float | None:
    values = [float(bool(row[field])) for row in records if row.get(field) is not None]
    return statistics.mean(values) if values else None


def points(result: str) -> float:
    return 1.0 if result == "win" else 0.5 if result == "draw" else 0.0


def bootstrap_mean_ci(values: list[float], *, seed: int, samples: int = 5_000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(values) for _ in range(len(values))) / len(values)
        for _ in range(samples)
    )
    return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]


def summarize_candidate(records: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    outcomes = Counter(record["result"] for record in records)
    candidate_times = [
        float(record["candidate_move_median_ms"])
        for record in records
    ]
    paired_openings: dict[tuple[int, int], list[float]] = defaultdict(list)
    for record in records:
        paired_openings[(record["opening_seed"], record["opening_index"])].append(
            points(record["result"])
        )
    pair_scores = [statistics.mean(values) for values in paired_openings.values()]
    by_side = {}
    for side in ("X", "O"):
        group = [record for record in records if record["candidate_side"] == side]
        side_outcomes = Counter(record["result"] for record in group)
        by_side[side] = {
            "games": len(group),
            "wins": side_outcomes["win"],
            "draws": side_outcomes["draw"],
            "losses": side_outcomes["loss"],
            "score_rate": statistics.mean(points(record["result"]) for record in group) if group else None,
        }
    return {
        "games": len(records),
        "wins": outcomes["win"],
        "draws": outcomes["draw"],
        "losses": outcomes["loss"],
        "score_rate": statistics.mean(points(record["result"]) for record in records),
        "paired_opening_score_rate": statistics.mean(pair_scores),
        "paired_opening_score_95pct_bootstrap_ci": bootstrap_mean_ci(
            pair_scores, seed=seeds[0] + len(records)
        ),
        "by_side": by_side,
        "candidate_move_median_ms_median": statistics.median(candidate_times),
        "candidate_move_median_ms_p95": percentile(candidate_times, 0.95),
        "median_search_depth": median_field(records, "hybrid_completed_depth_median"),
        "median_nodes": median_field(records, "hybrid_nodes_median"),
        "median_model_evaluations": median_field(records, "hybrid_model_evaluations_median"),
        "model_budget_exhaustion_rate": mean_field(records, "hybrid_budget_exhaustion_rate"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--games-per-side", type=int, default=25, help="paired openings per color, per seed")
    parser.add_argument("--seeds", type=str, default="43027,43028")
    parser.add_argument(
        "--only-candidates",
        type=str,
        help="optional comma-separated strategy names, e.g. alphabeta_d5,hybrid_d5_no_model_value",
    )
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if args.games_per_side < 1 or not seeds or args.threads < 1:
        parser.error("games-per-side, seeds, and threads must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    torch.set_num_threads(args.threads)
    candidates = build_candidates(args.checkpoint)
    if args.only_candidates:
        requested = {name.strip() for name in args.only_candidates.split(",") if name.strip()}
        if not requested:
            parser.error("--only-candidates must contain at least one strategy name")
        available = {candidate.name for candidate in candidates}
        unknown = requested - available
        if unknown:
            parser.error(f"unknown candidates: {', '.join(sorted(unknown))}; choose from {', '.join(sorted(available))}")
        candidates = [candidate for candidate in candidates if candidate.name in requested]
    all_records: dict[str, list[dict[str, Any]]] = {candidate.name: [] for candidate in candidates}
    started = time.perf_counter()
    cpu_started = time.process_time()
    for seed in seeds:
        openings = make_openings(args.games_per_side, seed)
        for candidate in candidates:
            print(f"seed={seed} candidate={candidate.name} openings={len(openings)} x both colors", flush=True)
            baseline = AlphaBetaTeacher()
            for opening_index, opening in enumerate(openings):
                for candidate_player in (PLAYER_ONE, -PLAYER_ONE):
                    record = play_one(candidate, opening, candidate_player, baseline)
                    record["opening_seed"] = seed
                    record["opening_index"] = opening_index
                    all_records[candidate.name].append(record)
            print(f"  score={statistics.mean(points(r['result']) for r in all_records[candidate.name]):.1%}", flush=True)

    def opening_pair_scores(records: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
        grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
        for row in records:
            grouped[(row["opening_seed"], row["opening_index"])].append(points(row["result"]))
        return {key: statistics.mean(values) for key, values in grouped.items()}

    paired_differences = {}
    available_references = [
        name for name in ("transformer_guarded", "alphabeta_d5") if name in all_records
    ]
    for name, records in all_records.items():
        references = {}
        for reference in available_references:
            if name == reference:
                continue
            candidate_by_opening = opening_pair_scores(records)
            reference_by_opening = opening_pair_scores(all_records[reference])
            differences = [
                candidate_by_opening[key] - reference_by_opening[key]
                for key in reference_by_opening
            ]
            references[reference] = {
                "mean_paired_score_delta": statistics.mean(differences),
                "paired_delta_95pct_bootstrap_ci": bootstrap_mean_ci(
                    differences, seed=seeds[0] + 911
                ),
            }
        if references:
            paired_differences[name] = references

    report = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "seeds": seeds,
            "paired_openings_per_seed": args.games_per_side,
            "games_per_candidate_per_seed": 2 * args.games_per_side,
            "colors_swapped_on_each_same_opening": True,
            "opponent": "alpha_beta_depth3 (5,000-node cap)",
            "cpu_threads": args.threads,
            "candidates_included": [candidate.name for candidate in candidates],
            "strategies": {
                "transformer_guarded": "one Transformer pass + exact immediate-tactic guard",
                "alphabeta_d5": "plain alpha-beta depth 5, 5,000 nodes, center move ordering, no Transformer calls",
                "hybrid_d3": "Transformer root ordering + alpha-beta depth 3, 1,500 nodes, <=64 neural leaves",
                "hybrid_d5_value_032": "Transformer root ordering + alpha-beta depth 5, 5,000 nodes, <=32 neural leaves, 0.35 learned leaf value",
                "hybrid_d5_value_035": "Transformer root ordering + alpha-beta depth 5, 5,000 nodes, <=128 neural leaves, 0.35 learned leaf value",
                "hybrid_d5_no_model_value": "same depth-5 search, learned leaf value weight 0 (heuristic-only leaf ablation)",
            },
        },
        "summary": {
            name: summarize_candidate(records, seeds)
            for name, records in all_records.items()
        },
        "paired_strategy_deltas_by_reference": paired_differences,
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "process_cpu_seconds": time.process_time() - cpu_started,
            "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_rss_units": "bytes on macOS; KiB on Linux",
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "games": [record for records in all_records.values() for record in records],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nSummary against depth-3 alpha-beta:")
    for name, summary in report["summary"].items():
        ci = summary["paired_opening_score_95pct_bootstrap_ci"]
        latency = summary["candidate_move_median_ms_median"]
        print(
            f"{name:<27} {summary['wins']}W/{summary['draws']}D/{summary['losses']}L "
            f"score={summary['score_rate']:.1%} paired95%=[{ci[0]:.1%},{ci[1]:.1%}] "
            f"move median={latency:.1f}ms"
        )
    print(f"saved report: {args.report}")


if __name__ == "__main__":
    main()
