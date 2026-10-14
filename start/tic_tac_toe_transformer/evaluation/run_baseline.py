"""Run the complete strength and CPU baseline, then save a JSON report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any

import torch

from ..agents import MinimaxAgent, RandomAgent, TransformerAgent
from ..systems.cpu_profile import profile_model
from .position_test import evaluate_positions
from .tournament import run_matchup


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "artifacts" / "minimax_policy.pt"
DEFAULT_DATASET = PACKAGE_ROOT / "artifacts" / "minimax_positions.jsonl"
DEFAULT_REPORT = PACKAGE_ROOT / "artifacts" / "reports" / "baseline_v0.json"


def _transformer_results(matchup: dict[str, Any]) -> dict[str, int]:
    transformer_mark = "X" if matchup["x_agent"] == "transformer" else "O"
    opponent_mark = "O" if transformer_mark == "X" else "X"
    return {
        "wins": int(matchup[f"{transformer_mark.lower()}_wins"]),
        "losses": int(matchup[f"{opponent_mark.lower()}_wins"]),
        "draws": int(matchup["draws"]),
    }


def print_summary(report: dict[str, Any]) -> None:
    positions = report["positions"]
    print("\nPOSITION QUALITY")
    print(f"  optimal moves       {positions['optimal_accuracy']:.1%}")
    print(f"  any blunder         {positions['blunder_rate']:.1%}")
    print(f"  losing blunder      {positions['losing_blunder_rate']:.1%}")
    print(f"  mean Minimax regret {positions['mean_regret']:.4f}")

    print("\nCOMPLETE GAMES (Transformer perspective)")
    for name, matchup in report["tournaments"].items():
        result = _transformer_results(matchup)
        print(
            f"  {name:<24} {result['wins']:>3} wins  "
            f"{result['draws']:>3} draws  {result['losses']:>3} losses"
        )

    print("\nCPU INFERENCE")
    for name, profile in report["cpu"].items():
        latency = profile["latency"]
        print(
            f"  {name:<10} p50 {latency['p50_ms']:.3f} ms  "
            f"p95 {latency['p95_ms']:.3f} ms  "
            f"{profile['examples_per_second']:.0f} positions/s  "
            f"~{profile['average_cores']:.2f} cores"
        )
    print(f"  peak process memory {report['cpu']['batch_1']['peak_rss_mib']:.1f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--games-per-side", type=int, default=100)
    parser.add_argument("--profile-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.games_per_side <= 0 or args.profile_iterations <= 0:
        parser.error("games-per-side and profile-iterations must be positive")

    transformer = TransformerAgent(args.checkpoint)
    print("Evaluating all labelled board positions...")
    position_report = evaluate_positions(transformer, args.dataset)

    print("Running complete games against Random and Minimax...")
    tournaments = {
        "transformer_X_vs_random": run_matchup(
            transformer, RandomAgent(args.seed), args.games_per_side
        ),
        "transformer_O_vs_random": run_matchup(
            RandomAgent(args.seed + 1), transformer, args.games_per_side
        ),
        # These are deterministic, so repeating them would create fake sample size.
        "transformer_X_vs_minimax": run_matchup(transformer, MinimaxAgent(), 1),
        "transformer_O_vs_minimax": run_matchup(MinimaxAgent(), transformer, 1),
    }

    print("Profiling CPU inference after warmup...")
    cpu = {
        "batch_1": profile_model(
            transformer.model, batch_size=1, iterations=args.profile_iterations
        ),
        "batch_128": profile_model(
            transformer.model, batch_size=128, iterations=args.profile_iterations
        ),
    }
    report = {
        "report_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata": transformer.checkpoint_metadata,
        "dataset": str(args.dataset.resolve()),
        "environment": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "positions": position_report,
        "tournaments": tournaments,
        "cpu": cpu,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
        output_file.write("\n")

    print_summary(report)
    print(f"\nSaved reproducible baseline report to {args.output}")


if __name__ == "__main__":
    main()
