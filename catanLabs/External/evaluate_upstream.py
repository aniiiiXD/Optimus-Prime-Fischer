"""Seat-rotated evaluation of the upstream Rust AlphaBot on its own engine.

This is deliberately separate from Catanatron: its action/observation codecs,
visibility, and stochastic game rules are not interchangeable with ours.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "catan-rl"
DEFAULT_BINARY = UPSTREAM / "rust/target/release/catan-sim"
DEFAULT_NET = UPSTREAM / "models/catan-512.ctnn"


def wilson(wins: int, games: int) -> list[float]:
    if not games:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games**2)) / denominator
    return [round(max(0.0, center - radius), 4), round(min(1.0, center + radius), 4)]


def run(args: argparse.Namespace) -> dict:
    if args.games < 1:
        raise ValueError("--games must be positive")
    if args.opponent not in {"H", "V"}:
        raise ValueError("--opponent must be H or V")
    args.output = args.output.resolve()
    args.binary = args.binary.resolve(strict=True)
    args.net = args.net.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_games = []
    by_seat = []
    started = time.perf_counter()
    for seat in range(4):
        players = [args.opponent] * 4
        players[seat] = args.candidate
        metrics_path = args.output.with_name(f"{args.output.stem}.seat{seat}.jsonl")
        command = [str(args.binary), "--games", str(args.games), "--seed", str(args.seed),
                   "--players", ",".join(players), "--victory-target", str(args.target),
                   "--net", str(args.net), "--alpha-config", args.alpha_config,
                   "--metrics", str(metrics_path)]
        completed = subprocess.run(command, cwd=UPSTREAM / "rust", capture_output=True,
                                   text=True, check=True)
        events = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        games = [event for event in events if event["t"] == "game"]
        if len(games) != args.games or any(event["winner"] < 0 or event["cap"] for event in games):
            raise RuntimeError(f"seat {seat}: missing/capped games; inspect {metrics_path}")
        wins = sum(event["winner"] == seat for event in games)
        by_seat.append({"seat": seat, "games": len(games), "wins": wins,
                        "win_rate": round(wins / len(games), 4),
                        "mean_vp": round(sum(event["vp"][seat] for event in games) / len(games), 3),
                        "simulator_stdout": completed.stdout.strip()})
        all_games.extend({"seat": seat, "seed_index": event["i"], "winner": event["winner"],
                          "won": event["winner"] == seat, "turns": event["turns"],
                          "vp": event["vp"][seat]} for event in games)
        print(f"seat {seat}: {wins}/{len(games)} ({wins/len(games):.1%})", flush=True)
    wins = sum(row["won"] for row in all_games)
    result = {
        "protocol": "upstream-catan-rl-local-seat-rotation-v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                  cwd=UPSTREAM, text=True).strip(),
        "source_modified": bool(subprocess.check_output(["git", "status", "--porcelain"],
                                                       cwd=UPSTREAM, text=True).strip()),
        "model": str(args.net.resolve()),
        "model_sha256": hashlib.sha256(args.net.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "candidate": {"A": "alpha-search", "T": "alpha-search-with-trades",
                       "P": "policy-only-alpha-action-set", "Q": "policy-only-all-legal-actions"}[args.candidate],
        "rules": {"players": 4, "target_vp": args.target,
                  "root_observation": "perfect information",
                  "rollouts": "clone of actual full state; random policy" if args.candidate in {"A", "T"} else "none",
                  "trade_proposals": args.candidate in {"T", "Q"}},
        "opponent": "heuristic-v1" if args.opponent == "H" else "heuristic-v2",
        "alpha_config": args.alpha_config,
        "first_seed": args.seed,
        "games_per_seat": args.games,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "total": {"games": len(all_games), "wins": wins,
                  "win_rate": round(wins / len(all_games), 4),
                  "wilson95": wilson(wins, len(all_games)),
                  "mean_vp": round(sum(row["vp"] for row in all_games) / len(all_games), 3)},
        "by_seat": by_seat,
        "games": all_games,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--net", type=Path, default=DEFAULT_NET)
    parser.add_argument("--target", type=int, choices=(7, 10), default=7)
    parser.add_argument("--opponent", choices=("H", "V"), default="H")
    parser.add_argument("--candidate", choices=("A", "T", "P", "Q"), default="A")
    parser.add_argument("--alpha-config", default="8,96,300")
    parser.add_argument("--games", type=int, default=48, help="Games per seat")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"total": result["total"], "elapsed_seconds": result["elapsed_seconds"]},
                     indent=2))


if __name__ == "__main__":
    main()
