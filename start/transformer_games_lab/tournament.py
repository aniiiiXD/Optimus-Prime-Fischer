#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gamebench.evaluation.tournament import TournamentConfig, run_tournament
from gamebench.registry import make_agent, make_game


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a seat-balanced round-robin tournament.")
    parser.add_argument("--game", choices=("othello", "connect4"), default="othello")
    parser.add_argument("--agents", default="random,greedy,minimax:2")
    parser.add_argument("--games", type=int, default=20, help="Games per agent pair")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-games", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    game = make_game(args.game)
    agents = [
        make_agent(name.strip(), expected_game=args.game)
        for name in args.agents.split(",")
        if name.strip()
    ]
    result = run_tournament(
        game,
        agents,
        TournamentConfig(games_per_pair=args.games, seed=args.seed, record_games=args.record_games),
    )
    print(f"{args.game}: {args.games} games per pair")
    print(f"{'agent':<18} {'W':>4} {'D':>4} {'L':>4} {'score':>8} {'Elo':>8} {'move ms':>10}")
    for name, row in result["standings"].items():
        latency = row["decision_latency_ms"]["median"]
        print(f"{name:<18} {row['wins']:>4} {row['draws']:>4} {row['losses']:>4} {row['score_rate']:>8.3f} {row['elo']:>8.1f} {latency:>10.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
