"""Pool independent matched board sets and bootstrap paired win differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


def compare_series(pairs, samples=10000):
    reports = [(json.loads(a.read_text()), json.loads(b.read_text())) for a, b in pairs]
    protocol = reports[0][0]["protocol"]
    rules = reports[0][0]["rules"]
    for candidate, incumbent in reports:
        if (candidate["protocol"] != incumbent["protocol"] or
                candidate["protocol"] != protocol or
                candidate["rules"] != incumbent["rules"] or
                candidate["rules"] != rules):
            raise ValueError("Incompatible protocols or rules")
    tiers = set.intersection(*(set(a["tiers"]) & set(b["tiers"]) for a, b in reports))
    result = {"protocol": protocol, "rules": rules, "pairs": [[str(a), str(b)] for a, b in pairs], "tiers": {}}
    rng = random.Random(61015)
    for tier in sorted(tiers):
        candidate_games, incumbent_games = {}, {}
        for candidate, incumbent in reports:
            for source, destination in ((candidate, candidate_games), (incumbent, incumbent_games)):
                for game in source["tiers"][tier]["games"]:
                    key = (game["seed"], game["seat"])
                    if key in destination:
                        raise ValueError(f"Duplicate board and seat: {key}")
                    destination[key] = game
        if candidate_games.keys() != incumbent_games.keys():
            raise ValueError(f"Unmatched boards or seats for {tier}")
        seeds = sorted({seed for seed, _ in candidate_games})
        board_differences = []
        for seed in seeds:
            seats = sorted(seat for actual_seed, seat in candidate_games if actual_seed == seed)
            board_differences.append(statistics.fmean(
                int(candidate_games[seed, seat]["won"]) - int(incumbent_games[seed, seat]["won"])
                for seat in seats
            ))
        draws = sorted(statistics.fmean(rng.choices(board_differences, k=len(seeds)))
                       for _ in range(samples))
        games = len(candidate_games)
        candidate_wins = sum(game["won"] for game in candidate_games.values())
        incumbent_wins = sum(game["won"] for game in incumbent_games.values())
        result["tiers"][tier] = {
            "boards": len(seeds), "games": games,
            "candidate_wins": candidate_wins,
            "incumbent_wins": incumbent_wins,
            "candidate_win_rate": candidate_wins / games,
            "incumbent_win_rate": incumbent_wins / games,
            "paired_difference": statistics.fmean(board_differences),
            "board_bootstrap_95_difference": [draws[int(samples * 0.025)], draws[int(samples * 0.975) - 1]],
            "candidate_mean_vp": statistics.fmean(game["candidate_vp"] for game in candidate_games.values()),
            "incumbent_mean_vp": statistics.fmean(game["candidate_vp"] for game in incumbent_games.values()),
            "candidate_errors": sum(game["status"] == "error" for game in candidate_games.values()),
            "incumbent_errors": sum(game["status"] == "error" for game in incumbent_games.values()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, type=Path, action="append", required=True,
                        metavar=("CANDIDATE", "INCUMBENT"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare_series(args.pair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for tier, summary in result["tiers"].items():
        print(tier, summary["candidate_wins"], summary["incumbent_wins"],
              summary["board_bootstrap_95_difference"])


if __name__ == "__main__":
    main()
