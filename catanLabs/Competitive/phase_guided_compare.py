"""Matched phase-gate summary with seat-level outcomes and paired intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from catanLabs.Competitive.compare import compare


ARMS = ("opening", "turn", "all", "fair")


def summarize(reports):
    fair = reports["fair"]
    protocol, rules = fair["protocol"], fair["rules"]
    mixes = set(fair["tiers"])
    for name, report in reports.items():
        if report["protocol"] != protocol or report["rules"] != rules:
            raise ValueError(f"Protocol or rules mismatch: {name}")
        if set(report["tiers"]) != mixes:
            raise ValueError(f"Opponent mix mismatch: {name}")
    summary = {"protocol": protocol, "rules": rules, "by_mix": {},
               "paired_vs_fair": {name: compare(reports[name], fair)
                                  for name in ARMS if name != "fair"}}
    for mix in sorted(mixes):
        rows = {}
        for name, report in reports.items():
            indexed = {(game["seed"], game["seat"]): game
                       for game in report["tiers"][mix]["games"]}
            if len(indexed) != len(report["tiers"][mix]["games"]):
                raise ValueError(f"Duplicate seed/seat in {name}/{mix}")
            rows[name] = indexed
        keys = set(rows["fair"])
        if any(set(indexed) != keys for indexed in rows.values()):
            raise ValueError(f"Unmatched seed/seat grid for {mix}")
        seeds = {seed for seed, _ in keys}
        expected_boards = fair.get("boards_per_mix")
        if (expected_boards is not None and len(seeds) != expected_boards
                or any({seat for actual_seed, seat in keys if actual_seed == seed}
                       != {0, 1, 2, 3} for seed in seeds)):
            raise ValueError(f"Incomplete four-seat board grid for {mix}")
        for key in keys:
            control = rows["fair"][key]
            for name, indexed in rows.items():
                game = indexed[key]
                if game["status"] != "completed":
                    raise ValueError(f"Incomplete game in {name}/{mix}/{key}")
                if (game["color"] != control["color"]
                        or game["opponent_assignments"] != control["opponent_assignments"]):
                    raise ValueError(f"Unmatched opponent assignment in {name}/{mix}/{key}")
        by_seat = {}
        for seat in sorted({seat for _, seat in keys}):
            seat_keys = sorted(key for key in keys if key[1] == seat)
            fair_wins = sum(bool(rows["fair"][key]["won"]) for key in seat_keys)
            by_seat[str(seat)] = {"games": len(seat_keys), "fair_wins": fair_wins,
                                 "arms": {}}
            for name in ARMS[:-1]:
                wins = sum(bool(rows[name][key]["won"]) for key in seat_keys)
                vp_diff = statistics.fmean(rows[name][key]["candidate_vp"]
                                           - rows["fair"][key]["candidate_vp"]
                                           for key in seat_keys)
                by_seat[str(seat)]["arms"][name] = {
                    "wins": wins, "paired_win_difference_games": wins - fair_wins,
                    "mean_vp_difference": vp_diff}
        summary["by_mix"][mix] = {"games": len(keys), "by_seat": by_seat}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ARMS:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {name: json.loads(getattr(args, name).read_text()) for name in ARMS}
    result = summarize(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for mix, data in result["by_mix"].items():
        print(mix)
        for seat, row in data["by_seat"].items():
            print(f"  seat {seat}: fair {row['fair_wins']}/{row['games']}; "
                  + ", ".join(f"{name} {value['wins']}/{row['games']} "
                              f"({value['paired_win_difference_games']:+d})"
                              for name, value in row["arms"].items()))


if __name__ == "__main__":
    main()
