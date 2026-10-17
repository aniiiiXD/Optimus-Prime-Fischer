"""Compare two paired reports written by ``family_eval.py``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


def compare(candidate, incumbent, samples=10_000):
    keys = ("protocol", "boards_per_variant", "first_seed", "opponent_composition")
    if any(candidate.get(key) != incumbent.get(key) for key in keys):
        raise ValueError("Family reports have incompatible evaluation grids")
    result = {"protocol": candidate["protocol"], "tiers": {}}
    rng = random.Random(61015)
    if candidate["tiers"].keys() != incumbent["tiers"].keys():
        raise ValueError("Family reports have different variants")
    for name in sorted(candidate["tiers"]):
        a, b = candidate["tiers"][name], incumbent["tiers"][name]
        if a["rules"] != b["rules"]:
            raise ValueError(f"Rule mismatch for {name}")
        by_a = {(g["seed"], g["seat"]): g for g in a["games"]}
        by_b = {(g["seed"], g["seat"]): g for g in b["games"]}
        if by_a.keys() != by_b.keys():
            raise ValueError(f"Unmatched boards or seats for {name}")
        if any(by_a[key]["opponent_assignments"] != by_b[key]["opponent_assignments"] for key in by_a):
            raise ValueError(f"Opponent assignments differ for {name}")
        boards = sorted({seed for seed, _ in by_a})
        win_diffs, vp_diffs = [], []
        for seed in boards:
            seats = [seat for actual_seed, seat in by_a if actual_seed == seed]
            win_diffs.append(statistics.fmean(int(by_a[seed, seat]["won"]) - int(by_b[seed, seat]["won"]) for seat in seats))
            vp_diffs.append(statistics.fmean(by_a[seed, seat]["candidate_vp"] - by_b[seed, seat]["candidate_vp"] for seat in seats))
        boot = sorted(statistics.fmean(rng.choices(win_diffs, k=len(boards))) for _ in range(samples))
        result["tiers"][name] = {
            "rules": a["rules"], "boards": len(boards), "games": len(by_a),
            "candidate_wins": a["summary"]["wins"], "incumbent_wins": b["summary"]["wins"],
            "win_rate_difference": statistics.fmean(win_diffs),
            "board_bootstrap_95_difference": [boot[int(.025 * samples)], boot[int(.975 * samples) - 1]],
            "mean_vp_difference": statistics.fmean(vp_diffs),
            "candidate_latency_ms": a["summary"]["decision_latency_ms"],
            "incumbent_latency_ms": b["summary"]["decision_latency_ms"],
            "candidate_noncompleted": sum(g["status"] != "completed" for g in by_a.values()),
            "incumbent_noncompleted": sum(g["status"] != "completed" for g in by_b.values()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("incumbent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.candidate.read_text()), json.loads(args.incumbent.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, tier in result["tiers"].items():
        print(name, tier["candidate_wins"], tier["incumbent_wins"],
              "delta", round(tier["win_rate_difference"], 3),
              "95%", [round(x, 3) for x in tier["board_bootstrap_95_difference"]])


if __name__ == "__main__":
    main()
