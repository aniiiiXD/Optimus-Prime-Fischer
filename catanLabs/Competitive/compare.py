"""Compare two tournament reports with board-seed clustered intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


def compare(candidate, reference, bootstrap_samples=10000):
    if candidate["protocol"] != reference["protocol"] or candidate["rules"] != reference["rules"]:
        raise ValueError("Tournament protocols or rules differ")
    result = {"protocol": candidate["protocol"], "rules": candidate["rules"], "tiers": {}}
    rng = random.Random(61015)
    for tier in sorted(candidate["tiers"].keys() & reference["tiers"].keys()):
        a = {(g["seed"], g["seat"]): g for g in candidate["tiers"][tier]["games"]}
        b = {(g["seed"], g["seat"]): g for g in reference["tiers"][tier]["games"]}
        if a.keys() != b.keys():
            raise ValueError(f"Unmatched board seeds or seats for {tier}")
        board_seeds = sorted({seed for seed, _ in a})
        board_differences = []
        vp_differences = []
        for seed in board_seeds:
            seats = sorted(seat for actual_seed, seat in a if actual_seed == seed)
            board_differences.append(statistics.fmean(int(a[seed, seat]["won"]) - int(b[seed, seat]["won"]) for seat in seats))
            vp_differences.append(statistics.fmean(a[seed, seat]["candidate_vp"] - b[seed, seat]["candidate_vp"] for seat in seats))
        bootstrapped = sorted(
            statistics.fmean(rng.choices(board_differences, k=len(board_seeds)))
            for _ in range(bootstrap_samples)
        )
        a_summary = candidate["tiers"][tier]["summary"]
        b_summary = reference["tiers"][tier]["summary"]
        result["tiers"][tier] = {
            "boards": len(board_seeds), "games": len(a),
            "candidate_wins": a_summary["wins"],
            "reference_wins": b_summary["wins"],
            "candidate_win_rate": a_summary["win_rate_all_attempts"],
            "reference_win_rate": b_summary["win_rate_all_attempts"],
            "win_rate_difference": statistics.fmean(board_differences),
            "board_bootstrap_95_difference": [
                bootstrapped[int(0.025 * bootstrap_samples)],
                bootstrapped[int(0.975 * bootstrap_samples) - 1],
            ],
            "mean_vp_difference": statistics.fmean(vp_differences),
            "candidate_mean_vp": a_summary["mean_final_vp"],
            "reference_mean_vp": b_summary["mean_final_vp"],
            "candidate_latency_ms": a_summary["decision_latency_ms"],
            "reference_latency_ms": b_summary["decision_latency_ms"],
            "candidate_errors": sum(g["status"] == "error" for g in a.values()),
            "reference_errors": sum(g["status"] == "error" for g in b.values()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    reference = json.loads(args.reference.read_text())
    result = compare(candidate, reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, tier in result["tiers"].items():
        print(name, tier["candidate_wins"], tier["reference_wins"],
              "delta", round(tier["win_rate_difference"], 3),
              "95%", [round(x, 3) for x in tier["board_bootstrap_95_difference"]])


if __name__ == "__main__":
    main()
