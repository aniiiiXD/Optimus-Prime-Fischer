"""Pool independent matched upstream search tournaments by board seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", nargs="+", type=Path, required=True)
    parser.add_argument("--control", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    if len(args.candidate) != len(args.control):
        parser.error("Candidate and control need the same number of series")
    board_differences = []
    pairs = []
    candidate_wins = control_wins = 0
    reference = None
    for candidate_path, control_path in zip(args.candidate, args.control):
        candidate = json.loads(candidate_path.read_text())
        control = json.loads(control_path.read_text())
        for report in (candidate, control):
            signature = (report["opponent"], report["rules"]["players"],
                         report["rules"]["target_vp"], report["alpha_config"])
            if reference is None:
                reference = signature
            elif signature != reference:
                raise ValueError("Series do not share opponent, game rules, and search settings")
        if (candidate["first_seed"], candidate["games_per_seat"]) != (
                control["first_seed"], control["games_per_seat"]):
            raise ValueError("Candidate and control seeds/boards differ")
        candidate_games = {(g["seed_index"], g["seat"]): int(g["won"])
                           for g in candidate["games"]}
        control_games = {(g["seed_index"], g["seat"]): int(g["won"])
                         for g in control["games"]}
        if candidate_games.keys() != control_games.keys():
            raise ValueError("Candidate and control game keys differ")
        boards = sorted({key[0] for key in candidate_games})
        board_differences.extend(sum(candidate_games[(board, seat)] - control_games[(board, seat)]
                                     for seat in range(4)) / 4 for board in boards)
        candidate_wins += sum(candidate_games.values())
        control_wins += sum(control_games.values())
        pairs.append({"candidate": str(candidate_path.resolve()),
                      "control": str(control_path.resolve()), "boards": len(boards),
                      "candidate_wins": sum(candidate_games.values()),
                      "control_wins": sum(control_games.values())})
    differences = np.asarray(board_differences, dtype=float)
    rng = np.random.default_rng(20261015)
    samples = rng.integers(0, len(differences), size=(args.resamples, len(differences)))
    lower, upper = np.quantile(differences[samples].mean(axis=1), [0.025, 0.975])
    result = {
        "protocol": "upstream-board-clustered-paired-bootstrap-v1",
        "pairs": pairs, "opponent": reference[0], "target_vp": reference[2],
        "boards": len(differences), "games_each": len(differences) * 4,
        "candidate_wins": candidate_wins, "control_wins": control_wins,
        "win_rate_difference": round(float(differences.mean()), 4),
        "board_bootstrap_95": [round(float(lower), 4), round(float(upper), 4)],
        "resamples": args.resamples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
