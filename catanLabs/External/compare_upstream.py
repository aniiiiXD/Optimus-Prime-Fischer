"""Paired board-seed bootstrap for two seat-rotated Rust search reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    new = json.loads(args.candidate.read_text())
    old = json.loads(args.control.read_text())
    keys = ("protocol", "opponent", "alpha_config", "first_seed", "games_per_seat")
    for key in keys:
        if new[key] != old[key]:
            raise ValueError(f"Reports differ in {key}: {new[key]} versus {old[key]}")
    for key in ("players", "target_vp", "root_observation"):
        if new["rules"][key] != old["rules"][key]:
            raise ValueError(f"Reports differ in rules.{key}")
    new_games = {(g["seed_index"], g["seat"]): int(g["won"]) for g in new["games"]}
    old_games = {(g["seed_index"], g["seat"]): int(g["won"]) for g in old["games"]}
    if new_games.keys() != old_games.keys():
        raise ValueError("Reports have different board/seat keys")
    boards = sorted({key[0] for key in new_games})
    board_diffs = np.array([sum(new_games[(board, seat)] - old_games[(board, seat)]
                                for seat in range(4)) / 4 for board in boards], dtype=float)
    rng = np.random.default_rng(20261015)
    sampled = rng.integers(0, len(boards), size=(args.resamples, len(boards)))
    interval = np.quantile(board_diffs[sampled].mean(axis=1), [0.025, 0.975]).tolist()
    report = {
        "candidate": str(args.candidate.resolve()),
        "control": str(args.control.resolve()),
        "candidate_agent": new.get("candidate", "alpha-search"),
        "control_agent": old.get("candidate", "alpha-search"),
        "boards": len(boards),
        "games_each": len(new_games),
        "candidate_wins": sum(new_games.values()),
        "control_wins": sum(old_games.values()),
        "win_rate_difference": round(float(board_diffs.mean()), 4),
        "board_bootstrap_95": [round(float(x), 4) for x in interval],
        "resamples": args.resamples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
