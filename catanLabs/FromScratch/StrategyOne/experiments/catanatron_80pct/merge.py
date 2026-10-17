"""Merge completed, non-overlapping board shards into one validated report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=HERE / "mlp_vs_web_80pct.json")
    parser.add_argument("--shards", type=Path, nargs="*",
                        default=[HERE / f"shard_{index}.json" for index in range(1, 5)])
    parser.add_argument("--output", type=Path, default=HERE / "mlp_vs_web_80pct_merged.json")
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in [args.base, *args.shards]]
    base = reports[0]
    selected = base["selected_boards"]
    expected = {(board["seed"], seat) for board in selected for seat in (0, 1)}
    seen = {}
    for report in reports:
        for key in ("candidate", "opponent", "engine", "config", "fraction",
                    "selection_seed", "rules", "seat_protocol"):
            if report[key] != base[key]:
                raise ValueError(f"Mismatched {key} in a shard")
        for game in report["games"]:
            key = (game["seed"], game["seat"])
            if key in seen:
                raise ValueError(f"Duplicate game {key}")
            seen[key] = game
    if set(seen) != expected:
        raise ValueError(f"Expected {len(expected)} game keys, found {len(seen)}; "
                         f"missing={len(expected - set(seen))}, extra={len(set(seen) - expected)}")
    games = sorted(seen.values(), key=lambda row: (row["game_id"], row["seat"]))
    wins_by_seed = Counter(game["seed"] for game in games if game["ours_won"])
    summary = {
        "games": len(games),
        "completed": sum(game["status"] == "completed" for game in games),
        "capped": sum(game["status"] == "capped" for game in games),
        "errors": sum(game["status"] == "error" for game in games),
        "ours_wins": sum(game["ours_won"] for game in games),
        "catanatron_wins": sum(game["catanatron_won"] for game in games),
        "by_board": {
            "ours_won_both_seats": sum(wins_by_seed[board["seed"]] == 2 for board in selected),
            "ours_won_one_seat": sum(wins_by_seed[board["seed"]] == 1 for board in selected),
            "ours_won_neither_seat": sum(wins_by_seed[board["seed"]] == 0 for board in selected),
        },
        "by_seat": {
            str(seat): {
                "games": sum(game["seat"] == seat for game in games),
                "ours_wins": sum(game["seat"] == seat and game["ours_won"] for game in games),
            }
            for seat in (0, 1)
        },
        "by_dataset_split": {
            split: {
                "games": sum((game["game_id"] % 5 == 0) == is_holdout for game in games),
                "ours_wins": sum((game["game_id"] % 5 == 0) == is_holdout and game["ours_won"] for game in games),
            }
            for split, is_holdout in (("teacher_train_boards", False), ("teacher_held_out_boards", True))
        },
        "move_counts": {
            label: dict(sum((Counter(game["move_counts"][label]) for game in games), Counter()))
            for label in ("ours", "catanatron")
        },
    }
    merged = {key: value for key, value in base.items() if key not in ("games", "summary")}
    merged["games"] = games
    merged["summary"] = summary
    args.output.write_text(json.dumps(merged, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
