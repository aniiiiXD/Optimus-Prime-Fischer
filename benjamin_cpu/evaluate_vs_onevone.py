"""Paired-seat Benjamin CPU checkpoint vs plain OneVOne on fresh boards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import secrets
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent / "Hyde"
sys.path.insert(0, str(MAIN))

from benjamin_cpu.catanatron_adapter import BenjaminCatanatronAgent  # noqa: E402
from catanLabs.Catanatron.tournament_protocol import (  # noqa: E402
    canonical_action_key,
    install_v5_protocol,
)
from catanLabs.OneVOne.agent import DEFAULT_CONFIG, create_agent as create_onevone  # noqa: E402
from catanatron import Color, Game, RandomPlayer  # noqa: E402


def play(seed: int, benjamin_seat: int, checkpoint: Path, max_seconds: float) -> dict:
    random.seed(seed * 2 + benjamin_seat)
    game = Game(
        [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
        seed=seed,
        vps_to_win=10,
        discard_limit=7,
        friendly_robber=False,
        number_placement="official_spiral",
    )
    game.playable_actions.sort(key=canonical_action_key)
    ben_color = game.state.colors[benjamin_seat]
    one_color = next(color for color in game.state.colors if color != ben_color)
    agents = {
        ben_color: BenjaminCatanatronAgent(ben_color, checkpoint),
        one_color: create_onevone(one_color, str(DEFAULT_CONFIG)),
    }
    moves = {"Benjamin": Counter(), "OneVOne": Counter()}
    status, error = "capped", None
    actions = 0
    started = time.perf_counter()
    try:
        while actions < 10000 and game.state.num_turns < 500:
            if game.winning_color() is not None:
                status = "completed"
                break
            if time.perf_counter() - started > max_seconds:
                break
            actor = game.state.current_color()
            action = agents[actor].decide(game, game.playable_actions)
            if action not in game.playable_actions:
                raise ValueError(f"illegal action: {action!r}")
            moves["Benjamin" if actor == ben_color else "OneVOne"][action.action_type.name] += 1
            game.execute(action, validate_action=True)
            game.playable_actions.sort(key=canonical_action_key)
            actions += 1
        if game.winning_color() is not None:
            status = "completed"
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
    winner = game.winning_color()
    vp = {
        name: int(game.state.player_state[
            f"P{game.state.color_to_index[color]}_ACTUAL_VICTORY_POINTS"
        ])
        for name, color in (("Benjamin", ben_color), ("OneVOne", one_color))
    }
    return {
        "seed": seed,
        "benjamin_seat": benjamin_seat,
        "status": status,
        "winner": "Benjamin" if winner == ben_color else "OneVOne" if winner == one_color else None,
        "vp": vp,
        "turns": game.state.num_turns,
        "actions": actions,
        "seconds": round(time.perf_counter() - started, 3),
        "move_counts": {name: dict(counter) for name, counter in moves.items()},
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-pairs", type=int, default=25)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path,
                        default=ROOT / "benjamin_cpu/reports/benjamin_vs_onevone_50.jsonl")
    parser.add_argument("--max-game-seconds", type=float, default=120)
    args = parser.parse_args()
    torch.set_num_threads(4)
    install_v5_protocol()
    meta_path = args.report.with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["checkpoint"] != str(args.checkpoint) or len(meta["seeds"]) != args.board_pairs:
            raise ValueError("Existing report uses a different checkpoint or board count")
    else:
        selection_seed = secrets.randbits(64)
        seeds = random.Random(selection_seed).sample(range(1, 2**31 - 1), args.board_pairs)
        meta = {
            "selection_seed": selection_seed,
            "seeds": seeds,
            "checkpoint": str(args.checkpoint),
            "onevone_config": str(DEFAULT_CONFIG),
            "rules": "Catanatron 1v1, 10 VP, official spiral, discard 7, no friendly robber",
            "seats": "Benjamin and OneVOne swap seats on each seed",
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    rows = ([json.loads(line) for line in args.report.read_text().splitlines() if line]
            if args.report.exists() else [])
    completed = {(row["seed"], row["benjamin_seat"]) for row in rows}
    for seed in meta["seeds"]:
        for benjamin_seat in (0, 1):
            if (seed, benjamin_seat) in completed:
                continue
            row = play(seed, benjamin_seat, args.checkpoint, args.max_game_seconds)
            with args.report.open("a") as out:
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
            rows.append(row)
            if len(rows) % 5 == 0 or row["status"] != "completed":
                print(f"{len(rows)}/{args.board_pairs * 2} "
                      f"Benjamin={sum(r['winner'] == 'Benjamin' for r in rows)} "
                      f"OneVOne={sum(r['winner'] == 'OneVOne' for r in rows)} "
                      f"last={row['status']}", flush=True)
    summary = {
        "games": len(rows),
        "boards": len(meta["seeds"]),
        "benjamin_wins": sum(row["winner"] == "Benjamin" for row in rows),
        "onevone_wins": sum(row["winner"] == "OneVOne" for row in rows),
        "completed": sum(row["status"] == "completed" for row in rows),
        "errors": [row["error"] for row in rows if row["status"] == "error"],
        "capped": sum(row["status"] == "capped" for row in rows),
        "benjamin_wins_by_seat": {
            str(seat): sum(row["winner"] == "Benjamin" and row["benjamin_seat"] == seat
                           for row in rows)
            for seat in (0, 1)
        },
        "mean_vp": {name: statistics.mean(row["vp"][name] for row in rows)
                    for name in ("Benjamin", "OneVOne")},
    }
    summary_path = args.report.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
