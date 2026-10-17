"""Paired-seat benchmark of the pinned Catanatron web AlphaBeta bot.

Run from the workspace root with start/.venv/bin/python. The report is written
after every game so a slow benchmark can be stopped without losing results.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
STRATEGY = WORKSPACE / "catanLabs/FromScratch/StrategyOne"
sys.path[:0] = [str(WORKSPACE), str(STRATEGY)]

from catanLabs.Catanatron.engine import source_info  # noqa: E402
from catanLabs.Catanatron.tournament_protocol import (  # noqa: E402
    canonical_action_key,
    install_v5_protocol,
)

ENGINE = source_info()
sys.path.insert(0, ENGINE["python_root"])

from catanatron import Color, Game, RandomPlayer  # noqa: E402
from catanatron.players.minimax import AlphaBetaPlayer  # noqa: E402
from catanatron.web.players import Catanatron  # noqa: E402
from generate_dataset import make_bot  # noqa: E402


def run_game(seed: int, seat: int, opponent_name: str, variant: str, max_seconds: float, max_actions: int) -> tuple[dict, list[float]]:
    colors = [Color.RED, Color.BLUE]
    game = Game(
        [RandomPlayer(color) for color in colors],
        seed=seed,
        vps_to_win=10,
        discard_limit=7,
        friendly_robber=False,
        number_placement="official_spiral",
    )
    game.playable_actions.sort(key=canonical_action_key)
    candidate_color = game.state.colors[seat]
    opponent_color = next(color for color in game.state.colors if color != candidate_color)
    candidate = (
        Catanatron(candidate_color, Catanatron.Params())
        if variant == "web"
        else AlphaBetaPlayer(candidate_color, AlphaBetaPlayer.Params(depth=2, value_fn="base", prunning=False, epsilon=None))
    )
    opponent = make_bot(opponent_name, opponent_color)
    policies = {candidate_color: candidate, opponent_color: opponent}
    started = time.perf_counter()
    latencies = []
    status = "capped"
    error = None
    actions = 0
    try:
        while actions < max_actions and game.state.num_turns < 500:
            if game.winning_color() is not None:
                status = "completed"
                break
            if time.perf_counter() - started > max_seconds:
                break
            actor = game.state.current_color()
            before = time.perf_counter_ns()
            action = policies[actor].decide(game, game.playable_actions)
            if actor == candidate_color:
                latencies.append((time.perf_counter_ns() - before) / 1e6)
            if action not in game.playable_actions:
                raise ValueError(f"Illegal action from {type(policies[actor]).__name__}: {action}")
            game.execute(action, validate_action=True)
            game.playable_actions.sort(key=canonical_action_key)
            actions += 1
        if game.winning_color() is not None:
            status = "completed"
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    winner = game.winning_color()
    candidate_prefix = f"P{game.state.color_to_index[candidate_color]}_"
    opponent_prefix = f"P{game.state.color_to_index[opponent_color]}_"
    record = {
        "seed": seed,
        "seat": seat,
        "opponent": opponent_name,
        "status": status,
        "won": status == "completed" and winner == candidate_color,
        "winner": winner.value if winner is not None else None,
        "candidate_vp": int(game.state.player_state[candidate_prefix + "ACTUAL_VICTORY_POINTS"]),
        "opponent_vp": int(game.state.player_state[opponent_prefix + "ACTUAL_VICTORY_POINTS"]),
        "actions": actions,
        "turns": game.state.num_turns,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    if error is not None:
        record["error"] = error
    return record, latencies


def summarize(games: list[dict], latencies: list[float]) -> dict:
    sorted_latencies = sorted(latencies)
    return {
        "games": len(games),
        "wins": sum(g["won"] for g in games),
        "completed": sum(g["status"] == "completed" for g in games),
        "capped": sum(g["status"] == "capped" for g in games),
        "errors": sum(g["status"] == "error" for g in games),
        "median_decision_ms": statistics.median(latencies) if latencies else None,
        "p95_decision_ms": sorted_latencies[int(.95 * (len(sorted_latencies) - 1))] if sorted_latencies else None,
        "max_decision_ms": max(latencies) if latencies else None,
        "decisions": len(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=16, help="Each board is played from both seats")
    parser.add_argument("--seed", type=int, default=20268100)
    parser.add_argument("--opponents", default="F_BASE,F_CONTENDER")
    parser.add_argument("--variant", choices=("web", "official"), default="web")
    parser.add_argument("--max-game-seconds", type=float, default=120.0)
    parser.add_argument("--max-actions", type=int, default=10000)
    parser.add_argument("--report", type=Path, default=HERE / "results.json")
    args = parser.parse_args()
    opponents = [x.strip() for x in args.opponents.split(",") if x.strip()]
    if args.boards <= 0 or any(x not in {"F_BASE", "F_CONTENDER"} for x in opponents):
        parser.error("Use positive boards and F_BASE,F_CONTENDER opponents")
    install_v5_protocol()
    report = {
        "strategy": "pinned Catanatron AlphaBeta depth=2, base value, pruning=" + str(args.variant == "web"),
        "engine": ENGINE,
        "rules": {"players": 2, "vps_to_win": 10, "discard_limit": 7, "friendly_robber": False, "number_placement": "official_spiral"},
        "seed_start": args.seed,
        "boards_per_opponent": args.boards,
        "seat_rotation": "both seats per seed",
        "caps": {"max_game_seconds": args.max_game_seconds, "max_actions": args.max_actions, "max_turns": 500},
        "results": {},
    }
    for opponent in opponents:
        games = []
        latencies = []
        for index in range(args.boards):
            for seat in range(2):
                record, samples = run_game(args.seed + index, seat, opponent, args.variant, args.max_game_seconds, args.max_actions)
                games.append(record)
                latencies.extend(samples)
                report["results"][opponent] = {"summary": summarize(games, latencies), "games": games}
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                print(f"{opponent} {len(games)}/{args.boards * 2}: {record['status']} {'WIN' if record['won'] else 'NO WIN'} {record['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})
    main()
