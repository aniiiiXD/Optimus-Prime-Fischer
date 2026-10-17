"""Held-out tournaments against mixed Catanatron opponent compositions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
RUNNER = WORKSPACE / "catanLabs/Benchmarks&Evals/run.py"
spec = importlib.util.spec_from_file_location("catanlabs_benchmark", RUNNER)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
from catanLabs.Catanatron.tournament_protocol import PROTOCOL, install_v5_protocol

COMPOSITIONS = {
    "mixed_diverse": ("weighted", "value", "alphabeta1"),
    "mixed_strong": ("value", "alphabeta1", "alphabeta1"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--agent", choices=("value", "alphabeta1"))
    selection.add_argument("--agent-file", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--boards", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261700)
    parser.add_argument("--max-game-seconds", type=float, default=90.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.boards < 1 or args.max_game_seconds <= 0:
        parser.error("boards and max-game-seconds must be positive")
    engine = benchmark.source_info()
    sys.path.insert(0, engine["python_root"])
    from catanatron import Color, Game, RandomPlayer
    from catanatron.players.weighted_random import WeightedRandomPlayer
    from catanatron.players.value import ValueFunctionPlayer
    from catanatron.players.minimax import AlphaBetaPlayer
    install_v5_protocol()

    classes = {"weighted": WeightedRandomPlayer, "value": ValueFunctionPlayer}
    def baseline(name, color):
        if name == "alphabeta1":
            return AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=1))
        return classes[name](color)

    if args.agent_file:
        agent_path = args.agent_file.resolve(strict=True)
        factory = benchmark.load_agent_factory(agent_path)
        candidate = {"type": "external", "file": str(agent_path),
                     "sha256": benchmark.sha256(agent_path)}
    else:
        factory = lambda color, checkpoint, seed: baseline(args.agent, color)
        candidate = {"type": args.agent}
    checkpoint = args.checkpoint.resolve(strict=True) if args.checkpoint else None
    if checkpoint:
        candidate.update({"checkpoint": str(checkpoint),
                          "checkpoint_sha256": benchmark.sha256(checkpoint)})
    report = {
        "protocol": f"{PROTOCOL}-mixed",
        "engine": engine,
        "rules": {"players": 4, "target_vp": 10, "discard_limit": 7,
                  "friendly_robber": False, "number_placement": "official_spiral",
                  "player_trade_offers": False, "isolated_copy_rng": True,
                  "forced_search_outcomes": True},
        "candidate": candidate,
        "boards_per_mix": args.boards,
        "first_seed": args.seed,
        "compositions": COMPOSITIONS,
        "platform": platform.platform(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "runner_sha256": benchmark.sha256(Path(__file__)),
        "tiers": {},
    }
    colors = list(Color)
    for mix_name, composition in COMPOSITIONS.items():
        games = []
        latencies = []
        for board_index in range(args.boards):
            seed = args.seed + board_index
            for seat in range(4):
                game = Game([RandomPlayer(color) for color in colors], seed=seed,
                            vps_to_win=10, discard_limit=7, friendly_robber=False,
                            number_placement="official_spiral")
                game.playable_actions.sort(key=benchmark.canonical_action_key)
                color = game.state.colors[seat]
                player = factory(color, checkpoint, seed * 4 + seat)
                opponent_colors = [other for other in game.state.colors if other != color]
                rotation = (board_index + seat) % 3
                assignments = {other: composition[(index + rotation) % 3]
                               for index, other in enumerate(opponent_colors)}
                opponents = {other: baseline(name, other) for other, name in assignments.items()}
                started = time.perf_counter()
                actions = 0
                status = "capped"
                error = None
                try:
                    while actions < 15000 and game.state.num_turns < 500:
                        if game.winning_color() is not None:
                            status = "completed"
                            break
                        if time.perf_counter() - started > args.max_game_seconds:
                            break
                        actor = game.state.current_color()
                        bot = player if actor == color else opponents[actor]
                        before = time.perf_counter_ns()
                        action = bot.decide(game, game.playable_actions)
                        if actor == color:
                            latencies.append((time.perf_counter_ns() - before) / 1e6)
                        if action not in game.playable_actions:
                            raise ValueError(f"Illegal action from {type(bot).__name__}: {action}")
                        game.execute(action, validate_action=True)
                        actions += 1
                    if game.winning_color() is not None:
                        status = "completed"
                except Exception as exc:
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
                winner = game.winning_color()
                vp = game.state.player_state[f"P{game.state.color_to_index[color]}_ACTUAL_VICTORY_POINTS"]
                result = {"seed": seed, "seat": seat, "color": color.value,
                          "opponent": mix_name, "opponent_assignments": {
                              other.value: name for other, name in assignments.items()},
                          "status": status, "winner": winner.value if winner else None,
                          "won": status == "completed" and winner == color,
                          "candidate_vp": vp, "actions": actions,
                          "turns": game.state.num_turns,
                          "wall_seconds": round(time.perf_counter() - started, 3)}
                if error:
                    result["error"] = error
                games.append(result)
        report["tiers"][mix_name] = {"summary": benchmark.summarize(games, latencies), "games": games}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        summary = report["tiers"][mix_name]["summary"]
        print(f"{mix_name}: {summary['wins']}/{summary['attempts']} wins; "
              f"{summary['completed']} completed", flush=True)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
