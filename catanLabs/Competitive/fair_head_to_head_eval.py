"""Paired tactical-vs-fair screen with a FairValue bot in every game.

Run once with ``--candidate tactical`` and again with ``--candidate fair``.
The seed, seat and opponent-color assignments then match exactly, and the
JSON reports can be passed to ``compare.py``.
"""

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


PROTOCOL_NAME = f"{PROTOCOL}-fair-head-to-head"
TIER = "fair_value_alphabeta1"
COMPOSITION = ("fair", "value", "alphabeta1")
ROOT = Path(__file__).resolve().parent


def opponent_assignments(colors, candidate_color, board_index: int, seat: int):
    """Use mixed_eval's rotation so each opponent occupies every other seat."""
    others = [color for color in colors if color != candidate_color]
    rotation = (board_index + seat) % len(COMPOSITION)
    return {color: COMPOSITION[(index + rotation) % len(COMPOSITION)]
            for index, color in enumerate(others)}


def configs(fair_path: Path, tactical_path: Path):
    fair = json.loads(fair_path.read_text())
    tactical = json.loads(tactical_path.read_text())
    if not isinstance(fair, dict) or not isinstance(tactical, dict):
        raise ValueError("Agent configurations must be JSON objects")
    tactical_base = {key: value for key, value in tactical.items()
                     if not key.startswith("tactical_")}
    if tactical_base != fair:
        raise ValueError("Tactical and fair candidates must share identical FairValue settings")
    return fair, tactical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("tactical", "fair"), required=True)
    parser.add_argument("--fair-config", type=Path, default=ROOT / "configs/fair_army_finish.json")
    parser.add_argument("--tactical-config", type=Path, default=ROOT / "configs/tactical_vp_research.json")
    parser.add_argument("--boards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20271960)
    parser.add_argument("--max-game-seconds", type=float, default=90.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.boards < 1 or args.max_game_seconds <= 0:
        parser.error("boards and max-game-seconds must be positive")

    fair_path = args.fair_config.resolve(strict=True)
    tactical_path = args.tactical_config.resolve(strict=True)
    fair_config, tactical_config = configs(fair_path, tactical_path)
    engine = benchmark.source_info()
    sys.path.insert(0, engine["python_root"])
    from catanatron import Color, Game, RandomPlayer
    from catanatron.players.value import ValueFunctionPlayer
    from catanatron.players.minimax import AlphaBetaPlayer
    from catanLabs.Competitive.fair_value_agent import FairValueAgent
    from catanLabs.Competitive.tactical_vp_agent import TacticalVPAgent

    install_v5_protocol()
    candidate_cls = TacticalVPAgent if args.candidate == "tactical" else FairValueAgent
    candidate_config = tactical_config if args.candidate == "tactical" else fair_config

    def opponent(name, color):
        if name == "fair":
            return FairValueAgent(color, fair_config)
        if name == "value":
            return ValueFunctionPlayer(color)
        if name == "alphabeta1":
            return AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=1))
        raise ValueError(f"Unknown opponent: {name}")

    report = {
        "protocol": PROTOCOL_NAME,
        "engine": engine,
        "rules": {"players": 4, "target_vp": 10, "discard_limit": 7,
                  "friendly_robber": False, "number_placement": "official_spiral",
                  "player_trade_offers": False, "isolated_copy_rng": True,
                  "forced_search_outcomes": True,
                  "opponent_composition": COMPOSITION,
                  "fair_config_sha256": benchmark.sha256(fair_path)},
        "candidate": {"type": args.candidate,
                      "file": str(ROOT / ("tactical_vp_agent.py" if args.candidate == "tactical"
                                          else "fair_value_agent.py")),
                      "sha256": benchmark.sha256(ROOT / ("tactical_vp_agent.py" if args.candidate == "tactical"
                                                               else "fair_value_agent.py")),
                      "checkpoint": str(tactical_path if args.candidate == "tactical" else fair_path),
                      "checkpoint_sha256": benchmark.sha256(tactical_path if args.candidate == "tactical" else fair_path)},
        "opponents": {"composition": COMPOSITION, "fair_config": str(fair_path),
                      "fair_config_sha256": benchmark.sha256(fair_path),
                      "fair_source_sha256": benchmark.sha256(ROOT / "fair_value_agent.py")},
        "boards_per_mix": args.boards,
        "first_seed": args.seed,
        "caps": {"max_actions": 15000, "max_turns": 500,
                 "max_game_seconds": args.max_game_seconds},
        "platform": platform.platform(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "runner_sha256": benchmark.sha256(Path(__file__)),
        "tiers": {},
    }
    colors = list(Color)
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
            candidate = candidate_cls(color, candidate_config)
            assignments = opponent_assignments(game.state.colors, color, board_index, seat)
            opponents = {other: opponent(name, other)
                         for other, name in assignments.items()}
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
                    bot = candidate if actor == color else opponents[actor]
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
            vp = game.state.player_state[
                f"P{game.state.color_to_index[color]}_ACTUAL_VICTORY_POINTS"
            ]
            result = {"seed": seed, "seat": seat, "color": color.value,
                      "opponent": TIER,
                      "opponent_assignments": {other.value: name for other, name in assignments.items()},
                      "status": status, "winner": winner.value if winner else None,
                      "won": status == "completed" and winner == color,
                      "candidate_vp": vp, "actions": actions,
                      "turns": game.state.num_turns,
                      "wall_seconds": round(time.perf_counter() - started, 3)}
            if error:
                result["error"] = error
            games.append(result)
    report["tiers"][TIER] = {"summary": benchmark.summarize(games, latencies),
                             "games": games}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = report["tiers"][TIER]["summary"]
    print(f"{TIER}: {summary['wins']}/{summary['attempts']} wins; "
          f"{summary['completed']} completed; report {args.report}", flush=True)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
