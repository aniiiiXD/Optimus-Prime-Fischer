"""Paired Catan-family robustness screen for frozen FairValue-derived agents.

Each run evaluates one candidate against the same fixed mixed-strong opponent
table. Run once for ``tactical`` and once for ``fair`` with the same seed,
then pass both reports to ``compare.py``. This is a rule-transfer drill, not a
promotion benchmark for an unknown competition contract.
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
assert spec.loader is not None
spec.loader.exec_module(benchmark)

from catanLabs.Catanatron.tournament_protocol import PROTOCOL, install_v5_protocol


ROOT = Path(__file__).resolve().parent
COMPOSITION = ("value", "alphabeta1", "alphabeta1")
VARIANTS = {
    "target_vp_8": {"vps_to_win": 8, "discard_limit": 7,
                    "friendly_robber": False, "number_placement": "official_spiral"},
    "target_vp_12": {"vps_to_win": 12, "discard_limit": 7,
                     "friendly_robber": False, "number_placement": "official_spiral"},
    "discard_limit_5": {"vps_to_win": 10, "discard_limit": 5,
                        "friendly_robber": False, "number_placement": "official_spiral"},
    "friendly_robber": {"vps_to_win": 10, "discard_limit": 7,
                        "friendly_robber": True, "number_placement": "official_spiral"},
    "random_numbers": {"vps_to_win": 10, "discard_limit": 7,
                       "friendly_robber": False, "number_placement": "random"},
}


def opponent_assignments(colors, candidate_color, board_index, seat):
    others = [color for color in colors if color != candidate_color]
    rotation = (board_index + seat) % len(COMPOSITION)
    return {color: COMPOSITION[(index + rotation) % len(COMPOSITION)]
            for index, color in enumerate(others)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("tactical", "fair"), required=True)
    parser.add_argument("--fair-config", type=Path,
                        default=ROOT / "configs/fair_army_finish.json")
    parser.add_argument("--tactical-config", type=Path,
                        default=ROOT / "configs/tactical_vp_min0.json")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--boards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20274500)
    parser.add_argument("--max-game-seconds", type=float, default=20.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    requested = tuple(part.strip() for part in args.variants.split(",") if part.strip())
    if not requested or any(name not in VARIANTS for name in requested):
        parser.error(f"--variants must be drawn from {','.join(VARIANTS)}")
    if args.boards < 1 or args.max_game_seconds <= 0:
        parser.error("boards and max-game-seconds must be positive")

    fair_path = args.fair_config.resolve(strict=True)
    tactical_path = args.tactical_config.resolve(strict=True)
    fair_config = json.loads(fair_path.read_text())
    tactical_config = json.loads(tactical_path.read_text())
    tactical_base = {key: value for key, value in tactical_config.items()
                     if not key.startswith("tactical_")}
    if tactical_base != fair_config:
        raise ValueError("Tactical and FairValue settings must match")

    engine = benchmark.source_info()
    sys.path.insert(0, engine["python_root"])
    from catanatron import Color, Game, RandomPlayer
    from catanatron.players.minimax import AlphaBetaPlayer
    from catanatron.players.value import ValueFunctionPlayer
    from catanLabs.Competitive.fair_value_agent import FairValueAgent
    from catanLabs.Competitive.tactical_vp_agent import TacticalVPAgent

    install_v5_protocol()
    candidate_cls = TacticalVPAgent if args.candidate == "tactical" else FairValueAgent
    candidate_config = tactical_config if args.candidate == "tactical" else fair_config

    def opponent(name, color):
        if name == "value":
            return ValueFunctionPlayer(color)
        return AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=1))

    report = {
        "protocol": f"{PROTOCOL}-family-mixed-strong",
        "engine": engine,
        "candidate": {"type": args.candidate,
                      "file": str(ROOT / ("tactical_vp_agent.py" if args.candidate == "tactical" else "fair_value_agent.py")),
                      "checkpoint": str(tactical_path if args.candidate == "tactical" else fair_path),
                      "checkpoint_sha256": benchmark.sha256(tactical_path if args.candidate == "tactical" else fair_path)},
        "boards_per_variant": args.boards,
        "first_seed": args.seed,
        "opponent_composition": COMPOSITION,
        "caps": {"max_actions": 15000, "max_turns": 500,
                 "max_game_seconds": args.max_game_seconds},
        "platform": platform.platform(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "runner_sha256": benchmark.sha256(Path(__file__)),
        "tiers": {},
    }
    colors = list(Color)
    for variant_index, name in enumerate(requested):
        rules = VARIANTS[name]
        games, latencies = [], []
        for board_index in range(args.boards):
            seed = args.seed + variant_index * 10000 + board_index
            for seat in range(4):
                game = Game([RandomPlayer(color) for color in colors], seed=seed, **rules)
                game.playable_actions.sort(key=benchmark.canonical_action_key)
                color = game.state.colors[seat]
                candidate = candidate_cls(color, candidate_config)
                assignments = opponent_assignments(game.state.colors, color, board_index, seat)
                opponents = {other: opponent(kind, other) for other, kind in assignments.items()}
                started, actions, status, error = time.perf_counter(), 0, "capped", None
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
                    status, error = "error", f"{type(exc).__name__}: {exc}"
                winner = game.winning_color()
                vp_key = f"P{game.state.color_to_index[color]}_ACTUAL_VICTORY_POINTS"
                result = {"seed": seed, "seat": seat, "color": color.value,
                          "opponent_assignments": {other.value: kind for other, kind in assignments.items()},
                          "status": status, "winner": winner.value if winner else None,
                          "won": status == "completed" and winner == color,
                          "candidate_vp": game.state.player_state[vp_key],
                          "actions": actions, "turns": game.state.num_turns,
                          "wall_seconds": round(time.perf_counter() - started, 3)}
                if error:
                    result["error"] = error
                games.append(result)
        report["tiers"][name] = {"rules": {"players": 4, "player_trade_offers": False,
                                             "isolated_copy_rng": True, "forced_search_outcomes": True,
                                             **rules},
                                 "summary": benchmark.summarize(games, latencies), "games": games}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        summary = report["tiers"][name]["summary"]
        print(f"{name}: {summary['wins']}/{summary['attempts']} wins; {summary['completed']} completed", flush=True)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})
    main()
