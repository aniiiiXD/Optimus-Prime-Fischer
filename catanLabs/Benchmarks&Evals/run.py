"""Reproducible four-player Catan tournament for our future agent."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from catanLabs.Catanatron.engine import source_info  # noqa: E402
from catanLabs.Catanatron.tournament_protocol import (  # noqa: E402
    PROTOCOL, canonical_action_key, install_v5_protocol,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson95(wins: int, attempts: int) -> list[float]:
    if not attempts:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = wins / attempts
    denominator = 1 + z * z / attempts
    center = (p + z * z / (2 * attempts)) / denominator
    radius = z * ((p * (1 - p) / attempts + z * z / (4 * attempts**2)) ** 0.5) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(games: list[dict], latencies_ms: list[float]) -> dict:
    wins = sum(game["won"] for game in games)
    attempts = len(games)
    ordered = sorted(latencies_ms)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else None
    return {
        "attempts": attempts,
        "completed": sum(game["status"] == "completed" for game in games),
        "wins": wins,
        "win_rate_all_attempts": wins / attempts if attempts else None,
        "win_rate_wilson95": wilson95(wins, attempts),
        "mean_final_vp": statistics.fmean(game["candidate_vp"] for game in games if game["candidate_vp"] is not None) if games else None,
        "decision_latency_ms": {
            "samples": len(ordered),
            "median": statistics.median(ordered) if ordered else None,
            "p95": p95,
        },
        "by_seat": {
            str(seat): {
                "attempts": sum(game["seat"] == seat for game in games),
                "wins": sum(game["seat"] == seat and game["won"] for game in games),
            }
            for seat in range(4)
        },
    }


def load_agent_factory(path: Path):
    spec = importlib.util.spec_from_file_location("catan_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load agent file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, "create_agent", None)
    if not callable(factory):
        raise ValueError(f"{path} must define create_agent(color, checkpoint)")
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--agent", choices=("random", "weighted", "value", "alphabeta1"))
    selection.add_argument("--agent-file", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--opponents", default="random,weighted,value")
    parser.add_argument("--boards", type=int, default=1, help="Boards per opponent tier; each uses four seats")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--max-actions", type=int, default=15000)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--max-game-seconds", type=float, default=60)
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parent / "reports/latest.json")
    args = parser.parse_args()
    if args.boards < 1 or args.max_actions < 1 or args.max_turns < 1 or args.max_game_seconds <= 0:
        parser.error("boards and all game limits must be positive")
    if args.checkpoint and not args.agent_file:
        parser.error("--checkpoint requires --agent-file")
    opponent_names = [name.strip() for name in args.opponents.split(",")]
    if not opponent_names or any(name not in ("random", "weighted", "value", "alphabeta1") for name in opponent_names):
        parser.error("--opponents must be a comma-separated list of random,weighted,value,alphabeta1")
    if len(set(opponent_names)) != len(opponent_names):
        parser.error("--opponents contains a duplicate tier")

    engine = source_info()
    sys.path.insert(0, engine["python_root"])
    from catanatron import Color, Game, RandomPlayer
    from catanatron.players.weighted_random import WeightedRandomPlayer
    from catanatron.players.value import ValueFunctionPlayer
    from catanatron.players.minimax import AlphaBetaPlayer

    install_v5_protocol()

    classes = {"random": RandomPlayer, "weighted": WeightedRandomPlayer, "value": ValueFunctionPlayer}
    def make_baseline(name, color):
        if name == "alphabeta1":
            return AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=1))
        return classes[name](color)

    if args.agent_file:
        agent_path = args.agent_file.resolve(strict=True)
        factory = load_agent_factory(agent_path)
        agent_info = {"type": "external", "file": str(agent_path), "sha256": sha256(agent_path)}
    else:
        factory = lambda color, checkpoint, seed: make_baseline(args.agent, color)
        agent_info = {"type": args.agent}
    checkpoint = args.checkpoint.resolve(strict=True) if args.checkpoint else None
    if checkpoint:
        agent_info["checkpoint"] = str(checkpoint)
        agent_info["checkpoint_sha256"] = sha256(checkpoint)

    report = {
        "protocol": PROTOCOL,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": {"source": engine["source"], "commit": engine["commit"]},
        "rules": {"players": 4, "target_vp": 10, "discard_limit": 7,
                  "friendly_robber": False, "number_placement": "official_spiral",
                  "player_trade_offers": False, "isolated_copy_rng": True,
                  "forced_search_outcomes": True},
        "candidate": agent_info,
        "opponents": opponent_names,
        "boards_per_tier": args.boards,
        "first_seed": args.seed,
        "caps": {"max_actions": args.max_actions, "max_turns": args.max_turns,
                 "max_game_seconds": args.max_game_seconds},
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "runner_sha256": sha256(Path(__file__)),
        "tiers": {},
    }
    colors = list(Color)
    for opponent_name in opponent_names:
        results: list[dict] = []
        latencies_ms: list[float] = []
        for board_index in range(args.boards):
            seed = args.seed + board_index
            for seat in range(4):
                # The engine shuffles input colors. The candidate is assigned
                # after that shuffle so each board is played in all four seats.
                game = Game([RandomPlayer(color) for color in colors], seed=seed,
                            vps_to_win=10, discard_limit=7, friendly_robber=False,
                            number_placement="official_spiral")
                game.playable_actions.sort(key=canonical_action_key)
                candidate_color = game.state.colors[seat]
                candidate = factory(candidate_color, checkpoint, seed * 4 + seat)
                if not callable(getattr(candidate, "decide", None)):
                    raise TypeError("create_agent must return an object with decide(game, playable_actions)")
                opponents = {color: make_baseline(opponent_name, color) for color in colors}
                started = time.perf_counter()
                actions = 0
                status = "capped"
                error = None
                try:
                    while actions < args.max_actions and game.state.num_turns < args.max_turns:
                        winner = game.winning_color()
                        if winner is not None:
                            status = "completed"
                            break
                        if time.perf_counter() - started > args.max_game_seconds:
                            break
                        actor = game.state.current_color()
                        player = candidate if actor == candidate_color else opponents[actor]
                        decision_start = time.perf_counter_ns()
                        action = player.decide(game, game.playable_actions)
                        if actor == candidate_color:
                            latencies_ms.append((time.perf_counter_ns() - decision_start) / 1e6)
                        if action not in game.playable_actions:
                            raise ValueError(f"Illegal action from {type(player).__name__}: {action}")
                        game.execute(action, validate_action=True)
                        actions += 1
                    if game.winning_color() is not None:
                        status = "completed"
                except Exception as exc:
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
                winner = game.winning_color()
                vp = game.state.player_state[f"P{seat}_ACTUAL_VICTORY_POINTS"]
                result = {"seed": seed, "seat": seat, "color": candidate_color.value,
                          "opponent": opponent_name, "status": status,
                          "winner": winner.value if winner else None,
                          "won": status == "completed" and winner == candidate_color,
                          "candidate_vp": vp, "actions": actions,
                          "turns": game.state.num_turns,
                          "wall_seconds": round(time.perf_counter() - started, 3)}
                if error:
                    result["error"] = error
                results.append(result)
        report["tiers"][opponent_name] = {"summary": summarize(results, latencies_ms), "games": results}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        summary = report["tiers"][opponent_name]["summary"]
        print(f"{opponent_name}: {summary['wins']}/{summary['attempts']} wins; "
              f"{summary['completed']} completed; report {args.report}", flush=True)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
