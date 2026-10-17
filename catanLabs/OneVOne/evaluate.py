"""Paired-seed, seat-swapped 1v1 evaluation for the local Catanatron agent."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/trained.json"
if not DEFAULT_CONFIG.is_file():
    DEFAULT_CONFIG = ROOT / "configs/default.json"
sys.path.insert(0, str(WORKSPACE))

from catanLabs.Catanatron.engine import source_info  # noqa: E402
from catanLabs.Catanatron.tournament_protocol import (  # noqa: E402
    PROTOCOL,
    canonical_action_key,
    install_v5_protocol,
)


OPPONENTS = (
    "random", "weighted", "value", "alphabeta1", "alphabeta2",
    "catanatron", "fair", "tactical", "onevone",
)
FAIR_CONFIG_PATH = WORKSPACE / "catanLabs/Competitive/configs/fair_army_finish.json"
TACTICAL_CONFIG_PATH = WORKSPACE / "catanLabs/Competitive/configs/tactical_vp_min0.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson95(wins: int, attempts: int) -> list[float] | None:
    if attempts == 0:
        return None
    z = 1.959963984540054
    p = wins / attempts
    denominator = 1 + z * z / attempts
    center = (p + z * z / (2 * attempts)) / denominator
    radius = z * math.sqrt(
        p * (1 - p) / attempts + z * z / (4 * attempts**2)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def summarize(games: list[dict], latencies: list[float]) -> dict:
    wins = sum(game["won"] for game in games)
    attempts = len(games)
    vp_margins = [game["candidate_vp"] - game["opponent_vp"] for game in games]
    return {
        "attempts": attempts,
        "completed": sum(game["status"] == "completed" for game in games),
        "capped": sum(game["status"] == "capped" for game in games),
        "errors": sum(game["status"] == "error" for game in games),
        "wins": wins,
        "win_rate_all_attempts": wins / attempts if attempts else None,
        "win_rate_wilson95": wilson95(wins, attempts),
        "mean_final_vp": statistics.fmean(game["candidate_vp"] for game in games)
        if games
        else None,
        "mean_vp_margin": statistics.fmean(vp_margins) if vp_margins else None,
        "candidate_decision_latency_ms": {
            "samples": len(latencies),
            "median": statistics.median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "by_seat": {
            str(seat): {
                "attempts": sum(game["seat"] == seat for game in games),
                "wins": sum(game["seat"] == seat and game["won"] for game in games),
                "win_rate": (
                    sum(game["seat"] == seat and game["won"] for game in games)
                    / sum(game["seat"] == seat for game in games)
                    if any(game["seat"] == seat for game in games)
                    else None
                ),
            }
            for seat in range(2)
        },
    }


def load_factory(path: Path):
    spec = importlib.util.spec_from_file_location("catan_onevone_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import agent file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, "create_agent", None)
    if not callable(factory):
        raise ValueError(f"{path} must define create_agent(color, checkpoint, seed)")
    return factory


@lru_cache(maxsize=2)
def _baseline_config(name):
    path = FAIR_CONFIG_PATH if name == "fair" else TACTICAL_CONFIG_PATH
    return json.loads(path.read_text())


def make_opponent(name, color, _candidate_config=None):
    from catanLabs.Competitive.fair_value_agent import FairValueAgent
    from catanLabs.Competitive.tactical_vp_agent import TacticalVPAgent
    from catanLabs.OneVOne.agent import OneVOneAgent
    from catanatron import RandomPlayer
    from catanatron.players.minimax import AlphaBetaPlayer
    from catanatron.players.value import ValueFunctionPlayer
    from catanatron.players.weighted_random import WeightedRandomPlayer
    from catanatron.web.players import Catanatron as CatanatronBot

    if name == "random":
        return RandomPlayer(color)
    if name == "weighted":
        return WeightedRandomPlayer(color)
    if name == "value":
        return ValueFunctionPlayer(color)
    if name == "alphabeta1":
        return AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=1))
    if name == "alphabeta2":
        return AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=2))
    if name == "catanatron":
        return CatanatronBot(color, CatanatronBot.Params())
    if name == "fair":
        return FairValueAgent(color, _baseline_config("fair"))
    if name == "tactical":
        return TacticalVPAgent(color, _baseline_config("tactical"))
    if name == "onevone":
        return OneVOneAgent(color, _candidate_config or {})
    raise ValueError(f"Unsupported opponent: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-file", type=Path, default=ROOT / "agent.py")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--opponents", default="value,alphabeta1,tactical")
    parser.add_argument("--boards", type=int, default=1,
                        help="Board seeds per opponent; every seed is played from both seats")
    parser.add_argument("--seed", type=int, default=20271000)
    parser.add_argument("--max-actions", type=int, default=15000)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--max-game-seconds", type=float, default=60.0)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/latest.json")
    args = parser.parse_args()

    requested = [name.strip() for name in args.opponents.split(",") if name.strip()]
    if not requested or any(name not in OPPONENTS for name in requested):
        parser.error(f"--opponents must be a comma-separated subset of {','.join(OPPONENTS)}")
    if len(set(requested)) != len(requested):
        parser.error("--opponents contains a duplicate")
    if args.boards < 1 or min(args.max_actions, args.max_turns) < 1:
        parser.error("boards, max-actions, and max-turns must be positive")
    if args.max_game_seconds <= 0:
        parser.error("max-game-seconds must be positive")

    agent_path = args.agent_file.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text())
    rules = config.get("rules", {})
    if rules.get("players") != 2:
        parser.error("This evaluator requires rules.players=2")
    if rules.get("trade_offers") is not True:
        # Catanatron controls domestic trades through legal actions, not this flag.
        # The flag records the expected contract; the runner never disables engine rules.
        parser.error("Set rules.trade_offers=true to record the local engine's trade support")

    engine = source_info()
    sys.path.insert(0, engine["python_root"])
    from catanatron import Color, Game, RandomPlayer

    install_v5_protocol()
    factory = load_factory(agent_path)
    colors = [Color.RED, Color.BLUE]
    game_rules = {
        "vps_to_win": int(rules["target_vp"]),
        "discard_limit": int(rules["discard_limit"]),
        "friendly_robber": bool(rules["friendly_robber"]),
        "number_placement": str(rules["number_placement"]),
    }
    report = {
        "protocol": f"{PROTOCOL}-paired-1v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": engine,
        "rules": {**rules, **game_rules},
        "candidate": {
            "file": str(agent_path),
            "sha256": sha256(agent_path),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
        },
        "opponents": requested,
        "opponent_configs": {
            name: {
                "file": str(path),
                "sha256": sha256(path),
            }
            for name, path in (("fair", FAIR_CONFIG_PATH), ("tactical", TACTICAL_CONFIG_PATH))
            if name in requested
        },
        "boards_per_opponent": args.boards,
        "first_seed": args.seed,
        "seat_protocol": "same seed and rules, candidate in each of the two engine seats",
        "caps": {
            "max_actions": args.max_actions,
            "max_turns": args.max_turns,
            "max_game_seconds": args.max_game_seconds,
        },
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tiers": {},
    }

    for opponent_name in requested:
        games = []
        latencies_ms = []
        for board_index in range(args.boards):
            seed = args.seed + board_index
            for candidate_seat in range(2):
                game = Game(
                    [RandomPlayer(color) for color in colors],
                    seed=seed,
                    **game_rules,
                )
                game.playable_actions.sort(key=canonical_action_key)
                candidate_color = game.state.colors[candidate_seat]
                opponent_color = next(
                    color for color in game.state.colors if color != candidate_color
                )
                candidate = factory(candidate_color, str(config_path), seed)
                opponent = make_opponent(opponent_name, opponent_color, config)
                started = time.perf_counter()
                actions = 0
                status = "capped"
                error = None
                try:
                    while actions < args.max_actions and game.state.num_turns < args.max_turns:
                        if game.winning_color() is not None:
                            status = "completed"
                            break
                        if time.perf_counter() - started > args.max_game_seconds:
                            break
                        actor = game.state.current_color()
                        bot = candidate if actor == candidate_color else opponent
                        before = time.perf_counter_ns()
                        action = bot.decide(game, game.playable_actions)
                        elapsed_ms = (time.perf_counter_ns() - before) / 1e6
                        if actor == candidate_color:
                            latencies_ms.append(elapsed_ms)
                        if action not in game.playable_actions:
                            raise ValueError(f"Illegal action from {type(bot).__name__}: {action}")
                        game.execute(action, validate_action=True)
                        game.playable_actions.sort(key=canonical_action_key)
                        actions += 1
                    if game.winning_color() is not None:
                        status = "completed"
                except Exception as exc:  # Preserve failures in the report and continue the batch.
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"

                winner = game.winning_color()
                candidate_prefix = f"P{game.state.color_to_index[candidate_color]}_"
                opponent_prefix = f"P{game.state.color_to_index[opponent_color]}_"
                record = {
                    "seed": seed,
                    "seat": candidate_seat,
                    "candidate_color": candidate_color.value,
                    "opponent": opponent_name,
                    "status": status,
                    "winner": winner.value if winner is not None else None,
                    "won": status == "completed" and winner == candidate_color,
                    "candidate_vp": int(game.state.player_state[
                        candidate_prefix + "ACTUAL_VICTORY_POINTS"
                    ]),
                    "opponent_vp": int(game.state.player_state[
                        opponent_prefix + "ACTUAL_VICTORY_POINTS"
                    ]),
                    "actions": actions,
                    "turns": game.state.num_turns,
                    "wall_seconds": round(time.perf_counter() - started, 3),
                }
                if error is not None:
                    record["error"] = error
                games.append(record)

        report["tiers"][opponent_name] = {
            "summary": summarize(games, latencies_ms),
            "games": games,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        summary = report["tiers"][opponent_name]["summary"]
        print(
            f"{opponent_name}: {summary['wins']}/{summary['attempts']} wins, "
            f"{summary['completed']} completed; report {args.report}",
            flush=True,
        )


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(
            sys.executable,
            [sys.executable, *sys.argv],
            {**os.environ, "PYTHONHASHSEED": "0"},
        )
    main()
