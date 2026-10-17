"""Matched v5 screen of three fair rescues and frozen model/fair controls."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import torch


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))
RUNNER = WORKSPACE / "catanLabs/Benchmarks&Evals/run.py"
spec = importlib.util.spec_from_file_location("catanlabs_benchmark", RUNNER)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

from catanLabs.Catanatron.tournament_protocol import PROTOCOL, install_v5_protocol
from catanLabs.Competitive.compare import compare
from catanLabs.Competitive.selective_fair_rescue_agent import (
    INTERVENTIONS, SelectiveFairRescueAgent, load_config,
)


ROOT = Path(__file__).resolve().parent
CONFIGS = {arm: ROOT / "configs" / f"selective_rescue_{arm}.json"
           for arm in INTERVENTIONS}
ARMS = ("model", "fair", *INTERVENTIONS)
COMPOSITIONS = {
    "mixed_diverse": ("weighted", "value", "alphabeta1"),
    "mixed_strong": ("value", "alphabeta1", "alphabeta1"),
}
PROTOCOL_NAME = f"{PROTOCOL}-selective-rescue"
SPLIT = WORKSPACE / "catanLabs/Transformer/runs/gate4_merged_v3/dataset.split.json"


def validate_assets() -> dict:
    configs = {arm: load_config(path) for arm, path in CONFIGS.items()}
    model_paths = {config["model_checkpoint_path"] for config in configs.values()}
    fair_paths = {config["fair_config_path"] for config in configs.values()}
    model_hashes = {config["model_checkpoint_sha256"] for config in configs.values()}
    fair_hashes = {config["fair_config_sha256"] for config in configs.values()}
    if len(model_paths) != 1 or len(fair_paths) != 1 or len(model_hashes) != 1 or len(fair_hashes) != 1:
        raise ValueError("All rescue arms must use the same frozen model and fair configuration")
    return {"configs": configs, "model_path": model_paths.pop(), "fair_path": fair_paths.pop(),
            "model_sha256": model_hashes.pop(), "fair_sha256": fair_hashes.pop()}


def assert_fresh_boards(first_seed: int, boards: int) -> str:
    split = json.loads(SPLIT.read_text())
    used = set().union(*(set(split["board_seeds"][name])
                         for name in ("train", "validation", "final")))
    proposed = set(range(first_seed, first_seed + boards))
    overlap = proposed & used
    if overlap:
        raise ValueError(f"Proposed boards overlap frozen dataset split: {sorted(overlap)}")
    return benchmark.sha256(SPLIT)


def write_json_atomic(path: Path, payload: dict) -> None:
    """Leave the previous complete JSON intact if a write is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def persist_partial(output_dir: Path, report: dict, arm: str, mix_name: str,
                    games: list[dict], latencies: list[float],
                    aggregate: Counter, expected_games: int) -> None:
    """Record one arm/mix after each game; these files are never final reports."""
    tier = {"summary": benchmark.summarize(games, latencies), "games": games}
    if arm in INTERVENTIONS:
        tier["rescue_summary"] = dict(sorted(aggregate.items()))
    partial = {
        "schema": "catanlabs-selective-rescue-partial-v1",
        "status": "unverified_partial",
        "arm": arm, "mix": mix_name,
        "attempts_recorded": len(games), "expected_games": expected_games,
        "protocol": report["protocol"], "rules": report["rules"],
        "candidate": report["candidate"],
        "first_seed": report["first_seed"],
        "boards_per_mix": report["boards_per_mix"],
        "split_file_sha256": report["split_file_sha256"],
        "runner_sha256": report["runner_sha256"],
        "tier": tier,
    }
    write_json_atomic(output_dir / "partial" / f"{mix_name}__{arm}.json", partial)


def validate_matched_grid(reports: dict, boards: int, first_seed: int) -> None:
    """Require every arm to have the same complete board/seat/assignment grid."""
    expected = {(first_seed + board, seat) for board in range(boards) for seat in range(4)}
    for mix_name in COMPOSITIONS:
        reference = None
        for arm in ARMS:
            tier = reports[arm]["tiers"].get(mix_name)
            if tier is None:
                raise RuntimeError(f"Missing {mix_name} tier for {arm}")
            rows = tier["games"]
            keys = [(row["seed"], row["seat"]) for row in rows]
            if len(rows) != len(expected) or set(keys) != expected or len(set(keys)) != len(keys):
                raise RuntimeError(f"Incomplete board/seat grid for {mix_name} {arm}")
            current = [(row["seed"], row["seat"], row["color"], row["opponent_assignments"])
                       for row in rows]
            if reference is None:
                reference = current
            elif current != reference:
                raise RuntimeError(f"Candidate/control assignments differ for {mix_name} {arm}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20273500)
    parser.add_argument("--max-game-seconds", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.boards < 1 or args.max_game_seconds <= 0:
        parser.error("boards and max-game-seconds must be positive")

    assets = validate_assets()
    split_hash = assert_fresh_boards(args.seed, args.boards)
    engine = benchmark.source_info()
    sys.path.insert(0, engine["python_root"])
    from catanatron import Color, Game, RandomPlayer
    from catanatron.players.weighted_random import WeightedRandomPlayer
    from catanatron.players.value import ValueFunctionPlayer
    from catanatron.players.minimax import AlphaBetaPlayer
    from catanLabs.Competitive.fair_value_agent import FairValueAgent
    from catanLabs.Transformer.agent import TransformerAgent
    from catanLabs.Transformer.visible_observation import VisibleAgentAdapter

    install_v5_protocol()
    torch.set_num_threads(1)
    fair_config = json.loads(assets["fair_path"].read_text())
    colors = list(Color)

    def baseline(name, color):
        if name == "alphabeta1":
            return AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=1))
        return {"weighted": WeightedRandomPlayer,
                "value": ValueFunctionPlayer}[name](color)

    def candidate(arm, color):
        if arm == "model":
            return VisibleAgentAdapter(TransformerAgent(color, assets["model_path"]))
        if arm == "fair":
            return FairValueAgent(color, fair_config)
        return SelectiveFairRescueAgent(color, CONFIGS[arm])

    rules = {"players": 4, "target_vp": 10, "discard_limit": 7,
             "friendly_robber": False, "number_placement": "official_spiral",
             "player_trade_offers": False, "isolated_copy_rng": True,
             "forced_search_outcomes": True,
             "frozen_model_sha256": assets["model_sha256"],
             "fair_config_sha256": assets["fair_sha256"]}
    reports = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        source = (ROOT / "selective_fair_rescue_agent.py" if arm in INTERVENTIONS
                  else WORKSPACE / ("catanLabs/Transformer/agent.py" if arm == "model"
                                    else "catanLabs/Competitive/fair_value_agent.py"))
        reports[arm] = {
            "protocol": PROTOCOL_NAME, "engine": engine, "rules": rules,
            "candidate": {"type": arm, "file": str(source),
                          "sha256": benchmark.sha256(source),
                          "checkpoint": str(CONFIGS[arm] if arm in INTERVENTIONS
                                            else assets["model_path"] if arm == "model"
                                            else assets["fair_path"]),
                          "checkpoint_sha256": benchmark.sha256(CONFIGS[arm] if arm in INTERVENTIONS
                                                                 else assets["model_path"] if arm == "model"
                                                                 else assets["fair_path"])},
            "model_checkpoint": str(assets["model_path"]),
            "fair_config": str(assets["fair_path"]),
            "split_file": str(SPLIT), "split_file_sha256": split_hash,
            "boards_per_mix": args.boards, "first_seed": args.seed,
            "compositions": COMPOSITIONS,
            "caps": {"max_actions": 15000, "max_turns": 500,
                     "max_game_seconds": args.max_game_seconds},
            "platform": platform.platform(),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "runner_sha256": benchmark.sha256(Path(__file__)),
            "tiers": {},
        }

    for mix_name, composition in COMPOSITIONS.items():
        for arm in ARMS:
            games = []
            latencies = []
            aggregate = Counter()
            for board_index in range(args.boards):
                seed = args.seed + board_index
                for seat in range(4):
                    game = Game([RandomPlayer(color) for color in colors], seed=seed,
                                vps_to_win=10, discard_limit=7,
                                friendly_robber=False, number_placement="official_spiral")
                    game.playable_actions.sort(key=benchmark.canonical_action_key)
                    color = game.state.colors[seat]
                    player = candidate(arm, color)
                    other_colors = [other for other in game.state.colors if other != color]
                    rotation = (board_index + seat) % 3
                    assignments = {other: composition[(index + rotation) % 3]
                                   for index, other in enumerate(other_colors)}
                    opponents = {other: baseline(name, other)
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
                    vp = game.state.player_state[
                        f"P{game.state.color_to_index[color]}_ACTUAL_VICTORY_POINTS"
                    ]
                    result = {"seed": seed, "seat": seat, "color": color.value,
                              "opponent": mix_name,
                              "opponent_assignments": {other.value: name for other, name in assignments.items()},
                              "status": status, "winner": winner.value if winner else None,
                              "won": status == "completed" and winner == color,
                              "candidate_vp": vp, "actions": actions,
                              "turns": game.state.num_turns,
                              "wall_seconds": round(time.perf_counter() - started, 3)}
                    if error:
                        result["error"] = error
                    if arm in INTERVENTIONS:
                        result["rescue"] = player.summary()
                        for key in ("decisions", "forced", "consulted", "disagreed", "overrides"):
                            aggregate[key] += result["rescue"][key]
                        for key, count in result["rescue"]["trade_directions"].items():
                            aggregate[key] += count
                    games.append(result)
                    persist_partial(args.output_dir, reports[arm], arm, mix_name, games,
                                    latencies, aggregate, args.boards * 4)
            tier = {"summary": benchmark.summarize(games, latencies), "games": games}
            if arm in INTERVENTIONS:
                tier["rescue_summary"] = dict(sorted(aggregate.items()))
            reports[arm]["tiers"][mix_name] = tier
            print(f"{mix_name} {arm}: {tier['summary']['wins']}/{tier['summary']['attempts']} wins; "
                  f"{tier['summary']['completed']} completed; overrides {aggregate['overrides']}", flush=True)

    validate_matched_grid(reports, args.boards, args.seed)
    paired = {arm: compare(reports[arm], reports["model"]) for arm in INTERVENTIONS}
    for arm in ARMS:
        write_json_atomic(args.output_dir / f"{arm}.json", reports[arm])
    for arm in INTERVENTIONS:
        write_json_atomic(args.output_dir / f"{arm}_vs_model.json", paired[arm])
    print(f"Saved paired reports to {args.output_dir}", flush=True)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
