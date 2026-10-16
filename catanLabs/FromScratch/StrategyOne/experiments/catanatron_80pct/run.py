"""Paired-seat head-to-head on 80% of the 1,028 recorded board seeds.

This is a new tournament on those seeds, not a replay of saved decisions.
Writes a resumable JSON report after each completed board pair.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from catanLabs.Catanatron.engine import source_info  # noqa: E402
from catanLabs.Catanatron.tournament_protocol import (  # noqa: E402
    canonical_action_key,
    install_v5_protocol,
)

ENGINE = source_info()
sys.path.insert(0, ENGINE["python_root"])

from catanatron import Color, Game, RandomPlayer  # noqa: E402
from catanatron.web.players import Catanatron  # noqa: E402
from catanLabs.OneVOne.agent import create_agent as create_onevone  # noqa: E402


META = ROOT / "catanLabs/FromScratch/StrategyOne/data/catanatron_diverse.meta.json"
CONFIG = ROOT / "catanLabs/OneVOne/configs/default.json"
EXPERIMENT_ROOT = ROOT.parent / "Hyde-catan-experiments"
EXPERIMENT_STRATEGY = EXPERIMENT_ROOT / "catanLabs/FromScratch/StrategyOne"
MLP_CHECKPOINT = EXPERIMENT_STRATEGY / "runs/calibrated_ranker_v1/model.pt"
SETTLEMENT_HEAD = EXPERIMENT_STRATEGY / "runs/settlement_target_v1/epoch_4.pt"
DEFAULT_REPORT = Path(__file__).resolve().parent / "mlp_vs_web_80pct.json"


def select_boards(meta_path: Path, fraction: float, selection_seed: int) -> list[dict]:
    games = json.loads(meta_path.read_text())["games"]
    count = round(len(games) * fraction)
    chosen = random.Random(selection_seed).sample(games, count)
    return sorted(({"game_id": game["game_id"], "seed": game["seed"]} for game in chosen),
                  key=lambda row: row["game_id"])


def load_candidate(kind: str):
    if kind == "onevone":
        return lambda color, seed: create_onevone(color, str(CONFIG), seed)

    import torch

    sys.path.insert(0, str(EXPERIMENT_STRATEGY))
    evaluation = importlib.import_module("evaluate_win_rate")
    checkpoint = torch.load(MLP_CHECKPOINT, map_location="cpu", weights_only=True)
    model = evaluation.LegalRanker(checkpoint["state_size"], checkpoint["action_size"])
    model.load_state_dict(checkpoint["base_model"])
    model.eval()
    calibration = evaluation.ActionCalibrationAdapter(checkpoint["state_size"])
    calibration.load_state_dict(checkpoint["adapter"])
    calibration.eval()
    target_checkpoint = torch.load(SETTLEMENT_HEAD, map_location="cpu", weights_only=True)
    settlement = evaluation.SettlementTargetHead(target_checkpoint["input_size"])
    settlement.load_state_dict(target_checkpoint["head"])
    settlement.eval()
    with (EXPERIMENT_STRATEGY / "data/catanatron_diverse.jsonl").open() as handle:
        first = json.loads(next(handle))
    edge_index = evaluation.stable_edge_index(first["observation"])
    device = torch.device("cpu")

    def create(color, seed):
        del seed
        return evaluation.MLPPlayer(
            color, model, edge_index, device, calibration, settlement,
            same_turn_search=True, same_turn_depth=4,
        )

    return create


def run_game(seed: int, seat: int, *, candidate_factory, max_seconds: float,
             max_actions: int, max_turns: int, trace: bool) -> dict:
    random.seed(seed * 2 + seat)
    game = Game(
        [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
        seed=seed,
        vps_to_win=10,
        discard_limit=7,
        friendly_robber=False,
        number_placement="official_spiral",
    )
    game.playable_actions.sort(key=canonical_action_key)
    our_color = game.state.colors[seat]
    their_color = next(color for color in game.state.colors if color != our_color)
    ours = candidate_factory(our_color, seed)
    theirs = Catanatron(their_color, Catanatron.Params())
    policies = {our_color: ours, their_color: theirs}
    counts = {"ours": Counter(), "catanatron": Counter()}
    trace_rows = []
    latencies = {"ours": [], "catanatron": []}
    started = time.perf_counter()
    actions = 0
    status = "capped"
    error = None

    try:
        while actions < max_actions and game.state.num_turns < max_turns:
            if game.winning_color() is not None:
                status = "completed"
                break
            if time.perf_counter() - started > max_seconds:
                break
            actor = game.state.current_color()
            label = "ours" if actor == our_color else "catanatron"
            before = time.perf_counter_ns()
            action = policies[actor].decide(game, game.playable_actions)
            latencies[label].append((time.perf_counter_ns() - before) / 1e6)
            if action not in game.playable_actions:
                raise ValueError(f"Illegal action from {label}: {action!r}")
            counts[label][action.action_type.name] += 1
            if trace:
                trace_rows.append({
                    "ply": actions,
                    "turn": game.state.num_turns,
                    "actor": label,
                    "action": action.action_type.name,
                    "value": repr(action.value),
                })
            game.execute(action, validate_action=True)
            game.playable_actions.sort(key=canonical_action_key)
            actions += 1
        if game.winning_color() is not None:
            status = "completed"
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    winner = game.winning_color()
    ours_key = f"P{game.state.color_to_index[our_color]}_ACTUAL_VICTORY_POINTS"
    theirs_key = f"P{game.state.color_to_index[their_color]}_ACTUAL_VICTORY_POINTS"
    result = {
        "seed": seed,
        "seat": seat,
        "status": status,
        "ours_won": status == "completed" and winner == our_color,
        "catanatron_won": status == "completed" and winner == their_color,
        "ours_vp": int(game.state.player_state[ours_key]),
        "catanatron_vp": int(game.state.player_state[theirs_key]),
        "actions": actions,
        "turns": game.state.num_turns,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "move_counts": {label: dict(counter) for label, counter in counts.items()},
        "decision_ms": {
            label: {
                "count": len(values),
                "median": statistics.median(values) if values else None,
                "p95": sorted(values)[int(.95 * (len(values) - 1))] if values else None,
            }
            for label, values in latencies.items()
        },
    }
    if error:
        result["error"] = error
    if trace:
        result["trace"] = trace_rows
    return result


def write_report(report: dict, path: Path) -> None:
    games = report["games"]
    report["summary"] = {
        "games": len(games),
        "completed": sum(row["status"] == "completed" for row in games),
        "capped": sum(row["status"] == "capped" for row in games),
        "errors": sum(row["status"] == "error" for row in games),
        "ours_wins": sum(row["ours_won"] for row in games),
        "catanatron_wins": sum(row["catanatron_won"] for row in games),
        "move_counts": {
            label: dict(sum((Counter(row["move_counts"][label]) for row in games), Counter()))
            for label in ("ours", "catanatron")
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("strategyone", "onevone"), default="strategyone")
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--selection-seed", type=int, default=20268116)
    parser.add_argument("--max-boards", type=int, help="Pilot cap; omit for full 80%% sample")
    parser.add_argument("--board-start", type=int, default=0,
                        help="First index in the deterministically selected board list")
    parser.add_argument("--board-stop", type=int,
                        help="Exclusive last index in the selected board list")
    parser.add_argument("--max-game-seconds", type=float, default=60.0)
    parser.add_argument("--max-actions", type=int, default=10000)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("fraction must be in (0, 1]")
    install_v5_protocol()
    candidate_factory = load_candidate(args.candidate)
    config_path = CONFIG if args.candidate == "onevone" else MLP_CHECKPOINT
    boards = select_boards(META, args.fraction, args.selection_seed)
    if args.max_boards is not None:
        boards = boards[:args.max_boards]
    if args.board_start < 0 or args.board_start > len(boards):
        parser.error("board-start is outside the selected board list")
    if args.board_stop is not None and not args.board_start < args.board_stop <= len(boards):
        parser.error("board-stop must be after board-start and within the selected board list")
    boards = boards[args.board_start:args.board_stop]
    report_path = args.report.resolve()
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["selected_boards"] != boards or report["config"] != str(config_path):
            raise ValueError("Existing report has a different selection or config")
    else:
        report = {
            "candidate": ("OneVOne default" if args.candidate == "onevone"
                          else "StrategyOne calibrated MLP + settlement head + same-turn search"),
            "opponent": "pinned Catanatron web bot (AlphaBeta depth=2, pruning=True)",
            "engine": ENGINE,
            "config": str(config_path),
            "settlement_head": str(SETTLEMENT_HEAD) if args.candidate == "strategyone" else None,
            "dataset_meta": str(META),
            "fraction": args.fraction,
            "selection_seed": args.selection_seed,
            "board_start": args.board_start,
            "board_stop": args.board_stop,
            "selected_boards": boards,
            "rules": {"players": 2, "target_vp": 10, "number_placement": "official_spiral",
                      "discard_limit": 7, "friendly_robber": False},
            "seat_protocol": "both seats per selected board seed",
            "caps": {"max_game_seconds": args.max_game_seconds,
                     "max_actions": args.max_actions, "max_turns": args.max_turns},
            "games": [],
        }
    completed_keys = {(row["seed"], row["seat"]) for row in report["games"]}
    total = 2 * len(boards)
    for board_index, board in enumerate(boards):
        for seat in (0, 1):
            key = (board["seed"], seat)
            if key in completed_keys:
                continue
            result = run_game(board["seed"], seat, candidate_factory=candidate_factory,
                              max_seconds=args.max_game_seconds,
                              max_actions=args.max_actions,
                              max_turns=args.max_turns,
                              trace=board_index == 0)
            result["game_id"] = board["game_id"]
            report["games"].append(result)
            write_report(report, report_path)
            if len(report["games"]) % 20 == 0 or len(report["games"]) == total:
                summary = report["summary"]
                print(f"{len(report['games'])}/{total}: ours {summary['ours_wins']}, "
                      f"Catanatron {summary['catanatron_wins']}, "
                      f"caps {summary['capped']}, errors {summary['errors']}", flush=True)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
