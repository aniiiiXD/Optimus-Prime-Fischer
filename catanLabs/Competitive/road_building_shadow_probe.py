"""Small shadow probe: FairValue plays; Road Building continuations are logged."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanatron import Color, Game, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer

from catanLabs.Catanatron.tournament_protocol import canonical_action_key, install_v5_protocol
from catanLabs.Competitive.check_visibility import perturbed_hidden
from catanLabs.Competitive.fair_value_agent import FairValueAgent
from catanLabs.Competitive.road_building_agent import RoadBuildingAgent


def percentile(values, fraction):
    values = sorted(values)
    rank = (len(values) - 1) * fraction
    lower, upper = int(rank), min(len(values) - 1, int(rank) + 1)
    return round(values[lower] + (rank - lower) * (values[upper] - values[lower]), 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20271200)
    parser.add_argument("--max-game-seconds", type=float, default=20)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.boards < 1 or args.max_game_seconds <= 0:
        parser.error("boards and max-game-seconds must be positive")
    install_v5_protocol()
    config = json.loads((Path(__file__).parent / "configs/road_building_research.json").read_text())
    totals = {"own_decisions": 0, "card_available": 0, "overrides": 0,
              "first_roads": 0, "second_roads": 0, "truncated": 0,
              "visibility_checks": 0}
    latencies, examples, games = [], [], []
    colors = list(Color)
    for board_index in range(args.boards):
        seed = args.seed + board_index
        for seat in range(4):
            game = Game([RandomPlayer(color) for color in colors], seed=seed,
                        vps_to_win=10, discard_limit=7, friendly_robber=False,
                        number_placement="official_spiral")
            game.playable_actions.sort(key=canonical_action_key)
            color = game.state.colors[seat]
            fair = FairValueAgent(color, config)
            scanner = RoadBuildingAgent(color, config)
            others = [other for other in game.state.colors if other != color]
            mix = ("value", "alphabeta1", "alphabeta1")
            rotation = (board_index + seat) % 3
            opponents = {}
            for index, other in enumerate(others):
                name = mix[(index + rotation) % 3]
                opponents[other] = (ValueFunctionPlayer(other) if name == "value" else
                                    AlphaBetaPlayer(other, AlphaBetaPlayer.Params(depth=1)))
            started = time.perf_counter()
            status, actions = "capped", 0
            try:
                while actions < 15000 and game.state.num_turns < 500:
                    if game.winning_color() is not None:
                        status = "completed"
                        break
                    if time.perf_counter() - started > args.max_game_seconds:
                        break
                    actor, legal = game.state.current_color(), game.playable_actions
                    if actor == color:
                        rng_before = game.state.random.getstate()
                        baseline = fair.decide(game, legal)
                        before = time.perf_counter_ns()
                        alternative = scanner.find_override(game, legal, baseline)
                        scan_ms = (time.perf_counter_ns() - before) / 1e6
                        if game.state.random.getstate() != rng_before:
                            raise AssertionError("Shadow scoring advanced live RNG")
                        info = dict(scanner.last_scan)
                        totals["own_decisions"] += 1
                        if info["eligible"]:
                            totals["card_available"] += 1
                            totals["first_roads"] += info["first_roads"]
                            totals["second_roads"] += info["second_roads"]
                            totals["overrides"] += int(info["override"])
                            totals["truncated"] += int(info["truncated"])
                            latencies.append(scan_ms)
                            if totals["visibility_checks"] < 8:
                                changed = perturbed_hidden(game, color)
                                hidden_baseline = fair.decide(changed, changed.playable_actions)
                                hidden_alternative = scanner.find_override(
                                    changed, changed.playable_actions, hidden_baseline)
                                if hidden_baseline != baseline or hidden_alternative != alternative:
                                    raise AssertionError("Hidden-state perturbation changed choice")
                                scanner.last_scan = info
                                totals["visibility_checks"] += 1
                            if len(examples) < 20:
                                examples.append({"seed": seed, "seat": seat,
                                                 "turn": game.state.num_turns,
                                                 "baseline": baseline.action_type.name,
                                                 "override": bool(alternative),
                                                 "best_line": info.get("best_line"),
                                                 "baseline_score": info.get("baseline_score"),
                                                 "best_score": info.get("best_score"),
                                                 "scan_ms": round(scan_ms, 3)})
                        action = baseline
                    else:
                        action = opponents[actor].decide(game, legal)
                    if action not in legal:
                        raise ValueError(f"Illegal action: {action}")
                    game.execute(action, validate_action=True)
                    actions += 1
                if game.winning_color() is not None:
                    status = "completed"
            except Exception as exc:
                status = f"error: {type(exc).__name__}: {exc}"
            games.append({"seed": seed, "seat": seat, "status": status,
                          "actions": actions, "candidate_vp": scanner._vp(game),
                          "wall_seconds": round(time.perf_counter() - started, 3)})
    report = {"probe": "Road Building shadowed against FairValue",
              "seed": args.seed, "boards": args.boards, "seats_per_board": 4,
              "config": config, "totals": totals,
              "scan_latency_ms": {"median": round(statistics.median(latencies), 3) if latencies else None,
                                  "p95": percentile(latencies, 0.95) if len(latencies) >= 20 else None,
                                  "max": round(max(latencies), 3) if latencies else None,
                                  "samples": [round(value, 3) for value in latencies]},
              "examples": examples, "games": games}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"totals": totals,
                      "latency_ms": report["scan_latency_ms"],
                      "completed": sum(game["status"] == "completed" for game in games),
                      "games": len(games)}, indent=2))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
