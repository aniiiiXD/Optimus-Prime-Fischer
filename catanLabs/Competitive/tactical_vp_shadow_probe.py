"""Small shadow probe: run FairValue while logging exact VP tactics it misses."""

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
from catanLabs.Competitive.tactical_vp_agent import TacticalVPAgent


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))], 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20271200)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-game-seconds", type=float, default=20)
    args = parser.parse_args()
    if args.boards < 1 or args.max_game_seconds <= 0:
        parser.error("boards and max-game-seconds must be positive")
    install_v5_protocol()
    config = json.loads((Path(__file__).parent / "configs/tactical_vp_research.json").read_text())
    records, scan_times, examples = [], [], []
    totals = {"own_decisions": 0, "eligible_decisions": 0, "found_lines": 0,
              "overrides": 0, "truncated": 0, "roots": 0, "followups": 0,
              "visibility_checks": 0}
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
            scanner = TacticalVPAgent(color, config)
            opponent_colors = [other for other in game.state.colors if other != color]
            names = ("value", "alphabeta1", "alphabeta1")
            rotation = (board_index + seat) % 3
            opponents = {}
            for index, other in enumerate(opponent_colors):
                name = names[(index + rotation) % 3]
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
                    actor = game.state.current_color()
                    legal = game.playable_actions
                    if actor == color:
                        rng_before = game.state.random.getstate()
                        baseline = fair.decide(game, legal)
                        before = time.perf_counter_ns()
                        alternative = scanner.find_override(game, legal, baseline)
                        elapsed = (time.perf_counter_ns() - before) / 1e6
                        if game.state.random.getstate() != rng_before:
                            raise AssertionError("Shadow scoring advanced live RNG")
                        totals["own_decisions"] += 1
                        info = dict(scanner.last_scan)
                        if info["eligible"]:
                            totals["eligible_decisions"] += 1
                            scan_times.append(elapsed)
                            if totals["visibility_checks"] < 8:
                                changed = perturbed_hidden(game, color)
                                hidden_baseline = fair.decide(changed, changed.playable_actions)
                                hidden_alternative = scanner.find_override(
                                    changed, changed.playable_actions, hidden_baseline)
                                if (hidden_baseline != baseline or hidden_alternative != alternative):
                                    raise AssertionError("Hidden-state perturbation changed choice")
                                scanner.last_scan = info
                                totals["visibility_checks"] += 1
                        totals["found_lines"] += int(info["found_line"])
                        totals["overrides"] += int(info["override"])
                        totals["truncated"] += int(info["truncated"])
                        totals["roots"] += info["roots"]
                        totals["followups"] += info["followups"]
                        if alternative is not None and len(examples) < 10:
                            examples.append({"seed": seed, "seat": seat,
                                             "turn": game.state.num_turns,
                                             "vp": scanner._vp(game),
                                             "baseline": baseline.action_type.name,
                                             "tactical": alternative.action_type.name,
                                             "line_vp": info["best_line_vp"],
                                             "line": info.get("line"),
                                             "scan_ms": round(elapsed, 3)})
                        action = baseline  # Keep the actual game on FairValue.
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
            records.append({"seed": seed, "seat": seat, "status": status,
                            "actions": actions, "winner": (
                                game.winning_color().value if game.winning_color() else None),
                            "candidate_vp": scanner._vp(game),
                            "wall_seconds": round(time.perf_counter() - started, 3)})
    report = {"probe": "fair-value shadowed by deterministic late-turn VP scanner",
              "seed": args.seed, "boards": args.boards, "seats_per_board": 4,
              "config": config, "totals": totals,
              "scan_latency_ms": {"median": percentile(scan_times, 0.5),
                                  "p95": percentile(scan_times, 0.95),
                                  "max": round(max(scan_times), 3) if scan_times else None,
                                  "mean": round(statistics.mean(scan_times), 3) if scan_times else None},
              "examples": examples, "games": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"totals": totals, "scan_latency_ms": report["scan_latency_ms"],
                      "completed": sum(r["status"] == "completed" for r in records),
                      "games": len(records)}, indent=2))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
