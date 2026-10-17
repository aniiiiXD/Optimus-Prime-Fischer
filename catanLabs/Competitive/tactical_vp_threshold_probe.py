"""Compare tactical VP gates on identical FairValue games with one scan/state."""

from __future__ import annotations

import argparse
from collections import Counter
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


THRESHOLDS = (7, 5, 0)


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower, upper = int(index), min(len(values) - 1, int(index) + 1)
    return round(values[lower] + (index - lower) * (values[upper] - values[lower]), 3)


def vp_bucket(vp):
    return "0-4" if vp < 5 else "5-6" if vp < 7 else "7-9"


def threshold_summary(rows, threshold):
    selected = [row for row in rows if row["vp"] >= threshold]
    latency = [row["scan_ms"] for row in selected]
    overrides = [row for row in selected if row["override"]]
    by_bucket = {}
    for bucket in ("0-4", "5-6", "7-9"):
        bucket_rows = [row for row in selected if vp_bucket(row["vp"]) == bucket]
        bucket_overrides = [row for row in bucket_rows if row["override"]]
        bucket_latency = [row["scan_ms"] for row in bucket_rows]
        by_bucket[bucket] = {
            "eligible_decisions": len(bucket_rows),
            "override_decisions": len(bucket_overrides),
            "planned_followups_checked": sum(row.get("continuation_match") is not None
                                             for row in bucket_overrides),
            "planned_followups_matched": sum(row.get("continuation_match") is True
                                             for row in bucket_overrides),
            "games_with_overrides": len({(row["seed"], row["seat"])
                                         for row in bucket_overrides}),
            "turns_with_overrides": len({(row["seed"], row["seat"], row["turn"])
                                         for row in bucket_overrides}),
            "truncated_scans": sum(row["truncated"] for row in bucket_rows),
            "scan_latency_ms": {
                "median": round(statistics.median(bucket_latency), 3)
                if bucket_latency else None,
                "p95": percentile(bucket_latency, 0.95),
                "max": round(max(bucket_latency), 3) if bucket_latency else None},
        }
    return {"eligible_decisions": len(selected),
            "found_lines": sum(row["found_line"] for row in selected),
            "override_decisions": len(overrides),
            "planned_followups_checked": sum(row.get("continuation_match") is not None
                                             for row in overrides),
            "planned_followups_matched": sum(row.get("continuation_match") is True
                                             for row in overrides),
            "games_with_overrides": len({(row["seed"], row["seat"]) for row in overrides}),
            "turns_with_overrides": len({(row["seed"], row["seat"], row["turn"])
                                        for row in overrides}),
            "truncated_scans": sum(row["truncated"] for row in selected),
            "roots": sum(row["roots"] for row in selected),
            "followups": sum(row["followups"] for row in selected),
            "by_vp_bucket": by_bucket,
            "scan_latency_ms": {
                "median": round(statistics.median(latency), 3) if latency else None,
                "p95": percentile(latency, 0.95),
                "max": round(max(latency), 3) if latency else None}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20274000)
    parser.add_argument("--max-game-seconds", type=float, default=20)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.boards < 1 or args.max_game_seconds <= 0:
        parser.error("boards and max-game-seconds must be positive")
    install_v5_protocol()
    config = json.loads((Path(__file__).parent / "configs/tactical_vp_research.json").read_text())
    scanner_config = {**config, "tactical_min_vp": 0}
    rows, games, examples = [], [], []
    visibility_checks = Counter()
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
            scanner = TacticalVPAgent(color, scanner_config)
            other_colors = [other for other in game.state.colors if other != color]
            names = ("value", "alphabeta1", "alphabeta1")
            rotation = (board_index + seat) % 3
            opponents = {}
            for index, other in enumerate(other_colors):
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
                    actor, legal = game.state.current_color(), game.playable_actions
                    if actor == color:
                        rng_before = game.state.random.getstate()
                        baseline = fair.decide(game, legal)
                        if scanner._eligible(game) and len(legal) > 1:
                            vp = scanner._vp(game)
                            before = time.perf_counter_ns()
                            alternative = scanner.find_override(game, legal, baseline)
                            elapsed = (time.perf_counter_ns() - before) / 1e6
                            info = dict(scanner.last_scan)
                            if not info["eligible"]:
                                raise AssertionError("Zero-VP scanner unexpectedly ineligible")
                            continuation_match = None
                            if alternative is not None:
                                if alternative not in legal:
                                    raise AssertionError("Scanner selected an illegal root")
                                trial = game.copy()
                                trial.execute(alternative, validate_action=True)
                                planned_second = info["line"]["second"]
                                if planned_second is not None:
                                    expected = next((action for action in trial.playable_actions
                                                     if action.action_type.name == planned_second[0]
                                                     and repr(action.value) == planned_second[1]), None)
                                    if expected is None:
                                        raise AssertionError("Searched follow-up is not legal after root")
                                    trial_rng = trial.state.random.getstate()
                                    continuation_match = scanner.decide(
                                        trial, trial.playable_actions) == expected
                                    if trial.state.random.getstate() != trial_rng:
                                        raise AssertionError("Follow-up check advanced copied RNG")
                                if len(examples) < 50:
                                    examples.append({"seed": seed, "seat": seat,
                                                     "turn": game.state.num_turns,
                                                     "vp": vp,
                                                     "baseline": baseline.action_type.name,
                                                     "root": alternative.action_type.name,
                                                     "line": info.get("line"),
                                                     "continuation_match": continuation_match,
                                                     "scan_ms": round(elapsed, 3)})
                            bucket = vp_bucket(vp)
                            if visibility_checks[bucket] < 3:
                                changed = perturbed_hidden(game, color)
                                hidden_baseline = fair.decide(changed, changed.playable_actions)
                                hidden_alternative = scanner.find_override(
                                    changed, changed.playable_actions, hidden_baseline)
                                if hidden_baseline != baseline or hidden_alternative != alternative:
                                    raise AssertionError("Hidden perturbation changed action")
                                visibility_checks[bucket] += 1
                                scanner.last_scan = info
                            rows.append({"seed": seed, "seat": seat,
                                         "turn": game.state.num_turns,
                                         "vp": vp, "scan_ms": round(elapsed, 6),
                                         "found_line": bool(info["found_line"]),
                                         "override": bool(info["override"]),
                                         "continuation_match": continuation_match,
                                         "truncated": bool(info["truncated"]),
                                         "roots": info["roots"],
                                         "followups": info["followups"]})
                        if game.state.random.getstate() != rng_before:
                            raise AssertionError("Shadow decision advanced live RNG")
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
                          "actions": actions,
                          "wall_seconds": round(time.perf_counter() - started, 3)})
    report = {"probe": "FairValue-played matched tactical VP threshold shadow",
              "seed": args.seed, "boards": args.boards,
              "thresholds": THRESHOLDS,
              "opponents": ["value", "alphabeta1", "alphabeta1"],
              "config": config,
              "visibility_checks_by_vp_bucket": dict(visibility_checks),
              "by_threshold": {str(threshold): threshold_summary(rows, threshold)
                               for threshold in THRESHOLDS},
              "examples": examples, "decisions": rows, "games": games}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"by_threshold": report["by_threshold"],
                      "visibility_checks_by_vp_bucket": dict(visibility_checks),
                      "completed": sum(game["status"] == "completed" for game in games),
                      "games": len(games)}, indent=2))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
