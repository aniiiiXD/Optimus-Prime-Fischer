"""Setup-only shadow: FairValue opens while a joint-pair planner is logged."""

from __future__ import annotations

import argparse
import hashlib
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
from catanLabs.Competitive.opening_pair_agent import OpeningPairAgent


def percentile(values, fraction):
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower, upper = int(rank), min(len(ordered) - 1, int(rank) + 1)
    return round(ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower]), 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20272700)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.boards < 1:
        parser.error("boards must be positive")
    install_v5_protocol()
    config_file = Path(__file__).parent / "configs/opening_pair_research.json"
    config = json.loads(config_file.read_text())
    results, latencies = [], []
    for seed in range(args.seed, args.seed + args.boards):
        game = Game([RandomPlayer(color) for color in Color], seed=seed,
                    vps_to_win=10, discard_limit=7, friendly_robber=False,
                    number_placement="official_spiral")
        game.playable_actions.sort(key=canonical_action_key)
        last = game.state.colors[-1]
        opponents = {color: (ValueFunctionPlayer(color) if i == 0 else
                             AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=1)))
                     for i, color in enumerate(game.state.colors[:-1])}
        while game.state.current_color() != last:
            actor = game.state.current_color()
            choice = opponents[actor].decide(game, game.playable_actions)
            if choice not in game.playable_actions:
                raise AssertionError("Opponent emitted illegal setup action")
            game.execute(choice, validate_action=True)

        fair = FairValueAgent(last, config)
        scanner = OpeningPairAgent(last, config)
        before_rng = game.state.random.getstate()
        start = time.perf_counter_ns()
        candidate_first = scanner.find_opening_plan(game, game.playable_actions)
        latency_ms = (time.perf_counter_ns() - start) / 1e6
        latencies.append(latency_ms)
        if game.state.random.getstate() != before_rng:
            raise AssertionError("Opening shadow advanced live RNG")
        candidate_line = scanner._candidate_plan
        if len(candidate_line) != 4:
            raise AssertionError("Opening shadow did not produce four moves")
        changed = perturbed_hidden(game, last)
        hidden_scanner = OpeningPairAgent(last, config)
        hidden_scanner.find_opening_plan(changed, changed.playable_actions)
        if hidden_scanner._candidate_plan != candidate_line:
            raise AssertionError("Hidden-state perturbation changed opening plan")

        # Verify that the planned continuation survives the real legal-action API.
        branch = game.copy()
        for action in candidate_line:
            if action not in branch.playable_actions:
                raise AssertionError("Planned opening was illegal in replay")
            branch.execute(action, validate_action=True)

        fair_line = []
        for _ in range(4):
            if game.state.current_color() != last:
                raise AssertionError("Last-seat opening was interrupted")
            choice = fair.decide(game, game.playable_actions)
            if choice not in game.playable_actions:
                raise AssertionError("FairValue emitted illegal setup action")
            fair_line.append(choice)
            game.execute(choice, validate_action=True)

        production, starting = scanner._node_data(game)
        candidate_pair_score = scanner._pair_score(
            game, candidate_line[0].value, candidate_line[2].value, production, starting)
        fair_pair_score = scanner._pair_score(
            game, fair_line[0].value, fair_line[2].value, production, starting)
        results.append({
            "seed": seed,
            "seat": 3,
            "color": last.value,
            "first_action_flip": candidate_first != fair_line[0],
            "full_line_flip": tuple(fair_line) != candidate_line,
            "candidate_line": [(a.action_type.name, repr(a.value)) for a in candidate_line],
            "fair_line": [(a.action_type.name, repr(a.value)) for a in fair_line],
            "candidate_pair_score": round(candidate_pair_score, 3),
            "fair_pair_score": round(fair_pair_score, 3),
            "scan": scanner.last_scan,
            "scan_ms": round(latency_ms, 3),
            "legal_replay": True,
            "visibility_invariant": True,
            "live_rng_unchanged": True,
        })

    report = {
        "probe": "last-seat opening shadow; FairValue made actual moves",
        "rules": "provisional four-player Catanatron v5, first to 10, no domestic offers",
        "boards": args.boards,
        "first_seed": args.seed,
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "agent_sha256": hashlib.sha256((Path(__file__).parent / "opening_pair_agent.py").read_bytes()).hexdigest(),
        "first_action_flips": sum(row["first_action_flip"] for row in results),
        "full_line_flips": sum(row["full_line_flip"] for row in results),
        "latency_ms": {"median": round(statistics.median(latencies), 3),
                       "p95": percentile(latencies, 0.95),
                       "max": round(max(latencies), 3)},
        "games": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("boards", "first_action_flips", "full_line_flips", "latency_ms")}, indent=2))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
