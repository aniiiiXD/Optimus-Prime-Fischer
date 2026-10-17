"""Outcome-tune the invest/convert coefficients; confirm on fresh paired seeds."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import statistics
import sys

from train_legal_ranker import HERE

MAIN = HERE.parents[2].parent / "Hyde"
BENCHMARK = (MAIN / "catanLabs/FromScratch/StrategyOne/experiments/"
             "catanatron_80pct/run.py")
spec = importlib.util.spec_from_file_location("catanatron_80pct_run", BENCHMARK)
benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(benchmark)

from invest_convert import Coefficients, InvestConvertPlayer  # noqa: E402


ARMS = {
    "balanced": Coefficients(),
    "growth": Coefficients(vp=7.0, production=6.0, access=2.0,
                           horizon_base=14.0),
    "convert": Coefficients(vp=11.0, production=3.0, access=1.0,
                            horizon_base=12.0),
    "urgent": Coefficients(vp=11.0, production=4.0, access=1.25,
                           horizon_opp_vp=1.4),
    "static": Coefficients(vp=11.0, production=3.0, access=1.0,
                           horizon_base=6.0, horizon_opp_vp=0.0,
                           horizon_own_vp=0.0),
}


def factory_from_baseline(baseline_factory, name):
    if name == "mlp":
        return baseline_factory
    if name == "onevone":
        return benchmark.load_candidate("onevone")
    if name == "hybrid":
        turn_factory = factory_from_baseline(baseline_factory, "convert")
        other_factory = benchmark.load_candidate("onevone")

        class Hybrid:
            def __init__(self, turn, other):
                self.turn = turn
                self.other = other

            def decide(self, game, playable_actions):
                if game.state.current_prompt.name == "PLAY_TURN":
                    prefix = f"P{game.state.color_to_index[self.turn.color]}_"
                    if game.state.player_state[prefix + "HAS_ROLLED"]:
                        return self.turn.decide(game, playable_actions)
                return self.other.decide(game, playable_actions)

        return lambda color, seed: Hybrid(turn_factory(color, seed),
                                          other_factory(color, seed))
    coefficients = ARMS[name]

    def create(color, seed):
        base = baseline_factory(color, seed)
        return InvestConvertPlayer(
            color, base.model, base.edge_index, base.device, base.adapter,
            base.settlement_head, coefficients=coefficients,
        )

    return create


def report_summary(rows):
    summary = {}
    for arm in sorted({row["arm"] for row in rows}):
        games = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            "games": len(games),
            "wins": sum(row["ours_won"] for row in games),
            "win_rate": sum(row["ours_won"] for row in games) / len(games),
            "mean_vp": statistics.mean(row["ours_vp"] for row in games),
            "caps": sum(row["status"] == "capped" for row in games),
            "errors": sum(row["status"] == "error" for row in games),
            "move_counts": dict(sum((Counter(row["move_counts"]["ours"])
                                     for row in games), Counter())),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("probe", "tune", "confirm"), required=True)
    parser.add_argument("--arms", nargs="+", choices=("mlp", "onevone", "hybrid", *ARMS),
                        default=["mlp", "balanced", "growth", "convert", "urgent"])
    parser.add_argument("--boards", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    defaults = {"probe": (2, 20269000), "tune": (16, 20269020),
                "confirm": (64, 20269100)}
    board_count, first_seed = defaults[args.phase]
    board_count = args.boards if args.boards is not None else board_count
    first_seed = args.seed if args.seed is not None else first_seed
    if board_count < 1:
        parser.error("boards must be positive")

    benchmark.install_v5_protocol()
    baseline_factory = benchmark.load_candidate("strategyone")
    factories = {name: factory_from_baseline(baseline_factory, name)
                 for name in args.arms}
    config = {
        "phase": args.phase, "seed": first_seed, "boards": board_count,
        "arms": args.arms, "rules": {"players": 2, "vps_to_win": 10,
                                   "number_placement": "official_spiral",
                                   "discard_limit": 7, "friendly_robber": False},
        "opponent": "pinned Catanatron web bot depth2",
        "coefficients": {name: arm.json() for name, arm in ARMS.items()},
    }
    path = args.report.resolve()
    if path.exists():
        report = json.loads(path.read_text())
        if report["config"] != config:
            raise ValueError("Existing report has different configuration")
    else:
        report = {"config": config, "games": [], "summary": {}}
    completed = {(row["arm"], row["seed"], row["seat"])
                 for row in report["games"]}

    for board in range(board_count):
        seed = first_seed + board
        for seat in (0, 1):
            for arm in args.arms:
                if (arm, seed, seat) in completed:
                    continue
                game = benchmark.run_game(
                    seed, seat, candidate_factory=factories[arm],
                    max_seconds=60, max_actions=10000, max_turns=500,
                    trace=False,
                )
                game["arm"] = arm
                report["games"].append(game)
                report["summary"] = report_summary(report["games"])
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text(json.dumps(report, indent=2) + "\n")
                tmp.replace(path)
                if len(report["games"]) % 16 == 0:
                    print(len(report["games"]),
                          {name: (data["wins"], data["games"])
                           for name, data in report["summary"].items()}, flush=True)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    if __import__("os").environ.get("PYTHONHASHSEED") != "0":
        import os
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
