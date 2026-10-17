"""Bounded direct-outcome search over six fixed FairValue configurations.

This is a research tuner, not a promotion mechanism. The full run uses four
stage-one boards, then eight fresh stage-two boards for the top two variants.
The --smoke path exercises only one stage-one board and never selects a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import sys
import time


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))
BENCHMARK_FILE = WORKSPACE / "catanLabs/Benchmarks&Evals/run.py"
spec = importlib.util.spec_from_file_location("catanlabs_benchmark", BENCHMARK_FILE)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

from catanLabs.Catanatron.tournament_protocol import PROTOCOL, install_v5_protocol


SCHEMA = "catanlabs-fair-value-policy-search-v1"
SPLIT = WORKSPACE / "catanLabs/Transformer/runs/gate4_merged_v3/dataset.split.json"
BASE_CONFIG = ROOT / "configs/fair_army_finish.json"
RULES_MANIFEST = WORKSPACE / "catanLabs/rules_manifest.json"
STAGE1 = (20276000, 4)
STAGE2 = (20276100, 8)
SMOKE = (20275951, 1)
COMPOSITIONS = {
    "mixed_diverse": ("weighted", "value", "alphabeta1"),
    "mixed_strong": ("value", "alphabeta1", "alphabeta1"),
}
VARIANTS = {
    # These alter the scores that break otherwise dominant VP/production ties.
    "reach_army_up": {"reachable_production_0": 1.5, "army_size": 1.5},
    "reach_army_down": {"reachable_production_0": 0.5, "army_size": 0.5},
    "reach_up_sites_down": {"reachable_production_0": 1.5, "buildable_nodes": 0.5},
    "reach_down_sites_up": {"reachable_production_0": 0.5, "buildable_nodes": 1.5},
    "hand_up_army_down": {"hand_synergy": 1.5, "army_size": 0.5},
    "hand_down_army_up": {"hand_synergy": 0.5, "army_size": 1.5},
}
ARMS = ("fair", *VARIANTS)


class BudgetExceeded(RuntimeError):
    pass


def canonical_bytes(payload) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def configs_from_base(base: dict, defaults: dict) -> dict[str, dict]:
    """Never mutate the incumbent configuration."""
    if not isinstance(base.get("value_weights"), dict):
        raise ValueError("FairValue base config has no value_weights")
    result = {"fair": json.loads(json.dumps(base))}
    for name, changes in VARIANTS.items():
        config = json.loads(json.dumps(base))
        for key, multiplier in changes.items():
            if key not in config["value_weights"] and key not in defaults:
                raise ValueError(f"Missing incumbent/default weight: {key}")
            if not 0.5 <= multiplier <= 1.5:
                raise ValueError("Weight perturbation exceeds predeclared bound")
            original = config["value_weights"].get(key, defaults.get(key))
            config["value_weights"][key] = original * multiplier
        result[name] = config
    if len({canonical_bytes(config) for config in result.values()}) != len(result):
        raise ValueError("Policy search generated duplicate configurations")
    return result


def assert_fresh_seeds(stages: tuple[tuple[int, int], ...]) -> str:
    split = json.loads(SPLIT.read_text())
    used = set().union(*(set(split["board_seeds"][part])
                         for part in ("train", "validation", "final")))
    seen = set()
    for start, boards in stages:
        proposed = set(range(start, start + boards))
        if proposed & (used | seen):
            raise ValueError("Policy-search boards overlap each other or the frozen dataset split")
        seen |= proposed
    return benchmark.sha256(SPLIT)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def objective(candidate: dict, control: dict) -> dict:
    """Equal-weight mixes; paired wins primary, paired VP only a tie-break."""
    differences = {}
    for mix in COMPOSITIONS:
        a = {(row["seed"], row["seat"]): row for row in candidate["tiers"][mix]["games"]}
        b = {(row["seed"], row["seat"]): row for row in control["tiers"][mix]["games"]}
        if a.keys() != b.keys() or not a:
            raise ValueError(f"Unmatched or empty game grid in {mix}")
        for key in a:
            x, y = a[key], b[key]
            if (x["status"] != "completed" or y["status"] != "completed"
                    or x["color"] != y["color"]
                    or x["opponent_assignments"] != y["opponent_assignments"]):
                raise ValueError(f"Incomplete or unmatched candidate/control game: {mix} {key}")
        win_delta = statistics.fmean(int(a[key]["won"]) - int(b[key]["won"]) for key in a)
        vp_delta = statistics.fmean(a[key]["candidate_vp"] - b[key]["candidate_vp"] for key in a)
        differences[mix] = {"win_rate_difference": win_delta, "mean_vp_difference": vp_delta}
    mean_win = statistics.fmean(v["win_rate_difference"] for v in differences.values())
    mean_vp = statistics.fmean(v["mean_vp_difference"] for v in differences.values())
    return {"mean_win_rate_difference": mean_win,
            "mean_vp_difference": mean_vp,
            "by_mix": differences}


def rank_variants(reports: dict) -> list[dict]:
    ranked = [{"name": name, **objective(reports[name], reports["fair"])}
              for name in VARIANTS if name in reports]
    return sorted(ranked, key=lambda row: (-row["mean_win_rate_difference"],
                                           -row["mean_vp_difference"], row["name"]))


def divergence_by_phase(candidate: dict, control: dict) -> dict:
    """Compare first action changes while paired trajectories still match."""
    counts = {"setup": 0, "early": 0, "mid": 0, "late": 0,
              "trace_length": 0, "identical": 0}
    for mix in COMPOSITIONS:
        ref = {(row["seed"], row["seat"]): row
               for row in control["tiers"][mix]["games"]}
        for row in candidate["tiers"][mix]["games"]:
            baseline = ref[row["seed"], row["seat"]]
            trace = row["decision_trace"]
            fair_trace = baseline["decision_trace"]
            mismatch = next((idx for idx, (a, b) in enumerate(zip(trace, fair_trace))
                             if a[1] != b[1]), None)
            if mismatch is None:
                counts["identical" if len(trace) == len(fair_trace)
                       else "trace_length"] += 1
            else:
                counts[trace[mismatch][0]] += 1
    counts["total_games"] = sum(counts.values())
    counts["first_action_divergence_fraction"] = (
        1 - counts["identical"] / counts["total_games"])
    return counts


def check_budget(clock: dict) -> None:
    if (time.perf_counter() >= clock["work_deadline"]
            or time.process_time() >= clock["cpu_deadline"]):
        raise BudgetExceeded("Reserved wall/CPU shutdown margin reached")


def _alarm_handler(signum, frame):
    del signum, frame
    raise BudgetExceeded("Hard wall-clock timer expired")


def run_stage(stage_name: str, first_seed: int, boards: int, arms: tuple[str, ...],
              configs: dict, output_dir: Path, metadata: dict, clock: dict,
              max_game_seconds: float, engine: dict) -> dict:
    sys.path.insert(0, engine["python_root"])
    from catanatron import Color, Game, RandomPlayer
    from catanatron.players.minimax import AlphaBetaPlayer
    from catanatron.players.value import ValueFunctionPlayer
    from catanatron.players.weighted_random import WeightedRandomPlayer
    from catanLabs.Competitive.fair_value_agent import FairValueAgent

    def opponent(name, color):
        if name == "weighted":
            return WeightedRandomPlayer(color)
        if name == "value":
            return ValueFunctionPlayer(color)
        return AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=1))

    reports = {}
    for arm in arms:
        reports[arm] = {
            "protocol": f"{PROTOCOL}-fair-value-policy-search",
            "engine": engine,
            "rules": metadata["rules"],
            "candidate": {"type": arm, "file": str(ROOT / "fair_value_agent.py"),
                          "sha256": metadata["fair_source_sha256"],
                          "config_sha256": metadata["config_sha256"][arm]},
            "stage": stage_name,
            "boards_per_mix": boards,
            "first_seed": first_seed,
            "compositions": COMPOSITIONS,
            "split_file_sha256": metadata["split_file_sha256"],
            "runner_sha256": metadata["runner_sha256"],
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "tiers": {},
        }

    colors = list(Color)
    for mix_name, composition in COMPOSITIONS.items():
        for arm in arms:
            games, latencies = [], []
            for board_index in range(boards):
                seed = first_seed + board_index
                for seat in range(4):
                    check_budget(clock)
                    game = Game([RandomPlayer(color) for color in colors], seed=seed,
                                vps_to_win=10, discard_limit=7,
                                friendly_robber=False, number_placement="official_spiral")
                    game.playable_actions.sort(key=benchmark.canonical_action_key)
                    color = game.state.colors[seat]
                    candidate = FairValueAgent(color, configs[arm])
                    others = [other for other in game.state.colors if other != color]
                    rotation = (board_index + seat) % 3
                    assignments = {other: composition[(index + rotation) % 3]
                                   for index, other in enumerate(others)}
                    opponents = {other: opponent(name, other)
                                 for other, name in assignments.items()}
                    started = time.perf_counter()
                    status, error, actions = "capped", None, 0
                    decision_trace = []
                    try:
                        while actions < 15000 and game.state.num_turns < 500:
                            if game.winning_color() is not None:
                                status = "completed"
                                break
                            check_budget(clock)
                            if time.perf_counter() - started >= max_game_seconds:
                                break
                            actor = game.state.current_color()
                            bot = candidate if actor == color else opponents[actor]
                            before = time.perf_counter_ns()
                            action = bot.decide(game, game.playable_actions)
                            if actor == color:
                                latencies.append((time.perf_counter_ns() - before) / 1e6)
                                phase = ("setup" if game.state.is_initial_build_phase else
                                         "early" if game.state.num_turns < 20 else
                                         "mid" if game.state.num_turns < 60 else "late")
                                decision_trace.append((phase, benchmark.canonical_action_key(action)))
                            if action not in game.playable_actions:
                                raise ValueError(f"Illegal action: {action}")
                            game.execute(action, validate_action=True)
                            actions += 1
                        if game.winning_color() is not None:
                            status = "completed"
                    except BudgetExceeded as exc:
                        status, error = "budget_capped", str(exc)
                    except Exception as exc:
                        status, error = "error", f"{type(exc).__name__}: {exc}"
                    winner = game.winning_color()
                    vp = game.state.player_state[
                        f"P{game.state.color_to_index[color]}_ACTUAL_VICTORY_POINTS"]
                    row = {"seed": seed, "seat": seat, "color": color.value,
                           "opponent": mix_name,
                           "opponent_assignments": {other.value: name for other, name in assignments.items()},
                           "status": status, "winner": winner.value if winner else None,
                           "won": status == "completed" and winner == color,
                           "candidate_vp": vp, "actions": actions,
                           "turns": game.state.num_turns,
                           "wall_seconds": round(time.perf_counter() - started, 3),
                           "decision_trace": decision_trace}
                    if error:
                        row["error"] = error
                    games.append(row)
                    partial = {"schema": SCHEMA, "status": "unverified_partial",
                               "stage": stage_name, "arm": arm, "mix": mix_name,
                               "attempts_recorded": len(games), "expected_games": boards * 4,
                               "metadata": metadata,
                               "tier": {"summary": benchmark.summarize(games, latencies),
                                        "games": games}}
                    write_json_atomic(output_dir / "partial" / stage_name /
                                      f"{mix_name}__{arm}.json", partial)
                    if status != "completed":
                        raise RuntimeError(f"{stage_name} {mix_name} {arm} seed {seed} seat {seat}: {status}")
            reports[arm]["tiers"][mix_name] = {
                "summary": benchmark.summarize(games, latencies), "games": games}
            print(f"{stage_name} {mix_name} {arm}: "
                  f"{sum(row['won'] for row in games)}/{len(games)} wins", flush=True)

    expected = {(first_seed + board, seat) for board in range(boards) for seat in range(4)}
    for mix_name in COMPOSITIONS:
        reference = None
        for arm in arms:
            rows = reports[arm]["tiers"][mix_name]["games"]
            keys = [(row["seed"], row["seat"]) for row in rows]
            if len(rows) != len(expected) or set(keys) != expected or len(set(keys)) != len(keys):
                raise RuntimeError(f"Incomplete matched grid: {stage_name} {mix_name} {arm}")
            assignment = [(row["seed"], row["seat"], row["color"], row["opponent_assignments"])
                          for row in rows]
            if reference is None:
                reference = assignment
            elif reference != assignment:
                raise RuntimeError(f"Mismatched opponent grid: {stage_name} {mix_name} {arm}")
    for arm, report in reports.items():
        report["divergence_from_fair"] = (
            {"total_games": sum(len(report["tiers"][mix]["games"])
                                for mix in COMPOSITIONS),
             "first_action_divergence_fraction": 0.0}
            if arm == "fair" else divergence_by_phase(report, reports["fair"]))
        write_json_atomic(output_dir / stage_name / f"{arm}.json", report)
    return reports


def main():
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="One stage-one board; no selection")
    parser.add_argument("--wall-budget", type=float, default=540.0)
    parser.add_argument("--max-game-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if not 0 < args.wall_budget <= 590 or args.max_game_seconds <= 0:
        parser.error("wall-budget must be in (0, 590], max-game-seconds positive")
    work_deadline = wall_started + max(0.0, args.wall_budget - 5.0)
    clock = {"work_deadline": work_deadline,
             "cpu_deadline": cpu_started + max(0.0, args.wall_budget - 5.0)}
    if hasattr(signal, "setitimer"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, args.wall_budget - (time.perf_counter() - wall_started)))

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output-dir must be empty to prevent stale selection artifacts")
    stages = (SMOKE,) if args.smoke else (STAGE1, STAGE2)
    try:
        split_hash = assert_fresh_seeds(stages)
        engine = benchmark.source_info()
        sys.path.insert(0, engine["python_root"])
        install_v5_protocol()
        from catanatron.players.value import DEFAULT_WEIGHTS
        base = json.loads(BASE_CONFIG.read_text())
        configs = configs_from_base(base, DEFAULT_WEIGHTS)
        source_hash = benchmark.sha256(ROOT / "fair_value_agent.py")
        metadata = {
            "schema": SCHEMA,
            "mode": "one_board_smoke" if args.smoke else "successive_halving_4_then_8",
            "incumbent_config": str(BASE_CONFIG),
            "incumbent_config_sha256": benchmark.sha256(BASE_CONFIG),
            "config_sha256": {name: sha256_bytes(canonical_bytes(config))
                              for name, config in configs.items()},
            "variant_multipliers": VARIANTS,
            "base_weights": {key: base["value_weights"].get(key, DEFAULT_WEIGHTS[key])
                             for changes in VARIANTS.values() for key in changes},
            "fair_source_sha256": source_hash,
            "runner_sha256": benchmark.sha256(Path(__file__)),
            "split_file_sha256": split_hash,
            "rules_manifest_sha256": benchmark.sha256(RULES_MANIFEST),
            "engine": engine,
            "rules": {"players": 4, "target_vp": 10, "discard_limit": 7,
                      "friendly_robber": False, "number_placement": "official_spiral",
                      "player_trade_offers": False, "isolated_copy_rng": True,
                      "forced_search_outcomes": True},
            "stages": [{"name": "smoke" if args.smoke else "stage1", "first_seed": stages[0][0],
                        "boards": stages[0][1], "arms": list(ARMS)}] +
                      ([] if args.smoke else [{"name": "stage2", "first_seed": STAGE2[0],
                                                "boards": STAGE2[1], "arms": "top_two_plus_fair"}]),
            "objective": "equal-mix paired win-rate difference; equal-mix paired mean VP tie-break; variant name final tie-break",
            "max_game_seconds": args.max_game_seconds,
            "wall_budget_seconds": args.wall_budget,
            "cpu_budget_seconds": args.wall_budget,
            "platform": platform.platform(),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        }
        write_json_atomic(output / "manifest.json", metadata)
        first_seed, boards = stages[0]
        first_name = "smoke" if args.smoke else "stage1"
        stage1 = run_stage(first_name, first_seed, boards, ARMS,
                           configs, output, metadata, clock,
                           args.max_game_seconds, engine)
        check_budget(clock)
        if args.smoke:
            write_json_atomic(output / "status.json", {
                "schema": SCHEMA, "status": "smoke_complete_no_selection",
                "wall_seconds": round(time.perf_counter() - wall_started, 3),
                "cpu_seconds": round(time.process_time() - cpu_started, 3),
                "games_completed": len(ARMS) * len(COMPOSITIONS) * boards * 4})
            return
        ranked1 = rank_variants(stage1)
        write_json_atomic(output / "stage1_ranking.json", {
            "schema": SCHEMA, "ranking": ranked1,
            "selected_for_stage2": [row["name"] for row in ranked1[:2]]})
        stage2_arms = ("fair", *(row["name"] for row in ranked1[:2]))
        stage2 = run_stage("stage2", STAGE2[0], STAGE2[1], stage2_arms,
                           configs, output, metadata, clock,
                           args.max_game_seconds, engine)
        check_budget(clock)
        ranked2 = rank_variants(stage2)
        selected = ranked2[0]
        config_file = output / "selected_config.json"
        config_temp = config_file.with_name(config_file.name + ".tmp")
        config_temp.write_bytes(canonical_bytes(configs[selected["name"]]))
        os.replace(config_temp, config_file)
        config_hash = benchmark.sha256(config_file)
        if config_hash != metadata["config_sha256"][selected["name"]]:
            raise AssertionError("Selected config hash mismatch")
        write_json_atomic(output / "selection.json", {
            "schema": SCHEMA, "status": "research_candidate_not_promoted",
            "selected_variant": selected["name"],
            "selected_config": str(config_file), "selected_config_sha256": config_hash,
            "stage1_ranking": ranked1, "stage2_ranking": ranked2,
            "stage2_seed_range": [STAGE2[0], STAGE2[0] + STAGE2[1] - 1],
            "wall_seconds": round(time.perf_counter() - wall_started, 3),
            "cpu_seconds": round(time.process_time() - cpu_started, 3),
            "games_completed": (len(ARMS) * len(COMPOSITIONS) * STAGE1[1] * 4
                                + len(stage2_arms) * len(COMPOSITIONS) * STAGE2[1] * 4),
            "fresh_game_test_required": True})
        print(f"Selected unpromoted variant {selected['name']} ({config_hash})", flush=True)
    except Exception as exc:
        write_json_atomic(output / "status.json", {
            "schema": SCHEMA, "status": "incomplete_no_selection",
            "reason": f"{type(exc).__name__}: {exc}",
            "wall_seconds": round(time.perf_counter() - wall_started, 3),
            "cpu_seconds": round(time.process_time() - cpu_started, 3)})
        raise
    finally:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
