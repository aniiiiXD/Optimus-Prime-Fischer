"""Train 1v1 strategy parameters by league-based evolutionary search.

The optimizer mutates the existing legal-action policy, evaluates every
candidate on matched seeds and both seats, then checks the selected candidate
on a separate held-out seed range before promotion. Win rate is the primary
objective; VP margin breaks close ties. No training or validation seeds are
reused for the final confirmation tournament.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

from catanLabs.Catanatron.engine import source_info  # noqa: E402
from catanLabs.Catanatron.tournament_protocol import (  # noqa: E402
    PROTOCOL,
    canonical_action_key,
    install_v5_protocol,
)


STRONG_OPPONENTS = ("value", "alphabeta1", "alphabeta2", "tactical")
VALUE_KEYS = (
    "reachable_production_0",
    "reachable_production_1",
    "buildable_nodes",
    "hand_synergy",
    "army_size",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def mutate(parent: dict, rng: random.Random) -> dict:
    child = copy.deepcopy(parent)
    value_weights = child.setdefault("value_weights", {})
    robber = child.setdefault("onevone_robber", {})
    robber_weights = child.setdefault("robber_weights", {})

    operations = [
        "value_weight", "value_weight", "resource_cost", "lead_pressure",
        "theft", "robber_scale", "dev_purchase", "tactical_threshold",
        "tactical_budget",
    ]
    for operation in rng.sample(operations, k=rng.randint(1, 3)):
        if operation == "value_weight":
            key = rng.choice(VALUE_KEYS)
            current = float(value_weights.get(key, 0.0))
            if current <= 0:
                current = 100.0
            factor = rng.choice((0.5, 0.7, 0.85, 1.2, 1.5, 2.0))
            value_weights[key] = round(current * factor, 4)
        elif operation == "resource_cost":
            current = float(robber.get("own_production_cost", 1.2))
            robber["own_production_cost"] = round(
                max(0.5, min(2.0, current * rng.choice((0.65, 0.8, 1.2, 1.45)))), 4
            )
        elif operation == "lead_pressure":
            robber["opponent_lead_sensitivity"] = rng.choice((0.0, 0.05, 0.1, 0.16, 0.24, 0.32))
            robber["max_opponent_pressure"] = rng.choice((1.2, 1.4, 1.6, 1.8))
        elif operation == "theft":
            robber["theft_base"] = rng.choice((0.0, 0.1, 0.2, 0.35, 0.5))
            robber["theft_per_public_card"] = rng.choice((0.0, 0.04, 0.08, 0.12, 0.18))
        elif operation == "robber_scale":
            current = float(robber_weights.get("robber", 9.0))
            robber_weights["robber"] = round(
                max(4.0, min(18.0, current * rng.choice((0.65, 0.8, 1.25, 1.5)))), 4
            )
        elif operation == "dev_purchase":
            child["dev_buy_vp_threshold"] = rng.choice((5, 6, 7, 8, 9))
            current = float(child.get("dev_buy_late_bonus", 30000.0))
            child["dev_buy_late_bonus"] = round(
                max(0.0, min(100000.0, current * rng.choice((0.0, 0.5, 0.8, 1.25, 1.75)))),
                4,
            )
        elif operation == "tactical_threshold":
            # The scanner searches only deterministic same-turn VP lines. Its
            # minimum VP is a meaningful phase gate, not a score coefficient.
            child["tactical_min_vp"] = rng.choice((0, 3, 5, 7))
        elif operation == "tactical_budget":
            child["tactical_max_roots"] = rng.choice((32, 64, 128))
            child["tactical_max_followups"] = rng.choice((128, 256, 512))
    return child


def play_batch(config: dict, opponent_names: tuple[str, ...], seeds: tuple[int, ...],
               rules: dict, max_actions: int, max_turns: int,
               max_game_seconds: float) -> tuple[list[dict], list[float]]:
    from catanLabs.OneVOne.agent import OneVOneAgent
    from catanLabs.OneVOne.evaluate import make_opponent
    from catanatron import Color, Game, RandomPlayer

    colors = [Color.RED, Color.BLUE]
    game_rules = {
        "vps_to_win": int(rules["target_vp"]),
        "discard_limit": int(rules["discard_limit"]),
        "friendly_robber": bool(rules["friendly_robber"]),
        "number_placement": str(rules["number_placement"]),
    }
    games = []
    latencies = []
    for opponent_name in opponent_names:
        for seed in seeds:
            for candidate_seat in range(2):
                game = Game([RandomPlayer(color) for color in colors], seed=seed, **game_rules)
                game.playable_actions.sort(key=canonical_action_key)
                candidate_color = game.state.colors[candidate_seat]
                opponent_color = next(color for color in game.state.colors if color != candidate_color)
                candidate = OneVOneAgent(candidate_color, config)
                opponent = make_opponent(opponent_name, opponent_color, config)
                started = time.perf_counter()
                actions = 0
                status = "capped"
                error = None
                try:
                    while actions < max_actions and game.state.num_turns < max_turns:
                        if game.winning_color() is not None:
                            status = "completed"
                            break
                        if time.perf_counter() - started > max_game_seconds:
                            break
                        actor = game.state.current_color()
                        bot = candidate if actor == candidate_color else opponent
                        before = time.perf_counter_ns()
                        action = bot.decide(game, game.playable_actions)
                        if actor == candidate_color:
                            latencies.append((time.perf_counter_ns() - before) / 1e6)
                        if action not in game.playable_actions:
                            raise ValueError(f"Illegal action from {type(bot).__name__}: {action}")
                        game.execute(action, validate_action=True)
                        game.playable_actions.sort(key=canonical_action_key)
                        actions += 1
                    if game.winning_color() is not None:
                        status = "completed"
                except Exception as exc:
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"

                winner = game.winning_color()
                candidate_prefix = f"P{game.state.color_to_index[candidate_color]}_"
                opponent_prefix = f"P{game.state.color_to_index[opponent_color]}_"
                record = {
                    "seed": seed,
                    "seat": candidate_seat,
                    "opponent": opponent_name,
                    "status": status,
                    "won": status == "completed" and winner == candidate_color,
                    "candidate_vp": int(game.state.player_state[candidate_prefix + "ACTUAL_VICTORY_POINTS"]),
                    "opponent_vp": int(game.state.player_state[opponent_prefix + "ACTUAL_VICTORY_POINTS"]),
                    "actions": actions,
                    "turns": game.state.num_turns,
                    "wall_seconds": round(time.perf_counter() - started, 3),
                }
                if error:
                    record["error"] = error
                games.append(record)
    return games, latencies


def metrics(games: list[dict]) -> dict:
    by_opponent = {}
    for name in sorted({game["opponent"] for game in games}):
        rows = [game for game in games if game["opponent"] == name]
        wins = sum(row["won"] for row in rows)
        by_opponent[name] = {
            "wins": wins,
            "games": len(rows),
            "win_rate": wins / len(rows) if rows else None,
            "completed": sum(row["status"] == "completed" for row in rows),
            "errors": sum(row["status"] == "error" for row in rows),
            "capped": sum(row["status"] == "capped" for row in rows),
            "mean_vp_margin": statistics.fmean(
                row["candidate_vp"] - row["opponent_vp"] for row in rows
            ) if rows else None,
        }
    macro_win_rate = statistics.fmean(item["win_rate"] for item in by_opponent.values()) if by_opponent else 0.0
    mean_margin = statistics.fmean(
        row["candidate_vp"] - row["opponent_vp"] for row in games
    ) if games else 0.0
    return {
        "macro_win_rate": macro_win_rate,
        "mean_win_rate": sum(item["wins"] for item in by_opponent.values()) / len(games) if games else 0.0,
        "mean_vp_margin": mean_margin,
        "completed": sum(row["status"] == "completed" for row in games),
        "capped": sum(row["status"] == "capped" for row in games),
        "errors": sum(row["status"] == "error" for row in games),
        "by_opponent": by_opponent,
        "by_seat": {
            str(seat): {
                "wins": sum(row["seat"] == seat and row["won"] for row in games),
                "games": sum(row["seat"] == seat for row in games),
            }
            for seat in range(2)
        },
    }


def seed_win_rates(games: list[dict]) -> dict[int, float]:
    buckets: dict[int, list[bool]] = {}
    for game in games:
        buckets.setdefault(game["seed"], []).append(bool(game["won"]))
    return {seed: sum(values) / len(values) for seed, values in buckets.items()}


def paired_bootstrap(candidate: list[dict], incumbent: list[dict], seed: int,
                     repetitions: int = 10000) -> dict:
    candidate_by_seed = seed_win_rates(candidate)
    incumbent_by_seed = seed_win_rates(incumbent)
    shared = sorted(set(candidate_by_seed) & set(incumbent_by_seed))
    if not shared:
        raise ValueError("Paired bootstrap needs identical board seeds")
    differences = [candidate_by_seed[item] - incumbent_by_seed[item] for item in shared]
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        samples.append(sum(sample) / len(sample))
    samples.sort()
    low = samples[int(0.025 * repetitions)]
    high = samples[min(repetitions - 1, int(0.975 * repetitions))]
    return {
        "boards": len(shared),
        "mean_difference": statistics.fmean(differences),
        "board_cluster_bootstrap_95": [low, high],
        "promote": statistics.fmean(differences) > 0 and low > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    parser.add_argument("--opponents", default=",".join(STRONG_OPPONENTS))
    parser.add_argument("--trials", type=int, default=12,
                        help="Number of parameter mutations after the incumbent")
    parser.add_argument("--train-boards", type=int, default=8)
    parser.add_argument("--validation-boards", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=20271200)
    parser.add_argument("--validation-seed", type=int, default=20271300)
    parser.add_argument("--max-actions", type=int, default=15000)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--max-game-seconds", type=float, default=45.0)
    parser.add_argument("--budget-seconds", type=float, default=480.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=61016)
    args = parser.parse_args()

    opponent_names = tuple(part.strip() for part in args.opponents.split(",") if part.strip())
    if not opponent_names or any(name not in STRONG_OPPONENTS for name in opponent_names):
        parser.error(f"opponents must be selected from {','.join(STRONG_OPPONENTS)}")
    if args.trials < 1 or args.train_boards < 2 or args.validation_boards < 2:
        parser.error("trials must be positive and each seed split needs at least two boards")
    if min(args.max_actions, args.max_turns) < 1 or args.max_game_seconds <= 0 or args.budget_seconds <= 0:
        parser.error("game caps and budget must be positive")

    config_path = args.config.resolve(strict=True)
    baseline_config = json.loads(config_path.read_text())
    rules = baseline_config["rules"]
    if rules.get("players") != 2:
        parser.error("Training requires rules.players=2")
    train_seeds = tuple(args.train_seed + offset for offset in range(args.train_boards))
    validation_seeds = tuple(args.validation_seed + offset for offset in range(args.validation_boards))
    if set(train_seeds) & set(validation_seeds):
        parser.error("training and validation seed ranges must be disjoint")

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    output_dir = args.output_dir or ROOT / "runs" / f"evolution-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    engine = source_info()
    sys.path.insert(0, engine["python_root"])
    install_v5_protocol()
    source_path = Path(__file__).resolve()
    report = {
        "protocol": f"{PROTOCOL}-1v1-evolution-v1",
        "engine": engine,
        "candidate_source": str(source_path),
        "candidate_source_sha256": sha256_bytes(source_path.read_bytes()),
        "baseline_config": str(config_path),
        "baseline_config_sha256": sha256_bytes(config_path.read_bytes()),
        "rules": rules,
        "opponents": list(opponent_names),
        "train_seeds": list(train_seeds),
        "validation_seeds": list(validation_seeds),
        "objective": "macro-average win rate across opponents, both seats",
        "training": {
            "algorithm": "seeded evolutionary parameter search over value, robber, dev-card, and tactical-scan parameters",
            "trials": args.trials,
            "train_boards": args.train_boards,
            "validation_boards": args.validation_boards,
            "seed": args.seed,
            "wall_budget_seconds": args.budget_seconds,
            "game_caps": {
                "max_actions": args.max_actions,
                "max_turns": args.max_turns,
                "max_game_seconds": args.max_game_seconds,
            },
        },
        "trials": [],
    }

    run_started = time.perf_counter()
    cache: dict[str, dict] = {}

    def score(config: dict, seeds: tuple[int, ...], label: str):
        digest = config_hash(config)
        cache_key = f"{label}:{digest}"
        if cache_key not in cache:
            games, latencies = play_batch(
                config, opponent_names, seeds, rules,
                args.max_actions, args.max_turns, args.max_game_seconds,
            )
            cache[cache_key] = {
                "games": games,
                "latencies": latencies,
                "metrics": metrics(games),
            }
        return cache[cache_key]

    incumbent_score = score(baseline_config, train_seeds, "train")
    population = [(baseline_config, incumbent_score["metrics"], "baseline")]
    report["trials"].append({
        "index": 0,
        "label": "baseline",
        "config_sha256": config_hash(baseline_config),
        "metrics": incumbent_score["metrics"],
    })
    elites = [baseline_config]
    for index in range(1, args.trials + 1):
        if time.perf_counter() - run_started >= args.budget_seconds * 0.6:
            report["training"]["stopped_early"] = "preserve budget for held-out validation"
            break
        parent = elites[rng.randrange(min(len(elites), 4))]
        candidate_config = mutate(parent, rng)
        if candidate_config == parent:
            candidate_config = mutate(baseline_config, rng)
        candidate_score = score(candidate_config, train_seeds, "train")
        candidate_metrics = candidate_score["metrics"]
        population.append((candidate_config, candidate_metrics, f"mutation_{index}"))
        report["trials"].append({
            "index": index,
            "label": f"mutation_{index}",
            "parent_config_sha256": config_hash(parent),
            "config_sha256": config_hash(candidate_config),
            "config": candidate_config,
            "metrics": candidate_metrics,
            "candidate_latency_ms": {
                "median": statistics.median(candidate_score["latencies"])
                if candidate_score["latencies"] else None,
                "p95": sorted(candidate_score["latencies"])[
                    min(len(candidate_score["latencies"]) - 1,
                        math.ceil(0.95 * len(candidate_score["latencies"])) - 1)
                ] if candidate_score["latencies"] else None,
            },
            "elapsed_seconds": round(time.perf_counter() - run_started, 3),
        })
        ranked = sorted(
            population,
            key=lambda item: (
                item[1]["macro_win_rate"],
                -item[1]["errors"],
                -item[1]["capped"],
                item[1]["mean_vp_margin"],
            ),
            reverse=True,
        )
        elites = [item[0] for item in ranked[:4]]
        print(
            f"trial {index}/{args.trials}: {candidate_metrics['macro_win_rate']:.3f} macro win rate; "
            f"best {ranked[0][1]['macro_win_rate']:.3f}; "
            f"{time.perf_counter() - run_started:.1f}s",
            flush=True,
        )

    ranked = sorted(
        population,
        key=lambda item: (
            item[1]["macro_win_rate"],
            -item[1]["errors"],
            -item[1]["capped"],
            item[1]["mean_vp_margin"],
        ),
        reverse=True,
    )
    selected_config, selected_train_metrics, selected_label = ranked[0]
    if time.perf_counter() - run_started >= args.budget_seconds * 0.85:
        selected_config, selected_train_metrics, selected_label = population[0]
        report["training"]["stopped_early"] = "validation budget reached; retained baseline"

    validation_baseline = score(baseline_config, validation_seeds, "validation")
    validation_selected = score(selected_config, validation_seeds, "validation")
    paired = paired_bootstrap(
        validation_selected["games"], validation_baseline["games"],
        seed=args.seed + 99,
    )
    promoted = bool(
        paired["promote"]
        and validation_selected["metrics"]["errors"] == 0
        and validation_selected["metrics"]["capped"] == 0
    )

    (output_dir / "selected_candidate.json").write_text(
        json.dumps(selected_config, indent=2) + "\n"
    )
    (output_dir / "validation_selected.json").write_text(
        json.dumps(validation_selected["games"], indent=2) + "\n"
    )
    (output_dir / "validation_baseline.json").write_text(
        json.dumps(validation_baseline["games"], indent=2) + "\n"
    )
    if promoted:
        trained_path = ROOT / "configs/trained.json"
        trained_path.write_text(json.dumps(selected_config, indent=2) + "\n")
        report["promoted_config"] = str(trained_path)
        report["promotion_decision"] = "promoted: paired held-out board interval above zero"
    else:
        report["promoted_config"] = None
        report["promotion_decision"] = "not promoted: held-out win-rate evidence did not clear the gate"

    report["selection"] = {
        "label": selected_label,
        "config_sha256": config_hash(selected_config),
        "training_metrics": selected_train_metrics,
        "validation_baseline_metrics": validation_baseline["metrics"],
        "validation_selected_metrics": validation_selected["metrics"],
        "paired_win_rate_difference": paired,
        "promoted": promoted,
        "promoted_config": report["promoted_config"],
    }
    report["elapsed_seconds"] = round(time.perf_counter() - run_started, 3)
    report["training"]["completed_trials"] = len(report["trials"]) - 1
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "event": "training_complete",
        "elapsed_seconds": report["elapsed_seconds"],
        "selected_train_macro_win_rate": selected_train_metrics["macro_win_rate"],
        "validation_baseline_macro_win_rate": validation_baseline["metrics"]["macro_win_rate"],
        "validation_selected_macro_win_rate": validation_selected["metrics"]["macro_win_rate"],
        "paired_difference": paired,
        "promoted": promoted,
        "report": str(output_dir / "training_report.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
