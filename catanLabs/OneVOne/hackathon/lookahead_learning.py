"""Warm a leaf-value MLP, learn from paired plan continuations, then play fresh boards."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor/kit"))
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from catan_kit import _patch
from catan_kit.obs import masked_copy
from catanatron import Color, Game, RandomPlayer
from action_learning import alternatives, trace_summary
from advantage_learning import _Opponent, _rollout, _export_mlp
from catanLabs.OneVOne.learning import heuristic_fingerprint
from catanLabs.OneVOne.lookahead_value import VERSION, HIDDEN, LookaheadPlayer, select
from resource_experiments import paired_difference, write_json
import tune_then_learn as league

MARGINS = (0., .03, .06, .1)


def collect_board(seed, config, opponent_name, positions, repeats, model_path=None):
    _patch.apply()
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=seed)
    random.seed(seed ^ 0x27483)
    focal = Color.RED if seed % 2 == 0 else Color.BLUE
    enemy = next(c for c in game.state.colors if c != focal)
    opponent = _Opponent(opponent_name, enemy, seed ^ 0x8A5)
    player = LookaheadPlayer(focal, config, model_path)
    rng = random.Random(seed)
    rows, last_turn, attempted, failures = [], -20, 0, 0
    for step in range(6000):
        if game.winning_color() is not None or game.state.num_turns >= 1000:
            break
        actions = list(game.playable_actions)
        color, turn = game.state.current_color(), game.state.num_turns
        prefix = f"P{game.state.color_to_index[focal]}_"
        if color == focal and len(actions) > 1:
            before = random.getstate()
            world = masked_copy(game, focal, random.Random(seed * 61 + step))
            try:
                eligible = (len(rows) < positions and 8 <= turn <= 100 and turn - last_turn >= 8
                            and game.state.current_prompt.name == "PLAY_TURN"
                            and game.state.player_state[prefix + "HAS_ROLLED"]
                            and len({a.action_type for a in actions}) > 1)
                reply_used_before = player.reply_used
                action = player.decide(world, actions, capture=eligible)
                eligible = eligible and not player.plan.winning.any()
                if eligible:
                    plan, options = player.plan, player.options
                    baseline = int(np.argmax(plan.heuristic))
                    roots, _ = alternatives(plan, 3, rng)
                    selected = [max((j for j, o in enumerate(options) if o["root"] == i),
                                    key=lambda j: options[j]["score"]) for i in roots]
                    choices_by_margin, reply_states = {}, {}
                    if player.model is not None:
                        for margin in MARGINS:
                            proposed = LookaheadPlayer(focal, config, model_path)
                            proposed.model["gate"]["margin"] = margin
                            proposed.reply_used = reply_used_before
                            proposed.decide(world, actions, capture=False)
                            index = proposed.last_decision["executed_index"]
                            choices_by_margin[str(margin)] = index
                            reply_states[margin] = proposed.reply_used
                            selected.append(next(j for j, o in enumerate(options) if o["root"] == index))
                    selected = list(dict.fromkeys(selected))
                    trials = []
                    scores = np.asarray([o["score"] for o in options])
                    score_scale = max(1., float(np.median(np.abs(scores - np.median(scores)))))
                    for j in selected:
                        option = options[j]
                        relative = np.clip((option["score"] - np.median(scores)) / score_scale, -4, 4)
                        teacher = .5 + .2 * float(plan.heuristic[option["root"]]) + .05 * float(relative)
                        for margin in MARGINS if model_path else (None,):
                            outcomes = [_rollout(game, plan.actions[option["root"]], focal, config,
                                                 seed * 100003 + turn * 103 + k, opponent_name,
                                                 lookahead=True, model_path=model_path, gate_margin=margin,
                                                 reply_used=reply_states.get(margin, player.reply_used)) for k in range(repeats)]
                            attempted += repeats
                            failures += sum(t["outcome"] is None for t in outcomes)
                            complete = all(t["outcome"] is not None for t in outcomes)
                            trials.append({"option_index": j, "root": option["root"], "margin": margin,
                                           "features": option["features"].tolist(),
                                           "line": [[a.action_type.name, repr(a.value)] for a in option["line"]],
                                           "execution": "First action, then replan with the deployed policy",
                                           "teacher": float(np.clip(teacher, 0, 1)),
                                           "target": float(np.mean([t["outcome"] for t in outcomes])) if complete else None,
                                           "trials": outcomes})
                    rows.append({"board_seed": seed, "turn": turn, "opponent": opponent_name,
                                 "baseline": baseline, "heuristic": plan.heuristic.tolist(),
                                 "choices_by_margin": choices_by_margin,
                                 "action_types": [a.action_type.name for a in actions],
                                 "options": [{"root": o["root"], "features": o["features"].tolist()} for o in options],
                                 "trials": trials})
                    last_turn = turn
            finally:
                random.setstate(before)
        elif color == focal:
            action = actions[0]
        else:
            action = opponent.choose(game, actions, seed * 61 + step)
        game.execute(action)
    return {"board_seed": seed, "positions": rows, "attempted": attempted, "failures": failures,
            "winner": None if game.winning_color() is None else game.winning_color().value}


def collect(output, seeds, config, workers, positions, repeats, model_path=None):
    output.mkdir()
    rows, attempted, failures = [], 0, 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(collect_board, seed, config, ("F", "Nano", "OneVOne")[i % 3],
                               positions, repeats, model_path) for i, seed in enumerate(seeds)]
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            write_json(output / f"board_{result['board_seed']}.json", result)
            rows.extend(result["positions"])
            attempted += result["attempted"]
            failures += result["failures"]
            print(f"{output.name}: {n}/{len(seeds)} boards; {attempted-failures}/{attempted} rollouts", flush=True)
    rows.sort(key=lambda r: (r["board_seed"], r["turn"]))
    report = {"positions": rows, "rollouts": attempted, "failures": failures}
    write_json(output / "data.json", report)
    return report


def fit(rows, config):
    from sklearn.neural_network import MLPRegressor
    flat = [dict(trial, board_seed=row["board_seed"]) for row in rows for trial in row["trials"] if trial["target"] is not None]
    boards = sorted({r["board_seed"] for r in flat})
    if len(boards) < 4:
        raise ValueError("Need at least four training boards")
    validation = set(boards[-max(1, len(boards)//4):])
    training = [r for r in flat if r["board_seed"] not in validation]
    valid = [r for r in flat if r["board_seed"] in validation]
    scale = np.maximum(np.std([r["features"] for r in training], axis=0), .05)
    vx = np.asarray([r["features"] for r in valid]) / scale
    vy = np.asarray([r["target"] for r in valid])
    train_boards = sorted(set(boards) - validation)
    members, losses = [], []
    for member in range(3):
        rng = random.Random(7341 + member)
        boot = rng.choices(train_boards, k=len(train_boards))
        batch = [r for board in boot for r in training if r["board_seed"] == board]
        x = np.asarray([r["features"] for r in batch]) / scale
        y = np.asarray([r["target"] for r in batch])
        teacher = np.asarray([r["teacher"] for r in batch])
        network = MLPRegressor(hidden_layer_sizes=(HIDDEN,), activation="tanh", alpha=.1,
                               batch_size=min(64, len(batch)), learning_rate_init=.003,
                               random_state=173 + member)
        for _ in range(20):
            network.partial_fit(x, teacher)
        best, weights, stale = float("inf"), None, 0
        for epoch in range(150):
            network.partial_fit(x, y)
            if epoch % 5 == 0:
                loss = float(np.mean((np.clip(network.predict(vx), 0, 1) - vy) ** 2))
                if loss < best:
                    best, stale = loss, 0
                    weights = ([v.copy() for v in network.coefs_], [v.copy() for v in network.intercepts_])
                else:
                    stale += 1
                if stale >= 8:
                    break
        network.coefs_, network.intercepts_ = weights
        members.append(_export_mlp(network))
        losses.append(best)
    return {"version": VERSION, "config_sha256": heuristic_fingerprint(config),
            "scale": scale.tolist(), "members": members,
            "gate": {"enabled": True, "z": 1., "margin": 0.}}, {
                "training_boards": sorted(set(boards) - validation), "validation_boards": sorted(validation),
                "training_choices": len(training), "validation_choices": len(valid),
                "validation_mse": losses, "warm_epochs": 20,
                "objective": "Terminal win fractions from paired, legally executed plan continuations"}


def calibrate(model, rows):
    trials = []
    for margin in MARGINS:
        model["gate"]["margin"] = margin
        effects, overrides, changed_boards, unmeasured = {}, 0, set(), 0
        for row in rows:
            plan = SimpleNamespace(heuristic=np.asarray(row["heuristic"]),
                                   actions=[SimpleNamespace(action_type=t) for t in row["action_types"]],
                                   winning=np.zeros(len(row["heuristic"]), dtype=bool))
            index, diagnostic = select(plan, row["options"], model)
            index = row.get("choices_by_margin", {}).get(str(margin), index)
            effect = 0.
            if index != row["baseline"]:
                chosen = next((t for t in row["trials"] if t["root"] == index and t.get("margin", margin) == margin), None)
                baseline = next((t for t in row["trials"] if t["root"] == row["baseline"] and t.get("margin", margin) == margin), None)
                if chosen is not None and baseline is not None and chosen["target"] is not None and baseline["target"] is not None:
                    effect = chosen["target"] - baseline["target"]
                    overrides += 1
                    changed_boards.add(row["board_seed"])
                else:
                    unmeasured += 1
            effects.setdefault(row["board_seed"], []).append(effect)
        means = [float(np.mean(v)) for v in effects.values()]
        rng = random.Random(17)
        samples = sorted(np.mean(rng.choices(means, k=len(means))) for _ in range(2000)) if means else []
        lower = float(samples[50]) if samples else None
        trials.append({"margin": margin, "overrides": overrides, "boards": len(changed_boards), "unmeasured_overrides": unmeasured,
                       "mean_gain": float(np.mean(means)) if means else 0., "board_bootstrap_lower": lower,
                       "accepted": not unmeasured and overrides >= 6 and len(changed_boards) >= 3 and lower is not None and lower > 0})
    accepted = next((r for r in trials if r["accepted"]), None)
    model["gate"].update(enabled=accepted is not None, margin=accepted["margin"] if accepted else 0.)
    return {"trials": trials, "accepted": accepted is not None}


def profile(output, name, model=None):
    source = Path(os.path.relpath(HERE, output)).as_posix()
    path = output / f"{name}_agent.py"
    path.write_text("from pathlib import Path\nimport sys\n"
                    f"sys.path.insert(0,str((Path(__file__).parent/{source!r}).resolve()))\n"
                    "from lookahead_agent import Agent as Base\n"
                    "class Agent(Base):\n    def __init__(self):\n"
                    "        directory=Path(__file__).parent\n"
                    f"        super().__init__(directory/'config.json', model=directory/{model!r} if {model!r} else None, trace_path=directory/'traces'/{name!r})\n")
    return path


def main():
    _patch.ensure_hash_seed()
    _patch.apply()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-boards", type=int, default=12)
    parser.add_argument("--calibration-boards", type=int, default=6)
    parser.add_argument("--positions", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--beam-width", type=int, default=4, choices=range(1, 17))
    parser.add_argument("--reply-check", action="store_true")
    parser.add_argument("--seed", type=int, default=2033000000)
    args = parser.parse_args()
    if min(args.workers, args.positions, args.repeats) < 1 or min(args.train_boards, args.calibration_boards) < 4 or args.games < 6 or args.games % 2:
        parser.error("Use positive counts, >=4 boards per split and even games >=6")
    if max(args.train_boards, args.calibration_boards, args.games) > 1000:
        parser.error("Counts exceed the separate seed ranges")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["ONEVONE_UNCLOCKED"] = "0"
    config = {"planner": "nano", "nano": {"search_depth": 4, "beam_width": args.beam_width}}
    if args.reply_check:
        config["reply_check"] = True
    write_json(output / "config.json", config)
    sources = [HERE.parent / "lookahead_value.py", HERE.parent / "nano_planner.py",
               HERE.parent / "reply_search.py",
               HERE / "advantage_learning.py", HERE / "lookahead_agent.py", Path(__file__),
               HERE.parents[2] / "my_team/nano.py", HERE / "action_learning.py",
               HERE / "critic_experiment.py", HERE / "tune_then_learn.py", HERE / "agent.py",
               HERE.parent / "learning.py", HERE.parent / "action_planner.py", HERE.parent / "agent.py",
               HERE.parent / "configs/default.json", HERE / "vendor/kit/examples/f_agent.py"]
    hashes = {str(p.relative_to(HERE.parents[2])): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(output / "protocol.json", {"arguments": vars(args) | {"output": str(output)},
               "train_seed0": args.seed, "calibration_seed0": args.seed + 10000, "eval_seed0": args.seed + 20000,
               "source_sha256": hashes, "clock": "Unclocked training; official-clock matches", "opponents": ["F", "Nano", "OneVOne"],
               "scope": "First action then deployed replanning; all legal roots and depth-four search"})
    started = time.time()
    train = collect(output / "training", list(range(args.seed, args.seed + args.train_boards)),
                    config, args.workers, args.positions, args.repeats)
    candidate, fitting = fit(train["positions"], config)
    write_json(output / "candidate_model.json", candidate)
    calibration = collect(output / "calibration", list(range(args.seed + 10000, args.seed + 10000 + args.calibration_boards)),
                          config, args.workers, args.positions, args.repeats, str(output / "candidate_model.json"))
    safe = json.loads(json.dumps(candidate))
    gate = calibrate(safe, calibration["positions"])
    write_json(output / "model.json", safe)
    write_json(output / "training.json", {"fitting": fitting, "calibration": gate,
               "training_rollouts": train["rollouts"], "calibration_rollouts": calibration["rollouts"],
               "rollout_failures": train["failures"] + calibration["failures"]})
    league.evaluation.WORKERS = args.workers
    (output / "audits").mkdir()
    os.environ["ONEVONE_AUDIT_DIR"] = str(output / "audits")
    results = {}
    for name, model in (("heuristic", None), ("candidate", "model.json")):
        os.environ["ONEVONE_AUDIT_ARM"] = name
        path = profile(output, name, model)
        league.run_mixed(output, name + "_smoke", path, 6, args.seed + 19000)
        results[name] = league.run_mixed(output, name, path, args.games, args.seed + 20000)
    changes = {name: trace_summary(output, name) for name in results}
    assert all(not row["unmatched_proposals"] and not row["rejected_proposals"] for row in changes.values())
    assert all(hashlib.sha256((HERE.parents[2] / p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    report = {"summaries": {k: v["summary"] for k, v in results.items()},
              "candidate_minus_heuristic": paired_difference(results["candidate"], results["heuristic"]),
              "changes": changes, "calibrated_gate_enabled": safe["gate"]["enabled"],
              "default_changed": False, "elapsed_s": time.time() - started,
              "note": "Only the calibrated model plays; rejected models retain Nano choices. No automatic promotion."}
    profile(output, "calibrated", "model.json")
    write_json(output / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
