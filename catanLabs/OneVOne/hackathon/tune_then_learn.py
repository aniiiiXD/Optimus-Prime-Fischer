"""Tune a frozen heuristic, train an MLP league, and screen on untouched boards."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import action_learning as actions
import critic_experiment as evaluation
import numpy as np
from catanLabs.OneVOne.agent import DEFAULT_CONFIG, TRAINED_CONFIG
from catanLabs.OneVOne.learning import heuristic_fingerprint
from resource_experiments import paired_difference, write_json


HERE = Path(__file__).resolve().parent
DIMENSIONS = ("production", "expansion", "army", "early_wood_brick",
              "early_wheat", "early_ore", "sheep", "public_vp")
TUNE_SEED, VALIDATION_SEED = 2031500000, 2031600000
LEARN_SEED, TEST_SEED = 2031700000, 2031900000
ROOT = HERE.parents[2]
OPPONENTS = {"F": HERE / "vendor/kit/examples/f_agent.py",
             "Nano": ROOT / "my_team/nano.py", "OneVOne": HERE / "agent.py"}


def run_mixed(output, name, path, games, seed):
    records, snapshots, summaries = [], [], {}
    boards, extra = divmod(games // 2, len(OPPONENTS))
    offset = 0
    for index, (opponent, opponent_path) in enumerate(OPPONENTS.items()):
        count = 2 * (boards + (index < extra))
        if not count:
            continue
        result, _ = evaluation.run(output, f"{name}_{opponent}", path, count, seed,
                                    game_start=offset, opponent=opponent_path)
        records.extend(dict(row, opponent=opponent) for row in result["records"])
        snapshots.extend(result["early_building"])
        summaries[opponent] = result["summary"]
        offset += count
    from catan_kit.evaluate import summarise
    result = {"summary": summarise(records), "records": records,
              "early_building": snapshots, "opponents": summaries}
    write_json(output / f"{name}.json", result)
    print(f"{name}: {result['summary']['wins']}/{games} league wins", flush=True)
    return result


def priorities(base, vector):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (len(DIMENSIONS),) or not np.isfinite(vector).all():
        raise ValueError("Expected eight finite priority coordinates")
    config = copy.deepcopy(base)
    preferences = config["nano"]
    for name, coordinate in (("production_scale", 0), ("expansion_scale", 1),
                              ("army_scale", 2), ("vp_scale", 7)):
        preferences[name] = base["nano"][name] * math.exp(vector[coordinate])
    resources = np.asarray(base["nano"]["resource_weights"]).copy()
    resources[[0, 1]] *= math.exp(vector[3])
    resources[3] *= math.exp(vector[4])
    resources[4] *= math.exp(vector[5])
    resources[2] *= math.exp(vector[6])
    preferences["resource_weights"] = resources.tolist()
    return config


def profile(output, name, config_name, model_name=None, trace=False):
    source = Path(os.path.relpath(HERE, output)).as_posix()
    model = f"str(Path(__file__).parent / {model_name!r})" if model_name else "None"
    trace_path = f"str(Path(__file__).parent / 'traces' / {name!r})" if trace else "None"
    path = output / f"{name}_agent.py"
    path.write_text("from pathlib import Path\nimport sys\n"
                    f"sys.path.insert(0, str((Path(__file__).parent / {source!r}).resolve()))\n"
                    "from learning_agent import Agent as LearningAgent\n"
                    "class Agent(LearningAgent):\n    def __init__(self):\n"
                    f"        super().__init__(config_path=str(Path(__file__).parent / {config_name!r}), "
                    f"model={model}, trace_path={trace_path})\n")
    return path


def tune(output, base, args):
    import cma
    write_json(output / "base_config.json", base)
    incumbent = profile(output, "tuning_control", "base_config.json")
    native = run_mixed(output, "depth_native", incumbent, args.tune_games, TUNE_SEED - 1000)
    depth_config = copy.deepcopy(base)
    depth_config["nano"]["search_depth"] = 4
    write_json(output / "depth_four_config.json", depth_config)
    depth_path = profile(output, "depth_four", "depth_four_config.json")
    shallow = run_mixed(output, "depth_four", depth_path, args.tune_games, TUNE_SEED - 1000)
    depth_report = {"native": native["summary"], "depth_four": shallow["summary"], "accepted": False}
    if shallow["summary"]["wins"] > native["summary"]["wins"]:
        control = run_mixed(output, "depth_validation_native", incumbent,
                            args.validation_games, VALIDATION_SEED + 1000)
        candidate = run_mixed(output, "depth_validation_four", depth_path,
                              args.validation_games, VALIDATION_SEED + 1000)
        depth_report["validation"] = {"native": control["summary"], "depth_four": candidate["summary"]}
        depth_report["accepted"] = candidate["summary"]["wins"] > control["summary"]["wins"]
    if depth_report["accepted"]:
        base, incumbent = depth_config, depth_path
    write_json(output / "depth_screen.json", depth_report)
    selected, selected_config, history = incumbent, base, []
    optimizer = cma.CMAEvolutionStrategy([0.] * len(DIMENSIONS), .65, {
        "seed": 173, "popsize": args.population, "verbose": -9, "verb_log": 0,
        "bounds": [-2., 2.],
    })
    for generation in range(args.generations):
        seed = TUNE_SEED + 1000 * generation
        control = run_mixed(output, f"tuning_g{generation}_incumbent", selected,
                            args.tune_games, seed)
        vectors, scores, trials = optimizer.ask(), [], []
        for index, vector in enumerate(vectors):
            tag = f"tuning_g{generation}_c{index}"
            config = priorities(base, vector)
            write_json(output / f"{tag}_config.json", config)
            path = profile(output, tag, f"{tag}_config.json")
            result = run_mixed(output, tag, path, args.tune_games, seed)
            scores.append(-result["summary"]["win_rate"])
            trials.append({"vector": list(map(float, vector)), "config": config,
                           "agent": path.name, "summary": result["summary"]})
        optimizer.tell(vectors, scores)
        best = min(range(len(trials)), key=lambda i: scores[i])
        if trials[best]["summary"]["wins"] > control["summary"]["wins"]:
            selected = output / trials[best]["agent"]
            selected_config = trials[best]["config"]
        history.append({"generation": generation, "seed": seed,
                        "incumbent": control["summary"], "trials": trials,
                        "retained": selected.name})
        write_json(output / "tuning.json", {"dimensions": DIMENSIONS, "history": history})
    report = {"dimensions": DIMENSIONS, "history": history, "accepted": False}
    if selected != incumbent:
        baseline = run_mixed(output, "tuning_validation_control", incumbent,
                             args.validation_games, VALIDATION_SEED)
        candidate = run_mixed(output, "tuning_validation_candidate", selected,
                              args.validation_games, VALIDATION_SEED)
        report["validation"] = {"baseline": baseline["summary"], "candidate": candidate["summary"],
                                **paired_difference(candidate, baseline)}
        report["accepted"] = candidate["summary"]["wins"] > baseline["summary"]["wins"]
    if not report["accepted"]:
        selected_config = base
    report["reason"] = ("Practice selection improved wins on separate validation boards."
                        if report["accepted"] else "No fresh validation gain; retained the starting heuristic.")
    write_json(output / "tuning.json", report)
    write_json(output / "frozen_config.json", selected_config)
    return selected_config


def main():
    actions._patch.ensure_hash_seed()
    actions._patch.apply()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--tune-games", type=int, default=8)
    parser.add_argument("--validation-games", type=int, default=24)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--practice-boards", type=int, default=24)
    parser.add_argument("--calibration-boards", type=int, default=8)
    parser.add_argument("--positions", type=int, default=4)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Reuse frozen tuning and unfinished practice checkpoints")
    args = parser.parse_args()
    if (min(args.workers, args.generations, args.positions, args.rollouts) < 1
            or args.population < 4 or args.practice_boards <= args.calibration_boards
            or args.calibration_boards < 3
            or any(n < 6 or n % 2 for n in (args.games, args.tune_games, args.validation_games))):
        parser.error("Use positive counts, population >=4, >=3 calibration boards, and even game counts")
    if max(args.games, args.tune_games, args.validation_games, args.practice_boards) > 1000 or args.generations > 20:
        parser.error("Counts exceed the separate seed ranges")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    evaluation.WORKERS = args.workers
    os.environ["ONEVONE_UNCLOCKED"] = "1"
    os.environ.pop("ONEVONE_AUDIT_DIR", None)
    started = time.perf_counter()
    config_path = TRAINED_CONFIG if TRAINED_CONFIG.is_file() else DEFAULT_CONFIG
    base = {"planner": "nano", "nano": {"resource_weights": [.88, .83, .78, 1.20, 1.18],
            "production_scale": 1., "expansion_scale": 1., "army_scale": 1., "vp_scale": 1.}}
    source_paths = [HERE.parent / "learning.py", HERE.parent / "action_planner.py",
                    HERE / "learning_agent.py", HERE / "action_learning.py", Path(__file__),
                    HERE / "advantage_learning.py", HERE / "critic_experiment.py", config_path,
                    ROOT / "my_team/nano.py", HERE.parent / "nano_planner.py"]
    hashes = {str(p.relative_to(HERE.parents[2])): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    if args.resume:
        previous = json.loads((output / "protocol.json").read_text())
        if not (output / "frozen_config.json").exists() or (output / "audits").exists():
            parser.error("Resume requires frozen tuning and an evaluation that has not started")
        for name, digest in previous["source_sha256"].items():
            if name not in (str(Path(__file__).relative_to(ROOT)), str((HERE / "advantage_learning.py").relative_to(ROOT))):
                if hashes.get(name) != digest:
                    parser.error(f"Policy changed since tuning: {name}; choose a fresh output")
        for name, value in previous["arguments"].items():
            if name not in ("output", "resume") and getattr(args, name) != value:
                parser.error(f"Resume settings differ: {name}")
        if not (output / "protocol_before_resume.json").exists():
            write_json(output / "protocol_before_resume.json", previous)
    protocol = {"opponents": list(OPPONENTS), "clock": "unclocked", "both_seats": True,
                "baseline": "Nano full purchase planner",
                "model": "MLP advantage ensemble; tanh hidden layer of 32 units",
                "tuning_seed": TUNE_SEED, "tuning_validation_seed": VALIDATION_SEED,
                "learning_seed": LEARN_SEED, "fresh_test_seed": TEST_SEED,
                "arguments": vars(args) | {"output": str(output)}, "source_sha256": hashes,
                "policy": "Identical all-legal-root planner, search depth and budgets; only scoring differs."}
    protocol["tuning_reused"] = args.resume
    protocol["wall_time_scope"] = "Current invocation; resumed runs exclude completed tuning"
    write_json(output / "protocol.json", protocol)
    config = json.loads((output / "frozen_config.json").read_text()) if args.resume else tune(output, base, args)
    from advantage_learning import collect, fit_advantage
    seeds = list(range(LEARN_SEED, LEARN_SEED + args.practice_boards))
    calibration = set(seeds[-args.calibration_boards:])
    train_seeds = [seed for seed in seeds if seed not in calibration]
    train = collect(output / "practice_train", config, train_seeds, workers=args.workers,
                    positions=args.positions, choices=3, repeats=args.rollouts)
    warm_model, _ = fit_advantage(train, set(train_seeds), set(), config)
    # Freeze the ensemble before sampling its actual all-root choices on held-out boards.
    heldout = collect(output / "practice_calibration", config, sorted(calibration),
                      workers=args.workers, positions=args.positions, choices=3,
                      repeats=args.rollouts, candidate_model=warm_model)
    artifact = {key: train[key] + heldout[key] for key in ("positions", "pairs", "practice_games")}
    model, training = fit_advantage(artifact, set(seeds) - calibration, calibration, config)
    if model["members"] != warm_model["members"] or model["feature_scale"] != warm_model["feature_scale"]:
        raise RuntimeError("Calibration changed the frozen ensemble")
    training["collection"] = {"train": train["collection"], "calibration": heldout["collection"]}
    assert model["heuristic_sha256"] == heuristic_fingerprint(config)
    write_json(output / "advantage_model.json", model)
    write_json(output / "learning.json", training)
    for path in source_paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(HERE.parents[2]))]:
            raise RuntimeError(f"Source changed during practice: {path}")
    (output / "audits").mkdir()
    os.environ["ONEVONE_AUDIT_DIR"] = str(output / "audits")
    nano_path = output / "nano_agent.py"
    nano_source = Path(os.path.relpath(ROOT, output)).as_posix()
    nano_path.write_text("from pathlib import Path\nimport sys\n"
                         f"sys.path.insert(0, str((Path(__file__).parent / {nano_source!r}).resolve()))\n"
                         "from my_team.nano import Agent\n")
    arms = {"nano": nano_path,
            "heuristic": profile(output, "heuristic", "frozen_config.json", trace=True),
            "learned": profile(output, "learned", "frozen_config.json", "advantage_model.json", trace=True)}
    results, records = {}, {}
    for name, path in arms.items():
        os.environ["ONEVONE_AUDIT_ARM"] = name
        tick = time.perf_counter()
        result = run_mixed(output, name, path, args.games, TEST_SEED)
        records[name] = result
        results[name] = {"summary": result["summary"], "opponents": result["opponents"],
                         "trace": actions.trace_summary(output, name),
                         "wall_s": time.perf_counter() - tick}
        if name != "nano":
            results[name]["versus_baseline"] = paired_difference(result, records["nano"])
        if name == "learned":
            results[name]["versus_heuristic"] = paired_difference(result, records["heuristic"])
        write_json(output / "results.json", {"arms": results, "wall_s": time.perf_counter() - started})
    confirmed = results["learned"]["versus_heuristic"]["board_bootstrap_ci95"][0] > 0
    recommended = "nano_agent.py"
    if results["heuristic"]["versus_baseline"]["board_bootstrap_ci95"][0] > 0:
        recommended = "heuristic_agent.py"
    if confirmed and results["learned"]["versus_baseline"]["board_bootstrap_ci95"][0] > 0:
        recommended = "learned_agent.py"
    write_json(output / "decision.json", {"ml_improvement_confirmed": confirmed,
               "ml_gate_enabled": model["gate"]["enabled"], "default_changed": False,
               "recommended_experimental_profile": recommended,
               "note": "Experimental profiles and all evidence retained; defaults require stronger confirmation."})
    print(json.dumps({"results": results, "gate": model["gate"],
                      "ml_improvement_confirmed": confirmed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
