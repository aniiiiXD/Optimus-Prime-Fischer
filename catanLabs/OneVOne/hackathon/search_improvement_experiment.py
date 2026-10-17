"""Measure wider search, next-turn checks, and corrected ML against frozen Nano."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import lookahead_learning as learning
from catan_kit import match
from catan_kit.evaluate import summarise
from resource_experiments import paired_difference, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def match_job(job):
    agent, opponent, indices, seed = job
    records, _ = match.play_chunk(str(agent), str(opponent),
                                  [seed + i // 2 for i in indices], indices)
    return records


def faults(result):
    return any(result["summary"][key] for key in
               ("errors", "timeouts", "crashes", "games_over_budget", "random_moves"))


def profile(output, name, config, model=None):
    write_json(output / f"{name}_config.json", config)
    path = output / f"{name}_agent.py"
    path.write_text("from pathlib import Path\nimport sys\n"
                    f"sys.path.insert(0, {str(HERE)!r})\n"
                    "from lookahead_agent import Agent as Base\n"
                    "class Agent(Base):\n    def __init__(self):\n"
                    "        directory=Path(__file__).parent\n"
                    f"        super().__init__(directory/{(name + '_config.json')!r}, "
                    f"model={str(model) if model else None!r}, trace_path=directory/'traces'/{name!r})\n")
    return path


def stage(pool, output, label, agents, opponents, games, seed):
    directory = output / label
    directory.mkdir()
    records = {name: [] for name in agents}
    jobs = {}
    for name, agent in agents.items():
        for oi, (opponent_name, opponent) in enumerate(opponents.items()):
            for start in range(0, games, 10):
                indices = list(range(start, min(start + 10, games)))
                future = pool.submit(match_job, (agent, opponent, indices, seed + oi * 1000))
                jobs[future] = name, opponent_name
    for future in as_completed(jobs):
        name, opponent = jobs[future]
        records[name].extend(dict(row, opponent=opponent) for row in future.result())
        print(f"{label} {name}: {len(records[name])}/{games * len(opponents)} games", flush=True)
        write_json(directory / f"{name}_checkpoint.json", records[name])
    results = {}
    for name, rows in records.items():
        rows.sort(key=lambda row: (row["opponent"], row["game"]))
        results[name] = {"summary": summarise(rows), "opponent_summary": summarise(rows, "b"), "records": rows,
                         "opponents": {opponent: summarise([r for r in rows if r["opponent"] == opponent])
                                       for opponent in opponents}}
        write_json(directory / f"{name}.json", results[name])
    for name, result in results.items():
        if name != "nano":
            result["difference"] = paired_difference(result, results["nano"])
    write_json(directory / "results.json", results)
    return results


def main():
    learning._patch.ensure_hash_seed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games", type=int, default=20, help="Games per opponent in each screen")
    parser.add_argument("--seed", type=int, default=2033200000)
    args = parser.parse_args()
    if args.workers < 1 or args.games < 2 or args.games % 2:
        parser.error("Positive workers and even games >=2 required")
    output = args.output.resolve()
    if (output / "protocol.json").exists():
        parser.error("Choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "before/my_team/nano.py"
    if not frozen.exists():
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "my_team/nano.py", frozen)
    os.environ["ONEVONE_UNCLOCKED"] = "0"
    os.environ.pop("ONEVONE_AUDIT_DIR", None)
    sources = list((ROOT / "my_team").glob("*.py"))
    for directory in (HERE.parent, HERE, HERE.parent / "configs", HERE / "vendor/kit/catan_kit", HERE / "vendor/kit/examples"):
        sources.extend(directory.glob("*.py"))
        sources.extend(directory.glob("*.json"))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for name in hashes:
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    opponents = {"F": HERE / "vendor/kit/examples/f_agent.py", "OneVOne": HERE / "agent.py"}
    base = {"planner": "nano", "nano": {"search_depth": 4, "beam_width": 4}}
    configs = {"adapter": base,
               "wide": dict(base, nano={"search_depth": 4, "beam_width": 8}),
               "reply": dict(base, reply_check=True),
               "wide_reply": dict(base, nano={"search_depth": 4, "beam_width": 8}, reply_check=True)}
    agents = {name: profile(output, name, config) for name, config in configs.items()}
    control = output / "nano_agent.py"
    control.write_text("from pathlib import Path\nimport importlib.util\n"
                       "spec=importlib.util.spec_from_file_location('frozen_control',Path(__file__).parent/'before/my_team/nano.py')\n"
                       "module=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\n"
                       "class Agent(module.Agent):\n    def __init__(self):\n        super().__init__({'search_depth':4})\n")
    agents = {"nano": control, **agents}
    protocol = {"clock": "Official clock enforced for both seats; training rollouts unclocked",
                "seed_ranges": {"smoke": args.seed, "screen": args.seed + 10000,
                                "training": args.seed + 20000, "calibration": args.seed + 30000,
                                "ml_smoke": args.seed + 40000, "ml_screen": args.seed + 50000,
                                "confirmation": args.seed + 60000},
                "arguments": vars(args) | {"output": str(output)}, "configs": configs,
                "baseline": "Frozen pre-change Nano with search_depth=4",
                "source_sha256": hashes, "sol": "Browser unavailable; Sol not consulted"}
    write_json(output / "protocol.json", protocol)
    started = time.monotonic()
    report = {"promotion_supported": False}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        smoke = stage(pool, output, "smoke", agents, opponents, 2, args.seed)
        if any(smoke["nano"]["summary"][key] for key in ("errors", "timeouts", "crashes")):
            raise RuntimeError("Frozen Nano had an agent error; inspect smoke records")
        # Baseline clock overruns are comparison data; preserve them instead of dropping the control.
        playable = {name: agent for name, agent in agents.items() if name == "nano" or not faults(smoke[name])}
        screen = stage(pool, output, "screen", playable, opponents, args.games, args.seed + 10000)
        report["search"] = {name: {k: v for k, v in result.items() if k != "records"}
                            for name, result in screen.items()}
        write_json(output / "results.json", report)
    # The ML ablation retains Nano's depth and width, isolating the training repair.
    training = learning.collect(output / "training", list(range(args.seed + 20000, args.seed + 20016)),
                                base, args.workers, 2, 4)
    model, fitting = learning.fit(training["positions"], base)
    write_json(output / "candidate_model.json", model)
    calibration = learning.collect(output / "calibration", list(range(args.seed + 30000, args.seed + 30008)),
                                   base, args.workers, 2, 4, str(output / "candidate_model.json"))
    gate = learning.calibrate(model, calibration["positions"])
    write_json(output / "model.json", model)
    report["training"] = {"fitting": fitting, "gate": gate,
                          "training_rollouts": training["rollouts"],
                          "calibration_rollouts": calibration["rollouts"],
                          "failures": training["failures"] + calibration["failures"]}
    if report["training"]["failures"]:
        raise RuntimeError("Incomplete training rollouts; inspect checkpoints")
    ml_agent = profile(output, "ml", base, output / "model.json")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        ml_smoke = stage(pool, output, "ml_smoke", {"nano": control, "ml": ml_agent}, opponents, 2, args.seed + 40000)
        report["ml_smoke"] = {name: r["summary"] for name, r in ml_smoke.items()}
        if model["gate"]["enabled"] and not faults(ml_smoke["ml"]):
            ml_screen = stage(pool, output, "ml_screen", {"nano": control, "ml": ml_agent}, opponents, args.games, args.seed + 50000)
            report["ml_screen"] = {name: {k: v for k, v in r.items() if k != "records"} for name, r in ml_screen.items()}
            screen["ml"] = ml_screen["ml"]
            agents["ml"] = ml_agent
        promising = [name for name, result in screen.items() if name not in ("nano", "adapter")
                     and not faults(result) and result["difference"]["board_bootstrap_ci95"][0] > 0
                     and all(result["opponents"][opp]["wins"] >=
                             (ml_screen if name == "ml" else screen)["nano"]["opponents"][opp]["wins"]
                             for opp in opponents)]
        if promising:
            selected = max(promising, key=lambda name: screen[name]["difference"]["win_rate_difference"])
            confirmation = stage(pool, output, "confirmation", {"nano": control, selected: agents[selected]},
                                 opponents, 200, args.seed + 60000)
            report["confirmation"] = {name: {k: v for k, v in r.items() if k != "records"} for name, r in confirmation.items()}
            report["promotion_supported"] = bool(not faults(confirmation[selected]) and
                confirmation[selected]["difference"]["board_bootstrap_ci95"][0] > 0 and
                all(confirmation[selected]["opponents"][o]["wins"] >= confirmation["nano"]["opponents"][o]["wins"] for o in opponents))
            report["selected"] = selected
    report["wall_s"] = time.monotonic() - started
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    report["source_hashes_verified"] = True
    report["default_policy_changed"] = False
    write_json(output / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
