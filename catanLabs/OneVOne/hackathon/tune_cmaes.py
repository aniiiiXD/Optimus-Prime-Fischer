"""CMA-ES priority search; accept only a win-rate gain confirmed on fresh boards."""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import critic_experiment as experiment
from resource_experiments import paired_difference, write_json


def priorities(vector):
    if len(vector) != 4 or any(not math.isfinite(v) for v in vector):
        raise ValueError("Expected four finite CMA-ES coordinates")
    production, expansion, army, vp = map(float, vector)
    return {
        "production": 1e8 * math.exp(production),
        "reachable_production_0": 1e5 * math.exp(expansion),
        "reachable_production_1": 3e4 * math.exp(expansion),
        "buildable_nodes": 1e4 * math.exp(expansion),
        "army_size": 3e4 * math.exp(army),
        "public_vps": 1e8 * 10 ** vp,
    }


def profile(output, name, weights=None):
    path = output / f"{name}.py"
    path.write_text(
        "import sys\n" + f"sys.path.insert(0, {str(experiment.HERE)!r})\n"
        "from critic_agent import Agent as HeuristicAgent\n"
        "class Agent(HeuristicAgent):\n    def __init__(self):\n"
        f"        super().__init__(value_weights={weights!r})\n"
    )
    return path


def main():
    experiment.ensure_hash_seed()
    import cma
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--validation-games", type=int, default=24)
    parser.add_argument("--confirm-games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=experiment.HERE.parent / "reports/cmaes_20261017")
    args = parser.parse_args()
    if min(args.generations, args.games, args.validation_games, args.confirm_games, args.workers) < 1 or args.population < 4 or any(n % 2 for n in (args.games, args.validation_games, args.confirm_games)):
        parser.error("Positive counts, population >= 4, and even game counts required")
    if args.games * args.generations > 10000 or args.validation_games > 10000:
        parser.error("Use smaller batches to preserve disjoint train/validation/confirmation seed ranges")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    experiment.WORKERS = args.workers
    control = profile(output, "control")
    manifest = {
        "method": "CMA-ES/pycma", "cma_version": importlib.metadata.version("cma"),
        "source": "https://cma-es.github.io/index.html", "opponent": "official F",
        "dimensions": ["log production", "log expansion", "log army", "log10 VP / 1e8"],
        "search_budget": {"tactical_roots": 8, "tactical_followups": 16},
        "training_seed0": 1200000, "validation_seed0": 1220000, "confirmation_seed0": 1240000,
        "starting_point": "Default production/expansion/army; economy-scaled VP seed of 3e8. Original 3e14 VP control is retained separately.",
        "objective": "Win rate first; CPU overruns break ties", "arguments": vars(args) | {"output": str(output)}, "generations": [],
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), experiment.HERE / "critic_agent.py", experiment.HERE / "critic_experiment.py")},
    }
    write_json(output / "manifest.json", manifest)
    optimizer = cma.CMAEvolutionStrategy([0, 0, 0, math.log10(3)], 0.6, {
        "seed": 17, "popsize": args.population, "bounds": [[-1.5, -2, -2, -1], [1.5, 2, 2, 7]],
        "CMA_stds": [1, 1, 1, 2], "verbose": -9, "verb_log": 0,
    })
    best_path, best_weights = control, None
    for generation in range(args.generations):
        seed = 1200000 + generation * args.games // 2
        incumbent, _ = experiment.run(output, f"g{generation}_incumbent", best_path, args.games, seed)
        vectors, scores, trials = optimizer.ask(), [], []
        for i, vector in enumerate(vectors):
            weights = priorities(vector)
            path = profile(output, f"g{generation}_candidate{i}", weights)
            result, _ = experiment.run(output, f"g{generation}_candidate{i}", path, args.games, seed)
            s = result["summary"]
            scores.append(-s["win_rate"] + 1e-4 * s["games_over_budget"] / s["games"])
            trials.append({"vector": list(map(float, vector)), "weights": weights, "summary": s, "agent": str(path)})
        optimizer.tell(vectors, scores)
        chosen = min(range(len(scores)), key=lambda i: scores[i])
        if trials[chosen]["summary"]["wins"] > incumbent["summary"]["wins"]:
            best_path, best_weights = Path(trials[chosen]["agent"]), trials[chosen]["weights"]
        manifest["generations"].append({"incumbent": incumbent["summary"], "trials": trials, "retained_weights": best_weights})
        write_json(output / "manifest.json", manifest)
    if best_weights is None:
        manifest.update(accepted=False, reason="No training-batch improvement; original priorities retained")
    else:
        baseline, _ = experiment.run(output, "validation_control", control, args.validation_games, 1220000)
        candidate, _ = experiment.run(output, "validation_candidate", best_path, args.validation_games, 1220000)
        manifest["validation"] = {"control": baseline["summary"], "candidate": candidate["summary"]}
        if candidate["summary"]["wins"] <= baseline["summary"]["wins"]:
            manifest.update(accepted=False, reason="Training gain did not improve wins on fresh validation boards")
        else:
            baseline, _ = experiment.run(output, "confirmation_control", control, args.confirm_games, 1240000)
            candidate, _ = experiment.run(output, "confirmation_candidate", best_path, args.confirm_games, 1240000)
            difference = paired_difference(candidate, baseline)
            manifest["confirmation"] = {"control": baseline["summary"], "candidate": candidate["summary"], **difference}
            accepted = difference["board_bootstrap_ci95"][0] > 0 and candidate["summary"]["games_over_budget"] <= baseline["summary"]["games_over_budget"]
            manifest.update(accepted=accepted, reason="Fresh paired-board confirmation gate" if accepted else "Fresh-board improvement was not confirmed")
            if accepted:
                write_json(output / "accepted_weights.json", best_weights)
                profile(output, "accepted_agent", best_weights)
    manifest["best_weights"] = best_weights
    write_json(output / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("accepted", "reason", "best_weights")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
