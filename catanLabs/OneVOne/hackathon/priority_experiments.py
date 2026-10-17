"""Compare handwritten resource priorities, tune them, and confirm on fresh boards."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor/kit"))

from catan_kit._patch import ensure_hash_seed
from resource_experiments import paired_difference, play, write_json
from priority_policy import normalize_priorities


def profile(output, name, weights):
    path = output / f"{name}.py"
    path.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "from priority_policy import Agent as PriorityAgent\n"
        "class Agent(PriorityAgent):\n"
        "    def __init__(self):\n"
        f"        super().__init__(priorities={weights!r})\n"
    )
    return path


def main():
    ensure_hash_seed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "reports/priority_tuning_20261017")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    priors = {
        "equal": [1, 1, 1, 1, 1],
        "handwritten": [1.15, 1.15, 0.85, 1.45, 1.35],
        "wheat_ore": [0.85, 0.85, 0.75, 1.25, 1.30],
        "expansion": [1.25, 1.25, 0.80, 0.95, 0.75],
    }
    priors = {name: normalize_priorities(w) for name, w in priors.items()}
    manifest = {
        "resource_order": ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"],
        "scope": "Own effective production only; original diversity, hand, opponent, and robber scoring retained",
        "priors": priors, "updates": [],
        "seeds": {"prior_screen": 1001000, "tuning": [1002000], "validation": 1003000, "confirmation": 1004000},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE / "priority_policy.py", HERE / "priority_experiments.py")},
    }
    write_json(output / "manifest.json", manifest)
    training = {}
    for name, weights in priors.items():
        training[name] = play(output, f"prior_{name}", profile(output, f"prior_{name}", weights), 4, 1001000)
    selected_prior = max(priors, key=lambda name: training[name]["summary"]["wins"])
    weights = priors[selected_prior].copy()
    manifest.update(selected_prior=selected_prior, prior_results={n: r["summary"] for n, r in training.items()})
    rng = random.Random(17)
    for generation in range(1):
        population = [weights] + [normalize_priorities([v * math.exp(rng.uniform(-0.45, 0.45)) for v in weights]) for _ in range(3)]
        results = []
        for i, candidate in enumerate(population):
            name = f"tune_{generation}_{i}"
            results.append(play(output, name, profile(output, name, candidate), 4, 1002000 + 4 * generation))
        selected = max(range(4), key=lambda i: results[i]["summary"]["wins"])
        weights = population[selected]
        manifest["updates"].append({"population": population, "wins": [r["summary"]["wins"] for r in results], "selected": selected})
        write_json(output / "manifest.json", manifest)
    arms = {
        "baseline": HERE / "agent.py",
        "best_handwritten": profile(output, "best_handwritten", priors[selected_prior]),
        "tuned": profile(output, "tuned", weights),
    }
    validations = {name: play(output, f"validation_{name}", path, 16, 1003000) for name, path in arms.items()}
    selected = max(("best_handwritten", "tuned"), key=lambda n: validations[n]["summary"]["wins"])
    manifest.update(weights=weights, validation={n: r["summary"] for n, r in validations.items()}, selected=selected)
    write_json(output / "manifest.json", manifest)
    if validations[selected]["summary"]["wins"] <= validations["baseline"]["summary"]["wins"]:
        manifest.update(promotable=False, confirmation_skipped="No held-out screening advantage over baseline")
        write_json(output / "manifest.json", manifest)
        print(json.dumps(manifest["validation"], indent=2), flush=True)
        return
    baseline = play(output, "confirmation_baseline", arms["baseline"], 32, 1004000)
    candidate = play(output, f"confirmation_{selected}", arms[selected], 32, 1004000)
    difference = paired_difference(candidate, baseline)
    manifest["confirmation"] = {"baseline": baseline["summary"], "candidate": candidate["summary"], **difference}
    manifest["promotable"] = difference["board_bootstrap_ci95"][0] > 0 and candidate["summary"]["games_over_budget"] == 0
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"selected": selected, "prior": selected_prior, "weights": weights, "confirmation": manifest["confirmation"], "promotable": manifest["promotable"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
