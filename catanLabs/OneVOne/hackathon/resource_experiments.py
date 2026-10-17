"""Train small scoring corrections online, then test frozen ablations on fresh boards."""

import argparse
import hashlib
import json
import random
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor/kit"))

from catan_kit._patch import ensure_hash_seed
from catan_kit import match
from catan_kit.evaluate import summarise


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def profile(directory, name, params, exact=True):
    path = directory / f"{name}.py"
    path.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "from resource_policy import Agent as ResourceAgent\n"
        "class Agent(ResourceAgent):\n"
        "    def __init__(self):\n"
        f"        super().__init__(exact_monopoly={exact!r}, params={params!r})\n"
    )
    return path


def play(directory, name, path, boards, seed):
    records, info = match.play_chunk(
        str(path), str(HERE / "vendor/kit/examples/f_agent.py"),
        [seed + i // 2 for i in range(2 * boards)], list(range(2 * boards)),
    )
    result = {"summary": summarise(records), "info": info, "records": records}
    write_json(directory / f"{name}.json", result)
    s = result["summary"]
    print(f"{name}: {s['wins']}/{s['games']} wins; {s['games_over_budget']} over budget; {s['errors']} errors", flush=True)
    if s["errors"] or s["crashes"] or s["timeouts"]:
        raise RuntimeError(f"Broken experiment arm {name}; inspect its saved records")
    return result


def paired_difference(candidate, baseline):
    a, b = candidate["records"], baseline["records"]
    if len(a) != len(b) or len(a) % 2:
        raise ValueError("Need matched, paired-seat records")
    deltas = []
    for i in range(0, len(a), 2):
        for j in (i, i + 1):
            if (a[j]["board_seed"], a[j]["game"]) != (b[j]["board_seed"], b[j]["game"]):
                raise ValueError("Seed/seat mismatch")
        deltas.append(sum((a[j]["winner"] == "a") - (b[j]["winner"] == "a") for j in (i, i + 1)) / 2)
    rng = random.Random(17)
    samples = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(4000))
    return {"win_rate_difference": sum(deltas) / len(deltas), "board_bootstrap_ci95": [samples[100], samples[3899]]}


def main():
    ensure_hash_seed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "reports/resource_learning_20261017")
    parser.add_argument("--train-boards", type=int, default=4)
    parser.add_argument("--screen-boards", type=int, default=16)
    parser.add_argument("--confirm-boards", type=int, default=32)
    args = parser.parse_args()
    if min(args.train_boards, args.screen_boards, args.confirm_boards) < 1 or max(args.train_boards, args.screen_boards) > 500:
        parser.error("board counts must be positive; train/screen at most 500 to keep seed ranges separate")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"method": "warm-start online episodic coordinate search, not neural PPO", "opponent": "F", "training_seed0": 993000, "screen_seed0": 995000, "confirmation_seed0": 997000, "boards": vars(args) | {"output": str(output)}, "updates": []}
    manifest["source_sha256"] = {
        str(path.relative_to(HERE.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (HERE / "resource_policy.py", HERE / "resource_experiments.py", HERE / "agent.py", HERE.parent / "configs/default.json", HERE / "requirements.txt")
    }
    write_json(output / "manifest.json", manifest)
    params = [0.0, 0.0, 0.0]
    # Warm start keeps OneVOne's weights; only the exact-resource Monopoly correction is on.
    for axis, step in enumerate((2.0, 1.0, 1.0)):
        candidates = []
        for label, change in (("center", 0), ("minus", -step), ("plus", step)):
            trial = params.copy()
            trial[axis] += change
            name = f"train_{axis}_{label}"
            path = profile(output, name, trial)
            result = play(output, name, path, args.train_boards, 993000 + axis * args.train_boards)
            candidates.append((trial, result))
        # Ties keep the incumbent. No held-out result feeds back into training.
        best = max(range(3), key=lambda i: candidates[i][1]["summary"]["wins"])
        params = candidates[best][0]
        manifest["updates"].append({"axis": axis, "parameters": params, "wins": [r[1]["summary"]["wins"] for r in candidates]})
        write_json(output / "manifest.json", manifest)
    arms = {
        "baseline": HERE / "agent.py",
        "exact_monopoly": profile(output, "exact_monopoly", [0.0, 0.0, 0.0]),
        "resource_aware": profile(output, "resource_aware", [0.0, 0.0, 1.0]),
        "online_trained": profile(output, "online_trained", params),
    }
    screens = {name: play(output, f"screen_{name}", path, args.screen_boards, 995000) for name, path in arms.items()}
    selected = max((name for name in arms if name != "baseline"), key=lambda n: screens[n]["summary"]["wins"])
    manifest.update(parameters=params, selected=selected, screen={name: r["summary"] for name, r in screens.items()})
    write_json(output / "manifest.json", manifest)
    base = play(output, "confirm_baseline", arms["baseline"], args.confirm_boards, 997000)
    candidate = play(output, f"confirm_{selected}", arms[selected], args.confirm_boards, 997000)
    difference = paired_difference(candidate, base)
    manifest.update(confirmation={"baseline": base["summary"], "candidate": candidate["summary"], **difference})
    manifest["promotable"] = difference["board_bootstrap_ci95"][0] > 0 and candidate["summary"]["games_over_budget"] == 0
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"selected": selected, "parameters": params, "confirmation": manifest["confirmation"], "promotable": manifest["promotable"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
