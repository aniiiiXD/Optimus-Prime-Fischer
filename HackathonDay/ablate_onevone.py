"""Diagnostic component removals; no new heuristic or baseline-file edits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official-kit"))
from catan_kit import match
from catan_kit._patch import ensure_hash_seed
from catan_kit.evaluate import summarise


def paired(baseline, candidate):
    differences = [int(c["winner"] == "a") - int(b["winner"] == "a")
                   for b, c in zip(baseline, candidate)]
    boards = [sum(differences[i:i+2]) / 2 for i in range(0, len(differences), 2)]
    rng = random.Random(6241017)
    resamples = sorted(sum(rng.choices(boards, k=len(boards))) / len(boards) for _ in range(10000))
    return {"difference_percentage_points": 100 * sum(differences) / len(differences),
            "board_bootstrap_95_pp": [100 * resamples[250], 100 * resamples[9749]],
            "gained_games": sum(x == 1 for x in differences),
            "lost_games": sum(x == -1 for x in differences)}


def main():
    ensure_hash_seed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["no_tactics", "default_value_weights", "generic_robber", "no_late_dev_bonus"])
    args = parser.parse_args()
    out = ROOT / "reports/onevone_analysis/component_removals"
    out.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((ROOT / "reports/onevone_analysis/baseline_trace_200/games.json").read_text())
    controls = [g["harness"] for g in baseline]
    all_results = {}
    for name in args.variants:
        agent = out / name
        if agent.exists():
            raise FileExistsError(f"Refusing to overwrite {agent}")
        agent.mkdir()
        for src in (ROOT / "onevone_baseline").iterdir():
            if src.suffix in (".py", ".json") and src.name != "PROVENANCE.json":
                shutil.copy2(src, agent / src.name)
        cfg = json.loads((agent / "config.json").read_text())
        if name == "no_tactics":
            cfg["tactical_min_vp"] = 10
            change = "Disable tactical override by requiring 10 VP in a game that ends at 10."
        elif name == "default_value_weights":
            cfg["value_weights"] = {}
            change = "Replace the five OneVOne value-weight overrides with engine default values."
        elif name == "no_late_dev_bonus":
            cfg["dev_buy_late_bonus"] = 0
            change = "Remove only the development-buy bonus applied at >=7 VP."
        elif name == "generic_robber":
            entry = (agent / "agent.py").read_text()
            entry = entry.replace("from onevone import create_agent", "from onevone import create_agent\nfrom competitive import CompetitiveAgent")
            entry = entry.replace("self.policy = create_agent(obs.me)", "self.policy = create_agent(obs.me)\n            self.policy.robber = CompetitiveAgent(obs.me, self.policy.config.get('robber_weights', {}))")
            (agent / "agent.py").write_text(entry)
            change = "Use the inherited generic robber policy instead of the duel-specific scorer."
        else:
            raise ValueError(name)
        (agent / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"Starting {name}: {change}", flush=True)
        original_play = match.play_game
        count = [0, 0]
        def progress(*args, **kwargs):
            winner, turns = original_play(*args, **kwargs)
            count[0] += 1
            count[1] += winner == args[0].color
            if count[0] % 50 == 0:
                print(f"{name}: {count[0]}/200, wins {count[1]}", flush=True)
            return winner, turns
        match.play_game = progress
        try:
            records, info = match.play_chunk(str(agent / "agent.py"),
                                            str(ROOT / "official-kit/examples/f_agent.py"),
                                            [2026101800 + i // 2 for i in range(200)], list(range(200)))
        finally:
            match.play_game = original_play
        result = {"variant": name, "change": change, "summary": summarise(records),
                  "opponent_summary": summarise(records, "b"), "load": info,
                  "paired_vs_baseline": paired(controls, records),
                  "baseline_wins": 127, "records": records,
                  "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in agent.iterdir() if p.is_file()}}
        (out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        all_results[name] = {k: v for k, v in result.items() if k not in ("records", "files_sha256")}
        print(json.dumps(all_results[name]), flush=True)
    (out / "summary.json").write_text(json.dumps(all_results, indent=2) + "\n")


if __name__ == "__main__":
    main()
