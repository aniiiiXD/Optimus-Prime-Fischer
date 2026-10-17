"""Create a compact, version-controlled evidence snapshot from local raw runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SELECTIONS = [
    ("benjamin_checked.json", {"benjamin"}),
    ("baselines_final.json", {"settlers", "catanatron_1v1"}),
    ("eli_256.json", {"eli"}),
    ("weighted_vs_weighted.json", {"catanatron"}),
    ("value_vs_weighted.json", {"catanatron"}),
    ("ab1_vs_weighted.json", {"catanatron"}),
]


def main():
    output = {"sources": json.loads((ROOT / "sources.json").read_text()),
              "checkpoints": json.loads((ROOT / "artifacts/checkpoints/manifest.json").read_text()),
              "caution": "Different rules/opponents: no cross-repository Elo or strength ranking. "
                         "Wilson intervals are approximate, not adjusted for shared map seeds.",
              "experiments": []}
    for filename, names in SELECTIONS:
        path = ROOT / "artifacts/reports" / filename
        report = json.loads(path.read_text())
        for entry in report["results"]:
            if entry["agent"] not in names:
                continue
            if entry["status"] != "ok":
                raise RuntimeError(f"Cannot archive failed experiment: {path}")
            for variant in entry.get("variants", [entry]):
                compact = {k: v for k, v in variant.items()
                           if k not in {"timed_decision_ms", "checkpoint_training_config"}}
                if "checkpoint_training_config" in variant:
                    config = variant["checkpoint_training_config"]
                    compact["checkpoint_training_config"] = {
                        key: config[key] for key in ("hidden", "visibility", "victory_target", "seed", "train_seats")}
                compact["raw_report"] = str(path.relative_to(ROOT))
                compact["raw_report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                compact["protocol"] = {k: v for k, v in report["protocol"].items() if k != "output"}
                output["experiments"].append(compact)
    destination = ROOT / "reports/pilot_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Archived {len(output['experiments'])} experiment rows to {destination}")


if __name__ == "__main__":
    main()
