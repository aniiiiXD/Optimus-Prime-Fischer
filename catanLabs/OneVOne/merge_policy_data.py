"""Merge compatible gzip JSONL policy datasets and their manifests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(dataset: Path) -> Path:
    return Path(str(dataset) + ".manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    sources = [path.resolve(strict=True) for path in args.inputs]
    output = args.output.resolve()
    output_manifest = manifest_path(output)
    if len(sources) < 2:
        parser.error("provide at least two datasets to merge")
    if output.exists() or output_manifest.exists():
        parser.error(f"refusing to overwrite {output} or its manifest")

    manifests = []
    for source in sources:
        sidecar = manifest_path(source)
        if not sidecar.is_file():
            parser.error(f"missing dataset manifest: {sidecar}")
        manifests.append(json.loads(sidecar.read_text()))

    reference = manifests[0]
    compatible_fields = (
        "format", "features_version", "families", "action_feature_dim",
        "engine", "rules", "config_sha256", "teachers", "seat_protocol", "caps",
    )
    for source, manifest in zip(sources[1:], manifests[1:]):
        differences = [key for key in compatible_fields
                       if manifest.get(key) != reference.get(key)]
        if differences:
            parser.error(
                f"{source} is incompatible with {sources[0]} in: "
                + ", ".join(differences)
            )

    seed_ranges = []
    for source, manifest in zip(sources, manifests):
        first = int(manifest["first_seed"])
        count = int(manifest["boards_per_teacher"])
        current = set(range(first, first + count))
        if any(current.intersection(previous) for previous in seed_ranges):
            parser.error(f"board seed range overlaps for {source}")
        seed_ranges.append(current)

    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wb", compresslevel=6) as destination:
        for source in sources:
            with gzip.open(source, "rb") as dataset:
                shutil.copyfileobj(dataset, destination, length=1024 * 1024)

    merged = dict(reference)
    merged["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    merged["source_datasets"] = [
        {"path": str(source), "sha256": sha256(source)}
        for source in sources
    ]
    merged["boards_per_teacher"] = sum(
        manifest["boards_per_teacher"] for manifest in manifests
    )
    merged["unique_board_seeds"] = len(set.union(*seed_ranges))
    merged["first_seed"] = min(min(seed_range) for seed_range in seed_ranges)
    for field in ("games_total", "completed_games", "errors", "capped_games", "examples"):
        merged[field] = sum(int(manifest[field]) for manifest in manifests)
    merged["elapsed_seconds"] = sum(
        float(manifest.get("elapsed_seconds", 0)) for manifest in manifests
    )
    merged["game_records"] = [
        record for manifest in manifests for record in manifest["game_records"]
    ]
    merged["by_teacher"] = {}
    for teacher in reference["teachers"]:
        records = [
            record for manifest in manifests
            for record in manifest["game_records"]
            if record["teacher"] == teacher
        ]
        merged["by_teacher"][teacher] = {
            "games": len(records),
            "completed": sum(record["status"] == "completed" for record in records),
            "wins": sum(bool(record["won"]) for record in records),
            "examples": sum(
                int(manifest["by_teacher"][teacher]["examples"])
                for manifest in manifests
            ),
            "errors": [record for record in records if record["status"] == "error"],
            "capped": sum(record["status"] == "capped" for record in records),
        }
    merged["dataset_sha256"] = sha256(output)
    merged["label_notes"] = (
        reference.get("label_notes", "")
        + " Merged from compatible datasets with disjoint board seed ranges."
    )
    output_manifest.write_text(json.dumps(merged, indent=2) + "\n")
    print(json.dumps({
        "dataset": str(output),
        "manifest": str(output_manifest),
        "games": merged["games_total"],
        "completed": merged["completed_games"],
        "examples": merged["examples"],
        "unique_board_seeds": merged["unique_board_seeds"],
        "sha256": merged["dataset_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
