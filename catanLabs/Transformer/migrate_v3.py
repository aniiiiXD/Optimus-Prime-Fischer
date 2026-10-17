"""Re-encode audited v2 raw observations as v3 tensors without replaying games.

The raw records and board split are immutable inputs. V3 changes only the
encoding of already-public opponent card counts. No teacher decision is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from catanLabs.Transformer.audit_dataset import build_split, freeze_split, sha256
from catanLabs.Transformer.raw_storage import RAW_SCHEMA, read_raw_records


SOURCE_DATASET_SCHEMA = "catanlabs-transformer-dataset-v2"
SOURCE_FEATURE_SCHEMA = "visible-state-tokens-v2"
TARGET_DATASET_SCHEMA = "catanlabs-transformer-dataset-v3"
TARGET_FEATURE_SCHEMA = "visible-state-tokens-v3"
MIGRATION_SCHEMA = "catanlabs-visible-v2-to-v3-migration-v1"

TENSOR_DTYPES = {
    "state_features": np.float32,
    "state_kinds": np.int64,
    "state_ids": np.int64,
    "action_features": np.float32,
    "action_types": np.int64,
    "target_groups": np.int64,
    "target_ids": np.int64,
}
METADATA_NAMES = (
    "candidate_counts", "labels", "game_ids", "phases", "behaviors",
    "actor_colors", "outcomes", "final_vps",
)


def checked_inputs(source: Path, manifest_path: Path, audit_path: Path,
                   split_path: Path, raw_path: Path) -> tuple[dict, dict, dict, str]:
    """Bind every input to the previously passed v2 audit and frozen split."""
    source_hash = sha256(source)
    raw_hash = sha256(raw_path)
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    split = json.loads(split_path.read_text())
    checks = {
        "source dataset hash": manifest.get("dataset_sha256") == source_hash == audit.get("dataset_sha256") == split.get("dataset_sha256"),
        "source schema": manifest.get("dataset_schema") == SOURCE_DATASET_SCHEMA and manifest.get("feature_schema") == SOURCE_FEATURE_SCHEMA,
        "raw schema/hash": manifest.get("raw_schema") == RAW_SCHEMA and manifest.get("raw_sha256") == raw_hash == audit.get("raw", {}).get("sha256"),
        "source audit": audit.get("gate2_pass") is True and audit.get("raw", {}).get("status") == "checked",
        "source split": split.get("schema") == "catanlabs-board-seed-split-v1" and split.get("membership_sha256") == audit.get("split_membership_sha256") and sha256(split_path) == audit.get("split_file_sha256"),
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError("V2 provenance check failed: " + ", ".join(failed))
    return manifest, audit, split, source_hash


def migrate(source: Path, source_manifest: Path, source_audit: Path,
            source_split: Path, raw: Path, output: Path) -> dict:
    from catanLabs.Transformer import model

    if (model.DATASET_SCHEMA != TARGET_DATASET_SCHEMA
            or model.FEATURE_SCHEMA != TARGET_FEATURE_SCHEMA):
        raise RuntimeError("V3 encoder/schema is not installed; migration is refused")
    manifest, audit, split, source_hash = checked_inputs(
        source, source_manifest, source_audit, source_split, raw,
    )
    if output.exists() or output.with_suffix(".json").exists() or output.with_suffix(".split.json").exists():
        raise FileExistsError("V3 output exists; migration never overwrites a frozen dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as source_npz:
        if (source_npz["dataset_schema"].item() != SOURCE_DATASET_SCHEMA
                or source_npz["feature_schema"].item() != SOURCE_FEATURE_SCHEMA):
            raise ValueError("Source NPZ schema disagrees with audited v2 manifest")
        metadata = {name: source_npz[name] for name in METADATA_NAMES}
    count = len(metadata["labels"])
    if count != int(manifest["unique_examples"]) or count != int(manifest["raw_records"]):
        raise ValueError("V2 NPZ, manifest and raw record counts disagree")
    max_candidates = int(manifest["max_legal_actions"])
    shapes = {
        "state_features": (count, model.STATE_TOKEN_COUNT, model.FEATURE_DIM),
        "state_kinds": (count, model.STATE_TOKEN_COUNT),
        "state_ids": (count, model.STATE_TOKEN_COUNT),
        "action_features": (count, max_candidates, model.ACTION_FEATURE_DIM),
        "action_types": (count, max_candidates),
        "target_groups": (count, max_candidates),
        "target_ids": (count, max_candidates),
    }
    tensor_names = tuple(TENSOR_DTYPES)
    rows = 0
    with tempfile.TemporaryDirectory(prefix="catan-v3-", dir=output.parent) as tempdir:
        temp = Path(tempdir)
        tensors = {
            name: np.lib.format.open_memmap(temp / f"{name}.npy", mode="w+",
                                               dtype=TENSOR_DTYPES[name], shape=shapes[name])
            for name in tensor_names
        }
        for name in ("action_features", "action_types", "target_groups", "target_ids"):
            tensors[name][:] = 0  # Explicitly initialize padded candidate slots.
        for index, record in enumerate(read_raw_records(raw)):
            if index >= count:
                raise ValueError("Raw file contains more rows than source NPZ")
            obs = record.observation
            legal_count = len(obs.legal_actions)
            if (record.feature_schema != SOURCE_FEATURE_SCHEMA
                    or record.game_id != int(metadata["game_ids"][index])
                    or record.phase != str(metadata["phases"][index])
                    or record.behavior != str(metadata["behaviors"][index])
                    or record.actor_color != str(metadata["actor_colors"][index])
                    or record.outcome != int(metadata["outcomes"][index])
                    or record.final_vp != int(metadata["final_vps"][index])
                    or record.target_action_index != int(metadata["labels"][index])
                    or legal_count != int(metadata["candidate_counts"][index])
                    or legal_count > max_candidates):
                raise ValueError(f"Raw/source mismatch at row {index}")
            encoded = model.encode_visible_observation(obs)
            for name, array in zip(tensor_names, encoded):
                if name in ("action_features", "action_types", "target_groups", "target_ids"):
                    tensors[name][index, :legal_count] = array
                else:
                    tensors[name][index] = array
            rows += 1
        if rows != count:
            raise ValueError(f"Raw file has {rows} records; source NPZ has {count}")
        for tensor in tensors.values():
            tensor.flush()
        temporary_npz = temp / "dataset.npz"
        np.savez_compressed(
            temporary_npz,
            dataset_schema=np.asarray(TARGET_DATASET_SCHEMA),
            feature_schema=np.asarray(TARGET_FEATURE_SCHEMA),
            **tensors, **metadata,
        )
        target_hash = sha256(temporary_npz)
        completed = {int(game["game_id"]): int(game["seed"])
                     for game in manifest["games"] if game.get("status") == "completed"}
        target_split = build_split(completed, target_hash)
        if (target_split["board_seeds"] != split["board_seeds"]
                or target_split["game_ids"] != split["game_ids"]
                or target_split["split_sha256"] != split["split_sha256"]
                or target_split["membership_sha256"] != split["membership_sha256"]):
            raise ValueError("V3 board split changed from frozen v2 membership")
        migration = {
            "schema": MIGRATION_SCHEMA,
            "source_dataset": str(source),
            "source_dataset_sha256": source_hash,
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256(source_manifest),
            "source_audit": str(source_audit),
            "source_audit_sha256": sha256(source_audit),
            "source_split": str(source_split),
            "source_split_file_sha256": sha256(source_split),
            "source_split_membership_sha256": split["membership_sha256"],
            "source_feature_schema": SOURCE_FEATURE_SCHEMA,
            "raw_sha256": sha256(raw),
            "encoder_sha256": sha256(WORKSPACE / "catanLabs/Transformer/model.py"),
            "description": "Re-encode immutable visible snapshots; labels, outcomes, games and board membership copied unchanged.",
        }
        target_manifest = dict(manifest)
        target_manifest.update({
            "schema": TARGET_DATASET_SCHEMA,
            "dataset_schema": TARGET_DATASET_SCHEMA,
            "feature_schema": TARGET_FEATURE_SCHEMA,
            "dataset_sha256": target_hash,
            "raw_file": str(raw),
            "raw_provenance": {
                "role": "source_visible_snapshots",
                "source_feature_schema": SOURCE_FEATURE_SCHEMA,
                "raw_schema": RAW_SCHEMA,
                "source_dataset_sha256": source_hash,
                "raw_sha256": migration["raw_sha256"],
            },
            "migration": migration,
            "encoder_sha256": migration["encoder_sha256"],
        })
        os.replace(temporary_npz, output)
        output.with_suffix(".json").write_text(json.dumps(target_manifest, indent=2, allow_nan=False) + "\n")
        freeze_split(output.with_suffix(".split.json"), target_split)
    return {
        "dataset": str(output), "dataset_sha256": target_hash,
        "manifest": str(output.with_suffix(".json")),
        "split_file": str(output.with_suffix(".split.json")),
        "rows": rows,
        "split_membership_sha256": target_split["membership_sha256"],
        "source_dataset_sha256": source_hash,
        "raw_sha256": migration["raw_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = WORKSPACE / "catanLabs/Transformer/runs/gate2_native_v2/dataset.npz"
    parser.add_argument("--source-dataset", type=Path, default=default_source)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--source-split", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "catanLabs/Transformer/runs/gate2_native_v3/dataset.npz")
    args = parser.parse_args()
    source = args.source_dataset.resolve(strict=True)
    manifest = (args.source_manifest or source.with_suffix(".json")).resolve(strict=True)
    audit = (args.source_audit or source.with_suffix(".audit.json")).resolve(strict=True)
    split = (args.source_split or source.with_suffix(".split.json")).resolve(strict=True)
    raw = (args.raw or source.with_name(source.stem + ".raw.jsonl.gz")).resolve(strict=True)
    output = args.output.resolve()
    print(json.dumps(migrate(source, manifest, audit, split, raw, output), indent=2))


if __name__ == "__main__":
    main()
