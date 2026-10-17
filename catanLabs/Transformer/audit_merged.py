"""Audit a merged v3 dataset, sealed base final split, and raw row alignment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from catanLabs.Transformer.audit_dataset import canonical_bytes, encoded_digest, sha256
from catanLabs.Transformer.merge_on_policy import (
    ARRAY_NAMES, MERGE_SCHEMA, METADATA_NAMES,
    MIN_SUPPLEMENTAL_COMPLETED_GAMES, _source,
)
from catanLabs.Transformer.model import (
    DATASET_SCHEMA, FEATURE_SCHEMA, encode_visible_observation,
)
from catanLabs.Transformer.raw_storage import RAW_SCHEMA, read_raw_records


AUDIT_SCHEMA = "catanlabs-v3-on-policy-merge-audit-v1"


def audit_merge(dataset: Path, sample_size: int = 64) -> dict:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    dataset = dataset.resolve(strict=True)
    manifest = json.loads(dataset.with_suffix(".json").read_text())
    split_path = dataset.with_suffix(".split.json")
    split = json.loads(split_path.read_text())
    raw_path = Path(manifest["raw_file"]).resolve(strict=True)
    failures: list[str] = []
    dataset_hash = sha256(dataset)
    raw_hash = sha256(raw_path)
    if (manifest.get("merge_schema") != MERGE_SCHEMA
            or manifest.get("dataset_schema") != DATASET_SCHEMA
            or manifest.get("feature_schema") != FEATURE_SCHEMA
            or manifest.get("raw_schema") != RAW_SCHEMA):
        failures.append("merged dataset/schema mismatch")
    if manifest.get("dataset_sha256") != dataset_hash or split.get("dataset_sha256") != dataset_hash:
        failures.append("dataset/manifest/split hash mismatch")
    if manifest.get("raw_sha256") != raw_hash:
        failures.append("raw hash mismatch")
    if manifest.get("split_membership_sha256") != split.get("membership_sha256"):
        failures.append("split membership hash mismatch")
    if split.get("schema") != "catanlabs-board-seed-split-v1":
        failures.append("split schema mismatch")
    membership = {"board_seeds": split.get("board_seeds"), "game_ids": split.get("game_ids")}
    if sha256_bytes(canonical_bytes(membership)) != split.get("membership_sha256"):
        failures.append("split membership contents/hash mismatch")

    source_hashes = manifest["source_hashes"]
    base = _source(Path(source_hashes["base"]["dataset"]), audited=True)
    on_policy = _source(Path(source_hashes["on_policy"]["dataset"]), audited=False)
    for source_name, source in (("base", base), ("on_policy", on_policy)):
        expected = source_hashes[source_name]
        if (source["dataset_sha256"] != expected["dataset_sha256"]
                or sha256(source["manifest_path"]) != expected["manifest_sha256"]
                or source["raw_sha256"] != expected["raw_sha256"]):
            failures.append(f"{source_name} provenance hash mismatch")
    if (sha256(base["audit_path"]) != source_hashes["base"]["audit_sha256"]
            or sha256(base["split_path"]) != source_hashes["base"]["split_sha256"]
            or base["split"]["membership_sha256"] != source_hashes["base"]["split_membership_sha256"]):
        failures.append("base audit/split provenance mismatch")
    base_completed_count = sum(game.get("status") == "completed" for game in base["manifest"]["games"])
    on_completed_count = sum(game.get("status") == "completed" for game in on_policy["manifest"]["games"])
    if (base_completed_count < 100
            or on_completed_count < MIN_SUPPLEMENTAL_COMPLETED_GAMES
            or manifest.get("source_completed_games") != {
                "base": base_completed_count, "on_policy": on_completed_count,
            }
            or manifest.get("source_roles") != {
                "base": "audited_gate2_100_game_base",
                "on_policy": "supplemental_completed_games_not_gate2_base",
            }):
        failures.append("source roles or role-specific completed-game thresholds mismatch")

    names = ("train", "validation", "final")
    groups = {name: set(split["game_ids"][name]) for name in names}
    boards = {name: set(split["board_seeds"][name]) for name in names}
    if any(groups[a] & groups[b] or boards[a] & boards[b]
           for i, a in enumerate(names) for b in names[i + 1:]):
        failures.append("split game or board groups overlap")
    for name in names:
        if not set(base["split"]["game_ids"][name]) <= groups[name]:
            failures.append(f"base {name} game assignment changed")
        if not set(base["split"]["board_seeds"][name]) <= boards[name]:
            failures.append(f"base {name} board assignment changed")
    if (split["game_ids"]["final"] != base["split"]["game_ids"]["final"]
            or split["board_seeds"]["final"] != base["split"]["board_seeds"]["final"]):
        failures.append("base final partition changed")
    shifted = set(int(value) for value in manifest["on_policy_game_id_map"].values())
    if shifted & groups["final"]:
        failures.append("on-policy game entered final")
    completed = {int(game["game_id"]): int(game["seed"])
                 for game in manifest["games"] if game["status"] == "completed"}
    for name in names:
        if any(completed.get(game_id) not in boards[name] for game_id in groups[name]):
            failures.append(f"{name} game/board membership mismatch")

    with np.load(dataset, allow_pickle=False) as npz:
        if (npz["dataset_schema"].item() != DATASET_SCHEMA
                or npz["feature_schema"].item() != FEATURE_SCHEMA):
            failures.append("merged NPZ schema mismatch")
        metadata = {name: npz[name] for name in METADATA_NAMES}
        count = len(metadata["labels"])
        if count != manifest.get("unique_examples") or count != manifest.get("raw_records"):
            failures.append("merged NPZ/raw/manifest row count mismatch")
        if (set(int(value) for value in metadata["game_ids"])
                != set().union(*groups.values())):
            failures.append("split does not partition merged NPZ games")
        if any(int(value) not in completed for value in metadata["game_ids"]):
            failures.append("merged NPZ includes incomplete games")
        if np.any(metadata["labels"] < 0) or np.any(metadata["labels"] >= metadata["candidate_counts"]):
            failures.append("merged NPZ has invalid labels")
        expected_drop = sum(int(value) for value in manifest["duplicate_rows_dropped"].values())
        if sum(manifest["source_rows"].values()) - count != expected_drop:
            failures.append("duplicate accounting mismatch")
        if sum(int(value < manifest["game_id_shift"]) for value in metadata["game_ids"]) != manifest["source_rows"]["base"]:
            failures.append("base rows were removed or shifted")
        samples = set(random.Random(int(dataset_hash[:16], 16)).sample(
            range(count), min(count, sample_size),
        ))
        sample_arrays = {index: {} for index in samples}
        for name in ARRAY_NAMES:
            whole = npz[name]
            for index in samples:
                sample_arrays[index][name] = whole[index].copy()
            del whole

        raw_count = 0
        reconstructed = 0
        metadata_errors = 0
        hidden_errors = 0
        seen_keys: set[str] = set()
        duplicate_keys = 0
        phases = Counter()
        behaviors = Counter()
        for index, record in enumerate(read_raw_records(raw_path)):
            raw_count += 1
            if record.canonical_state_key in seen_keys:
                duplicate_keys += 1
            seen_keys.add(record.canonical_state_key)
            phases[record.phase] += 1
            behaviors[record.behavior] += 1
            if any(player.own_resources is not None or player.own_development_cards is not None
                   for player in record.observation.players[1:]):
                hidden_errors += 1
            if index >= count:
                metadata_errors += 1
                continue
            if (record.feature_schema != FEATURE_SCHEMA
                    or record.game_id != int(metadata["game_ids"][index])
                    or record.board_seed != completed.get(record.game_id)
                    or record.target_action_index != int(metadata["labels"][index])
                    or len(record.observation.legal_actions) != int(metadata["candidate_counts"][index])
                    or record.phase != str(metadata["phases"][index])
                    or record.behavior != str(metadata["behaviors"][index])
                    or record.actor_color != str(metadata["actor_colors"][index])
                    or record.outcome != int(metadata["outcomes"][index])
                    or record.final_vp != int(metadata["final_vps"][index])
                    or record.rules_manifest_sha256 != manifest["rules_manifest_sha256"]
                    or record.benchmark_protocol != manifest["benchmark_protocol"]
                    or record.engine_commit != manifest["engine"]["commit"]
                    or record.teacher != "fair"):
                metadata_errors += 1
            if index in samples:
                encoded = encode_visible_observation(record.observation)
                width = len(record.observation.legal_actions)
                matches = encoded_digest(encoded) == record.canonical_state_key
                for name, expected in zip(ARRAY_NAMES, encoded):
                    actual = sample_arrays[index][name]
                    if name in ("action_features", "action_types", "target_groups", "target_ids"):
                        actual = actual[:width]
                    if not np.array_equal(actual, expected):
                        matches = False
                reconstructed += int(matches)
        if raw_count != count or duplicate_keys or metadata_errors or hidden_errors or reconstructed != len(samples):
            failures.append("merged raw alignment, uniqueness or visibility check failed")
        if dict(phases) != manifest.get("phases") or dict(behaviors) != manifest.get("trajectory_policies"):
            failures.append("merged phase/behavior counts disagree with manifest")

    return {
        "schema": AUDIT_SCHEMA,
        "dataset": str(dataset), "dataset_sha256": dataset_hash,
        "raw_sha256": raw_hash, "split_file_sha256": sha256(split_path),
        "split_membership_sha256": split["membership_sha256"],
        "rows": count, "raw_records": raw_count,
        "duplicate_encoded_keys": duplicate_keys,
        "source_duplicate_rows_dropped": manifest["duplicate_rows_dropped"],
        "reconstructed_samples": reconstructed,
        "sample_size": len(samples),
        "metadata_errors": metadata_errors,
        "hidden_field_errors": hidden_errors,
        "split_rows": {name: int(np.isin(metadata["game_ids"], list(groups[name])).sum())
                       for name in names},
        "base_final_games": len(base["split"]["game_ids"]["final"]),
        "source_completed_games": {"base": base_completed_count,
                                   "on_policy": on_completed_count},
        "on_policy_validation_boards": manifest["on_policy_validation_boards"],
        "pass": not failures, "failures": failures,
    }


def sha256_bytes(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit_merge(args.dataset, args.sample_size)
    path = args.report or args.dataset.with_suffix(".audit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pass": report["pass"], "rows": report["rows"],
                      "failures": report["failures"], "report": str(path)}, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
