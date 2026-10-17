"""Audit a native Catan dataset and freeze board-disjoint split membership.

The report distinguishes metadata checks from the stronger raw-observation
checks. A legacy NPZ without raw visible snapshots cannot pass Gate 2.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from catanLabs.Catanatron.tournament_protocol import PROTOCOL
from catanLabs.Transformer.model import (
    DATASET_SCHEMA, FEATURE_SCHEMA, encode_visible_observation,
)
from catanLabs.Transformer.visible_observation import SCHEMA_VERSION
from catanLabs.Transformer.raw_storage import RAW_SCHEMA, read_raw_records


AUDIT_SCHEMA = "catanlabs-native-dataset-audit-v1"
SPLIT_SCHEMA = "catanlabs-board-seed-split-v1"
SPLIT_SALT = "catanlabs-gate2-sealed-board-splits-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def observation_digest(observation) -> str:
    return hashlib.sha256(canonical_bytes(asdict(observation))).hexdigest()


def encoded_digest(encoded) -> str:
    digest = hashlib.blake2b(digest_size=20)
    for array in encoded:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def build_split(completed_games: dict[int, int], dataset_sha256: str) -> dict:
    """Assign complete board seeds to one split with a fixed, public salt."""
    board_seeds = set(completed_games.values())
    ordered = sorted(board_seeds, key=lambda seed: (hashlib.sha256(
        f"{SPLIT_SALT}:{seed}".encode()).hexdigest(), seed))
    size = len(ordered)
    held = max(1, round(size * 0.1)) if size >= 3 else 0
    final = sorted(ordered[:held])
    validation = sorted(ordered[held:2 * held])
    train = sorted(ordered[2 * held:])
    splits = {"train": train, "validation": validation, "final": final}
    game_ids = {name: sorted(game_id for game_id, board_seed in completed_games.items()
                             if board_seed in set(seeds))
                for name, seeds in splits.items()}
    hashes = {name: hashlib.sha256(canonical_bytes({"board_seeds": splits[name],
                                                     "game_ids": game_ids[name]})).hexdigest()
              for name in splits}
    return {
        "schema": SPLIT_SCHEMA,
        "salt_sha256": hashlib.sha256(SPLIT_SALT.encode()).hexdigest(),
        "dataset_sha256": dataset_sha256,
        "board_seeds": splits,
        "game_ids": game_ids,
        "split_sha256": hashes,
        "membership_sha256": hashlib.sha256(canonical_bytes({"board_seeds": splits,
                                                               "game_ids": game_ids})).hexdigest(),
    }


def freeze_split(path: Path, split: dict) -> None:
    """A changed seed set requires a new split file, never silent overwrite."""
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != split:
            raise ValueError(f"Frozen split disagrees with dataset: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2, allow_nan=False) + "\n")


def _sample_indices(count: int, requested: int, dataset_hash: str) -> set[int]:
    seed = int(dataset_hash[:16], 16)
    return set(random.Random(seed).sample(range(count), min(count, requested)))


def _array_matches(arrays: dict[str, np.ndarray], encoded, count: int) -> bool:
    names = ("state_features", "state_kinds", "state_ids", "action_features",
             "action_types", "target_groups", "target_ids")
    for name, expected in zip(names, encoded):
        actual = arrays[name]
        if name.startswith("action_") or name.startswith("target_"):
            actual = actual[:count]
        if not np.array_equal(actual, expected):
            return False
    return True


def _check_migration(manifest: dict, raw_hash: str) -> tuple[list[str], dict | None]:
    """Confirm that a re-encoded dataset cites the passed source corpus."""
    migration = manifest.get("migration")
    if migration is None:
        return [], None
    failures = []
    if migration.get("schema") != "catanlabs-visible-v2-to-v3-migration-v1":
        failures.append("unsupported migration schema")
    if migration.get("source_feature_schema") != "visible-state-tokens-v2":
        failures.append("migration source feature schema mismatch")
    files = {
        "source_dataset": "source_dataset_sha256",
        "source_manifest": "source_manifest_sha256",
        "source_audit": "source_audit_sha256",
        "source_split": "source_split_file_sha256",
    }
    for path_name, hash_name in files.items():
        path = Path(str(migration.get(path_name, "")))
        if not path.is_file() or sha256(path) != migration.get(hash_name):
            failures.append(f"migration {path_name} hash mismatch")
    if migration.get("raw_sha256") != raw_hash:
        failures.append("migration raw hash mismatch")
    source_audit_path = Path(str(migration.get("source_audit", "")))
    source_split_path = Path(str(migration.get("source_split", "")))
    source_split = None
    if source_audit_path.is_file() and source_split_path.is_file():
        source_audit = json.loads(source_audit_path.read_text())
        source_split = json.loads(source_split_path.read_text())
        if source_audit.get("gate2_pass") is not True:
            failures.append("migration source audit did not pass")
        if source_audit.get("dataset_sha256") != migration.get("source_dataset_sha256"):
            failures.append("migration source audit/dataset mismatch")
        if source_split.get("membership_sha256") != migration.get("source_split_membership_sha256"):
            failures.append("migration source split membership mismatch")
    return failures, source_split


def audit_dataset(dataset_path: Path, manifest_path: Path, raw_path: Path,
                  rules_path: Path, sample_size: int = 32,
                  expected_teacher: str = "fair") -> tuple[dict, dict | None]:
    failures: list[str] = []
    manifest = json.loads(manifest_path.read_text())
    rules = json.loads(rules_path.read_text())
    dataset_hash = sha256(dataset_path)
    rules_hash = sha256(rules_path)
    if manifest.get("dataset_sha256") != dataset_hash:
        failures.append("dataset_sha256 mismatch")
    if manifest.get("rules_manifest_sha256") != rules_hash:
        failures.append("rules_manifest_sha256 mismatch")
    if manifest.get("dataset_schema") != DATASET_SCHEMA or manifest.get("feature_schema") != FEATURE_SCHEMA:
        failures.append("manifest dataset/feature schema mismatch")
    if manifest.get("raw_schema") != RAW_SCHEMA:
        failures.append("manifest raw schema mismatch")
    if manifest.get("benchmark_protocol") != PROTOCOL or rules.get("local_protocol") != PROTOCOL:
        failures.append("benchmark protocol mismatch")
    if manifest.get("teacher") != expected_teacher:
        failures.append(f"teacher {manifest.get('teacher')!r} != expected {expected_teacher!r}")
    if manifest.get("engine", {}).get("commit") != rules.get("engine", {}).get("commit"):
        failures.append("engine commit mismatch")
    migration_failures, source_split = _check_migration(manifest, sha256(raw_path) if raw_path.is_file() else "")
    failures.extend(migration_failures)
    raw_feature_schema = (manifest.get("migration") or {}).get("source_feature_schema", FEATURE_SCHEMA)

    games = manifest.get("games", [])
    game_ids_in_manifest = [int(game["game_id"]) for game in games]
    if len(set(game_ids_in_manifest)) != len(game_ids_in_manifest):
        failures.append("duplicate game IDs in manifest")
    completed = {int(game["game_id"]): int(game["seed"]) for game in games
                 if game.get("status") == "completed"}
    if len(completed) < 100:
        failures.append(f"completed games {len(completed)} < 100")
    board_seeds = set(completed.values())
    if len(board_seeds) < 100:
        failures.append(f"distinct completed board seeds {len(board_seeds)} < 100")

    with np.load(dataset_path, allow_pickle=False) as npz:
        if npz["dataset_schema"].item() != DATASET_SCHEMA or npz["feature_schema"].item() != FEATURE_SCHEMA:
            failures.append("NPZ dataset/feature schema mismatch")
        count = len(npz["labels"])
        if count != manifest.get("unique_examples"):
            failures.append("manifest unique_examples differs from NPZ")
        if count < 20_000:
            failures.append(f"unique decisions {count} < 20000")
        game_ids = npz["game_ids"].astype(int)
        if any(int(game_id) not in completed for game_id in game_ids):
            failures.append("NPZ contains decisions from incomplete/unknown games")
        phases = npz["phases"]
        behaviors = npz["behaviors"]
        phase_counts = Counter(str(value) for value in phases)
        behavior_counts = Counter(str(value) for value in behaviors)
        labels = npz["labels"].astype(int)
        candidate_counts = npz["candidate_counts"].astype(int)
        if np.any(labels < 0) or np.any(labels >= candidate_counts):
            failures.append("NPZ contains invalid action labels")

        raw_report = {"status": "missing", "path": str(raw_path), "records": None,
                      "duplicates": None, "reconstructed_samples": 0,
                      "hidden_field_errors": 0}
        if raw_path.is_file() and manifest.get("raw_schema") == RAW_SCHEMA:
            raw_hash = sha256(raw_path)
            if manifest.get("raw_sha256") != raw_hash:
                failures.append("raw_sha256 mismatch")
            if manifest.get("raw_file") and Path(manifest["raw_file"]).resolve() != raw_path.resolve():
                failures.append("manifest raw_file path mismatch")
            samples = _sample_indices(count, sample_size, dataset_hash)
            array_names = ("state_features", "state_kinds", "state_ids", "action_features",
                           "action_types", "target_groups", "target_ids")
            # NPZ has no row seeking. Decompress one tensor at a time and retain
            # only sampled rows, so auditing a large corpus does not hold every
            # dense feature tensor in memory at once.
            encoded_samples: dict[int, dict[str, np.ndarray]] = {index: {} for index in samples}
            for name in array_names:
                if not samples:
                    break
                whole_array = npz[name]
                for index in samples:
                    encoded_samples[index][name] = whole_array[index].copy()
                del whole_array
            seen: set[str] = set()
            duplicates = 0
            raw_count = 0
            reconstructed = 0
            hidden_errors = 0
            metadata_errors = 0
            phase_mismatches = 0
            teacher_counts: Counter[str] = Counter()
            target_kinds: Counter[str] = Counter()
            seat_counts: Counter[int] = Counter()
            actors = npz["actor_colors"] if "actor_colors" in npz else None
            outcomes = npz["outcomes"] if "outcomes" in npz else None
            final_vps = npz["final_vps"] if "final_vps" in npz else None
            for index, record in enumerate(read_raw_records(raw_path)):
                raw_count += 1
                obs = record.observation
                key = observation_digest(obs)
                if key in seen:
                    duplicates += 1
                seen.add(key)
                teacher_counts[str(record.teacher)] += 1
                seat_counts[int(record.actor_seat)] += 1
                if not 0 <= int(record.actor_seat) < 4:
                    metadata_errors += 1
                if (obs.schema_version != SCHEMA_VERSION or obs.acting_relative_seat != 0
                        or any(player.own_resources is not None or player.own_development_cards is not None
                               for player in obs.players[1:])):
                    hidden_errors += 1
                if (record.feature_schema != raw_feature_schema
                        or record.benchmark_protocol != PROTOCOL
                        or record.rules_manifest_sha256 != rules_hash
                        or record.engine_commit != rules.get("engine", {}).get("commit")
                        or record.teacher != manifest.get("teacher")
                        or record.teacher_config_sha256 != manifest.get("raw_teacher_config_sha256")
                        or record.teacher_source_sha256 != manifest.get("raw_teacher_source_sha256")):
                    metadata_errors += 1
                if index >= count:
                    metadata_errors += 1
                    continue
                target = int(record.target_action_index)
                if not (0 <= target < len(obs.legal_actions)):
                    metadata_errors += 1
                    continue
                target_kinds[obs.legal_actions[target].kind] += 1
                if (int(record.game_id) != int(game_ids[index])
                        or int(record.board_seed) != completed.get(int(record.game_id))
                        or target != int(labels[index])
                        or len(obs.legal_actions) != int(candidate_counts[index])
                        or record.behavior != str(behaviors[index])
                        or (actors is not None and record.actor_color != str(actors[index]))
                        or (outcomes is not None and record.outcome != int(outcomes[index]))
                        or (final_vps is not None and record.final_vp != int(final_vps[index]))):
                    metadata_errors += 1
                if obs.phase != str(phases[index]):
                    phase_mismatches += 1
                if index in samples:
                    encoded = encode_visible_observation(obs)
                    state_key_matches = (raw_feature_schema != FEATURE_SCHEMA
                                         or encoded_digest(encoded) == record.canonical_state_key)
                    if (_array_matches(encoded_samples[index], encoded, len(obs.legal_actions))
                            and state_key_matches):
                        reconstructed += 1
                    else:
                        metadata_errors += 1
            if raw_count != count:
                failures.append(f"raw records {raw_count} != NPZ rows {count}")
            if manifest.get("raw_records") != raw_count:
                failures.append("manifest raw_records differs from raw file")
            if duplicates:
                failures.append(f"exact visible-state duplicates {duplicates}")
            if hidden_errors:
                failures.append(f"hidden-field boundary failures {hidden_errors}")
            if metadata_errors or phase_mismatches:
                failures.append(f"raw/NPZ reconstruction or metadata failures {metadata_errors + phase_mismatches}")
            if reconstructed != len(samples):
                failures.append(f"reconstructed samples {reconstructed} != requested {len(samples)}")
            raw_report = {"status": "checked", "path": str(raw_path), "sha256": raw_hash,
                          "records": raw_count, "duplicates": duplicates,
                          "reconstructed_samples": reconstructed,
                          "sample_size": len(samples), "hidden_field_errors": hidden_errors,
                          "metadata_errors": metadata_errors, "phase_mismatches": phase_mismatches,
                          "teacher_counts": dict(sorted(teacher_counts.items())),
                          "actor_seat_counts": dict(sorted(seat_counts.items())),
                          "target_action_kinds": dict(sorted(target_kinds.items()))}
        elif raw_path.is_file():
            raw_report["status"] = "unsupported_schema"
            failures.append(f"raw schema {manifest.get('raw_schema')!r} cannot be reconstructed by {RAW_SCHEMA}")
        else:
            failures.append("raw visible snapshots missing; legal-action reconstruction unavailable")

    split = build_split(completed, dataset_hash) if len(board_seeds) >= 3 else None
    if split is not None and source_split is not None:
        if (split["board_seeds"] != source_split.get("board_seeds")
                or split["game_ids"] != source_split.get("game_ids")
                or split["membership_sha256"] != source_split.get("membership_sha256")):
            failures.append("migrated dataset changed frozen board/game split membership")
    split_row_counts = {}
    if split is not None:
        for name, seeds in split["board_seeds"].items():
            seed_set = set(seeds)
            split_row_counts[name] = sum(completed.get(int(game_id)) in seed_set for game_id in game_ids)
        if any(not seeds for seeds in split["board_seeds"].values()):
            failures.append("one or more board splits are empty")
    else:
        failures.append("too few complete boards to create train/validation/final splits")
    report = {
        "schema": AUDIT_SCHEMA,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "manifest": str(manifest_path),
        "rules_manifest": str(rules_path),
        "rules_manifest_sha256": rules_hash,
        "completed_games": len(completed),
        "distinct_completed_board_seeds": len(board_seeds),
        "unique_decisions": count,
        "phase_counts": dict(sorted(phase_counts.items())),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "raw": raw_report,
        "migration": manifest.get("migration"),
        "split_row_counts": split_row_counts,
        "split_membership_sha256": split["membership_sha256"] if split else None,
        "gate2_pass": not failures,
        "failures": failures,
    }
    return report, split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--rules", type=Path, default=WORKSPACE / "catanLabs/rules_manifest.json")
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--expected-teacher", default="fair")
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    dataset = args.dataset.resolve(strict=True)
    manifest = args.manifest or dataset.with_suffix(".json")
    raw = args.raw or dataset.with_suffix(".raw.jsonl.gz")
    report_path = args.report or dataset.with_suffix(".audit.json")
    split_path = args.split_file or dataset.with_suffix(".split.json")
    report, split = audit_dataset(dataset, manifest, raw, args.rules, args.sample_size,
                                  args.expected_teacher)
    if report["gate2_pass"] and split is not None:
        freeze_split(split_path, split)
        report["split_file"] = str(split_path)
        report["split_file_sha256"] = sha256(split_path)
    else:
        report["split_file"] = None
        report["split_file_sha256"] = None
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"gate2_pass": report["gate2_pass"],
                      "completed_games": report["completed_games"],
                      "unique_decisions": report["unique_decisions"],
                      "failures": report["failures"], "report": str(report_path)}, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
