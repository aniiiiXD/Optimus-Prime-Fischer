"""Merge audited v3 base data with fair-labelled v3 on-policy decisions.

Base rows have priority during exact visible-state deduplication. The frozen
base final games stay sealed. On-policy games receive new IDs and enter train,
apart from a deterministic validation reserve when there are enough boards.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from catanLabs.Transformer.audit_dataset import (
    SPLIT_SCHEMA, canonical_bytes, encoded_digest, sha256,
)
from catanLabs.Transformer.model import (
    ACTION_FEATURE_DIM, DATASET_SCHEMA, FEATURE_DIM, FEATURE_SCHEMA,
    STATE_TOKEN_COUNT, encode_visible_observation,
)
from catanLabs.Transformer.raw_storage import RAW_SCHEMA, RawRecordWriter, read_raw_records


MERGE_SCHEMA = "catanlabs-v3-on-policy-merge-v1"
VALIDATION_SALT = "catanlabs-on-policy-validation-boards-v1"
MIN_ON_POLICY_VALIDATION_BOARDS = 10
MIN_SUPPLEMENTAL_COMPLETED_GAMES = 10
ARRAY_NAMES = (
    "state_features", "state_kinds", "state_ids", "action_features",
    "action_types", "target_groups", "target_ids",
)
METADATA_NAMES = (
    "candidate_counts", "labels", "game_ids", "phases", "behaviors",
    "actor_colors", "outcomes", "final_vps",
)


def choose_on_policy_validation(board_seeds: set[int]) -> set[int]:
    """Reserve a stable 10% of on-policy boards only once ten exist."""
    if len(board_seeds) < MIN_ON_POLICY_VALIDATION_BOARDS:
        return set()
    count = max(1, round(len(board_seeds) * 0.1))
    ranked = sorted(board_seeds, key=lambda seed: (
        hashlib.sha256(f"{VALIDATION_SALT}:{seed}".encode()).hexdigest(), seed,
    ))
    return set(ranked[:count])


def merged_split(base_split: dict, added: dict[int, tuple[int, str]],
                 dataset_sha256: str) -> dict:
    """Keep every base assignment, then add shifted on-policy game IDs."""
    if base_split.get("schema") != SPLIT_SCHEMA:
        raise ValueError("Unsupported base split schema")
    boards = {name: set(base_split["board_seeds"][name])
              for name in ("train", "validation", "final")}
    games = {name: set(base_split["game_ids"][name])
             for name in ("train", "validation", "final")}
    for game_id, (board_seed, partition) in added.items():
        if partition not in ("train", "validation"):
            raise ValueError("On-policy games cannot enter final")
        if game_id in games["train"] | games["validation"] | games["final"]:
            raise ValueError("Shifted game ID overlaps base split")
        if board_seed in boards["train"] | boards["validation"] | boards["final"]:
            raise ValueError("On-policy board seed overlaps an existing split")
        games[partition].add(game_id)
        boards[partition].add(board_seed)
    board_groups = {name: sorted(values) for name, values in boards.items()}
    game_groups = {name: sorted(values) for name, values in games.items()}
    membership = {"board_seeds": board_groups, "game_ids": game_groups}
    return {
        "schema": SPLIT_SCHEMA,
        "salt_sha256": hashlib.sha256(VALIDATION_SALT.encode()).hexdigest(),
        "dataset_sha256": dataset_sha256,
        **membership,
        "split_sha256": {name: hashlib.sha256(canonical_bytes({
            "board_seeds": board_groups[name], "game_ids": game_groups[name],
        })).hexdigest() for name in board_groups},
        "membership_sha256": hashlib.sha256(canonical_bytes(membership)).hexdigest(),
        "base_membership_sha256": base_split["membership_sha256"],
        "on_policy_validation_min_boards": MIN_ON_POLICY_VALIDATION_BOARDS,
    }


def _source(path: Path, *, audited: bool) -> dict:
    path = path.resolve(strict=True)
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    dataset_hash = sha256(path)
    if (manifest.get("dataset_sha256") != dataset_hash
            or manifest.get("dataset_schema") != DATASET_SCHEMA
            or manifest.get("feature_schema") != FEATURE_SCHEMA
            or manifest.get("raw_schema") != RAW_SCHEMA
            or manifest.get("teacher") != "fair"):
        raise ValueError(f"Source dataset/teacher schema or hash mismatch: {path}")
    raw_path = Path(manifest["raw_file"])
    if not raw_path.is_absolute():
        raw_path = WORKSPACE / raw_path
    raw_path = raw_path.resolve(strict=True)
    raw_hash = sha256(raw_path)
    if raw_hash != manifest.get("raw_sha256"):
        raise ValueError(f"Source raw hash mismatch: {raw_path}")
    audit_path = path.with_suffix(".audit.json") if audited else None
    split_path = path.with_suffix(".split.json") if audited else None
    split = None
    if audited:
        audit = json.loads(audit_path.read_text())
        split = json.loads(split_path.read_text())
        if (audit.get("gate2_pass") is not True
                or audit.get("dataset_sha256") != dataset_hash
                or audit.get("raw", {}).get("sha256") != raw_hash
                or audit.get("split_file_sha256") != sha256(split_path)
                or split.get("dataset_sha256") != dataset_hash
                or split.get("membership_sha256") != audit.get("split_membership_sha256")):
            raise ValueError("Base dataset is not bound to its passed audit and frozen split")
    with np.load(path, allow_pickle=False) as npz:
        if (npz["dataset_schema"].item() != DATASET_SCHEMA
                or npz["feature_schema"].item() != FEATURE_SCHEMA):
            raise ValueError("Source NPZ feature schema mismatch")
        metadata = {name: npz[name] for name in METADATA_NAMES}
    if len(metadata["labels"]) != manifest.get("unique_examples") or len(metadata["labels"]) != manifest.get("raw_records"):
        raise ValueError("Source NPZ, manifest and raw record counts disagree")
    return {
        "path": path, "manifest_path": manifest_path, "manifest": manifest,
        "dataset_sha256": dataset_hash, "raw_path": raw_path,
        "raw_sha256": raw_hash, "audit_path": audit_path,
        "split_path": split_path, "split": split, "metadata": metadata,
    }


def _assert_compatible(base: dict, on_policy: dict) -> None:
    a, b = base["manifest"], on_policy["manifest"]
    for name in ("benchmark_protocol", "rules_manifest_sha256",
                 "raw_teacher_config_sha256", "raw_teacher_source_sha256"):
        if a.get(name) != b.get(name):
            raise ValueError(f"Base and on-policy {name} differ")
    if a.get("engine", {}).get("commit") != b.get("engine", {}).get("commit"):
        raise ValueError("Base and on-policy engine commits differ")
    if b.get("behavior") != "transformer_selfplay" or b.get("smoke_only") is not False or b.get("behavior_checkpoint_used") is not True:
        raise ValueError("On-policy source must be real v3 Transformer self-play")
    if b.get("collector_schema") != "catanlabs-on-policy-fair-labels-v1":
        raise ValueError("Unsupported on-policy collector schema")
    completed = [game for game in b.get("games", ()) if game.get("status") == "completed"]
    if len(completed) < MIN_SUPPLEMENTAL_COMPLETED_GAMES:
        raise ValueError(f"Supplemental on-policy corpus needs at least {MIN_SUPPLEMENTAL_COMPLETED_GAMES} complete games")


def _check_row(record, metadata: dict, index: int, source: dict, completed: dict[int, int]) -> None:
    if index >= len(metadata["labels"]):
        raise ValueError("Raw source has more rows than NPZ")
    if (record.game_id != int(metadata["game_ids"][index])
            or record.board_seed != completed.get(record.game_id)
            or record.phase != str(metadata["phases"][index])
            or record.behavior != str(metadata["behaviors"][index])
            or record.actor_color != str(metadata["actor_colors"][index])
            or record.target_action_index != int(metadata["labels"][index])
            or record.outcome != int(metadata["outcomes"][index])
            or record.final_vp != int(metadata["final_vps"][index])
            or len(record.observation.legal_actions) != int(metadata["candidate_counts"][index])
            or record.benchmark_protocol != source["manifest"]["benchmark_protocol"]
            or record.rules_manifest_sha256 != source["manifest"]["rules_manifest_sha256"]
            or record.engine_commit != source["manifest"]["engine"]["commit"]
            or record.teacher != "fair"
            or record.teacher_config_sha256 != source["manifest"]["raw_teacher_config_sha256"]
            or record.teacher_source_sha256 != source["manifest"]["raw_teacher_source_sha256"]):
        raise ValueError(f"Raw/NPZ provenance mismatch at {source['path']} row {index}")
    if any(player.own_resources is not None or player.own_development_cards is not None
           for player in record.observation.players[1:]):
        raise ValueError("Raw observation exposes opponent private cards")


def _tensor_files(folder: Path, upper_rows: int, max_candidates: int) -> dict[str, np.memmap]:
    specs = {
        "state_features": ((upper_rows, STATE_TOKEN_COUNT, FEATURE_DIM), np.float32),
        "state_kinds": ((upper_rows, STATE_TOKEN_COUNT), np.int64),
        "state_ids": ((upper_rows, STATE_TOKEN_COUNT), np.int64),
        "action_features": ((upper_rows, max_candidates, ACTION_FEATURE_DIM), np.float32),
        "action_types": ((upper_rows, max_candidates), np.int64),
        "target_groups": ((upper_rows, max_candidates), np.int64),
        "target_ids": ((upper_rows, max_candidates), np.int64),
    }
    tensors = {name: np.lib.format.open_memmap(folder / f"{name}.npy", mode="w+",
                                                  dtype=dtype, shape=shape)
               for name, (shape, dtype) in specs.items()}
    for name in ("action_features", "action_types", "target_groups", "target_ids"):
        tensors[name][:] = 0
    return tensors


def merge(base_dataset: Path, on_policy_dataset: Path, output: Path) -> dict:
    if DATASET_SCHEMA != "catanlabs-transformer-dataset-v3" or FEATURE_SCHEMA != "visible-state-tokens-v3":
        raise RuntimeError("Merge requires v3 encoder and dataset schemas")
    base = _source(base_dataset, audited=True)
    on_policy = _source(on_policy_dataset, audited=False)
    _assert_compatible(base, on_policy)
    output = output.resolve()
    output_raw = output.with_name(output.stem + ".raw.jsonl.gz")
    paths = (output, output_raw, output.with_suffix(".json"),
             output.with_suffix(".split.json"), output.with_suffix(".audit.json"))
    if any(path.exists() for path in paths):
        raise FileExistsError("Merged output exists; merge never overwrites an artifact")
    output.parent.mkdir(parents=True, exist_ok=True)

    base_manifest, on_manifest = base["manifest"], on_policy["manifest"]
    base_games = {int(game["game_id"]): int(game["seed"])
                  for game in base_manifest["games"] if game.get("status") == "completed"}
    on_games = {int(game["game_id"]): int(game["seed"])
                for game in on_manifest["games"] if game.get("status") == "completed"}
    if len(base_games) != len(set(base_games)) or len(on_games) != len(set(on_games)):
        raise ValueError("Duplicate game IDs")
    base_boards = set(base_games.values())
    if base_boards & set(on_games.values()):
        raise ValueError("On-policy board seeds overlap base corpus")
    if len(set(on_games.values())) != len(on_games):
        raise ValueError("On-policy board seeds repeat")
    shift = max(int(game["game_id"]) for game in base_manifest["games"]) + 1
    game_map = {old: shift + old for old in on_games}
    validation_boards = choose_on_policy_validation(set(on_games.values()))
    upper_rows = len(base["metadata"]["labels"]) + len(on_policy["metadata"]["labels"])
    max_candidates = max(int(base_manifest["max_legal_actions"]), int(on_manifest["max_legal_actions"]))
    duplicate_counts = Counter()
    phase_counts = Counter()
    behavior_counts = Counter()
    kept_by_game = Counter()
    metadata_lists = {name: [] for name in METADATA_NAMES}
    seen: set[str] = set()
    kept = 0

    with tempfile.TemporaryDirectory(prefix="catan-merge-", dir=output.parent) as tempdir:
        temp = Path(tempdir)
        tensors = _tensor_files(temp, upper_rows, max_candidates)
        temporary_raw = temp / "dataset.raw.jsonl.gz"
        writer = RawRecordWriter(temporary_raw)
        for source_name, source, complete in (
            ("base", base, base_games), ("on_policy", on_policy, on_games),
        ):
            source_count = 0
            source_feature_schema = (source["manifest"].get("migration") or {}).get(
                "source_feature_schema", FEATURE_SCHEMA,
            )
            for index, record in enumerate(read_raw_records(source["raw_path"])):
                source_count += 1
                _check_row(record, source["metadata"], index, source, complete)
                if record.feature_schema != source_feature_schema:
                    raise ValueError(f"Unexpected {source_name} raw feature schema")
                encoded = encode_visible_observation(record.observation)
                digest = encoded_digest(encoded)
                if source_feature_schema == FEATURE_SCHEMA and digest != record.canonical_state_key:
                    raise ValueError(f"{source_name} canonical state key changed at row {index}")
                if digest in seen:
                    duplicate_counts[source_name] += 1
                    continue
                seen.add(digest)
                target_game_id = record.game_id if source_name == "base" else game_map[record.game_id]
                raw_record = replace(record, game_id=target_game_id,
                                     canonical_state_key=digest,
                                     feature_schema=FEATURE_SCHEMA)
                writer.write(raw_record)
                width = len(record.observation.legal_actions)
                for name, array in zip(ARRAY_NAMES, encoded):
                    if name in ("action_features", "action_types", "target_groups", "target_ids"):
                        tensors[name][kept, :width] = array
                    else:
                        tensors[name][kept] = array
                for name in METADATA_NAMES:
                    value = source["metadata"][name][index]
                    metadata_lists[name].append(target_game_id if name == "game_ids" else value)
                phase_counts[record.phase] += 1
                behavior_counts[record.behavior] += 1
                kept_by_game[target_game_id] += 1
                kept += 1
            if source_count != len(source["metadata"]["labels"]):
                raise ValueError(f"{source_name} raw/NPZ row count mismatch")
        raw_hash = writer.close()
        if writer.count != kept:
            raise AssertionError("Merged raw and tensor row counts differ")
        if duplicate_counts["base"]:
            raise ValueError("Audited base contains duplicate v3 encoded states")
        for tensor in tensors.values():
            tensor.flush()
        arrays = {name: tensor[:kept] for name, tensor in tensors.items()}
        arrays.update({name: np.asarray(values) for name, values in metadata_lists.items()})
        temporary_npz = temp / "dataset.npz"
        np.savez_compressed(temporary_npz,
                            dataset_schema=np.asarray(DATASET_SCHEMA),
                            feature_schema=np.asarray(FEATURE_SCHEMA),
                            **arrays)
        dataset_hash = sha256(temporary_npz)
        added = {game_map[old]: (board_seed, "validation" if board_seed in validation_boards else "train")
                 for old, board_seed in on_games.items() if kept_by_game[game_map[old]]}
        split = merged_split(base["split"], added, dataset_hash)
        actual_game_ids = set(int(value) for value in metadata_lists["game_ids"])
        if actual_game_ids != set().union(*(set(group) for group in split["game_ids"].values())):
            raise ValueError("Merged split does not partition rows")
        for name in ("train", "validation", "final"):
            if set(base["split"]["game_ids"][name]) - set(split["game_ids"][name]):
                raise ValueError("Base game assignment changed")

        games = list(base_manifest["games"])
        for game in on_manifest["games"]:
            shifted = dict(game)
            shifted["source_game_id"] = int(game["game_id"])
            shifted["game_id"] = shift + int(game["game_id"])
            shifted["source"] = "on_policy"
            shifted["stored_examples"] = kept_by_game[shifted["game_id"]]
            if shifted["status"] == "completed" and not shifted["stored_examples"]:
                shifted["status"] = "duplicate_only"
            games.append(shifted)
        source_hashes = {
            "base": {"dataset": str(base["path"]), "dataset_sha256": base["dataset_sha256"],
                     "manifest_sha256": sha256(base["manifest_path"]),
                     "audit_sha256": sha256(base["audit_path"]),
                     "split_sha256": sha256(base["split_path"]),
                     "split_membership_sha256": base["split"]["membership_sha256"],
                     "raw_sha256": base["raw_sha256"]},
            "on_policy": {"dataset": str(on_policy["path"]),
                          "dataset_sha256": on_policy["dataset_sha256"],
                          "manifest_sha256": sha256(on_policy["manifest_path"]),
                          "raw_sha256": on_policy["raw_sha256"],
                          "behavior_checkpoint_sha256": on_manifest["behavior_checkpoint_sha256"]},
        }
        manifest = {
            "schema": DATASET_SCHEMA, "dataset_schema": DATASET_SCHEMA,
            "feature_schema": FEATURE_SCHEMA, "raw_schema": RAW_SCHEMA,
            "merge_schema": MERGE_SCHEMA,
            "dataset_sha256": dataset_hash,
            "raw_file": str(output_raw), "raw_records": kept,
            "raw_sha256": raw_hash,
            "teacher": "fair", "teacher_config_sha256": base_manifest["teacher_config_sha256"],
            "teacher_source_sha256": base_manifest["teacher_source_sha256"],
            "raw_teacher_config_sha256": base_manifest["raw_teacher_config_sha256"],
            "raw_teacher_source_sha256": base_manifest["raw_teacher_source_sha256"],
            "benchmark_protocol": base_manifest["benchmark_protocol"],
            "rules_manifest_sha256": base_manifest["rules_manifest_sha256"],
            "engine": base_manifest["engine"],
            "source_hashes": source_hashes,
            "game_id_shift": shift,
            "on_policy_game_id_map": {str(key): value for key, value in sorted(game_map.items())},
            "on_policy_validation_boards": sorted(validation_boards),
            "on_policy_validation_min_boards": MIN_ON_POLICY_VALIDATION_BOARDS,
            "base_final_game_ids": base["split"]["game_ids"]["final"],
            "base_final_board_seeds": base["split"]["board_seeds"]["final"],
            "source_rows": {"base": len(base["metadata"]["labels"]),
                            "on_policy": len(on_policy["metadata"]["labels"])},
            "source_roles": {"base": "audited_gate2_100_game_base",
                             "on_policy": "supplemental_completed_games_not_gate2_base"},
            "source_completed_games": {"base": len(base_games), "on_policy": len(on_games)},
            "minimum_supplemental_completed_games": MIN_SUPPLEMENTAL_COMPLETED_GAMES,
            "duplicate_rows_dropped": dict(duplicate_counts),
            "unique_examples": kept,
            "max_legal_actions": max_candidates,
            "games": games,
            "phases": dict(sorted(phase_counts.items())),
            "trajectory_policies": dict(sorted(behavior_counts.items())),
            "feature_boundary": "player-visible raw snapshots; opponent card identities absent",
            "teacher_action_contract": "Labels predate behavior actions; outcomes are not policy inputs.",
            "split_membership_sha256": split["membership_sha256"],
            "collector_sha256": sha256(Path(__file__)),
            "encoder_sha256": sha256(WORKSPACE / "catanLabs/Transformer/model.py"),
        }
        os.replace(temporary_npz, output)
        os.replace(temporary_raw, output_raw)
        output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        output.with_suffix(".split.json").write_text(json.dumps(split, indent=2, allow_nan=False) + "\n")
    return {"dataset": str(output), "rows": kept,
            "duplicate_rows_dropped": dict(duplicate_counts),
            "split_membership_sha256": split["membership_sha256"],
            "dataset_sha256": dataset_hash,
            "raw_sha256": raw_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
                        default=WORKSPACE / "catanLabs/Transformer/runs/gate2_native_v3/dataset.npz")
    parser.add_argument("--on-policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(merge(args.base, args.on_policy, args.output), indent=2))


if __name__ == "__main__":
    main()
