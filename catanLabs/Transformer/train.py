"""Train the first from-scratch Catan action Transformer on CPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from catanLabs.Transformer.model import (
    ACTION_FEATURE_DIM,
    CHECKPOINT_SCHEMA,
    DATASET_SCHEMA,
    FEATURE_DIM,
    FEATURE_SCHEMA,
    STATE_TOKEN_COUNT,
    CatanActionTransformer,
)


RARE_TACTICAL_PHASES = frozenset({"BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD", "MOVE_ROBBER"})


def validate_dataset_schema(raw):
    """Reject encoded tensors from an incompatible observation layout."""
    expected = {"dataset_schema": DATASET_SCHEMA, "feature_schema": FEATURE_SCHEMA}
    for field, value in expected.items():
        found = str(raw[field].item()) if field in raw.files else None
        if found != value:
            raise ValueError(
                f"Dataset {field} is {found!r}; expected {value!r}. "
                "Regenerate the encoded dataset with the current collector."
            )
    if raw["state_features"].shape[1:] != (STATE_TOKEN_COUNT, FEATURE_DIM):
        raise ValueError("Dataset state_features shape does not match the current feature schema")
    if raw["action_features"].shape[-1] != ACTION_FEATURE_DIM:
        raise ValueError("Dataset action_features shape does not match the current feature schema")


def teacher_from_manifest(dataset_path: Path, dataset_sha256: str) -> str:
    """Attach the collector's actual teacher identity to a checkpoint."""
    manifest_path = dataset_path.with_suffix(".json")
    if not manifest_path.exists():
        return "unknown (dataset manifest unavailable)"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Dataset manifest hash does not match the training NPZ")
    teacher = manifest.get("teacher")
    if not isinstance(teacher, str) or not teacher:
        raise ValueError("Dataset manifest has no teacher identity")
    return teacher


def choice_training_indices(train_indices, candidate_counts, include_forced=False):
    """Skip one-action rows: their masked cross-entropy and gradient are zero."""
    selected = np.asarray(train_indices, dtype=np.int64)
    if not include_forced:
        selected = selected[np.asarray(candidate_counts)[selected] > 1]
    if len(selected) == 0:
        raise ValueError("Training split contains no decisions with a choice")
    return selected


def epoch_indices(indices, phases, rare_phase_multiplier, rng):
    """Return one deterministic epoch draw, optionally emphasizing rare tactical phases."""
    indices = np.asarray(indices, dtype=np.int64)
    if rare_phase_multiplier == 1.0:
        return rng.permutation(indices)
    weights = np.ones(len(indices), dtype=np.float64)
    weights[np.isin(np.asarray(phases)[indices], list(RARE_TACTICAL_PHASES))] = rare_phase_multiplier
    return rng.choice(indices, size=len(indices), replace=True, p=weights / weights.sum())


def _metrics(model, data, indices, batch_size):
    if len(indices) == 0:
        return {"loss": None, "accuracy": None, "examples": 0}
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    device = torch.device("cpu")
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]
            counts = data["counts"][ids]
            width = int(counts.max())
            action_mask = torch.arange(width, device=device)[None, :] < torch.as_tensor(counts, device=device)[:, None]
            logits = model(
                data["state_features"][ids], data["state_kinds"][ids], data["state_ids"][ids],
                data["action_features"][ids, :width], data["action_types"][ids, :width],
                data["target_groups"][ids, :width], data["target_ids"][ids, :width], action_mask,
            )
            labels = data["labels"][ids]
            loss = F.cross_entropy(logits.masked_fill(~action_mask, -1e9), labels, reduction="sum")
            total_loss += float(loss)
            legal_logits = logits.masked_fill(~action_mask, -1e9)
            correct += int((legal_logits.argmax(dim=1) == labels).sum())
            total += len(ids)
    model.train()
    return {"loss": total_loss / total, "accuracy": correct / total, "examples": total}


def split_indices(game_ids, seed, split_file, dataset_sha256):
    """Return train/validation rows while keeping audited final games sealed."""
    unique_games = sorted(set(np.asarray(game_ids, dtype=int).tolist()))
    final_games: set[int] = set()
    split_file_sha256 = None
    if split_file is not None:
        split_bytes = Path(split_file).read_bytes()
        split = json.loads(split_bytes)
        if split.get("schema") != "catanlabs-board-seed-split-v1":
            raise ValueError("Unsupported board-seed split schema")
        if split.get("dataset_sha256") != dataset_sha256:
            raise ValueError("Split file belongs to another dataset")
        groups = split.get("game_ids", {})
        train_games = set(groups.get("train", ()))
        heldout_games = set(groups.get("validation", ()))
        final_games = set(groups.get("final", ()))
        if not train_games or not heldout_games or not final_games:
            raise ValueError("Split must include nonempty train, validation, and final game groups")
        if (train_games & heldout_games) or (train_games & final_games) or (heldout_games & final_games):
            raise ValueError("Split game groups overlap")
        if train_games | heldout_games | final_games != set(unique_games):
            raise ValueError("Split game groups do not partition the dataset")
        train_indices = np.flatnonzero(np.isin(game_ids, list(train_games)))
        val_indices = np.flatnonzero(np.isin(game_ids, list(heldout_games)))
        split_name = "audited disjoint board-seed split"
        split_file_sha256 = hashlib.sha256(split_bytes).hexdigest()
    else:
        heldout_games = set(unique_games[max(1, int(len(unique_games) * 0.8)):]) if len(unique_games) >= 3 else set()
        if heldout_games:
            val_indices = np.flatnonzero(np.isin(game_ids, list(heldout_games)))
            train_indices = np.flatnonzero(~np.isin(game_ids, list(heldout_games)))
            split_name = "held-out complete games"
        else:
            rng = np.random.default_rng(seed)
            permutation = rng.permutation(len(game_ids))
            val_count = max(1, int(0.15 * len(game_ids)))
            val_indices, train_indices = permutation[:val_count], permutation[val_count:]
            split_name = "random positions (too few games for group split)"
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("Dataset split produced an empty train or validation group")
    return train_indices, val_indices, heldout_games, final_games, split_name, split_file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "runs/pilot/dataset.npz")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Warm-start from a compatible Transformer; optimizer starts fresh")
    parser.add_argument("--split-file", type=Path, help="Audited board-seed split; final IDs are never used for training or validation")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs/pilot")
    parser.add_argument("--seconds", type=float, default=600.0, help="Hard optimizer wall-time cap; never exceeds 600 seconds")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=61015)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--include-forced-training", action="store_true",
                        help="Reproduce older runs that spent updates on one-action positions")
    parser.add_argument("--patience-epochs", type=int, default=6,
                        help="Stop after this many complete epochs without validation-loss improvement")
    parser.add_argument("--rare-phase-multiplier", type=float, default=1.0,
                        help="Oversample opening/robber rows per epoch; 1 preserves uniform training")
    args = parser.parse_args()
    if (args.seconds <= 0 or args.batch_size < 1 or args.threads < 1
            or args.patience_epochs < 1 or args.learning_rate <= 0
            or args.rare_phase_multiplier < 1.0):
        parser.error("seconds, batch-size, learning-rate, threads and patience-epochs must be positive")
    if args.split_file is None and args.dataset.with_suffix(".split.json").exists():
        parser.error("This dataset has an audited split; pass --split-file to keep final games sealed")
    # Leave a small shutdown margin so checkpoint/report writing stays below
    # the user's hard ten-minute CPU training cap.
    budget = min(float(args.seconds), 590.0)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw = np.load(args.dataset, allow_pickle=False)
    validate_dataset_schema(raw)
    dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    teacher_name = teacher_from_manifest(args.dataset, dataset_sha256)
    n = len(raw["labels"])
    if n < 2:
        raise ValueError(f"Need at least two training examples, found {n}")
    max_candidates = int(raw["candidate_counts"].max())
    data = {
        "state_features": torch.from_numpy(raw["state_features"].astype(np.float32, copy=False)),
        "state_kinds": torch.from_numpy(raw["state_kinds"].astype(np.int64, copy=False)),
        "state_ids": torch.from_numpy(raw["state_ids"].astype(np.int64, copy=False)),
        "action_features": torch.from_numpy(raw["action_features"].astype(np.float32, copy=False)),
        "action_types": torch.from_numpy(raw["action_types"].astype(np.int64, copy=False)),
        "target_groups": torch.from_numpy(raw["target_groups"].astype(np.int64, copy=False)),
        "target_ids": torch.from_numpy(raw["target_ids"].astype(np.int64, copy=False)),
        "counts": raw["candidate_counts"].astype(np.int64, copy=False),
        "labels": torch.from_numpy(raw["labels"].astype(np.int64, copy=False)),
    }
    game_ids = raw["game_ids"].astype(int)
    phases = raw["phases"].astype(str)
    unique_games = sorted(set(game_ids.tolist()))
    train_indices, val_indices, heldout_games, final_games, split_name, split_file_sha256 = split_indices(
        game_ids, args.seed, args.split_file, dataset_sha256
    )
    split_train_count = len(train_indices)
    train_indices = choice_training_indices(train_indices, data["counts"], args.include_forced_training)
    forced_excluded = split_train_count - len(train_indices)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_config = {"width": args.width, "layers": args.layers, "heads": args.heads}
    model = CatanActionTransformer(**model_config).cpu()
    initialization = None
    if args.init_checkpoint is not None:
        source_path = args.init_checkpoint.resolve(strict=True)
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        if (source.get("schema") != CHECKPOINT_SCHEMA
                or source.get("feature_schema") != FEATURE_SCHEMA
                or source.get("dataset_schema") != DATASET_SCHEMA
                or source.get("model_config") != model_config):
            raise ValueError("Warm-start checkpoint schema or architecture is incompatible")
        model.load_state_dict(source["model_state"])
        initialization = {
            "checkpoint": str(source_path),
            "checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "source_dataset_sha256": source.get("dataset_sha256"),
            "source_split_file_sha256": source.get("split_file_sha256"),
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    print(json.dumps({
        "event": "training_started", "device": "cpu", "optimizer_budget_seconds": budget,
        "examples": n, "train_examples": len(train_indices), "validation_examples": len(val_indices),
        "forced_training_rows_excluded": forced_excluded,
        "unique_games": len(unique_games), "split": split_name,
        "max_legal_actions": max_candidates, "state_tokens": STATE_TOKEN_COUNT,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": {"width": args.width, "layers": args.layers, "heads": args.heads},
        "batch_size": args.batch_size, "threads": args.threads,
        "learning_rate": args.learning_rate,
        "rare_phase_multiplier": args.rare_phase_multiplier,
        "rare_phase_names": sorted(RARE_TACTICAL_PHASES),
        "initialization": initialization,
    }), flush=True)

    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation = _metrics(model, data, val_indices, args.batch_size)
    best_loss = validation["loss"] if validation["loss"] is not None else float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    started = time.perf_counter()
    next_report = 0.0
    epoch = 0
    epochs_without_improvement = 0
    stopped_reason = "cpu_time_cap"

    while time.perf_counter() - started < budget:
        epoch += 1
        model.train()
        permutation = epoch_indices(train_indices, phases, args.rare_phase_multiplier, np.random)
        for start in range(0, len(permutation), args.batch_size):
            if time.perf_counter() - started >= budget:
                break
            ids = permutation[start:start + args.batch_size]
            counts = data["counts"][ids]
            width = int(counts.max())
            action_mask = torch.arange(width)[None, :] < torch.as_tensor(counts)[:, None]
            logits = model(
                data["state_features"][ids], data["state_kinds"][ids], data["state_ids"][ids],
                data["action_features"][ids, :width], data["action_types"][ids, :width],
                data["target_groups"][ids, :width], data["target_ids"][ids, :width], action_mask,
            )
            labels = data["labels"][ids]
            loss = F.cross_entropy(logits.masked_fill(~action_mask, -1e9), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        validation = _metrics(model, data, val_indices, args.batch_size)
        if validation["loss"] is not None and validation["loss"] < best_loss - 1e-4:
            best_loss = validation["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "schema": CHECKPOINT_SCHEMA,
                "model_config": {"width": args.width, "layers": args.layers, "heads": args.heads},
                "model_state": best_state,
                "seed": args.seed,
                "teacher": teacher_name,
                "include_forced_training": args.include_forced_training,
                "learning_rate": args.learning_rate,
                "rare_phase_multiplier": args.rare_phase_multiplier,
                "initialization": initialization,
                "feature_schema": FEATURE_SCHEMA,
                "dataset_schema": DATASET_SCHEMA,
                "dataset_sha256": dataset_sha256,
                "split_file_sha256": split_file_sha256,
                "validation": validation,
                "epoch": epoch,
            }, args.output_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        elapsed = time.perf_counter() - started
        if elapsed >= next_report:
            print(json.dumps({"event": "training_progress", "epoch": epoch,
                              "elapsed_seconds": round(elapsed, 1), "remaining_seconds": round(max(0, budget - elapsed), 1),
                              "validation": validation, "best_epoch": best_epoch,
                              "best_validation_loss": best_loss}), flush=True)
            next_report = elapsed + 15.0
        if epochs_without_improvement >= args.patience_epochs:
            stopped_reason = "validation_plateau"
            break

    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    final_validation = _metrics(model, data, val_indices, args.batch_size)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "model_config": {"width": args.width, "layers": args.layers, "heads": args.heads},
        "model_state": best_state,
        "seed": args.seed,
        "teacher": teacher_name,
        "include_forced_training": args.include_forced_training,
        "learning_rate": args.learning_rate,
        "rare_phase_multiplier": args.rare_phase_multiplier,
        "initialization": initialization,
        "feature_schema": FEATURE_SCHEMA,
        "dataset_schema": DATASET_SCHEMA,
        "dataset_sha256": dataset_sha256,
        "split_file_sha256": split_file_sha256,
        "validation": final_validation,
        "epoch": best_epoch,
    }
    torch.save(checkpoint, args.output_dir / "best.pt")
    result = {
        "device": "cpu",
        "optimizer_budget_seconds": budget,
        "optimizer_elapsed_seconds": round(training_seconds, 3),
        "budget_respected": training_seconds <= 600.0,
        "epochs": epoch,
        "stopped_reason": stopped_reason,
        "best_epoch": best_epoch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "examples": n,
        "train_examples": len(train_indices),
        "forced_training_rows_excluded": forced_excluded,
        "include_forced_training": args.include_forced_training,
        "learning_rate": args.learning_rate,
        "rare_phase_multiplier": args.rare_phase_multiplier,
        "initialization": initialization,
        "validation_examples": len(val_indices),
        "unique_games": len(unique_games),
        "heldout_games": sorted(heldout_games),
        "sealed_final_games": sorted(final_games),
        "split": split_name,
        "split_file": str(args.split_file) if args.split_file else None,
        "split_file_sha256": split_file_sha256,
        "validation": final_validation,
        "architecture": {"width": args.width, "layers": args.layers, "heads": args.heads},
        "checkpoint": str(args.output_dir / "best.pt"),
        "dataset": str(args.dataset),
        "dataset_sha256": checkpoint["dataset_sha256"],
        "dataset_schema": DATASET_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
    }
    (args.output_dir / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "training_finished", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
