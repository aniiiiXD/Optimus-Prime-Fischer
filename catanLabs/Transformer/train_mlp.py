"""Train a parameter-matched action MLP on the v2 Catan CPU dataset."""

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

from catanLabs.Transformer.mlp import CHECKPOINT_SCHEMA, CatanActionMLP
from catanLabs.Transformer.model import DATASET_SCHEMA, FEATURE_SCHEMA, STATE_TOKEN_COUNT
from catanLabs.Transformer.train import _metrics, split_indices, validate_dataset_schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "runs/pilot/dataset.npz")
    parser.add_argument("--split-file", type=Path, help="Audited whole-game train/validation/final split")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs/mlp")
    parser.add_argument("--seconds", type=float, default=600.0, help="CPU optimizer wall budget, capped at 590 seconds")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=61015)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--patience-epochs", type=int, default=6,
                        help="Stop after this many complete epochs without validation-loss improvement")
    args = parser.parse_args()
    if args.seconds <= 0 or args.batch_size < 1 or args.threads < 1 or args.width < 1 or args.patience_epochs < 1:
        parser.error("seconds, batch-size, threads, width, and patience-epochs must be positive")
    budget = min(float(args.seconds), 590.0)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    with np.load(args.dataset, allow_pickle=False) as raw:
        validate_dataset_schema(raw)
        n = len(raw["labels"])
        if n < 2:
            raise ValueError(f"Need at least two examples, found {n}")
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
    train_indices, val_indices, heldout_games, final_games, split_name, split_file_hash = split_indices(
        game_ids, args.seed, args.split_file, dataset_hash,
    )
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("Dataset split produced an empty train or validation group")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = CatanActionMLP(width=args.width).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({
        "event": "training_started", "device": "cpu", "optimizer_budget_seconds": budget,
        "examples": n, "train_examples": len(train_indices), "validation_examples": len(val_indices),
        "unique_games": len(set(game_ids.tolist())), "split": split_name,
        "sealed_final_games": len(final_games), "state_tokens": STATE_TOKEN_COUNT,
        "parameters": parameters, "architecture": {"type": "action_mlp", "width": args.width},
        "batch_size": args.batch_size, "threads": args.threads,
    }), flush=True)

    train_indices = np.asarray(train_indices, dtype=np.int64)
    best_loss = float("inf")
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
        permutation = np.random.permutation(train_indices)
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
            loss = F.cross_entropy(logits.masked_fill(~action_mask, -1e9), data["labels"][ids])
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
                "feature_schema": FEATURE_SCHEMA,
                "dataset_schema": DATASET_SCHEMA,
                "model_config": {"width": args.width},
                "model_state": best_state,
                "seed": args.seed,
                "teacher": "dataset labels",
                "dataset_sha256": dataset_hash,
                "split_file": str(args.split_file) if args.split_file else None,
                "split_file_sha256": split_file_hash,
                "validation": validation,
                "epoch": epoch,
            }, args.output_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        elapsed = time.perf_counter() - started
        if elapsed >= next_report:
            print(json.dumps({"event": "training_progress", "epoch": epoch,
                              "elapsed_seconds": round(elapsed, 1),
                              "remaining_seconds": round(max(0, budget - elapsed), 1),
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
        "feature_schema": FEATURE_SCHEMA,
        "dataset_schema": DATASET_SCHEMA,
        "model_config": {"width": args.width},
        "model_state": best_state,
        "seed": args.seed,
        "teacher": "dataset labels",
        "dataset_sha256": dataset_hash,
        "split_file": str(args.split_file) if args.split_file else None,
        "split_file_sha256": split_file_hash,
        "validation": final_validation,
        "epoch": best_epoch,
    }
    torch.save(checkpoint, args.output_dir / "best.pt")
    result = {
        "device": "cpu", "optimizer_budget_seconds": budget,
        "optimizer_elapsed_seconds": round(training_seconds, 3),
        "budget_respected": training_seconds <= 600.0,
        "epochs": epoch, "stopped_reason": stopped_reason, "best_epoch": best_epoch,
        "parameters": parameters, "examples": n,
        "train_examples": len(train_indices), "validation_examples": len(val_indices),
        "heldout_games": sorted(heldout_games), "sealed_final_games": sorted(final_games),
        "split": split_name, "validation": final_validation,
        "architecture": {"type": "action_mlp", "width": args.width},
        "checkpoint": str(args.output_dir / "best.pt"),
        "dataset": str(args.dataset), "dataset_sha256": dataset_hash,
        "dataset_schema": DATASET_SCHEMA, "feature_schema": FEATURE_SCHEMA,
        "split_file": checkpoint["split_file"],
        "split_file_sha256": checkpoint["split_file_sha256"],
    }
    (args.output_dir / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "training_finished", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
