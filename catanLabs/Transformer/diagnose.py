"""Held-out action and CPU inference diagnostics for a v2 Catan policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from catanLabs.Transformer.model import (
    ACTION_TYPES, CHECKPOINT_SCHEMA, DATASET_SCHEMA, FEATURE_SCHEMA,
    RESOURCES, CatanActionTransformer,
)


GROUPS = ("none", "tile", "node", "edge")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_action(features, action_type, group, target_id) -> str:
    """Render only information preserved in the v2 legal-action encoding."""
    kind = ACTION_TYPES[int(action_type)]
    target = f" {GROUPS[int(group)]} {int(target_id)}" if group else ""
    if kind in ("DISCARD_RESOURCE", "PLAY_MONOPOLY"):
        chosen = np.flatnonzero(features[:5] > 0)
        if len(chosen):
            return f"{kind} {RESOURCES[int(chosen[0])]}"
    if kind == "PLAY_YEAR_OF_PLENTY":
        resources = ", ".join(f"{RESOURCES[i]} x{int(features[i])}"
                              for i in np.flatnonzero(features[:5]))
        return f"{kind} ({resources})"
    if kind == "MARITIME_TRADE":
        offered = ", ".join(f"{RESOURCES[i]} x{int(features[5 + i])}"
                            for i in np.flatnonzero(features[5:10]))
        requested = [RESOURCES[i] for i in np.flatnonzero(features[10:15])]
        return f"{kind} offer ({offered}) for {', '.join(requested)}"
    if kind == "MOVE_ROBBER":
        victims = np.flatnonzero(features[15:19])
        victim = f", victim relative seat {int(victims[0])}" if len(victims) else ""
        return f"{kind}{target}{victim}"
    return kind + target


def row_example(position, row_index, predictions, data):
    count = int(data["counts"][position])
    teacher = int(data["labels"][position])
    policy = int(predictions[position])
    def action(index):
        return describe_action(
            data["action_features"][position, index],
            data["action_types"][position, index],
            data["target_groups"][position, index],
            data["target_ids"][position, index],
        )
    alternatives = [action(index) for index in range(count) if index not in (teacher, policy)]
    return {
        "dataset_row": int(row_index),
        "game_id": int(data["game_ids"][position]),
        "phase": str(data["phases"][position]),
        "legal_action_count": count,
        "teacher_choice": action(teacher),
        "policy_choice": action(policy),
        "other_legal_alternatives": alternatives[:5],
        "interpretation": "Action disagreement only; the teacher choice is not proven optimal.",
    }


def summary_by_key(keys, labels, predictions, counts):
    result = {}
    for key in sorted(set(str(value) for value in keys)):
        mask = np.asarray(keys) == key
        n = int(mask.sum())
        correct = int((labels[mask] == predictions[mask]).sum())
        choices = mask & (counts > 1)
        choice_n = int(choices.sum())
        choice_correct = int((labels[choices] == predictions[choices]).sum())
        result[key] = {
            "count": n, "agreement_count": correct,
            "agreement": correct / n if n else None,
            "nonforced_count": choice_n,
            "nonforced_agreement": choice_correct / choice_n if choice_n else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent / "runs/gate1_profile"
    parser.add_argument("--dataset", type=Path, default=base / "dataset.npz")
    parser.add_argument("--checkpoint", type=Path, default=base / "short_update/best.pt")
    parser.add_argument("--training-summary", type=Path, default=base / "short_update/training.json")
    parser.add_argument("--output", type=Path, default=base / "diagnostics.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--latency-samples", type=int, default=200)
    args = parser.parse_args()
    if args.threads < 1 or args.latency_samples < 1:
        parser.error("threads and latency-samples must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    dataset_hash = sha256(args.dataset)
    checkpoint_hash = sha256(args.checkpoint)
    training = json.loads(args.training_summary.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("Checkpoint does not use the current v2 feature schema")
    if checkpoint.get("dataset_sha256") != dataset_hash or training["dataset_sha256"] != dataset_hash:
        raise ValueError("Checkpoint and training summary must match the exact dataset hash")
    heldout_games = set(int(game_id) for game_id in training["heldout_games"])
    if not heldout_games:
        raise ValueError("A whole-game held-out split is required for this report")

    with np.load(args.dataset, allow_pickle=False) as raw:
        if str(raw["dataset_schema"].item()) != DATASET_SCHEMA or str(raw["feature_schema"].item()) != FEATURE_SCHEMA:
            raise ValueError("Dataset does not use the current v2 schemas")
        game_ids = raw["game_ids"]
        indices = np.flatnonzero(np.isin(game_ids, list(heldout_games)))
        data = {
            "state_features": raw["state_features"][indices],
            "state_kinds": raw["state_kinds"][indices],
            "state_ids": raw["state_ids"][indices],
            "action_features": raw["action_features"][indices],
            "action_types": raw["action_types"][indices],
            "target_groups": raw["target_groups"][indices],
            "target_ids": raw["target_ids"][indices],
            "counts": raw["candidate_counts"][indices],
            "labels": raw["labels"][indices],
            "phases": raw["phases"][indices],
            "game_ids": game_ids[indices],
        }

    model = CatanActionTransformer(**checkpoint["model_config"]).cpu()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    predictions = np.empty(len(indices), dtype=np.int64)
    unmasked_predictions = np.empty(len(indices), dtype=np.int64)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(indices), 64):
            end = min(start + 64, len(indices))
            counts = data["counts"][start:end]
            width = int(counts.max())
            mask = torch.arange(width)[None, :] < torch.from_numpy(counts)[:, None]
            logits = model(
                torch.from_numpy(data["state_features"][start:end]),
                torch.from_numpy(data["state_kinds"][start:end]),
                torch.from_numpy(data["state_ids"][start:end]),
                torch.from_numpy(data["action_features"][start:end, :width]),
                torch.from_numpy(data["action_types"][start:end, :width]),
                torch.from_numpy(data["target_groups"][start:end, :width]),
                torch.from_numpy(data["target_ids"][start:end, :width]),
                mask,
            )
            predictions[start:end] = logits.masked_fill(~mask, -1e9).argmax(dim=1).numpy()
            unmasked_predictions[start:end] = logits.argmax(dim=1).numpy()
    evaluation_seconds = time.perf_counter() - started

    teacher_types = np.asarray([
        ACTION_TYPES[int(data["action_types"][row, label])]
        for row, label in enumerate(data["labels"])
    ])
    by_phase = summary_by_key(data["phases"], data["labels"], predictions, data["counts"])
    by_teacher_type = summary_by_key(teacher_types, data["labels"], predictions, data["counts"])
    correct = int((predictions == data["labels"]).sum())
    nonforced = data["counts"] > 1
    nonforced_n = int(nonforced.sum())

    # Select one opening placement and one regular-turn disagreement when
    # available; examples are descriptive, not counterfactual game outcomes.
    examples = []
    for phase in ("BUILD_INITIAL_SETTLEMENT", "PLAY_TURN", "MOVE_ROBBER"):
        choices = np.flatnonzero((data["phases"] == phase) &
                                 (predictions != data["labels"]) & (data["counts"] > 2))
        if len(choices):
            position = int(choices[0])
            examples.append(row_example(position, indices[position], predictions, data))
        if len(examples) >= 2:
            break

    latency_rows = np.linspace(0, len(indices) - 1,
                               min(args.latency_samples, len(indices)), dtype=int)
    latency_inputs = []
    for position in latency_rows:
        width = int(data["counts"][position])
        latency_inputs.append((
            torch.from_numpy(data["state_features"][position:position + 1]),
            torch.from_numpy(data["state_kinds"][position:position + 1]),
            torch.from_numpy(data["state_ids"][position:position + 1]),
            torch.from_numpy(data["action_features"][position:position + 1, :width]),
            torch.from_numpy(data["action_types"][position:position + 1, :width]),
            torch.from_numpy(data["target_groups"][position:position + 1, :width]),
            torch.from_numpy(data["target_ids"][position:position + 1, :width]),
            torch.ones((1, width), dtype=torch.bool),
        ))
    elapsed_ms = []
    with torch.inference_mode():
        for tensors in latency_inputs[:min(20, len(latency_inputs))]:
            model(*tensors)
        for tensors in latency_inputs:
            before = time.perf_counter_ns()
            model(*tensors)
            elapsed_ms.append((time.perf_counter_ns() - before) / 1e6)

    result = {
        "description": "Held-out teacher-action imitation and model-only CPU forward latency; not win rate or action quality.",
        "dataset": str(args.dataset), "dataset_sha256": dataset_hash,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash,
        "feature_schema": FEATURE_SCHEMA, "dataset_schema": DATASET_SCHEMA,
        "heldout_games": sorted(heldout_games),
        "heldout_rows": len(indices), "correct": correct,
        "agreement": correct / len(indices),
        "legacy_unmasked_agreement": float((unmasked_predictions == data["labels"]).mean()),
        "nonforced_rows": nonforced_n,
        "nonforced_agreement": int((predictions[nonforced] == data["labels"][nonforced]).sum()) / nonforced_n,
        "by_phase": by_phase, "by_teacher_action_type": by_teacher_type,
        "disagreement_examples": examples,
        "cpu_forward_latency_ms": {
            "scope": "single-example PyTorch model forward only; excludes snapshot, encoding and legal-action sorting",
            "threads": args.threads, "samples": len(elapsed_ms), "warmups": min(20, len(latency_inputs)),
            "p50": float(np.percentile(elapsed_ms, 50)),
            "p95": float(np.percentile(elapsed_ms, 95)),
            "evaluation_seconds": evaluation_seconds,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "heldout_rows": len(indices),
                      "agreement": result["agreement"],
                      "nonforced_agreement": result["nonforced_agreement"],
                      "p50_forward_ms": result["cpu_forward_latency_ms"]["p50"],
                      "p95_forward_ms": result["cpu_forward_latency_ms"]["p95"],
                      "evaluation_seconds": evaluation_seconds,
                      "examples": len(examples)}, indent=2))


if __name__ == "__main__":
    main()
