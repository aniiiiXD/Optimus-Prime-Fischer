"""Compare MLP and Transformer on one audited validation split, never final."""

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

from catanLabs.Transformer.diagnose import describe_action, summary_by_key
from catanLabs.Transformer.mlp import CHECKPOINT_SCHEMA as MLP_SCHEMA, CatanActionMLP
from catanLabs.Transformer.model import (
    ACTION_TYPES, CatanActionTransformer,
)
from catanLabs.Transformer.train import split_indices


SCHEMA_PAIRS = {
    "catanlabs-transformer-dataset-v2": ("visible-state-tokens-v2", "catanlabs-action-transformer-v2"),
    "catanlabs-transformer-dataset-v3": ("visible-state-tokens-v3", "catanlabs-action-transformer-v3"),
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path, expected_schema: str, dataset_hash: str, split_hash: str,
               dataset_schema: str, feature_schema: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint.get("schema") != expected_schema
            or checkpoint.get("dataset_schema") != dataset_schema
            or checkpoint.get("feature_schema") != feature_schema):
        raise ValueError(f"Wrong model or feature schema in {path}")
    if checkpoint.get("dataset_sha256") != dataset_hash:
        raise ValueError(f"Checkpoint dataset hash differs from audited data: {path}")
    if checkpoint.get("split_file_sha256") != split_hash:
        raise ValueError(f"Checkpoint split hash differs from audited validation split: {path}")
    if expected_schema == MLP_SCHEMA:
        model = CatanActionMLP(**checkpoint["model_config"])
    else:
        model = CatanActionTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def model_inputs(data, start: int, end: int):
    counts = data["counts"][start:end]
    width = int(counts.max())
    mask = torch.arange(width)[None, :] < torch.from_numpy(counts)[:, None]
    return (
        torch.from_numpy(data["state_features"][start:end]),
        torch.from_numpy(data["state_kinds"][start:end]),
        torch.from_numpy(data["state_ids"][start:end]),
        torch.from_numpy(data["action_features"][start:end, :width]),
        torch.from_numpy(data["action_types"][start:end, :width]),
        torch.from_numpy(data["target_groups"][start:end, :width]),
        torch.from_numpy(data["target_ids"][start:end, :width]),
        mask,
    )


def predictions_and_latency(model, data, latency_rows):
    predictions = np.empty(len(data["labels"]), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(predictions), 64):
            end = min(start + 64, len(predictions))
            tensors = model_inputs(data, start, end)
            logits = model(*tensors)
            predictions[start:end] = logits.masked_fill(~tensors[-1], -1e9).argmax(1).numpy()
        single_inputs = [model_inputs(data, int(row), int(row) + 1) for row in latency_rows]
        for tensors in single_inputs[:min(20, len(single_inputs))]:
            model(*tensors)
        elapsed_ms = []
        for tensors in single_inputs:
            started = time.perf_counter_ns()
            model(*tensors)
            elapsed_ms.append((time.perf_counter_ns() - started) / 1e6)
    return predictions, {
        "scope": "single-row model forward only; excludes visible snapshot and encoding",
        "samples": len(elapsed_ms), "warmups": min(20, len(single_inputs)),
        "p50_ms": float(np.percentile(elapsed_ms, 50)),
        "p95_ms": float(np.percentile(elapsed_ms, 95)),
    }


def overlap(labels, transformer, mlp, mask):
    correct_transformer = transformer[mask] == labels[mask]
    correct_mlp = mlp[mask] == labels[mask]
    return {
        "rows": int(mask.sum()),
        "both_correct": int((correct_transformer & correct_mlp).sum()),
        "transformer_only_correct": int((correct_transformer & ~correct_mlp).sum()),
        "mlp_only_correct": int((~correct_transformer & correct_mlp).sum()),
        "both_wrong": int((~correct_transformer & ~correct_mlp).sum()),
    }


def describe_row(position, original_index, data, transformer, mlp):
    def action(index):
        return describe_action(
            data["action_features"][position, index],
            data["action_types"][position, index],
            data["target_groups"][position, index],
            data["target_ids"][position, index],
        )
    return {
        "dataset_row": int(original_index), "game_id": int(data["game_ids"][position]),
        "phase": str(data["phases"][position]),
        "legal_action_count": int(data["counts"][position]),
        "teacher": action(int(data["labels"][position])),
        "transformer": action(int(transformer[position])),
        "mlp": action(int(mlp[position])),
        "meaning": "Model/teacher choice difference only; no outcome or strategic-regret claim.",
    }


def markdown(report):
    names = {"BUILD_INITIAL_SETTLEMENT": "Initial settlement",
             "BUILD_INITIAL_ROAD": "Initial road", "PLAY_TURN": "Main turn",
             "MOVE_ROBBER": "Move robber", "DISCARD": "Discard"}
    lines = [
        "# Audited validation: MLP versus Transformer",
        "",
        "Both checkpoints are evaluated on the **same validation rows** from the audited whole-game split. Final games remain sealed. This is teacher-action imitation, **not a win-rate comparison**.",
        "",
        f"Rows: **{report['rows']:,}** from **{len(report['evaluated_games'])}** games "
        f"in the {len(report['validation_games'])}-game validation partition. "
        f"Choice rows (more than one legal action): **{report['nonforced_rows']:,}**.",
        "",
        "| Phase | Rows | Transformer exact | MLP exact | Transformer choice rows | MLP choice rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for phase in ("BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD", "PLAY_TURN", "DISCARD", "MOVE_ROBBER"):
        tf = report["transformer"]["by_phase"].get(phase)
        mlp = report["mlp"]["by_phase"].get(phase)
        if not tf:
            continue
        pct = lambda value: f"{100 * value:.1f}%" if value is not None else "—"
        lines.append(f"| {names[phase]} | {tf['count']:,} | {pct(tf['agreement'])} | "
                     f"{pct(mlp['agreement'])} | {pct(tf['nonforced_agreement'])} | "
                     f"{pct(mlp['nonforced_agreement'])} |")
    lines += [
        "",
        f"Overall exact agreement: Transformer **{100 * report['transformer']['agreement']:.1f}%**, "
        f"MLP **{100 * report['mlp']['agreement']:.1f}%**. On choice rows: Transformer "
        f"**{100 * report['transformer']['nonforced_agreement']:.1f}%**, "
        f"MLP **{100 * report['mlp']['nonforced_agreement']:.1f}%**.",
        "",
        "Model-forward CPU latency excludes snapshot construction and feature encoding. "
        f"Transformer p50/p95: **{report['transformer']['latency']['p50_ms']:.3f}/"
        f"{report['transformer']['latency']['p95_ms']:.3f} ms**; "
        f"MLP p50/p95: **{report['mlp']['latency']['p50_ms']:.3f}/"
        f"{report['mlp']['latency']['p95_ms']:.3f} ms**.",
        "",
        "The JSON report includes paired correctness overlap, teacher-action-type tables, checkpoint hashes, and descriptive disagreement examples. Agreement differences are diagnostic; the final split must remain untouched until model selection is frozen.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "start/.venv/bin/python catanLabs/Transformer/compare_diagnostics.py \\",
        f"  --dataset {report['dataset']} \\",
        f"  --split-file {report['split_file']} \\",
        f"  --transformer-checkpoint {report['transformer']['checkpoint']} \\",
        f"  --mlp-checkpoint {report['mlp']['checkpoint']} \\",
        f"  --latency-samples {report['latency_samples']} \\",
        *( [f"  --max-validation-rows {report['max_validation_rows']} \\"]
           if report['max_validation_rows'] is not None else [] ),
        f"  --output {report['output']}",
        "```",
        "",
        f"Dataset SHA-256: `{report['dataset_sha256']}`  ",
        f"Split SHA-256: `{report['split_file_sha256']}`  ",
        f"Transformer checkpoint SHA-256: `{report['transformer']['checkpoint_sha256']}`  ",
        f"MLP checkpoint SHA-256: `{report['mlp']['checkpoint_sha256']}`",
    ]
    if report.get("sample_limited"):
        lines.insert(2, "**Smoke sample only:** row limit was enabled; these figures are not the full validation result.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent / "runs/gate2_native_v2"
    parser.add_argument("--dataset", type=Path, default=base / "dataset.npz")
    parser.add_argument("--split-file", type=Path, default=base / "dataset.split.json")
    parser.add_argument("--transformer-checkpoint", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=base / "model_comparison.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--latency-samples", type=int, default=200)
    parser.add_argument("--max-validation-rows", type=int, help="Smoke only; limit validation rows")
    args = parser.parse_args()
    if args.threads < 1 or args.latency_samples < 1 or (args.max_validation_rows is not None and args.max_validation_rows < 1):
        parser.error("threads, latency-samples, and max-validation-rows must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    dataset_hash = file_hash(args.dataset)
    split_hash = file_hash(args.split_file)
    with np.load(args.dataset, allow_pickle=False) as raw:
        dataset_schema = str(raw["dataset_schema"].item())
        feature_schema = str(raw["feature_schema"].item())
        if dataset_schema not in SCHEMA_PAIRS or SCHEMA_PAIRS[dataset_schema][0] != feature_schema:
            raise ValueError("Unsupported dataset/feature schema pair")
        if raw["state_features"].shape[1:] != (150, 40) or raw["action_features"].shape[-1] != 20:
            raise ValueError("Dataset tensor shapes are incompatible with the diagnostic")
        game_ids = raw["game_ids"].astype(int)
        _, val_indices, validation_games, final_games, _, _ = split_indices(
            game_ids, seed=0, split_file=args.split_file, dataset_sha256=dataset_hash,
        )
        if args.max_validation_rows is not None:
            val_indices = val_indices[:args.max_validation_rows]
        if set(game_ids[val_indices]) & set(final_games):
            raise RuntimeError("Final game IDs must never enter diagnostics")
        data = {
            "state_features": raw["state_features"][val_indices],
            "state_kinds": raw["state_kinds"][val_indices],
            "state_ids": raw["state_ids"][val_indices],
            "action_features": raw["action_features"][val_indices],
            "action_types": raw["action_types"][val_indices],
            "target_groups": raw["target_groups"][val_indices],
            "target_ids": raw["target_ids"][val_indices],
            "counts": raw["candidate_counts"][val_indices],
            "labels": raw["labels"][val_indices],
            "phases": raw["phases"][val_indices],
            "game_ids": game_ids[val_indices],
        }
    transformer = load_model(args.transformer_checkpoint, SCHEMA_PAIRS[dataset_schema][1],
                             dataset_hash, split_hash, dataset_schema, feature_schema)
    mlp = load_model(args.mlp_checkpoint, MLP_SCHEMA, dataset_hash, split_hash,
                     dataset_schema, feature_schema)
    latency_rows = np.linspace(0, len(val_indices) - 1,
                               min(args.latency_samples, len(val_indices)), dtype=int)
    tf_predictions, tf_latency = predictions_and_latency(transformer, data, latency_rows)
    mlp_predictions, mlp_latency = predictions_and_latency(mlp, data, latency_rows)
    labels = data["labels"]
    choice_mask = data["counts"] > 1
    teacher_types = np.asarray([ACTION_TYPES[int(data["action_types"][row, label])]
                                for row, label in enumerate(labels)])

    def model_summary(predictions, latency, path):
        return {
            "checkpoint": str(path), "checkpoint_sha256": file_hash(path),
            "correct": int((predictions == labels).sum()),
            "agreement": float((predictions == labels).mean()),
            "nonforced_agreement": float((predictions[choice_mask] == labels[choice_mask]).mean())
            if choice_mask.any() else None,
            "by_phase": summary_by_key(data["phases"], labels, predictions, data["counts"]),
            "by_teacher_action_type": summary_by_key(teacher_types, labels, predictions, data["counts"]),
            "latency": latency,
        }

    phase_overlap = {phase: overlap(labels, tf_predictions, mlp_predictions, data["phases"] == phase)
                     for phase in sorted(set(data["phases"].tolist()))}
    examples = []
    for phase in ("BUILD_INITIAL_SETTLEMENT", "PLAY_TURN", "MOVE_ROBBER"):
        positions = np.flatnonzero((data["phases"] == phase) & (tf_predictions != mlp_predictions)
                                  & choice_mask)
        if len(positions):
            row = int(positions[0])
            examples.append(describe_row(row, val_indices[row], data, tf_predictions, mlp_predictions))
    result = {
        "description": "Audited validation imitation comparison; not game strength",
        "dataset": str(args.dataset), "dataset_sha256": dataset_hash,
        "split_file": str(args.split_file), "split_file_sha256": split_hash,
        "validation_games": sorted(validation_games),
        "evaluated_games": sorted(set(int(value) for value in data["game_ids"])),
        "final_games_evaluated": 0,
        "rows": len(val_indices), "nonforced_rows": int(choice_mask.sum()),
        "sample_limited": args.max_validation_rows is not None,
        "max_validation_rows": args.max_validation_rows,
        "latency_samples": args.latency_samples,
        "cpu_threads": args.threads,
        "transformer": model_summary(tf_predictions, tf_latency, args.transformer_checkpoint),
        "mlp": model_summary(mlp_predictions, mlp_latency, args.mlp_checkpoint),
        "overlap": overlap(labels, tf_predictions, mlp_predictions, np.ones(len(labels), dtype=bool)),
        "nonforced_overlap": overlap(labels, tf_predictions, mlp_predictions, choice_mask),
        "phase_overlap": phase_overlap,
        "examples": examples,
        "output": str(args.output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(result))
    print(json.dumps({"output": str(args.output), "rows": len(val_indices),
                      "final_games_evaluated": 0,
                      "transformer_agreement": result["transformer"]["agreement"],
                      "mlp_agreement": result["mlp"]["agreement"],
                      "sample_limited": result["sample_limited"]}, indent=2))


if __name__ == "__main__":
    main()
