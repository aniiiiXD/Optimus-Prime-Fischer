"""Behavior-clone a five-head legal-action MLP and train its outcome value head."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

from catanLabs.OneVOne.mlp_policy import (  # noqa: E402
    POLICY_ARCHITECTURE,
    FiveHeadPolicy,
)
from catanLabs.OneVOne.policy_features import (  # noqa: E402
    ACTION_FEATURE_DIM,
    FEATURES_VERSION,
    FAMILY_NAMES,
)


def read_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (len(row["state"]) < 1 or len(row["actions"]) < 1
                    or len(row["families"]) != len(row["actions"])):
                raise ValueError(f"Malformed row at {path}:{line_number}")
            if not 0 <= int(row["chosen"]) < len(row["actions"]):
                raise ValueError(f"Bad target index at {path}:{line_number}")
            if any(len(action) != ACTION_FEATURE_DIM for action in row["actions"]):
                raise ValueError(f"Action feature schema mismatch at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No samples found in {path}")
    return rows


def seed_is_validation(seed: int) -> bool:
    digest = hashlib.sha256(str(seed).encode()).hexdigest()
    return int(digest[:8], 16) % 5 == 0


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[int], list[int]]:
    all_seeds = sorted({int(row["seed"]) for row in rows})
    validation_seeds = {seed for seed in all_seeds if seed_is_validation(seed)}
    if not validation_seeds and len(all_seeds) > 1:
        validation_seeds.add(all_seeds[-1])
    if len(validation_seeds) == len(all_seeds) and len(all_seeds) > 1:
        validation_seeds.remove(all_seeds[0])
    train = [row for row in rows if int(row["seed"]) not in validation_seeds]
    valid = [row for row in rows if int(row["seed"]) in validation_seeds]
    if not valid:
        raise ValueError("Need at least two unique board seeds for validation")
    return train, valid, sorted(set(all_seeds) - validation_seeds), sorted(validation_seeds)


def collate(rows: list[dict], device: torch.device,
            loss_policy_weight: float = 0.25) -> dict[str, torch.Tensor]:
    batch_size = len(rows)
    max_actions = max(len(row["actions"]) for row in rows)
    state_dim = len(rows[0]["state"])
    actions = np.zeros((batch_size, max_actions, ACTION_FEATURE_DIM), dtype=np.float32)
    families = np.zeros((batch_size, max_actions), dtype=np.int64)
    mask = np.zeros((batch_size, max_actions), dtype=np.bool_)
    states = np.zeros((batch_size, state_dim), dtype=np.float32)
    targets = np.zeros(batch_size, dtype=np.int64)
    target_families = np.zeros(batch_size, dtype=np.int64)
    outcomes = np.zeros(batch_size, dtype=np.float32)
    policy_weights = np.zeros(batch_size, dtype=np.float32)
    policy_mask = np.zeros(batch_size, dtype=np.bool_)
    for index, row in enumerate(rows):
        n = len(row["actions"])
        states[index] = row["state"]
        actions[index, :n] = row["actions"]
        families[index, :n] = row["families"]
        mask[index, :n] = True
        targets[index] = int(row["chosen"])
        target_families[index] = int(row["families"][int(row["chosen"])])
        outcomes[index] = float(row["outcome"])
        policy_mask[index] = n > 1
        if n > 1:
            policy_weights[index] = 1.0 if float(row["outcome"]) > 0 else loss_policy_weight
    return {
        "states": torch.as_tensor(states, device=device),
        "actions": torch.as_tensor(actions, device=device),
        "families": torch.as_tensor(families, device=device),
        "mask": torch.as_tensor(mask, device=device),
        "targets": torch.as_tensor(targets, device=device),
        "target_families": torch.as_tensor(target_families, device=device),
        "outcomes": torch.as_tensor(outcomes, device=device),
        "policy_weights": torch.as_tensor(policy_weights, device=device),
        "policy_mask": torch.as_tensor(policy_mask, device=device),
    }


@torch.inference_mode()
def evaluate(model: FiveHeadPolicy, rows: list[dict], device: torch.device,
             batch_size: int, value_weight: float,
             loss_policy_weight: float,
             family_weights: torch.Tensor) -> dict:
    model.eval()
    loss_sum = family_loss_sum = action_loss_sum = value_loss_sum = 0.0
    action_correct = family_correct = conditional_action_correct = 0
    value_sign_correct = count = policy_count = 0
    for start in range(0, len(rows), batch_size):
        batch = collate(rows[start:start + batch_size], device, loss_policy_weight)
        logits, family_logits, value = model(
            batch["states"], batch["actions"], batch["families"], batch["mask"]
        )
        within_family_mask = batch["mask"] & batch["families"].eq(
            batch["target_families"].unsqueeze(1)
        )
        within_family_logits = logits.masked_fill(~within_family_mask, -1e9)
        family_ce = F.cross_entropy(
            family_logits, batch["target_families"], weight=family_weights,
            reduction="none",
        )
        action_ce = F.cross_entropy(
            within_family_logits, batch["targets"], reduction="none",
        )
        sample_weights = batch["policy_weights"]
        weight_sum = sample_weights.sum().clamp_min(1.0)
        family_loss = (family_ce * sample_weights).sum() / weight_sum
        action_loss = (action_ce * sample_weights).sum() / weight_sum
        policy_loss = 0.5 * (family_loss + action_loss)
        value_loss = F.mse_loss(value, batch["outcomes"])
        n = batch["states"].shape[0]
        loss_sum += (policy_loss + value_weight * value_loss).item() * n
        family_loss_sum += family_loss.item() * n
        action_loss_sum += action_loss.item() * n
        value_loss_sum += value_loss.item() * n
        predicted_family = family_logits.argmax(dim=1)
        selected_mask = batch["mask"] & batch["families"].eq(predicted_family.unsqueeze(1))
        predicted_action = logits.masked_fill(~selected_mask, -1e9).argmax(dim=1)
        decision_rows = batch["policy_mask"]
        action_correct += ((predicted_action == batch["targets"]) & decision_rows).sum().item()
        family_correct += ((predicted_family == batch["target_families"]) & decision_rows).sum().item()
        conditional_action_correct += (
            (within_family_logits.argmax(dim=1) == batch["targets"]) & decision_rows
        ).sum().item()
        value_sign_correct += ((value >= 0) == (batch["outcomes"] > 0)).sum().item()
        count += n
        policy_count += int(decision_rows.sum().item())
    return {
        "loss": loss_sum / count,
        "family_cross_entropy": family_loss_sum / count,
        "conditional_action_cross_entropy": action_loss_sum / count,
        "value_mse": value_loss_sum / count,
        "action_accuracy_nonforced": action_correct / max(policy_count, 1),
        "family_accuracy_nonforced": family_correct / max(policy_count, 1),
        "conditional_action_accuracy_nonforced": conditional_action_correct / max(policy_count, 1),
        "outcome_sign_accuracy": value_sign_correct / count,
        "samples": count,
        "nonforced_policy_samples": policy_count,
    }


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--action-width", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--value-weight", type=float, default=0.15)
    parser.add_argument("--loss-policy-weight", type=float, default=0.25,
                        help="Imitation loss weight for moves from games the teacher lost")
    parser.add_argument("--seed", type=int, default=8417)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.width, args.action_width) < 1:
        parser.error("epochs, batch size, and model widths must be positive")
    if (args.learning_rate <= 0 or args.value_weight < 0
            or not 0.0 <= args.loss_policy_weight <= 1.0):
        parser.error("learning rate must be positive; value and loss-policy weights must be nonnegative")

    data_path = args.data.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text())
    rows = read_rows(data_path)
    train_rows, valid_rows, train_seeds, valid_seeds = split_rows(rows)
    state_dim = len(rows[0]["state"])
    if any(len(row["state"]) != state_dim for row in rows):
        raise ValueError("State feature dimensions vary in the dataset")
    outcomes = {float(row["outcome"]) for row in rows}
    if not outcomes <= {-1.0, 1.0}:
        raise ValueError(f"Expected terminal +/-1 outcomes, got {outcomes}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    model = FiveHeadPolicy(state_dim, ACTION_FEATURE_DIM, args.width, args.action_width).to(device)
    family_counts = np.zeros(5, dtype=np.int64)
    for row in train_rows:
        if len(row["actions"]) > 1:
            family_counts[int(row["families"][int(row["chosen"])])] += 1
    raw_family_weights = np.ones(5, dtype=np.float32)
    present = family_counts > 0
    raw_family_weights[present] = np.sqrt(
        family_counts[present].sum() / (5.0 * family_counts[present])
    )
    if present.any():
        raw_family_weights /= np.average(raw_family_weights, weights=np.maximum(family_counts, 1))
    family_weights = torch.as_tensor(raw_family_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output_dir = args.output_dir or ROOT / "runs" / f"mlp-bootstrap-{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    best_loss = float("inf")
    best_epoch = 0
    patience = 5
    stale = 0
    history = []
    run_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = list(range(len(train_rows)))
        rng.shuffle(indices)
        train_loss_sum = train_policy_sum = train_value_sum = 0.0
        train_count = 0
        for start in range(0, len(indices), args.batch_size):
            batch_rows = [train_rows[i] for i in indices[start:start + args.batch_size]]
            batch = collate(batch_rows, device, args.loss_policy_weight)
            logits, family_logits, value = model(
                batch["states"], batch["actions"], batch["families"], batch["mask"]
            )
            within_family_mask = batch["mask"] & batch["families"].eq(
                batch["target_families"].unsqueeze(1)
            )
            within_family_logits = logits.masked_fill(~within_family_mask, -1e9)
            family_ce = F.cross_entropy(
                family_logits, batch["target_families"], weight=family_weights,
                reduction="none",
            )
            action_ce = F.cross_entropy(
                within_family_logits, batch["targets"], reduction="none",
            )
            weight_sum = batch["policy_weights"].sum().clamp_min(1.0)
            family_loss = (family_ce * batch["policy_weights"]).sum() / weight_sum
            action_loss = (action_ce * batch["policy_weights"]).sum() / weight_sum
            policy_loss = 0.5 * (family_loss + action_loss)
            value_loss = F.mse_loss(value, batch["outcomes"])
            loss = policy_loss + args.value_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            n = len(batch_rows)
            train_loss_sum += loss.item() * n
            train_policy_sum += policy_loss.item() * n
            train_value_sum += value_loss.item() * n
            train_count += n

        validation = evaluate(
            model, valid_rows, device, args.batch_size, args.value_weight,
            args.loss_policy_weight, family_weights,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_count,
            "train_hierarchical_policy_cross_entropy": train_policy_sum / train_count,
            "train_value_mse": train_value_sum / train_count,
            "validation": validation,
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch}/{args.epochs}: val loss {validation['loss']:.4f}, "
            f"action acc {validation['action_accuracy_nonforced']:.3f}, "
            f"value sign {validation['outcome_sign_accuracy']:.3f}",
            flush=True,
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_epoch = epoch
            stale = 0
            torch.save({
                "architecture_version": POLICY_ARCHITECTURE,
                "features_version": FEATURES_VERSION,
                "family_names": list(FAMILY_NAMES),
                "family_class_weights": raw_family_weights.tolist(),
                "state_dim": state_dim,
                "action_dim": ACTION_FEATURE_DIM,
                "width": args.width,
                "action_width": args.action_width,
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }, output_dir / "policy.pt")
        else:
            stale += 1
            if stale >= patience:
                break

    policy_config = dict(config)
    policy_config["policy_checkpoint"] = "policy.pt"
    (output_dir / "policy_config.json").write_text(json.dumps(policy_config, indent=2) + "\n")
    manifest_path = data_path.with_suffix(data_path.suffix + ".manifest.json")
    dataset_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    report = {
        "algorithm": "teacher-league behavior cloning + terminal outcome value regression",
        "architecture_version": POLICY_ARCHITECTURE,
        "features_version": FEATURES_VERSION,
        "policy_heads": list(FAMILY_NAMES),
        "data": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "dataset_manifest": dataset_manifest,
        "samples": len(rows),
        "train_samples": len(train_rows),
        "validation_samples": len(valid_rows),
        "train_nonforced_policy_samples": sum(len(row["actions"]) > 1 for row in train_rows),
        "validation_nonforced_policy_samples": sum(len(row["actions"]) > 1 for row in valid_rows),
        "train_seeds": train_seeds,
        "validation_seeds": valid_seeds,
        "model": {
            "state_dim": state_dim,
            "action_dim": ACTION_FEATURE_DIM,
            "width": args.width,
            "action_width": args.action_width,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "training": {
            "epochs_requested": args.epochs,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "value_loss_weight": args.value_weight,
            "loss_policy_weight": args.loss_policy_weight,
            "family_counts_nonforced": family_counts.tolist(),
            "family_class_weights": raw_family_weights.tolist(),
            "device": str(device),
            "seed": args.seed,
            "history": history,
        },
        "best_validation": min(history, key=lambda item: item["validation"]["loss"])["validation"],
        "checkpoint": str(output_dir / "policy.pt"),
        "policy_config": str(output_dir / "policy_config.json"),
        "promotion": "not promoted; evaluate against frozen bots and the current heuristic first",
        "elapsed_seconds": round(time.perf_counter() - run_started, 3),
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "event": "policy_training_complete",
        "samples": len(rows),
        "train_samples": len(train_rows),
        "validation_samples": len(valid_rows),
        "best_epoch": best_epoch,
        "best_validation": report["best_validation"],
        "checkpoint": report["checkpoint"],
        "policy_config": report["policy_config"],
        "report": str(output_dir / "training_report.json"),
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
