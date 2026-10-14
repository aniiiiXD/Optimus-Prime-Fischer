"""Train and profile the Connect Four candidate-action Transformer on CPU."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import time
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from connect_four_transformer.config import DEFAULT_CONFIG
from connect_four_transformer.data import (
    DEFAULT_DATASET,
    SearchDataset,
    canonical_position,
    grouped_split,
    load_rows,
)
from connect_four_transformer.model import ConnectFourTransformer, ModelOutput


DEFAULT_CHECKPOINT = Path(__file__).parent / "artifacts" / "checkpoints" / "candidate_v0.pt"
DEFAULT_REPORT = Path(__file__).parent / "artifacts" / "reports" / "training_v0.json"


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def losses(
    output: ModelOutput,
    batch: dict[str, torch.Tensor],
    *,
    q_weight: float,
    value_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_probabilities = F.log_softmax(output.policy_logits, dim=-1)
    per_position_policy = -(batch["policy_target"] * log_probabilities).sum(dim=-1)
    policy = (per_position_policy * batch["policy_weight"]).sum() / batch["policy_weight"].sum()
    q = F.smooth_l1_loss(
        output.action_values[batch["candidate_mask"]],
        batch["q_target"][batch["candidate_mask"]],
    )
    value = F.smooth_l1_loss(output.state_value, batch["state_value_target"])
    total = policy + q_weight * q + value_weight * value
    return total, {
        "total": float(total.detach()),
        "policy": float(policy.detach()),
        "q": float(q.detach()),
        "value": float(value.detach()),
    }


def forward(model: ConnectFourTransformer, batch: dict[str, torch.Tensor]) -> ModelOutput:
    return model(
        batch["board_tokens"],
        batch["candidate_ids"],
        batch["landing_rows"],
        batch["candidate_mask"],
    )


@torch.inference_mode()
def evaluate(
    model: ConnectFourTransformer,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    q_weight: float,
    value_weight: float,
) -> dict[str, float | int | None]:
    model.eval()
    totals = {"total": 0.0, "policy": 0.0, "q": 0.0, "value": 0.0}
    positions = 0
    correct = 0
    top2_correct = 0
    critical_correct = 0
    critical_count = 0
    regret_sum = 0.0
    q_error = 0.0
    q_count = 0
    value_error = 0.0
    for batch in loader:
        output = forward(model, batch)
        _, parts = losses(output, batch, q_weight=q_weight, value_weight=value_weight)
        batch_size = batch["board_tokens"].shape[0]
        for name, value in parts.items():
            totals[name] += value * batch_size
        positions += batch_size

        predicted = output.policy_logits.argmax(dim=-1)
        critical_mask = batch["critical_mask"]
        critical_count += int(critical_mask.any(dim=1).sum().item())
        critical_correct += int(critical_mask.gather(1, predicted.unsqueeze(1)).sum().item())
        teacher_values = batch["q_target"].masked_fill(~batch["candidate_mask"], -torch.inf)
        best_values = teacher_values.max(dim=-1).values
        chosen_values = teacher_values.gather(1, predicted.unsqueeze(1)).squeeze(1)
        correct += int(torch.isclose(chosen_values, best_values, atol=1e-6).sum().item())
        top2 = output.policy_logits.topk(k=2, dim=-1).indices
        top2_values = teacher_values.gather(1, top2)
        top2_correct += int(torch.isclose(top2_values, best_values.unsqueeze(1), atol=1e-6).any(dim=1).sum().item())
        regret_sum += float((best_values - chosen_values).sum().item())
        mask = batch["candidate_mask"]
        q_error += float((output.action_values[mask] - batch["q_target"][mask]).abs().sum().item())
        q_count += int(mask.sum().item())
        value_error += float((output.state_value - batch["state_value_target"]).abs().sum().item())

    return {
        "positions": positions,
        "loss": totals["total"] / positions,
        "policy_loss": totals["policy"] / positions,
        "q_loss": totals["q"] / positions,
        "value_loss": totals["value"] / positions,
        "best_move_accuracy": correct / positions,
        "top2_best_move_recall": top2_correct / positions,
        "critical_move_accuracy": critical_correct / critical_count if critical_count else None,
        "critical_positions": critical_count,
        "mean_teacher_regret": regret_sum / positions,
        "q_mae": q_error / q_count,
        "value_mae": value_error / positions,
    }


@torch.inference_mode()
def benchmark(
    model: ConnectFourTransformer,
    sample: dict[str, torch.Tensor],
    *,
    batch_size: int,
    iterations: int = 300,
) -> dict[str, float | int]:
    model.eval()
    batch = {key: value.unsqueeze(0).repeat(batch_size, *([1] * value.ndim)) for key, value in sample.items()}
    for _ in range(25):
        forward(model, batch)
    times = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        forward(model, batch)
        times.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(times)
    median = statistics.median(times)
    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "median_ms": median,
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "positions_per_second": 1000.0 * batch_size / median,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--extra-train", type=Path, action="append")
    parser.add_argument(
        "--init-checkpoint", type=Path,
        help="optional checkpoint to fine-tune; omitted means initialize from scratch",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="epoch cap; defaults to 30 unless a time budget is supplied",
    )
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        help="train for at least this much fit-loop wall time, stopping after an epoch",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--q-weight", type=float, default=0.5)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hard-tactical-targets", action="store_true")
    parser.add_argument("--tactical-weight", type=float, default=1.0)
    parser.add_argument(
        "--selection-metric",
        choices=("loss", "critical_accuracy", "balanced"),
        default="loss",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()
    if args.epochs is not None and args.epochs <= 0:
        parser.error("epochs must be positive")
    if min(args.batch_size, args.threads, args.log_every) <= 0:
        parser.error("batch-size, threads, and log-every must be positive")
    if args.time_budget_seconds is not None and args.time_budget_seconds <= 0:
        parser.error("time-budget-seconds must be positive")
    if args.tactical_weight < 1:
        parser.error("--tactical-weight must be at least 1")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows = load_rows(args.dataset)
    split = grouped_split(rows, seed=args.seed)
    duplicate_extras = 0
    known_positions = {canonical_position(row) for row in rows}
    for extra_path in args.extra_train or []:
        extra = load_rows(extra_path)
        for row in extra:
            identity = canonical_position(row)
            if identity in known_positions:
                duplicate_extras += 1
                continue
            known_positions.add(identity)
            split["train"].append(len(rows))
            rows.append(row)
    loaders = {
        name: DataLoader(
            SearchDataset(
                rows,
                indices,
                mirror_probability=0.5 if name == "train" else 0.0,
                hard_tactical_targets=args.hard_tactical_targets if name == "train" else False,
                tactical_weight=args.tactical_weight if name == "train" else 1.0,
            ),
            batch_size=args.batch_size,
            shuffle=name == "train",
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=0,
        )
        for name, indices in split.items()
    }
    model = ConnectFourTransformer(DEFAULT_CONFIG)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if initial["model_config"] != DEFAULT_CONFIG.to_dict():
            raise ValueError("initial checkpoint model configuration does not match this trainer")
        model.load_state_dict(initial["model_state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_loss = math.inf
    best_critical = -1.0
    best_balanced = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    curve = []
    optimizer_step_wall_seconds = 0.0
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    print(
        f"CPU | {len(split['train'])} train / {len(split['validation'])} validation / "
        f"{len(split['test'])} test | {len(known_positions):,} unique positions "
        f"({duplicate_extras:,} duplicate extras skipped) | {parameter_count(model):,} parameters"
    )
    if args.time_budget_seconds is not None:
        print(f"fit-loop time budget: {args.time_budget_seconds:.0f} s; checking after each epoch", flush=True)

    epoch_limit = args.epochs if args.epochs is not None else (
        None if args.time_budget_seconds is not None else 30
    )
    epoch = 1
    completed_epochs = 0
    while epoch_limit is None or epoch <= epoch_limit:
        model.train()
        train_loss = 0.0
        count = 0
        optimizer_epoch_started = time.perf_counter()
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            output = forward(model, batch)
            loss, _ = losses(
                output, batch, q_weight=args.q_weight, value_weight=args.value_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size = batch["board_tokens"].shape[0]
            train_loss += float(loss.item()) * batch_size
            count += batch_size
        optimizer_step_wall_seconds += time.perf_counter() - optimizer_epoch_started

        validation = evaluate(
            model,
            loaders["validation"],
            q_weight=args.q_weight,
            value_weight=args.value_weight,
        )
        critical_accuracy = validation["critical_move_accuracy"]
        balanced_score = (
            0.45 * float(critical_accuracy or 0.0)
            + 0.35 * float(validation["best_move_accuracy"])
            + 0.20 * (1.0 - min(2.0, float(validation["mean_teacher_regret"])) / 2.0)
        )
        if args.selection_metric == "loss":
            selected = float(validation["loss"]) < best_loss
        elif args.selection_metric == "critical_accuracy":
            selected = critical_accuracy is not None and (
                float(critical_accuracy) > best_critical
                or (float(critical_accuracy) == best_critical and float(validation["loss"]) < best_loss)
            )
        else:
            selected = balanced_score > best_balanced
        if selected:
            best_loss = float(validation["loss"])
            best_critical = float(critical_accuracy or 0.0)
            best_balanced = balanced_score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        completed_epochs = epoch
        if epoch == 1 or epoch % args.log_every == 0 or (
            epoch_limit is not None and epoch == epoch_limit
        ):
            curve.append({"epoch": epoch, "train_loss": train_loss / count, "validation": validation})
            print(
                f"epoch {epoch:>3}/{epoch_limit or 'time'} | train {train_loss / count:.4f} | "
                f"val {validation['loss']:.4f} | best move {validation['best_move_accuracy']:.1%} | "
                f"critical {validation['critical_move_accuracy']:.1%} | "
                f"regret {validation['mean_teacher_regret']:.3f} | "
                f"fit {time.perf_counter() - wall_started:.0f}s",
                flush=True,
            )

        # Budget mode finishes the current epoch, so the fit loop meets or
        # slightly exceeds the requested duration rather than stopping early.
        if (
            args.time_budget_seconds is not None
            and time.perf_counter() - wall_started >= args.time_budget_seconds
        ):
            break
        epoch += 1

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    fit_wall_seconds = time.perf_counter() - wall_started
    model.load_state_dict(best_state)
    test = evaluate(
        model, loaders["test"], q_weight=args.q_weight, value_weight=args.value_weight
    )
    sample = SearchDataset(rows, [split["test"][0]])[0]
    latency = {
        "batch_1": benchmark(model, sample, batch_size=1),
        "batch_64": benchmark(model, sample, batch_size=64),
    }
    run_wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mib = peak_rss / (1024 * 1024 if sys.platform == "darwin" else 1024)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": DEFAULT_CONFIG.to_dict(),
            "best_epoch": best_epoch,
            "test": test,
            "dataset": str(args.dataset),
            "extra_train": [str(path) for path in args.extra_train] if args.extra_train else [],
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "hard_tactical_targets": args.hard_tactical_targets,
            "tactical_weight": args.tactical_weight,
            "selection_metric": args.selection_metric,
            "best_validation_balanced_score": best_balanced,
        },
        args.checkpoint,
    )
    report: dict[str, Any] = {
        "model": "ConnectFourTransformer candidate policy/Q/value v0",
        "config": DEFAULT_CONFIG.to_dict(),
        "parameters": parameter_count(model),
        "dataset": {"path": str(args.dataset), "extra_train": [str(path) for path in args.extra_train] if args.extra_train else [], "rows": len(rows), "split": {key: len(value) for key, value in split.items()}},
        "training": {
            "epochs": completed_epochs,
            "epoch_limit": epoch_limit,
            "time_budget_seconds": args.time_budget_seconds,
            "fit_wall_seconds": fit_wall_seconds,
            "optimizer_step_wall_seconds": optimizer_step_wall_seconds,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "q_weight": args.q_weight,
            "value_weight": args.value_weight,
            "hard_tactical_targets": args.hard_tactical_targets,
            "tactical_weight": args.tactical_weight,
            "selection_metric": args.selection_metric,
            "best_validation_balanced_score": best_balanced,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "wall_seconds": fit_wall_seconds,
            "total_run_wall_seconds": run_wall_seconds,
            "cpu_seconds": cpu_seconds,
            "peak_rss_mib": peak_rss_mib,
            "curve": curve,
        },
        "test": test,
        "latency": latency,
        "environment": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"test best move={test['best_move_accuracy']:.1%} | top2={test['top2_best_move_recall']:.1%} | "
        f"regret={test['mean_teacher_regret']:.3f}"
    )
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved report: {args.report}")


if __name__ == "__main__":
    main()
