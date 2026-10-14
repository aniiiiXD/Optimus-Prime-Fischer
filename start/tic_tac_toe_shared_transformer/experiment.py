"""Fair CPU comparison of classical, candidate, and shared multi-task Transformers."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tic_tac_toe_transformer.model import TicTacToePolicy

from tic_tac_toe_shared_transformer.config import DEFAULT_CONFIG
from tic_tac_toe_shared_transformer.data import (
    DEFAULT_DATASET,
    DecisionDataset,
    all_candidates,
    grouped_split,
    load_rows,
    nested_training_subset,
)
from tic_tac_toe_shared_transformer.losses import LossWeights, shared_loss, soft_policy_loss
from tic_tac_toe_shared_transformer.model import SharedActionTransformer


ARCHITECTURES = ("classic_policy", "candidate_policy", "shared_multitask")
DEFAULT_REPORT = Path(__file__).parent / "artifacts" / "reports" / "comparison.json"
DEFAULT_CHECKPOINT_DIR = Path(__file__).parent / "artifacts" / "checkpoints"


def parse_csv_numbers(text: str, cast: type) -> list[Any]:
    values = [cast(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("provide at least one comma-separated value")
    return values


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def make_loader(
    rows: list[dict[str, Any]],
    indices: list[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        DecisionDataset(rows, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def policy_logits_for(
    architecture: str,
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Any | None]:
    if architecture == "classic_policy":
        return model(batch["tokens"]), None
    candidate_ids, candidate_mask = all_candidates(batch["tokens"].shape[0], torch.device("cpu"))
    output = model(batch["tokens"], candidate_ids, candidate_mask)
    return output.policy_logits, output


@torch.inference_mode()
def evaluate(
    architecture: str,
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
) -> dict[str, float | int]:
    model.eval()
    count = 0
    policy_loss_total = 0.0
    optimal = 0
    regret_sum = 0
    losing_blunders = 0
    q_correct = 0
    q_count = 0
    value_correct = 0
    legality_correct = 0
    legality_count = 0

    for batch in loader:
        logits, output = policy_logits_for(architecture, model, batch)
        policy_loss = soft_policy_loss(logits, batch["legal_mask"], batch["target_policy"])
        predictions = logits.masked_fill(~batch["legal_mask"], -torch.inf).argmax(dim=-1)
        chosen_classes = batch["action_value_targets"].gather(1, predictions.unsqueeze(1)).squeeze(1)
        best_classes = batch["state_value_target"]
        batch_optimal = batch["optimal_mask"].gather(1, predictions.unsqueeze(1)).squeeze(1)
        optimal += int(batch_optimal.sum().item())
        regrets = best_classes - chosen_classes
        regret_sum += int(regrets.sum().item())
        losing_blunders += int(((chosen_classes == 0) & (best_classes >= 1)).sum().item())
        batch_size = batch["tokens"].shape[0]
        policy_loss_total += float(policy_loss.item()) * batch_size
        count += batch_size

        if architecture == "shared_multitask":
            legal = batch["legal_mask"]
            q_predictions = output.action_value_logits.argmax(dim=-1)
            q_correct += int((q_predictions[legal] == batch["action_value_targets"][legal]).sum().item())
            q_count += int(legal.sum().item())
            value_correct += int(
                (output.state_value_logits.argmax(dim=-1) == batch["state_value_target"]).sum().item()
            )
            legal_predictions = output.legality_logits >= 0
            legality_correct += int((legal_predictions == legal).sum().item())
            legality_count += legal.numel()

    metrics: dict[str, float | int] = {
        "positions": count,
        "policy_loss": policy_loss_total / count,
        "optimal_move_accuracy": optimal / count,
        "mean_minimax_regret": regret_sum / count,
        "losing_blunder_rate": losing_blunders / count,
    }
    if q_count:
        metrics.update(
            {
                "legal_action_value_accuracy": q_correct / q_count,
                "state_value_accuracy": value_correct / count,
                "legality_accuracy": legality_correct / legality_count,
            }
        )
    return metrics


def build_model(architecture: str) -> nn.Module:
    if architecture == "classic_policy":
        return TicTacToePolicy()
    return SharedActionTransformer(DEFAULT_CONFIG)


def train_one(
    architecture: str,
    rows: list[dict[str, Any]],
    train_indices: list[int],
    validation_indices: list[int],
    test_indices: list[int],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    curve_epochs: set[int],
    loss_weights: LossWeights,
) -> tuple[dict[str, Any], nn.Module]:
    random.seed(seed)
    torch.manual_seed(seed)
    model = build_model(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_loader = make_loader(
        rows, train_indices, batch_size=batch_size, shuffle=True, seed=seed
    )
    validation_loader = make_loader(
        rows, validation_indices, batch_size=batch_size, shuffle=False, seed=seed
    )
    test_loader = make_loader(rows, test_indices, batch_size=batch_size, shuffle=False, seed=seed)

    best_key = (-1.0, -math.inf)
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    curve: list[dict[str, Any]] = []
    optimizer_steps = 0
    wall_started = time.perf_counter()
    cpu_started = time.process_time()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, output = policy_logits_for(architecture, model, batch)
            if architecture == "classic_policy":
                loss = soft_policy_loss(logits, batch["legal_mask"], batch["target_policy"])
            else:
                loss, _ = shared_loss(
                    output,
                    batch,
                    loss_weights,
                    multitask=architecture == "shared_multitask",
                )
            loss.backward()
            optimizer.step()
            optimizer_steps += 1

        validation = evaluate(architecture, model, validation_loader)
        key = (
            float(validation["optimal_move_accuracy"]),
            -float(validation["policy_loss"]),
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        if epoch in curve_epochs or epoch == epochs:
            curve.append({"epoch": epoch, "validation": validation})

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    test = evaluate(architecture, model, test_loader)
    return (
        {
            "architecture": architecture,
            "seed": seed,
            "train_examples": len(train_indices),
            "epochs": epochs,
            "optimizer_steps": optimizer_steps,
            "examples_seen": len(train_indices) * epochs,
            "best_epoch": best_epoch,
            "parameter_count": parameter_count(model),
            "training_wall_seconds": time.perf_counter() - wall_started,
            "training_cpu_seconds": time.process_time() - cpu_started,
            "validation_best": {
                "optimal_move_accuracy": best_key[0],
                "policy_loss": -best_key[1],
            },
            "test": test,
            "curve": curve,
        },
        model,
    )


@torch.inference_mode()
def latency_benchmark(
    architecture: str,
    model: nn.Module,
    sample_tokens: torch.Tensor,
    batch_size: int,
    iterations: int = 200,
) -> dict[str, float | int]:
    model.eval()
    tokens = sample_tokens.repeat(batch_size, 1)
    candidates, candidate_mask = all_candidates(batch_size, torch.device("cpu"))

    def run() -> None:
        if architecture == "classic_policy":
            model(tokens)
        else:
            model(tokens, candidates, candidate_mask)

    for _ in range(20):
        run()
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        run()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    median = statistics.median(samples)
    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "median_latency_ms": median,
        "p95_latency_ms": p95,
        "positions_per_second_at_median": 1000.0 * batch_size / median,
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["architecture"], run["train_fraction"])].append(run)
    aggregates = []
    for (architecture, fraction), group in sorted(groups.items()):
        accuracies = [float(run["test"]["optimal_move_accuracy"]) for run in group]
        blunders = [float(run["test"]["losing_blunder_rate"]) for run in group]
        aggregates.append(
            {
                "architecture": architecture,
                "train_fraction": fraction,
                "train_examples_mean": statistics.mean(run["train_examples"] for run in group),
                "optimal_move_accuracy_mean": statistics.mean(accuracies),
                "optimal_move_accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "losing_blunder_rate_mean": statistics.mean(blunders),
                "training_wall_seconds_mean": statistics.mean(
                    run["training_wall_seconds"] for run in group
                ),
            }
        )
    return aggregates


def checkpoint_payload(
    architecture: str, model: nn.Module, run: dict[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "architecture": architecture,
        "model_state_dict": model.state_dict(),
        "experiment": {key: value for key, value in run.items() if key != "curve"},
    }
    if architecture != "classic_policy":
        payload["model_config"] = DEFAULT_CONFIG.to_dict()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--fractions", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--seeds", default="13,42,73")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--architectures",
        default=",".join(ARCHITECTURES),
        help="comma-separated subset of classic_policy,candidate_policy,shared_multitask",
    )
    parser.add_argument("--action-value-weight", type=float, default=0.5)
    parser.add_argument("--state-value-weight", type=float, default=0.25)
    parser.add_argument("--legality-weight", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=2026)
    args = parser.parse_args()

    fractions = parse_csv_numbers(args.fractions, float)
    seeds = parse_csv_numbers(args.seeds, int)
    architectures = [value.strip() for value in args.architectures.split(",") if value.strip()]
    unknown_architectures = sorted(set(architectures) - set(ARCHITECTURES))
    if not architectures:
        parser.error("provide at least one architecture")
    if unknown_architectures:
        parser.error(f"unknown architectures: {', '.join(unknown_architectures)}")
    loss_weights = LossWeights(
        action_value=args.action_value_weight,
        state_value=args.state_value_weight,
        legality=args.legality_weight,
    )
    if args.epochs <= 0 or args.batch_size <= 0 or args.threads <= 0:
        parser.error("epochs, batch-size, and threads must be positive")
    if any(fraction <= 0 or fraction > 1 for fraction in fractions):
        parser.error("fractions must be in (0, 1]")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    rows = load_rows(args.dataset)
    splits = grouped_split(rows, seed=args.split_seed)
    sample_tokens = torch.tensor([rows[splits["test"][0]]["tokens"]], dtype=torch.long)
    curve_epochs = {1, 2, 5, 10, 20, args.epochs}

    print(
        f"Symmetry-safe split: {len(splits['train'])} train / "
        f"{len(splits['validation'])} validation / {len(splits['test'])} test"
    )
    print(
        f"CPU threads={args.threads} | epochs={args.epochs} | batch={args.batch_size} | "
        f"lr={args.learning_rate:g}"
    )

    runs: list[dict[str, Any]] = []
    benchmark_models: dict[str, nn.Module] = {}
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for fraction in fractions:
        train_indices = nested_training_subset(rows, splits["train"], fraction, args.split_seed)
        for seed in seeds:
            for architecture in architectures:
                print(
                    f"training {architecture:>18} | data={fraction:>5.0%} "
                    f"({len(train_indices):4d}) | seed={seed}"
                )
                run, model = train_one(
                    architecture,
                    rows,
                    train_indices,
                    splits["validation"],
                    splits["test"],
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    curve_epochs=curve_epochs,
                    loss_weights=loss_weights,
                )
                run["train_fraction"] = fraction
                runs.append(run)
                print(
                    f"  test optimal={run['test']['optimal_move_accuracy']:.2%} | "
                    f"losing blunders={run['test']['losing_blunder_rate']:.2%} | "
                    f"best epoch={run['best_epoch']} | {run['training_wall_seconds']:.2f}s"
                )
                if fraction == max(fractions):
                    checkpoint = args.checkpoint_dir / f"{architecture}_seed{seed}.pt"
                    torch.save(checkpoint_payload(architecture, model, run), checkpoint)
                    if seed == seeds[0]:
                        benchmark_models[architecture] = model

    benchmarks: dict[str, Any] = {}
    for architecture, model in benchmark_models.items():
        benchmarks[architecture] = {
            "parameters": parameter_count(model),
            "batch_1": latency_benchmark(architecture, model, sample_tokens, 1),
            "batch_128": latency_benchmark(architecture, model, sample_tokens, 128),
        }

    report = {
        "experiment": "classical vs candidate-action vs shared multi-task Transformer",
        "created_at_unix": time.time(),
        "environment": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
        "protocol": {
            "dataset": str(args.dataset),
            "total_positions": len(rows),
            "split": {key: len(value) for key, value in splits.items()},
            "split_seed": args.split_seed,
            "symmetry_grouped": True,
            "fractions": fractions,
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "AdamW",
            "loss_weights": asdict(loss_weights),
            "checkpoint_selection": "highest validation optimal accuracy, then lowest policy loss",
        },
        "runs": runs,
        "aggregates": aggregate_runs(runs),
        "inference_benchmarks": benchmarks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved report: {args.report}")
    print("\narchitecture          data   test optimal (mean +/- sd)   losing blunders")
    for row in report["aggregates"]:
        print(
            f"{row['architecture']:<21} {row['train_fraction']:>4.0%}   "
            f"{row['optimal_move_accuracy_mean']:>7.2%} +/- "
            f"{row['optimal_move_accuracy_std']:>6.2%}       "
            f"{row['losing_blunder_rate_mean']:>7.2%}"
        )


if __name__ == "__main__":
    main()
