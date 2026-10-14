"""Counterexample-guided training until every reachable position is optimal."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import resource
import sys
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import MODEL_CONFIG
from ..data.symmetry import symmetry_orbit
from ..model import TicTacToePolicy


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PACKAGE_ROOT / "artifacts" / "minimax_positions.jsonl"
DEFAULT_INITIAL = PACKAGE_ROOT / "artifacts" / "minimax_policy.pt"
DEFAULT_OUTPUT = PACKAGE_ROOT / "artifacts" / "checkpoints" / "optimal_policy.pt"
DEFAULT_HISTORY = PACKAGE_ROOT / "artifacts" / "training" / "hard_mining.jsonl"


class PositionRows(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], indices: list[int]) -> None:
        self.rows = rows
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = self.rows[self.indices[item]]
        return {
            "tokens": torch.tensor(row["tokens"], dtype=torch.long),
            "legal": torch.tensor(row["legal_mask"], dtype=torch.bool),
            "optimal": torch.tensor(
                [probability > 0 for probability in row["target_policy"]],
                dtype=torch.bool,
            ),
        }


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as dataset_file:
        rows = [json.loads(line) for line in dataset_file if line.strip()]
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


@torch.inference_mode()
def find_counterexamples(
    model: TicTacToePolicy,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
) -> tuple[list[int], dict[str, Any]]:
    """Return every row where greedy inference does not select an optimal move."""
    model.eval()
    loader = DataLoader(PositionRows(rows, list(range(len(rows)))), batch_size=batch_size)
    failures: list[int] = []
    offset = 0
    probability_on_optimal: list[float] = []
    margins: list[float] = []

    for batch in loader:
        logits = model(batch["tokens"]).masked_fill(~batch["legal"], -torch.inf)
        predictions = logits.argmax(dim=-1)
        correct = batch["optimal"].gather(1, predictions.unsqueeze(1)).squeeze(1)
        failures.extend(offset + index for index in (~correct).nonzero().flatten().tolist())

        probabilities = torch.softmax(logits, dim=-1)
        probability_on_optimal.extend(
            (probabilities * batch["optimal"]).sum(dim=-1).tolist()
        )
        best_good = logits.masked_fill(~batch["optimal"], -torch.inf).max(dim=-1).values
        bad_mask = batch["legal"] & ~batch["optimal"]
        best_bad = logits.masked_fill(~bad_mask, -torch.inf).max(dim=-1).values
        finite_bad = torch.isfinite(best_bad)
        margins.extend((best_good[finite_bad] - best_bad[finite_bad]).tolist())
        offset += batch["tokens"].shape[0]

    return failures, {
        "positions": len(rows),
        "failures": len(failures),
        "optimal_accuracy": 1 - len(failures) / len(rows),
        "mean_probability_on_optimal": sum(probability_on_optimal) / len(rows),
        "minimum_good_bad_margin": min(margins) if margins else math.inf,
    }


def expand_with_symmetries(
    failure_indices: list[int], rows: list[dict[str, Any]]
) -> set[int]:
    lookup = {(row["board"], row["current_player"]): index for index, row in enumerate(rows)}
    expanded: set[int] = set()
    for index in failure_indices:
        row = rows[index]
        for board in symmetry_orbit(row["board"]):
            symmetric_index = lookup.get((board, row["current_player"]))
            if symmetric_index is not None:
                expanded.add(symmetric_index)
    return expanded


def optimal_set_loss(
    logits: torch.Tensor,
    legal: torch.Tensor,
    optimal: torch.Tensor,
    *,
    margin: float,
    margin_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reward any optimal action and separate it from every suboptimal action."""
    masked_logits = logits.masked_fill(~legal, -torch.inf)
    log_probabilities = F.log_softmax(masked_logits, dim=-1)
    optimal_log_mass = torch.logsumexp(
        log_probabilities.masked_fill(~optimal, -torch.inf), dim=-1
    )
    set_cross_entropy = -optimal_log_mass.mean()

    best_good = masked_logits.masked_fill(~optimal, -torch.inf).max(dim=-1).values
    bad_mask = legal & ~optimal
    best_bad = masked_logits.masked_fill(~bad_mask, -torch.inf).max(dim=-1).values
    rows_with_bad_move = bad_mask.any(dim=-1)
    if rows_with_bad_move.any():
        margin_loss = F.relu(
            margin + best_bad[rows_with_bad_move] - best_good[rows_with_bad_move]
        ).mean()
    else:
        margin_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
    total = set_cross_entropy + margin_weight * margin_loss
    return total, set_cross_entropy, margin_loss


def train_round(
    model: TicTacToePolicy,
    rows: list[dict[str, Any]],
    indices: list[int],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    batch_size: int,
    margin: float,
    margin_weight: float,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        PositionRows(rows, indices),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    total_loss = 0.0
    total_set_loss = 0.0
    total_margin_loss = 0.0
    steps = 0
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, set_loss, margin_loss = optimal_set_loss(
                model(batch["tokens"]),
                batch["legal"],
                batch["optimal"],
                margin=margin,
                margin_weight=margin_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_set_loss += set_loss.item()
            total_margin_loss += margin_loss.item()
            steps += 1
    return {
        "loss": total_loss / steps,
        "set_cross_entropy": total_set_loss / steps,
        "margin_loss": total_margin_loss / steps,
        "optimizer_steps": steps,
    }


def peak_rss_mib() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024 * 1024 if sys.platform == "darwin" else 1024)


def save_checkpoint(
    path: Path,
    model: TicTacToePolicy,
    *,
    round_number: int,
    report: dict[str, Any],
    hard_buffer_size: int,
    source_checkpoint: Path,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training": "counterexample_guided_optimal_policy",
            "source_checkpoint": str(source_checkpoint.resolve()),
            "round": round_number,
            "hard_buffer_size": hard_buffer_size,
            "verification": report,
            "model_config": asdict(MODEL_CONFIG),
            "hyperparameters": {
                "learning_rate": args.learning_rate,
                "epochs_per_round": args.epochs_per_round,
                "batch_size": args.batch_size,
                "replay_ratio": args.replay_ratio,
                "margin": args.margin,
                "margin_weight": args.margin_weight,
                "seed": args.seed,
            },
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--initial-checkpoint", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--epochs-per-round", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--verify-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--replay-ratio", type=float, default=0.50)
    parser.add_argument("--margin", type=float, default=0.75)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if any(
        value <= 0
        for value in (
            args.max_rounds,
            args.epochs_per_round,
            args.batch_size,
            args.verify_batch_size,
            args.learning_rate,
            args.threads,
        )
    ):
        parser.error("rounds, epochs, batch sizes, learning rate, and threads must be positive")
    if args.replay_ratio < 0:
        parser.error("replay-ratio cannot be negative")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows = load_rows(args.dataset)
    checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
    model = TicTacToePolicy()
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    args.history.write_text("", encoding="utf-8")

    failures, report = find_counterexamples(
        model, rows, batch_size=args.verify_batch_size
    )
    hard_buffer = expand_with_symmetries(failures, rows)
    best_failure_count = len(failures)
    save_checkpoint(
        args.output,
        model,
        round_number=0,
        report=report,
        hard_buffer_size=len(hard_buffer),
        source_checkpoint=args.initial_checkpoint,
        args=args,
    )
    print(
        f"round  0 | failures {len(failures):>4}/{len(rows)} | "
        f"accuracy {report['optimal_accuracy']:.2%} | hard buffer {len(hard_buffer)}"
    )
    started = time.perf_counter()

    for round_number in range(1, args.max_rounds + 1):
        if not failures:
            break
        replay_candidates = list(set(range(len(rows))) - hard_buffer)
        replay_count = min(
            len(replay_candidates), math.ceil(len(hard_buffer) * args.replay_ratio)
        )
        training_indices = sorted(hard_buffer) + rng.sample(replay_candidates, replay_count)
        training = train_round(
            model,
            rows,
            training_indices,
            optimizer,
            epochs=args.epochs_per_round,
            batch_size=args.batch_size,
            margin=args.margin,
            margin_weight=args.margin_weight,
            seed=args.seed + round_number,
        )
        failures, report = find_counterexamples(
            model, rows, batch_size=args.verify_batch_size
        )
        hard_buffer.update(expand_with_symmetries(failures, rows))
        record = {
            "round": round_number,
            "trained_positions": len(training_indices),
            "hard_buffer_size": len(hard_buffer),
            "training": training,
            "verification": report,
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_mib": peak_rss_mib(),
        }
        with args.history.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(
            f"round {round_number:>2} | failures {len(failures):>4}/{len(rows)} | "
            f"accuracy {report['optimal_accuracy']:.2%} | hard buffer {len(hard_buffer):>4} | "
            f"trained {len(training_indices):>4} | loss {training['loss']:.4f}"
        )
        if len(failures) < best_failure_count:
            best_failure_count = len(failures)
            save_checkpoint(
                args.output,
                model,
                round_number=round_number,
                report=report,
                hard_buffer_size=len(hard_buffer),
                source_checkpoint=args.initial_checkpoint,
                args=args,
            )

    elapsed = time.perf_counter() - started
    if failures:
        print(
            f"Stopped with {len(failures)} counterexamples after {args.max_rounds} rounds; "
            f"best checkpoint has {best_failure_count}."
        )
    else:
        print(f"CERTIFIED: zero suboptimal moves across all {len(rows):,} positions.")
    print(f"Final trained weights: {args.output}")
    print(f"Training history: {args.history}")
    print(f"CPU wall time: {elapsed:.2f}s | peak RSS: {peak_rss_mib():.1f} MiB")


if __name__ == "__main__":
    main()
