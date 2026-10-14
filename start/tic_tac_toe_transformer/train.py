"""Train the Tic-Tac-Toe policy to imitate Minimax's optimal moves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tic_tac_toe_transformer.model import TicTacToePolicy


DEFAULT_DATASET = Path(__file__).parent / "artifacts" / "minimax_positions.jsonl"
DEFAULT_CHECKPOINT = Path(__file__).parent / "artifacts" / "minimax_policy.pt"


class PositionDataset(Dataset[dict[str, torch.Tensor]]):
    """Load fixed-length token positions and soft Minimax policy targets."""

    def __init__(self, path: Path) -> None:
        with path.open(encoding="utf-8") as dataset_file:
            rows = [json.loads(line) for line in dataset_file if line.strip()]
        if not rows:
            raise ValueError(f"dataset is empty: {path}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            "tokens": torch.tensor(row["tokens"], dtype=torch.long),
            "legal_mask": torch.tensor(row["legal_mask"], dtype=torch.bool),
            "target_policy": torch.tensor(row["target_policy"], dtype=torch.float32),
            "optimal_mask": torch.tensor(
                [move in row["optimal_moves"] for move in range(1, 10)],
                dtype=torch.bool,
            ),
        }


class CpuSnapshot:
    """Dependency-free process CPU-time and peak-RSS measurements."""

    def __init__(self) -> None:
        self.wall = time.perf_counter()
        self.cpu = time.process_time()

    def read(self) -> dict[str, float | None]:
        now_wall = time.perf_counter()
        now_cpu = time.process_time()
        wall_delta = now_wall - self.wall
        cpu_delta = now_cpu - self.cpu
        self.wall = now_wall
        self.cpu = now_cpu

        peak_rss_mib: float | None = None
        try:
            import resource

            rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            # macOS reports bytes; Linux and most other Unix systems report KiB.
            peak_rss_mib = rss / (1024 * 1024 if sys.platform == "darwin" else 1024)
        except (ImportError, OSError, ValueError):
            pass

        cores = cpu_delta / wall_delta if wall_delta > 0 else 0.0
        return {
            "wall_seconds": wall_delta,
            "cpu_seconds": cpu_delta,
            "cpu_percent": 100.0 * cores,
            "cores": cores,
            "peak_rss_mib": peak_rss_mib,
        }


def policy_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Soft-target cross-entropy over legal moves only."""
    legal = batch["legal_mask"]
    masked_logits = logits.masked_fill(~legal, -torch.inf)
    log_probabilities = F.log_softmax(masked_logits, dim=-1)
    log_probabilities = log_probabilities.masked_fill(~legal, 0.0)
    return -(batch["target_policy"] * log_probabilities).sum(dim=-1).mean()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader[Any], device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_total = 0.0
    correct = 0
    count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(batch["tokens"])
        loss = policy_loss(logits, batch)
        predictions = logits.masked_fill(~batch["legal_mask"], -torch.inf).argmax(dim=-1)
        correct += batch["optimal_mask"].gather(1, predictions.unsqueeze(1)).sum().item()
        batch_size = batch["tokens"].shape[0]
        loss_total += loss.item() * batch_size
        count += batch_size
    return loss_total / count, correct / count


def format_profile(stats: dict[str, float | None]) -> str:
    rss = stats["peak_rss_mib"]
    rss_text = f"{rss:.1f} MiB peak RSS" if rss is not None else "peak RSS n/a"
    return (
        f"CPU {stats['cpu_percent']:.0f}% (~{stats['cores']:.2f} cores), "
        f"{stats['cpu_seconds']:.2f}s process CPU / {stats['wall_seconds']:.2f}s wall, "
        f"{rss_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10, help="print training metrics every N epochs")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.log_every <= 0:
        parser.error("epochs, batch-size, and log-every must be positive")
    if not 0.0 <= args.val_fraction < 1.0:
        parser.error("val-fraction must be in [0, 1)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    dataset = PositionDataset(args.dataset)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    val_count = math.floor(len(indices) * args.val_fraction)
    if args.val_fraction > 0 and len(indices) > 1:
        val_count = max(1, val_count)
    train_indices = indices[val_count:]
    val_indices = indices[:val_count]
    if not train_indices:
        parser.error("validation split leaves no training examples")

    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size)

    model = TicTacToePolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Device: {device} | examples: {len(train_indices):,} train / "
        f"{len(val_indices):,} validation | parameters: {parameter_count:,}"
    )
    print(
        f"Training: AdamW, lr={args.learning_rate:g}, batch={args.batch_size}, "
        f"epochs={args.epochs}; metric = top move is any Minimax-optimal move"
    )

    profile = CpuSnapshot()
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        count = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["tokens"])
            loss = policy_loss(logits, batch)
            loss.backward()
            optimizer.step()
            batch_size = batch["tokens"].shape[0]
            loss_total += loss.item() * batch_size
            count += batch_size

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            train_loss = loss_total / count
            if val_indices:
                val_loss, val_accuracy = evaluate(model, val_loader, device)
                accuracy_text = f"val loss {val_loss:.4f} | optimal-move accuracy {val_accuracy:.1%}"
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "epoch": epoch,
                            "validation_accuracy": val_accuracy,
                            "config": {
                                "d_model": 32,
                                "n_heads": 4,
                                "n_blocks": 2,
                                "mlp_hidden": 64,
                            },
                        },
                        args.checkpoint,
                    )
            else:
                accuracy_text = "validation disabled"
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, args.checkpoint)
            stats = profile.read()
            print(
                f"epoch {epoch:>3}/{args.epochs} | train loss {train_loss:.4f} | "
                f"{accuracy_text} | {format_profile(stats)}"
            )

    print(f"Best checkpoint saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()
