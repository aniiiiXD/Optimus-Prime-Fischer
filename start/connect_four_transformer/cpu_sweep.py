"""Measure fixed-batch CPU inference and optimizer-step cost on one checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

import torch

from .config import ModelConfig
from .data import SearchDataset, load_rows
from .model import ConnectFourTransformer
from .train import forward, losses


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "connect_four_distillation/artifacts/checkpoints/candidate_pons_phases_v3_5min.pt"
)
DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "connect_four_distillation/artifacts/data/pons_exact_phases_v3.jsonl"
)


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def measure(
    checkpoint: dict, sample: dict[str, torch.Tensor],
    *, batch_size: int, threads: int, repeats: int, mode: str,
) -> dict[str, float | int | str]:
    torch.set_num_threads(threads)
    model = ConnectFourTransformer(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    batch = {
        name: value.unsqueeze(0).expand(batch_size, *value.shape).clone()
        for name, value in sample.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if mode == "train_step" else None

    def run_once() -> None:
        if optimizer is None:
            model.eval()
            with torch.inference_mode():
                forward(model, batch)
        else:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = forward(model, batch)
            loss, _ = losses(output, batch, q_weight=0.5, value_weight=0.25)
            loss.backward()
            optimizer.step()

    for _ in range(5):
        run_once()
    wall_ms = []
    cpu_ms = []
    for _ in range(repeats):
        wall_start = time.perf_counter_ns()
        cpu_start = time.process_time_ns()
        run_once()
        wall_ms.append((time.perf_counter_ns() - wall_start) / 1e6)
        cpu_ms.append((time.process_time_ns() - cpu_start) / 1e6)
    median = statistics.median(wall_ms)
    return {
        "mode": mode,
        "threads": threads,
        "batch_size": batch_size,
        "repeats": repeats,
        "median_ms": median,
        "p95_ms": percentile(wall_ms, 0.95),
        "positions_per_second": batch_size * 1000 / median,
        "mean_cpu_cores": sum(cpu_ms) / sum(wall_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--threads", default="1,2,4")
    parser.add_argument("--batch-sizes", default="1,32,128")
    parser.add_argument("--modes", default="forward,train_step")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    threads = [int(value) for value in args.threads.split(",")]
    batches = [int(value) for value in args.batch_sizes.split(",")]
    modes = args.modes.split(",")
    if not threads or not batches or min(threads + batches + [args.repeats]) <= 0:
        parser.error("threads, batches, and repeats must be positive")
    if any(mode not in ("forward", "train_step") for mode in modes):
        parser.error("modes must be forward and/or train_step")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    sample = SearchDataset(load_rows(args.dataset), [0])[0]
    results = []
    for mode in modes:
        for batch_size in batches:
            for thread_count in threads:
                row = measure(
                    checkpoint, sample, batch_size=batch_size,
                    threads=thread_count, repeats=args.repeats, mode=mode,
                )
                results.append(row)
                print(
                    f"{mode:<10} batch={batch_size:<4} threads={thread_count} "
                    f"median={row['median_ms']:.2f} ms "
                    f"throughput={row['positions_per_second']:.0f}/s",
                    flush=True,
                )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "model_config": checkpoint["model_config"],
        "machine": platform.platform(),
        "pytorch": torch.__version__,
        "peak_rss_mib": rss / (1024 * 1024 if sys.platform == "darwin" else 1024),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
