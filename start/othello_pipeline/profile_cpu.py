"""Reproducible CPU timing for the Othello student, not a strength test."""

from __future__ import annotations

import argparse
import json
import platform
import random
import resource
import statistics
import time
from pathlib import Path

import torch

from gamebench.games.othello import OthelloAdapter, OthelloState
from .model import ModelConfig, OthelloTransformer, PAD_ACTION, encode_board


def sampled_states(count: int, seed: int) -> list[OthelloState]:
    rng = random.Random(seed)
    states: list[OthelloState] = []
    while len(states) < count:
        state = OthelloAdapter().reset()
        while not state.is_terminal() and len(states) < count:
            states.append(state.clone())
            state.apply_action(rng.choice(state.legal_actions()))
    return states


def inputs(states: list[OthelloState], all_candidates: bool) -> tuple[torch.Tensor, ...]:
    board = torch.stack([encode_board(state.board, state.current_player) for state in states])
    side = torch.tensor([state.current_player for state in states], dtype=torch.long)
    legal_lists = [tuple(range(65)) if all_candidates else state.legal_actions() for state in states]
    width = max(map(len, legal_lists))
    candidates = torch.full((len(states), width), PAD_ACTION, dtype=torch.long)
    mask = torch.zeros((len(states), width), dtype=torch.bool)
    for row, legal in enumerate(legal_lists):
        candidates[row, : len(legal)] = torch.tensor(legal, dtype=torch.long)
        mask[row, : len(legal)] = True
    return board, side, candidates, mask


def timing(model: OthelloTransformer, tensors: tuple[torch.Tensor, ...], warmup: int, samples: int) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(*tensors)
        durations: list[float] = []
        for _ in range(samples):
            start = time.perf_counter_ns()
            model(*tensors)
            durations.append((time.perf_counter_ns() - start) / 1e6)
    sorted_times = sorted(durations)
    median = statistics.median(sorted_times)
    return {
        "median_ms": median,
        "p95_ms": sorted_times[min(len(sorted_times) - 1, int(0.95 * len(sorted_times)))],
        "positions_per_second": len(tensors[0]) * 1000.0 / median,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the untrained Othello Transformer on CPU.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-sizes", default="1,32,128")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sizes = [int(item) for item in args.batch_sizes.split(",")]
    if min(args.threads, args.warmup, args.samples, *sizes) <= 0:
        parser.error("threads, warmup, samples, and batch sizes must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = OthelloTransformer(ModelConfig()).cpu().eval()
    states = sampled_states(max(sizes), args.seed)
    results = []
    for batch_size in sizes:
        for all_candidates in (False, True):
            tensors = inputs(states[:batch_size], all_candidates)
            measurement = timing(model, tensors, args.warmup, args.samples)
            result = {"batch_size": batch_size, "candidates": "all_65" if all_candidates else "legal_only", **measurement}
            results.append(result)
            print(f"batch={batch_size:3d} {result['candidates']:10s} p50={measurement['median_ms']:.3f} ms p95={measurement['p95_ms']:.3f} ms")
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "purpose": "untrained model kernel/dispatch timing; not playing strength",
        "hardware": platform.platform(),
        "pytorch": torch.__version__,
        "threads": args.threads,
        "warmup": args.warmup,
        "samples": args.samples,
        "seed": args.seed,
        "model_config": model.config.to_dict(),
        "parameters": model.parameter_count,
        "peak_process_rss_bytes": maxrss if platform.system() == "Darwin" else maxrss * 1024,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
