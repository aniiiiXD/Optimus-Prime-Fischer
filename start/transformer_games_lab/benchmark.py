#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from gamebench.models.tiny_transformer import TinyTransformerConfig
from gamebench.registry import make_game
from gamebench.systems.benchmark import BenchmarkConfig, run_benchmark
from gamebench.systems.torch_workload import TransformerWorkload, TransformerWorkloadConfig


PRESETS = {
    "micro": {"d_model": 64, "n_heads": 4, "n_layers": 2, "d_ff": 128},
    "small": {"d_model": 128, "n_heads": 4, "n_layers": 4, "d_ff": 512},
    "othello-gpt-shape": {"d_model": 512, "n_heads": 8, "n_layers": 8, "d_ff": 2048},
}


def integers(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile tiny Transformer CPU forward/backward performance.")
    parser.add_argument("--game", choices=("othello", "connect4"), default="othello")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="micro")
    parser.add_argument("--batch-sizes", type=integers, default=[1, 8])
    parser.add_argument("--sequence-lengths", type=integers, default=[8, 32])
    parser.add_argument("--modes", type=strings, default=["forward"])
    parser.add_argument("--threads", type=integers, default=[1])
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--target-sample-ms", type=float, default=8.0)
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument("--target-cv", type=float, default=0.05)
    parser.add_argument("--memory-seconds", type=float, default=0.15)
    parser.add_argument("--memory-sample-ms", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    invalid_modes = set(args.modes) - {"forward", "forward_backward", "train_step"}
    if invalid_modes:
        parser.error(f"invalid modes: {sorted(invalid_modes)}")
    torch.set_num_interop_threads(args.interop_threads)
    game = make_game(args.game)
    maximum_sequence = max(args.sequence_lengths)
    model_config = TinyTransformerConfig(
        vocab_size=game.spec.token_vocab_size,
        action_size=game.spec.action_space_size,
        context_length=maximum_sequence,
        **PRESETS[args.preset],
    )
    benchmark_config = BenchmarkConfig(
        warmup_calls=args.warmup,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        target_sample_ms=args.target_sample_ms,
        min_total_seconds=args.min_seconds,
        target_robust_cv=args.target_cv,
        memory_seconds=args.memory_seconds,
        memory_sample_interval_ms=args.memory_sample_ms,
    )
    results: list[dict[str, Any]] = []
    for threads in args.threads:
        torch.set_num_threads(threads)
        for mode in args.modes:
            for sequence_length in args.sequence_lengths:
                for batch_size in args.batch_sizes:
                    workload = TransformerWorkload(TransformerWorkloadConfig(
                        model=model_config,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        mode=mode,  # type: ignore[arg-type]
                        dtype=args.dtype,
                        seed=args.seed,
                    ))
                    result = run_benchmark(workload, benchmark_config).as_dict()
                    result["workload"]["requested_torch_threads"] = threads
                    results.append(result)
                    latency = result["latency_ms"]["median"]
                    throughput = result["throughput_items_per_second"]
                    stable = "stable" if result["variability"]["stable"] else "noisy"
                    print(f"threads={threads:<2} {workload.name:<43} {latency:>9.3f} ms  {throughput:>10.1f} items/s  {stable}")
    payload = {
        "schema_version": 1,
        "benchmark_method": {
            "warmup": "untimed calls before calibration",
            "latency": "adaptive inner-loop samples using perf_counter_ns; GC disabled only during timing",
            "stability": "robust CV = 1.4826 * median absolute deviation / median",
            "memory": "separate process-RSS sampling phase",
            "estimates": "analytical dense FLOPs and one weight-stream byte pass; not hardware counters",
        },
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.output}")
    elif len(results) == 1:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
