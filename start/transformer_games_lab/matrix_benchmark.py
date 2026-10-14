#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gamebench.systems.benchmark import BenchmarkConfig, run_benchmark
from gamebench.systems.hardware import cpu_capabilities
from gamebench.systems.matrix_workload import MatrixMultiplyWorkload, MatrixWorkloadConfig


def integers(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value.strip()]


def strings(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile isolated CPU matrix multiplication.")
    parser.add_argument("--sizes", type=integers, default=[128, 512, 1024])
    parser.add_argument("--dtypes", type=strings, default=["float32"])
    parser.add_argument("--threads", type=integers, default=[1, 4])
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    invalid = set(args.dtypes) - {"float32", "bfloat16", "float16"}
    if invalid:
        parser.error(f"invalid dtypes: {sorted(invalid)}")
    config = BenchmarkConfig(min_total_seconds=args.min_seconds)
    results = []
    for threads in args.threads:
        torch.set_num_threads(threads)
        for dtype in args.dtypes:
            for size in args.sizes:
                workload = MatrixMultiplyWorkload(MatrixWorkloadConfig(size, size, size, dtype))
                result = run_benchmark(workload, config).as_dict()
                result["workload"]["requested_torch_threads"] = threads
                results.append(result)
                print(
                    f"threads={threads:<2} {dtype:<8} {size:>4}³  "
                    f"{result['latency_ms']['median']:>9.3f} ms  "
                    f"{result['estimates']['achieved_gflops_per_second']:>9.1f} GFLOP/s"
                )
    payload = {"schema_version": 1, "hardware": cpu_capabilities(), "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
