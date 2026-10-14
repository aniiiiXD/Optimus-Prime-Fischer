#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gamebench.systems.benchmark import BenchmarkConfig, run_benchmark
from gamebench.systems.hardware import cpu_capabilities
from gamebench.systems.native_workload import NativeDotConfig, NativeDotWorkload, build_native_library


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare native scalar and explicit SIMD dot products.")
    parser.add_argument("--elements", type=int, default=1_048_576)
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    library = build_native_library(root)
    config = BenchmarkConfig(min_total_seconds=args.min_seconds)
    results = []
    for kernel in ("scalar", "vector"):
        workload = NativeDotWorkload(NativeDotConfig(args.elements, kernel), library)
        result = run_benchmark(workload, config).as_dict()
        results.append(result)
        print(
            f"{kernel:<7} {result['latency_ms']['median']:>9.3f} ms  "
            f"{result['estimates']['achieved_gflops_per_second']:>8.2f} GFLOP/s"
        )
    speedup = results[0]["latency_ms"]["median"] / results[1]["latency_ms"]["median"]
    scalar_value = results[0]["workload"]["result"]
    vector_value = results[1]["workload"]["result"]
    relative_error = abs(scalar_value - vector_value) / max(abs(scalar_value), 1e-30)
    maximum_reference_error = max(
        result["workload"]["relative_error_vs_reference"] for result in results
    )
    if maximum_reference_error > 1e-3:
        raise RuntimeError(
            f"a kernel differs from the float64 reference by {maximum_reference_error:.3%}"
        )
    print(f"SIMD speedup: {speedup:.2f}x")
    print(f"result relative difference: {relative_error:.3e}")
    print(f"maximum error vs float64 accumulation: {maximum_reference_error:.3e}")
    payload = {
        "schema_version": 1,
        "hardware": cpu_capabilities(),
        "compiled_library": str(library),
        "simd_speedup": speedup,
        "result_relative_difference": relative_error,
        "maximum_relative_error_vs_float64_reference": maximum_reference_error,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
