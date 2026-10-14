"""Benchmark kernel with adaptive inner loops and stability diagnostics.

Timing and memory are intentionally separate phases: RSS sampling can distort
short latency measurements.  Workloads provide estimates, never fake measured
hardware counters; result field names keep that distinction explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
import math
import os
import platform
import statistics
import sys
import threading
import time
from typing import Any, Protocol

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency fallback
    psutil = None  # type: ignore[assignment]


class BenchmarkWorkload(Protocol):
    name: str

    def run(self) -> None: ...
    def synchronize(self) -> None: ...
    def metadata(self) -> dict[str, Any]: ...
    def work_items_per_call(self) -> int: ...
    def estimated_flops_per_call(self) -> float | None: ...
    def estimated_bytes_per_call(self) -> float | None: ...


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_calls: int = 10
    min_samples: int = 25
    max_samples: int = 100
    target_sample_ms: float = 8.0
    min_total_seconds: float = 0.5
    target_robust_cv: float = 0.05
    memory_seconds: float = 0.15
    memory_sample_interval_ms: float = 2.0

    def __post_init__(self) -> None:
        if self.warmup_calls < 0 or self.min_samples < 2 or self.max_samples < self.min_samples:
            raise ValueError("invalid warmup/sample counts")
        if self.target_sample_ms <= 0 or self.min_total_seconds <= 0:
            raise ValueError("benchmark durations must be positive")
        if self.target_robust_cv < 0:
            raise ValueError("target_robust_cv cannot be negative")
        if self.memory_seconds <= 0 or self.memory_sample_interval_ms <= 0:
            raise ValueError("memory profiling durations must be positive")


@dataclass
class BenchmarkResult:
    name: str
    config: dict[str, Any]
    workload: dict[str, Any]
    environment: dict[str, Any]
    samples: int
    inner_loops: int
    total_timed_seconds: float
    latency_ms: dict[str, float]
    variability: dict[str, float | bool]
    throughput_items_per_second: float
    cpu: dict[str, float]
    memory: dict[str, int | float | str | None]
    estimates: dict[str, float | str | None] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _robust_cv(values: list[float]) -> float:
    median = statistics.median(values)
    if median == 0:
        return math.inf
    mad = statistics.median(abs(value - median) for value in values)
    return 1.4826 * mad / median


def _environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    if psutil is not None:
        payload["physical_cpus"] = psutil.cpu_count(logical=False)
    try:
        import torch

        payload.update({
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        })
    except ImportError:
        payload["torch"] = None
    return payload


def _calibrate_inner_loops(workload: BenchmarkWorkload, target_seconds: float) -> int:
    loops = 1
    while loops < 1_000_000:
        started = time.perf_counter_ns()
        for _ in range(loops):
            workload.run()
        workload.synchronize()
        elapsed = (time.perf_counter_ns() - started) / 1e9
        if elapsed >= target_seconds:
            return loops
        if elapsed <= 0:
            loops *= 10
        else:
            loops = min(1_000_000, max(loops + 1, math.ceil(loops * target_seconds / elapsed)))
    return loops


def _measure_memory(workload: BenchmarkWorkload, duration: float, interval: float) -> dict[str, int | float | str | None]:
    if psutil is None:
        return {"method": "unavailable; install psutil", "rss_before_bytes": None, "peak_rss_bytes": None, "rss_delta_bytes": None}
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, process.memory_info().rss)
            stop.wait(interval)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    calls = 0
    try:
        while time.perf_counter() - started < duration:
            workload.run()
            calls += 1
        workload.synchronize()
    finally:
        stop.set()
        sampler.join(timeout=1.0)
    peak = max(peak, process.memory_info().rss)
    return {
        "method": "separate-phase process RSS sampling",
        "rss_before_bytes": baseline,
        "peak_rss_bytes": peak,
        "rss_delta_bytes": max(0, peak - baseline),
        "sampling_interval_ms": interval * 1_000,
        "profiled_calls": calls,
    }


def run_benchmark(workload: BenchmarkWorkload, config: BenchmarkConfig | None = None) -> BenchmarkResult:
    config = config or BenchmarkConfig()
    for _ in range(config.warmup_calls):
        workload.run()
    workload.synchronize()

    inner_loops = _calibrate_inner_loops(workload, config.target_sample_ms / 1_000)
    samples_ms: list[float] = []
    total_wall = 0.0
    total_cpu = 0.0
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        while len(samples_ms) < config.max_samples:
            cpu_started = time.process_time()
            started = time.perf_counter_ns()
            for _ in range(inner_loops):
                workload.run()
            workload.synchronize()
            wall = (time.perf_counter_ns() - started) / 1e9
            cpu = time.process_time() - cpu_started
            total_wall += wall
            total_cpu += cpu
            samples_ms.append(wall * 1_000 / inner_loops)
            enough = len(samples_ms) >= config.min_samples and total_wall >= config.min_total_seconds
            if enough and _robust_cv(samples_ms) <= config.target_robust_cv:
                break
    finally:
        if gc_was_enabled:
            gc.enable()

    mean_ms = statistics.fmean(samples_ms)
    median_ms = statistics.median(samples_ms)
    standard_deviation = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    robust_cv = _robust_cv(samples_ms)
    items = workload.work_items_per_call()
    flops = workload.estimated_flops_per_call()
    bytes_moved = workload.estimated_bytes_per_call()
    estimates: dict[str, float | str | None] = {
        "flops_per_call": flops,
        "bytes_per_call": bytes_moved,
        "arithmetic_intensity_flops_per_byte": flops / bytes_moved if flops and bytes_moved else None,
        "achieved_gflops_per_second": flops / (median_ms / 1_000) / 1e9 if flops else None,
        "note": "FLOPs and bytes are workload estimates, not hardware counters.",
    }
    return BenchmarkResult(
        name=workload.name,
        config=asdict(config),
        workload=workload.metadata(),
        environment=_environment(),
        samples=len(samples_ms),
        inner_loops=inner_loops,
        total_timed_seconds=total_wall,
        latency_ms={
            "mean": mean_ms,
            "median": median_ms,
            "p90": _percentile(samples_ms, 0.90),
            "p99": _percentile(samples_ms, 0.99),
            "minimum": min(samples_ms),
            "maximum": max(samples_ms),
        },
        variability={
            "standard_deviation_ms": standard_deviation,
            "coefficient_of_variation": standard_deviation / mean_ms if mean_ms else math.inf,
            "robust_coefficient_of_variation": robust_cv,
            "stable": robust_cv <= config.target_robust_cv,
        },
        throughput_items_per_second=items / (median_ms / 1_000),
        cpu={
            "process_cpu_seconds": total_cpu,
            "wall_seconds": total_wall,
            "average_core_equivalents": total_cpu / total_wall if total_wall else 0.0,
        },
        memory=_measure_memory(
            workload,
            config.memory_seconds,
            config.memory_sample_interval_ms / 1_000,
        ),
        estimates=estimates,
    )
