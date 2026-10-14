"""Repeatable CPU inference benchmark for the exact Transformer model."""

from __future__ import annotations

import resource
import sys
import time
from typing import Any

import torch

from ..config import MODEL_CONFIG
from ..evaluation.metrics import summarize_latencies
from ..model import TicTacToePolicy


def estimated_forward_flops(batch_size: int = 1) -> int:
    """Approximate matmul FLOPs; excludes norms, softmax, GELU, and embeddings."""
    config = MODEL_CONFIG
    tokens = config.sequence_length
    width = config.d_model
    hidden = config.mlp_hidden
    per_block = (
        2 * tokens * width * (3 * width)
        + 2 * tokens * tokens * width
        + 2 * tokens * tokens * width
        + 2 * tokens * width * width
        + 2 * tokens * width * hidden
        + 2 * tokens * hidden * width
    )
    policy_head = 2 * 9 * width
    return batch_size * (config.n_blocks * per_block + policy_head)


def peak_rss_mib() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024 * 1024 if sys.platform == "darwin" else 1024)


@torch.inference_mode()
def profile_model(
    model: TicTacToePolicy,
    *,
    batch_size: int = 1,
    warmup: int = 50,
    iterations: int = 500,
) -> dict[str, Any]:
    model.eval()
    sample = torch.tensor([[1] * 9 + [4]], dtype=torch.long).repeat(batch_size, 1)
    for _ in range(warmup):
        model(sample)

    latencies: list[float] = []
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for _ in range(iterations):
        started = time.perf_counter()
        model(sample)
        latencies.append((time.perf_counter() - started) * 1_000)
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start

    total_examples = batch_size * iterations
    flops_per_example = estimated_forward_flops()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    return {
        "batch_size": batch_size,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "torch_threads": torch.get_num_threads(),
        "parameter_count": parameter_count,
        "parameter_memory_mib": parameter_bytes / (1024 * 1024),
        "estimated_flops_per_example": flops_per_example,
        "latency": summarize_latencies(latencies),
        "examples_per_second": total_examples / wall_seconds,
        "estimated_gflops_per_second": (
            total_examples * flops_per_example / wall_seconds / 1e9
        ),
        "process_cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "average_cpu_percent": 100 * cpu_seconds / wall_seconds,
        "average_cores": cpu_seconds / wall_seconds,
        "peak_rss_mib": peak_rss_mib(),
        "flops_note": "matmul estimate only; Python, norms, softmax, GELU and masking are excluded",
    }
