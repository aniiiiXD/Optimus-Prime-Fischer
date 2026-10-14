"""Isolated GEMM workload for studying CPU matrix throughput."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MatrixWorkloadConfig:
    m: int = 512
    k: int = 512
    n: int = 512
    dtype: str = "float32"
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.m, self.k, self.n) <= 0:
            raise ValueError("matrix dimensions must be positive")
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("dtype must be float32, bfloat16, or float16")


class MatrixMultiplyWorkload:
    def __init__(self, config: MatrixWorkloadConfig) -> None:
        self.config = config
        dtype = getattr(torch, config.dtype)
        generator = torch.Generator().manual_seed(config.seed)
        self.left = torch.randn((config.m, config.k), dtype=dtype, generator=generator)
        self.right = torch.randn((config.k, config.n), dtype=dtype, generator=generator)
        self.output = torch.empty((config.m, config.n), dtype=dtype)
        self.name = f"matmul/{config.dtype}/{config.m}x{config.k}x{config.n}"

    def run(self) -> None:
        torch.mm(self.left, self.right, out=self.output)

    def synchronize(self) -> None:
        pass

    def metadata(self) -> dict[str, Any]:
        return {
            **asdict(self.config),
            "operation": "C[M,N] = A[M,K] @ B[K,N]",
            "backend_selection": "delegated to the installed PyTorch CPU build",
        }

    def work_items_per_call(self) -> int:
        return 1

    def estimated_flops_per_call(self) -> float:
        return float(2 * self.config.m * self.config.k * self.config.n)

    def estimated_bytes_per_call(self) -> float:
        element_bytes = {"float32": 4, "bfloat16": 2, "float16": 2}[self.config.dtype]
        elements = (
            self.config.m * self.config.k
            + self.config.k * self.config.n
            + self.config.m * self.config.n
        )
        return float(element_bytes * elements)
