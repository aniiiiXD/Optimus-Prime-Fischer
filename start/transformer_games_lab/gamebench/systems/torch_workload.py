"""Pluggable forward/backward/train-step workloads for the tiny Transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch.nn import functional as F

from gamebench.models.tiny_transformer import TinyGameTransformer, TinyTransformerConfig


Mode = Literal["forward", "forward_backward", "train_step"]


@dataclass(frozen=True)
class TransformerWorkloadConfig:
    model: TinyTransformerConfig
    batch_size: int = 1
    sequence_length: int = 8
    mode: Mode = "forward"
    dtype: str = "float32"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.sequence_length > self.model.context_length:
            raise ValueError("sequence_length exceeds model context_length")
        if self.batch_size <= 0 or self.sequence_length <= 0:
            raise ValueError("batch_size and sequence_length must be positive")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("CPU harness supports float32 or bfloat16")


class TransformerWorkload:
    def __init__(self, config: TransformerWorkloadConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self.model = TinyGameTransformer(config.model).cpu()
        if config.dtype == "bfloat16":
            self.model = self.model.to(torch.bfloat16)
        self.tokens = torch.randint(
            2,
            config.model.vocab_size,
            (config.batch_size, config.sequence_length),
            dtype=torch.long,
        )
        self.targets = torch.randint(0, config.model.action_size, (config.batch_size,), dtype=torch.long)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3) if config.mode == "train_step" else None
        self.model.train(config.mode != "forward")
        self.name = f"tiny_transformer/{config.mode}/b{config.batch_size}/t{config.sequence_length}"
        self._last: torch.Tensor | None = None

    def run(self) -> None:
        if self.config.mode == "forward":
            with torch.inference_mode():
                self._last = self.model(self.tokens)
            return
        self.model.zero_grad(set_to_none=True)
        logits = self.model(self.tokens)
        loss = F.cross_entropy(logits.float(), self.targets)
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
        self._last = loss.detach()

    def synchronize(self) -> None:
        # CPU kernels are synchronous from the caller's perspective.
        pass

    def work_items_per_call(self) -> int:
        return self.config.batch_size

    def _forward_flops(self) -> float:
        cfg = self.config.model
        batch, time, width = self.config.batch_size, self.config.sequence_length, cfg.d_model
        per_layer = (
            8 * batch * time * width * width
            + 4 * batch * time * width * cfg.d_ff
            + 4 * batch * time * time * width
        )
        # This policy model projects only the final real sequence position.
        output = 2 * batch * width * cfg.action_size
        return cfg.n_layers * per_layer + output

    def estimated_flops_per_call(self) -> float:
        forward = self._forward_flops()
        # Backprop through dense operators is approximately twice the forward
        # work. Optimizer arithmetic is intentionally not included.
        return forward if self.config.mode == "forward" else 3 * forward

    def estimated_bytes_per_call(self) -> float:
        element_bytes = 4 if self.config.dtype == "float32" else 2
        return float(self.model.parameter_count * element_bytes)

    def metadata(self) -> dict[str, Any]:
        return {
            **asdict(self.config),
            "model": self.config.model.as_dict(),
            "parameters": self.model.parameter_count,
            "parameter_bytes": self.estimated_bytes_per_call(),
            "bytes_estimate_kind": "one streaming pass over model weights",
            "flops_estimate_kind": "dense multiply-add analytical estimate",
        }
