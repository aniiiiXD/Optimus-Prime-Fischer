"""Losses for policy-only and shared multi-task training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .model import SharedTransformerOutput


@dataclass(frozen=True)
class LossWeights:
    policy: float = 1.0
    action_value: float = 0.5
    state_value: float = 0.25
    legality: float = 0.1


def soft_policy_loss(
    logits: torch.Tensor, legal_mask: torch.Tensor, target_policy: torch.Tensor
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~legal_mask, -torch.inf)
    log_probabilities = F.log_softmax(masked_logits, dim=-1).masked_fill(~legal_mask, 0.0)
    return -(target_policy * log_probabilities).sum(dim=-1).mean()


def shared_loss(
    output: SharedTransformerOutput,
    batch: dict[str, torch.Tensor],
    weights: LossWeights = LossWeights(),
    *,
    multitask: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy = soft_policy_loss(
        output.policy_logits, batch["legal_mask"], batch["target_policy"]
    )
    if not multitask:
        return policy, {"policy": float(policy.detach())}

    legal = batch["legal_mask"]
    action_value = F.cross_entropy(
        output.action_value_logits[legal], batch["action_value_targets"][legal]
    )
    state_value = F.cross_entropy(
        output.state_value_logits, batch["state_value_target"]
    )
    legality = F.binary_cross_entropy_with_logits(
        output.legality_logits, legal.float()
    )
    total = (
        weights.policy * policy
        + weights.action_value * action_value
        + weights.state_value * state_value
        + weights.legality * legality
    )
    return total, {
        "policy": float(policy.detach()),
        "action_value": float(action_value.detach()),
        "state_value": float(state_value.detach()),
        "legality": float(legality.detach()),
    }


__all__ = ["LossWeights", "shared_loss", "soft_policy_loss"]
