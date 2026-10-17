"""Action-conditioned MLP baseline for the same v2 Catan tensors."""

from __future__ import annotations

import torch
from torch import nn

from catanLabs.Transformer.model import (
    ACTION_FEATURE_DIM, ACTION_TYPES, FEATURE_DIM, GROUP_NONE, KIND_ACTION,
    STATE_TOKEN_COUNT,
)


CHECKPOINT_SCHEMA = "catanlabs-action-mlp-v1"


class CatanActionMLP(nn.Module):
    """Score each legal action from pooled state and its target entity.

    The shared token encoder has no attention or sequential recurrence. An
    action receives its own features, the global state summary, and the token
    for its tile/node/edge target. It uses exactly the Transformer input API.
    """

    def __init__(self, width: int = 32):
        super().__init__()
        self.width = width
        self.state_projection = nn.Linear(FEATURE_DIM, width)
        self.action_projection = nn.Linear(ACTION_FEATURE_DIM, width)
        self.kind_embedding = nn.Embedding(6, width)
        self.entity_embedding = nn.Embedding(73, width)
        self.action_type_embedding = nn.Embedding(len(ACTION_TYPES), width)
        self.target_group_embedding = nn.Embedding(4, width)
        self.state_norm = nn.LayerNorm(width)
        self.action_norm = nn.LayerNorm(width)
        self.token_mlp = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(width * 3, width * 2), nn.GELU(),
            nn.Linear(width * 2, 1),
        )

    def forward(
        self,
        state_features: torch.Tensor,
        state_kinds: torch.Tensor,
        state_ids: torch.Tensor,
        action_features: torch.Tensor,
        action_types: torch.Tensor,
        target_groups: torch.Tensor,
        target_ids: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        if state_features.shape[1] != STATE_TOKEN_COUNT:
            raise ValueError(f"Expected {STATE_TOKEN_COUNT} state tokens")
        state = self.state_projection(state_features)
        state = state + self.kind_embedding(state_kinds) + self.entity_embedding(state_ids)
        state = self.token_mlp(self.state_norm(state))
        context = (state[:, :1] + state.mean(dim=1, keepdim=True)) / 2.0
        context = context.expand(-1, action_features.shape[1], -1)

        action = self.action_projection(action_features)
        action = (action + self.kind_embedding.weight[KIND_ACTION]
                  + self.action_type_embedding(action_types)
                  + self.target_group_embedding(target_groups)
                  + self.entity_embedding(target_ids))
        action = self.action_norm(action)

        # Fixed v2 token layout: global 0, players 1..4, tiles 5..23,
        # nodes 24..77, edges 78..149. Group 0 has no target token.
        offsets = target_groups.new_tensor((0, 5, 24, 78))
        positions = offsets[target_groups] + target_ids
        if torch.any((positions < 0) | (positions >= STATE_TOKEN_COUNT)):
            raise ValueError("Action target ID is outside the v2 board tokens")
        targets = state.gather(1, positions.unsqueeze(-1).expand(-1, -1, self.width))
        targets = targets * (target_groups != GROUP_NONE).unsqueeze(-1)
        logits = self.action_head(torch.cat((context, action, targets), dim=-1)).squeeze(-1)
        return logits.masked_fill(~action_mask, -1e9)


def parameter_count(width: int = 32) -> int:
    return sum(parameter.numel() for parameter in CatanActionMLP(width).parameters())
