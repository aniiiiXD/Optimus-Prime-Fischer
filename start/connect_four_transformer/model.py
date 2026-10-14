"""Small board encoder with shared legal-column policy and value heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import DEFAULT_CONFIG, ModelConfig


@dataclass
class ModelOutput:
    policy_logits: torch.Tensor
    action_values: torch.Tensor
    state_value: torch.Tensor


class EncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class ConnectFourTransformer(nn.Module):
    """Encode a board once and reuse one scorer for every legal column."""

    def __init__(self, config: ModelConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.cell_embedding = nn.Embedding(3, config.d_model)
        self.row_embedding = nn.Embedding(6, config.d_model)
        self.column_embedding = nn.Embedding(7, config.d_model)
        self.state_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.state_slot_embedding = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.n_blocks))
        self.final_norm = nn.LayerNorm(config.d_model)

        self.action_embedding = nn.Embedding(8, config.d_model, padding_idx=7)
        self.landing_row_embedding = nn.Embedding(7, config.d_model, padding_idx=6)
        self.candidate_fusion = nn.Sequential(
            nn.Linear(4 * config.d_model, config.mlp_hidden),
            nn.GELU(),
            nn.Linear(config.mlp_hidden, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.policy_head = nn.Linear(config.d_model, 1)
        self.q_head = nn.Sequential(nn.Linear(config.d_model, 1), nn.Tanh())
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
            nn.Tanh(),
        )

        square_ids = torch.arange(42)
        self.register_buffer("row_ids", square_ids // 7, persistent=False)
        self.register_buffer("column_ids", square_ids % 7, persistent=False)

    def forward(
        self,
        board_tokens: torch.Tensor,
        candidate_ids: torch.Tensor,
        landing_rows: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> ModelOutput:
        if board_tokens.ndim != 2 or board_tokens.shape[1] != 42:
            raise ValueError("board_tokens must have shape (batch, 42)")
        if candidate_ids.shape != landing_rows.shape or candidate_ids.shape != candidate_mask.shape:
            raise ValueError("candidate tensors must have matching shapes")

        cells = (
            self.cell_embedding(board_tokens)
            + self.row_embedding(self.row_ids).unsqueeze(0)
            + self.column_embedding(self.column_ids).unsqueeze(0)
        )
        state = self.state_token.expand(board_tokens.shape[0], -1, -1) + self.state_slot_embedding
        x = torch.cat((cells, state), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        board_features = x[:, :42].reshape(-1, 6, 7, self.config.d_model)
        state_features = x[:, 42]

        safe_columns = candidate_ids.clamp_max(6)
        safe_rows = landing_rows.clamp_max(5)
        batch_ids = torch.arange(board_tokens.shape[0], device=board_tokens.device).unsqueeze(1)
        landing_features = board_features[batch_ids, safe_rows, safe_columns]
        column_features = board_features.mean(dim=1).gather(
            1, safe_columns.unsqueeze(-1).expand(-1, -1, self.config.d_model)
        )
        action_features = (
            self.action_embedding(candidate_ids)
            + self.landing_row_embedding(landing_rows)
        )
        global_features = state_features.unsqueeze(1).expand(-1, candidate_ids.shape[1], -1)
        candidates = self.candidate_fusion(
            torch.cat(
                (global_features, column_features, landing_features, action_features), dim=-1
            )
        )
        candidates = candidates * candidate_mask.unsqueeze(-1)
        minimum = torch.finfo(candidates.dtype).min
        policy_logits = self.policy_head(candidates).squeeze(-1).masked_fill(~candidate_mask, minimum)
        action_values = self.q_head(candidates).squeeze(-1).masked_fill(~candidate_mask, 0.0)
        return ModelOutput(
            policy_logits=policy_logits,
            action_values=action_values,
            state_value=self.value_head(state_features).squeeze(-1),
        )


__all__ = ["ConnectFourTransformer", "ModelOutput"]
