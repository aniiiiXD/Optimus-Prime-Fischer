"""A state Transformer with an explicit, reusable candidate-action interface."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tic_tac_toe_transformer.blocks import TicTacToeTransformerBlocks
from tic_tac_toe_transformer.embeddings import TicTacToeEmbeddings

from .config import DEFAULT_CONFIG, SharedTransformerConfig


@dataclass
class SharedTransformerOutput:
    """All decisions produced by one encoded state."""

    policy_logits: torch.Tensor
    action_value_logits: torch.Tensor
    state_value_logits: torch.Tensor
    legality_logits: torch.Tensor


class SharedActionTransformer(nn.Module):
    """Encode the board once, then apply shared heads to any candidate moves.

    Candidate IDs use zero-based square indices 0..8. ``candidate_mask`` marks
    real entries when a variable-length list has been padded. It does not mean
    that a move is legal; game rules remain the source of legal actions.
    """

    def __init__(self, config: SharedTransformerConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config
        encoder_config = config.state_encoder_config()
        self.embeddings = TicTacToeEmbeddings(config.d_model)
        self.transformer = TicTacToeTransformerBlocks(encoder_config)

        self.action_embedding = nn.Embedding(
            config.action_count + 1,
            config.d_model,
            padding_idx=config.action_count,
        )
        self.action_row_embedding = nn.Embedding(3, config.d_model)
        self.action_column_embedding = nn.Embedding(3, config.d_model)
        self.candidate_fusion = nn.Sequential(
            nn.Linear(3 * config.d_model, config.mlp_hidden),
            nn.GELU(),
            nn.Linear(config.mlp_hidden, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        self.policy_head = nn.Linear(config.d_model, 1)
        self.action_value_head = nn.Linear(config.d_model, config.outcome_count)
        self.legality_head = nn.Linear(config.d_model, 1)
        self.state_value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.outcome_count),
        )

    def forward(
        self,
        state_tokens: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> SharedTransformerOutput:
        """Score candidates shaped ``(batch, candidates)`` in one pass."""
        if candidate_ids.ndim != 2 or candidate_ids.dtype != torch.long:
            raise TypeError("candidate_ids must be a rank-2 torch.long tensor")
        if state_tokens.shape[0] != candidate_ids.shape[0]:
            raise ValueError("state and candidate batches must have equal size")
        if candidate_mask is None:
            candidate_mask = candidate_ids != self.config.action_count
        if candidate_mask.shape != candidate_ids.shape or candidate_mask.dtype != torch.bool:
            raise TypeError("candidate_mask must be bool with candidate_ids shape")
        if torch.any(candidate_ids < 0) or torch.any(candidate_ids > self.config.action_count):
            raise ValueError("candidate IDs must be in 0..9, with 9 reserved for padding")

        contextual = self.transformer(self.embeddings(state_tokens))
        square_states = contextual[:, : self.config.action_count, :]
        state_summary = contextual[:, -1, :]

        safe_ids = candidate_ids.clamp_max(self.config.action_count - 1)
        gathered_squares = square_states.gather(
            1, safe_ids.unsqueeze(-1).expand(-1, -1, self.config.d_model)
        )
        action_features = (
            self.action_embedding(candidate_ids)
            + self.action_row_embedding(safe_ids // 3)
            + self.action_column_embedding(safe_ids % 3)
        )
        global_features = state_summary.unsqueeze(1).expand(-1, candidate_ids.shape[1], -1)
        candidate_features = self.candidate_fusion(
            torch.cat((gathered_squares, global_features, action_features), dim=-1)
        )
        candidate_features = candidate_features * candidate_mask.unsqueeze(-1)

        policy_logits = self.policy_head(candidate_features).squeeze(-1)
        legality_logits = self.legality_head(candidate_features).squeeze(-1)
        padding_value = torch.finfo(policy_logits.dtype).min
        policy_logits = policy_logits.masked_fill(~candidate_mask, padding_value)
        legality_logits = legality_logits.masked_fill(~candidate_mask, padding_value)

        return SharedTransformerOutput(
            policy_logits=policy_logits,
            action_value_logits=self.action_value_head(candidate_features),
            state_value_logits=self.state_value_head(state_summary),
            legality_logits=legality_logits,
        )


__all__ = ["SharedActionTransformer", "SharedTransformerOutput"]
