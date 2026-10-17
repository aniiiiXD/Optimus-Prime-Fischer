"""Small CPU-first Othello policy/Q/value Transformer.

The game adapter supplies legal moves. This module never decides legality by
itself; pass is action 64 and padding is action 65.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


SQUARES = 64
PASS_ACTION = 64
PAD_ACTION = 65


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    heads: int = 4
    blocks: int = 2
    mlp_width: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(self.width, self.heads, self.blocks, self.mlp_width) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass
class ModelOutput:
    policy_logits: torch.Tensor
    action_values: torch.Tensor
    state_value: torch.Tensor
    legality_logits: torch.Tensor


class EncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = nn.MultiheadAttention(
            config.width, config.heads, dropout=config.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, config.mlp_width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_width, config.width),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class OthelloTransformer(nn.Module):
    """Encode a 64-cell board once and score a variable action list in one pass.

    Board tokens are relative to side-to-move: 0 empty, 1 own, 2 opponent.
    `side_to_move` remains explicit, since absolute color and parity can matter.
    """

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.cell_embedding = nn.Embedding(3, width)
        self.row_embedding = nn.Embedding(8, width)
        self.col_embedding = nn.Embedding(8, width)
        self.side_embedding = nn.Embedding(2, width)
        self.state_token = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.blocks))
        self.final_norm = nn.LayerNorm(width)

        self.action_embedding = nn.Embedding(66, width, padding_idx=PAD_ACTION)
        self.candidate_fusion = nn.Sequential(
            nn.Linear(3 * width, config.mlp_width),
            nn.GELU(),
            nn.Linear(config.mlp_width, width),
            nn.LayerNorm(width),
        )
        self.policy_head = nn.Linear(width, 1)
        self.q_head = nn.Sequential(nn.Linear(width, 1), nn.Tanh())
        self.legality_head = nn.Linear(width, 1)
        self.value_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1), nn.Tanh())

        indices = torch.arange(SQUARES)
        self.register_buffer("row_ids", indices // 8, persistent=False)
        self.register_buffer("col_ids", indices % 8, persistent=False)

    def forward(
        self,
        board_tokens: torch.Tensor,
        side_to_move: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> ModelOutput:
        if board_tokens.ndim != 2 or board_tokens.shape[1] != SQUARES or board_tokens.dtype != torch.long:
            raise ValueError("board_tokens must be long [batch, 64]")
        batch = board_tokens.shape[0]
        if side_to_move.shape != (batch,) or side_to_move.dtype != torch.long:
            raise ValueError("side_to_move must be long [batch]")
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] != batch or candidate_ids.dtype != torch.long:
            raise ValueError("candidate_ids must be long [batch, candidates]")
        if candidate_mask.shape != candidate_ids.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be bool with candidate_ids shape")
        if torch.any((board_tokens < 0) | (board_tokens > 2)):
            raise ValueError("board tokens must be in 0..2")
        if torch.any((side_to_move < 0) | (side_to_move > 1)):
            raise ValueError("side_to_move must be 0 or 1")
        if torch.any((candidate_ids < 0) | (candidate_ids > PAD_ACTION)):
            raise ValueError("candidate IDs must be in 0..65")
        if torch.any(candidate_mask & (candidate_ids == PAD_ACTION)):
            raise ValueError("padding action cannot be a real candidate")

        side = self.side_embedding(side_to_move).unsqueeze(1)
        cells = (
            self.cell_embedding(board_tokens)
            + self.row_embedding(self.row_ids).unsqueeze(0)
            + self.col_embedding(self.col_ids).unsqueeze(0)
            + side
        )
        state = self.state_token.expand(batch, -1, -1) + side
        x = torch.cat((cells, state), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        state_features = x[:, SQUARES]

        safe_squares = candidate_ids.clamp_max(SQUARES - 1)
        square_features = x[:, :SQUARES].gather(
            1, safe_squares.unsqueeze(-1).expand(-1, -1, self.config.width)
        )
        # Pass and padding are not physical squares; do not give them H8's features.
        square_features = square_features * (candidate_ids < SQUARES).unsqueeze(-1)
        global_features = state_features.unsqueeze(1).expand(-1, candidate_ids.shape[1], -1)
        candidate_features = self.candidate_fusion(
            torch.cat((square_features, global_features, self.action_embedding(candidate_ids)), dim=-1)
        )
        candidate_features = candidate_features * candidate_mask.unsqueeze(-1)
        invalid = torch.finfo(candidate_features.dtype).min
        policy_logits = self.policy_head(candidate_features).squeeze(-1).masked_fill(~candidate_mask, invalid)
        legality_logits = self.legality_head(candidate_features).squeeze(-1).masked_fill(~candidate_mask, invalid)
        action_values = self.q_head(candidate_features).squeeze(-1).masked_fill(~candidate_mask, 0.0)
        state_value = self.value_head(state_features).squeeze(-1)
        return ModelOutput(policy_logits, action_values, state_value, legality_logits)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def encode_board(board: list[int], side_to_move: int) -> torch.Tensor:
    """The adapter uses -1 empty, 0 black, 1 white."""
    if len(board) != SQUARES or side_to_move not in (0, 1) or any(cell not in (-1, 0, 1) for cell in board):
        raise ValueError("invalid Othello board or side")
    return torch.tensor([0 if cell == -1 else 1 if cell == side_to_move else 2 for cell in board], dtype=torch.long)
