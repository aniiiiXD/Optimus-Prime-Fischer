"""Small legal-action Transformer distilled from OneVOne's visible decisions.

History contains only the agent's own earlier visible observations. It is
stored per agent instance and reset between games; no hidden engine fields enter.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from catanLabs.OneVOne.policy_features import encode_legal_actions, encode_state


class MemoryPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, width: int = 64,
                 history_len: int = 4):
        super().__init__()
        self.history_len = history_len
        self.state_proj = nn.Sequential(nn.Linear(state_dim, width), nn.LayerNorm(width))
        self.position = nn.Parameter(torch.zeros(history_len + 1, width))
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, 0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.action_proj = nn.Linear(action_dim, width)
        self.score = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(),
                                   nn.Linear(width, 1))

    def forward(self, states: torch.Tensor, history_mask: torch.Tensor,
                actions: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        # states: [batch, history_len + current, state_dim]. Left padded.
        x = self.state_proj(states) + self.position[None, :states.shape[1]]
        # Causal mask prevents earlier tokens from seeing future tokens.
        length = states.shape[1]
        causal = torch.triu(torch.ones(length, length, device=states.device,
                                       dtype=torch.bool), diagonal=1)
        # Left padding is represented by zero observations. A key-padding mask
        # combined with a causal mask would leave the earliest padded queries
        # with no valid keys; PyTorch then produces NaNs that contaminate later
        # tokens. Keeping zero padding as ordinary tokens is safe and finite.
        x = self.encoder(x, mask=causal)
        context = x[:, -1:, :].expand(-1, actions.shape[1], -1)
        logits = self.score(torch.cat((context, self.action_proj(actions)), -1)).squeeze(-1)
        return logits.masked_fill(~action_mask, -1e9)


def make_sequence(current: np.ndarray, prior: list[np.ndarray] | deque,
                  history_len: int) -> np.ndarray:
    seq = np.zeros((history_len + 1, len(current)), dtype=np.float32)
    prior = list(prior)[-history_len:]
    if prior:
        seq[history_len - len(prior):history_len] = np.stack(prior)
    seq[-1] = current
    return seq


class MemoryPolicyAgent:
    def __init__(self, color: Any, model: MemoryPolicy, use_history: bool = True):
        self.color = color
        self.model = model.cpu().eval()
        self.use_history = use_history
        self.prior: deque[np.ndarray] = deque(maxlen=model.history_len)

    @torch.inference_mode()
    def decide(self, game: Any, playable_actions: list[Any]) -> Any:
        state = encode_state(game, self.color)
        prior = self.prior if self.use_history else []
        seq = make_sequence(state, prior, self.model.history_len)
        self.prior.append(state)
        if len(playable_actions) == 1:
            return playable_actions[0]
        features, _ = encode_legal_actions(playable_actions, game.state.board.map)
        mask = np.any(seq != 0, axis=1)
        logits = self.model(torch.from_numpy(seq[None]),
                            torch.from_numpy(mask[None]),
                            torch.from_numpy(features[None]),
                            torch.ones((1, len(features)), dtype=torch.bool))
        return playable_actions[int(logits[0].argmax())]


def load_agent(color: Any, checkpoint: str | Path) -> MemoryPolicyAgent:
    item = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = MemoryPolicy(item["state_dim"], item["action_dim"],
                         item["width"], item["history_len"])
    model.load_state_dict(item["model"])
    return MemoryPolicyAgent(color, model, item["use_history"])
