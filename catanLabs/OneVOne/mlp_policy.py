"""Five-head MLP policy and Catanatron player adapter."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from catanLabs.Catanatron.engine import source_info  # noqa: E402

ENGINE = source_info()
sys.path.insert(0, ENGINE["python_root"])

from catanatron.models.player import Player  # noqa: E402
from catanLabs.OneVOne.policy_features import (  # noqa: E402
    FEATURES_VERSION,
    encode_legal_actions,
    encode_state,
)

POLICY_ARCHITECTURE = "five-family-legal-action-mlp-v2"


class FiveHeadPolicy(nn.Module):
    """Shared state/action representation with one score head per action family."""

    def __init__(self, state_dim: int, action_dim: int, width: int = 256,
                 action_width: int = 128):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.width = int(width)
        self.action_width = int(action_width)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.width),
            nn.LayerNorm(self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
            nn.GELU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.action_width),
            nn.LayerNorm(self.action_width),
            nn.GELU(),
            nn.Linear(self.action_width, self.action_width),
            nn.GELU(),
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(self.width + self.action_width, self.width),
            nn.LayerNorm(self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width // 2),
            nn.GELU(),
        )
        self.policy_heads = nn.ModuleList(
            nn.Linear(self.width // 2, 1) for _ in range(5)
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.width, self.width // 2),
            nn.GELU(),
            nn.Linear(self.width // 2, 1),
            nn.Tanh(),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor,
                families: torch.Tensor,
                mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score legal actions and aggregate their scores into five family scores."""
        state_embedding = self.state_encoder(states)
        action_embedding = self.action_encoder(actions)
        expanded_state = state_embedding.unsqueeze(1).expand(
            -1, action_embedding.shape[1], -1
        )
        joint = self.joint_encoder(torch.cat((expanded_state, action_embedding), dim=-1))
        all_scores = torch.stack(
            [head(joint).squeeze(-1) for head in self.policy_heads], dim=-1
        )
        logits = all_scores.gather(-1, families.long().unsqueeze(-1)).squeeze(-1)
        if mask is None:
            mask = torch.ones_like(families, dtype=torch.bool)
        family_scores = []
        for family_id in range(5):
            family_candidates = mask & families.eq(family_id)
            family_scores.append(
                logits.masked_fill(~family_candidates, -1e9).amax(dim=1)
            )
        family_logits = torch.stack(family_scores, dim=1)
        value = self.value_head(state_embedding).squeeze(-1)
        return logits, family_logits, value


class MLPPolicyAgent(Player):
    """Use a trained five-head policy to rank the current legal action list."""

    def __init__(self, color: Any, model: FiveHeadPolicy):
        Player.__init__(self, color)
        self.model = model.cpu().eval()

    @torch.inference_mode()
    def decide(self, game: Any, playable_actions: list[Any]) -> Any:
        if not playable_actions:
            raise ValueError("Catanatron supplied no legal actions")
        if len(playable_actions) == 1:
            return playable_actions[0]
        state = torch.from_numpy(encode_state(game, self.color)).unsqueeze(0)
        action_features, families = encode_legal_actions(
            playable_actions, game.state.board.map
        )
        actions = torch.from_numpy(action_features).unsqueeze(0)
        family_ids = torch.from_numpy(families).unsqueeze(0)
        mask = torch.ones_like(family_ids, dtype=torch.bool)
        logits, family_logits, _ = self.model(state, actions, family_ids, mask)
        selected_family = int(torch.argmax(family_logits[0]).item())
        family_mask = family_ids[0].eq(selected_family)
        selected = logits[0].masked_fill(~family_mask, -1e9)
        return playable_actions[int(torch.argmax(selected).item())]


def load_policy(path: Path) -> FiveHeadPolicy:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture_version") != POLICY_ARCHITECTURE:
        raise ValueError("Checkpoint does not use the hierarchical five-head policy architecture")
    if checkpoint.get("features_version") != FEATURES_VERSION:
        raise ValueError(
            f"Checkpoint feature schema {checkpoint.get('features_version')!r} "
            f"does not match {FEATURES_VERSION!r}"
        )
    model = FiveHeadPolicy(
        checkpoint["state_dim"], checkpoint["action_dim"],
        checkpoint["width"], checkpoint["action_width"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def create_agent(color: Any, checkpoint: str | None = None, seed: int | None = None):
    """Factory for evaluate.py; checkpoint is a policy-run JSON config."""
    del seed
    if checkpoint is None:
        raise ValueError("MLP policy requires a JSON config with policy_checkpoint")
    config_path = Path(checkpoint).resolve(strict=True)
    config = json.loads(config_path.read_text())
    relative_path = Path(config["policy_checkpoint"])
    model_path = relative_path if relative_path.is_absolute() else config_path.parent / relative_path
    return MLPPolicyAgent(color, load_policy(model_path.resolve(strict=True)))
