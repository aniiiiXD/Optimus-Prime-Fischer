"""Benchmark adapter for the trained legal-action Transformer."""

from __future__ import annotations

import torch

from catanLabs.Transformer.model import (
    CHECKPOINT_SCHEMA,
    FEATURE_SCHEMA,
    CatanActionTransformer,
    encode_visible_observation,
)
from catanLabs.Transformer.visible_observation import (
    VisibleAgentAdapter,
    VisibleObservation,
    snapshot,
)


class TransformerAgent:
    checkpoint_schema = CHECKPOINT_SCHEMA
    feature_schema = FEATURE_SCHEMA

    def __init__(self, color, checkpoint_path):
        self.color = color
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != self.checkpoint_schema or checkpoint.get("feature_schema") != self.feature_schema:
            raise ValueError(
                "Incompatible Catan Transformer checkpoint or feature schema; "
                f"expected {self.checkpoint_schema} / {self.feature_schema}."
            )
        self.model = CatanActionTransformer(**checkpoint["model_config"]).cpu()
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self._state_kinds = None
        self._state_ids = None

    def _encode(self, observation: VisibleObservation):
        return encode_visible_observation(observation)

    def score_visible(self, observation: VisibleObservation):
        """Return one logit per action in a frozen visible observation."""
        state_features, kinds, ids, action_features, action_types, groups, target_ids = self._encode(observation)
        with torch.inference_mode():
            logits = self.model(
                torch.from_numpy(state_features).unsqueeze(0),
                torch.from_numpy(kinds).unsqueeze(0),
                torch.from_numpy(ids).unsqueeze(0),
                torch.from_numpy(action_features).unsqueeze(0),
                torch.from_numpy(action_types).unsqueeze(0),
                torch.from_numpy(groups).unsqueeze(0),
                torch.from_numpy(target_ids).unsqueeze(0),
                torch.ones((1, len(observation.legal_actions)), dtype=torch.bool),
            )
        return logits.squeeze(0).tolist()

    def score_actions(self, game, playable_actions):
        """Legacy search hook; the network receives only a visible snapshot."""
        return self.score_visible(snapshot(game, tuple(playable_actions)))

    def decide_visible(self, observation: VisibleObservation) -> int:
        if len(observation.legal_actions) == 1:
            return 0
        scores = self.score_visible(observation)
        return max(range(len(scores)), key=scores.__getitem__)


def create_agent(color, checkpoint, seed):
    del seed
    if checkpoint is None:
        raise ValueError("The trained Transformer requires --checkpoint")
    return VisibleAgentAdapter(TransformerAgent(color, checkpoint))
