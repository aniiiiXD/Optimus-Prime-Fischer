"""Explicit replay adapter for the frozen v2 Transformer checkpoint.

V3 changed only two previously unused player-token columns. Zeroing those
columns reconstructs the exact v2 input without weakening the v3 loader's
checkpoint guard. This adapter is for historical comparisons, not new training.
"""

from __future__ import annotations

from catanLabs.Transformer.agent import TransformerAgent
from catanLabs.Transformer.model import encode_visible_observation
from catanLabs.Transformer.visible_observation import VisibleAgentAdapter, VisibleObservation


class LegacyV2TransformerAgent(TransformerAgent):
    checkpoint_schema = "catanlabs-action-transformer-v2"
    feature_schema = "visible-state-tokens-v2"

    def _encode(self, observation: VisibleObservation):
        encoded = encode_visible_observation(observation)
        encoded[0][1:5, 32:34] = 0.0
        return encoded


def create_agent(color, checkpoint, seed):
    del seed
    if checkpoint is None:
        raise ValueError("Legacy v2 Transformer requires --checkpoint")
    return VisibleAgentAdapter(LegacyV2TransformerAgent(color, checkpoint))
