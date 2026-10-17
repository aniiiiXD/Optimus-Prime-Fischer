"""Regression checks for the v3 public-count feature migration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanatron import Color, Game, RandomPlayer

from catanLabs.Transformer.agent import TransformerAgent
from catanLabs.Transformer.model import (
    CatanActionTransformer, FEATURE_SCHEMA, encode_visible_observation,
)
from catanLabs.Transformer.visible_observation import snapshot


class PublicCountsV3Tests(unittest.TestCase):
    def test_public_opponent_counts_change_only_reserved_player_slots(self):
        game = Game([RandomPlayer(color) for color in Color], seed=42,
                    vps_to_win=10, discard_limit=7, friendly_robber=False,
                    number_placement="official_spiral")
        observation = snapshot(game, tuple(game.playable_actions))
        opponent = observation.players[1]
        low = replace(opponent, resource_count=0, development_card_count=0)
        high = replace(opponent, resource_count=7, development_card_count=3)
        base = encode_visible_observation(replace(
            observation, players=(observation.players[0], low, *observation.players[2:])))
        changed = encode_visible_observation(replace(
            observation, players=(observation.players[0], high, *observation.players[2:])))

        self.assertEqual(FEATURE_SCHEMA, "visible-state-tokens-v3")
        self.assertEqual(float(base[0][2, 32]), 0.0)
        self.assertEqual(float(base[0][2, 33]), 0.0)
        self.assertAlmostEqual(float(changed[0][2, 32]), 7 / 19)
        self.assertAlmostEqual(float(changed[0][2, 33]), 3 / 25)
        state_delta = changed[0] - base[0]
        state_delta[2, 32:34] = 0
        np.testing.assert_array_equal(state_delta, np.zeros_like(state_delta))
        for old, new in zip(base[1:], changed[1:]):
            np.testing.assert_array_equal(old, new)

    def test_v2_checkpoint_rejected_before_loading_weights(self):
        model = CatanActionTransformer(width=16, layers=1, heads=4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "v2.pt"
            torch.save({
                "schema": "catanlabs-action-transformer-v2",
                "feature_schema": "visible-state-tokens-v2",
                "model_config": {"width": 16, "layers": 1, "heads": 4},
                "model_state": model.state_dict(),
            }, path)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                TransformerAgent(None, path)


if __name__ == "__main__":
    unittest.main()
