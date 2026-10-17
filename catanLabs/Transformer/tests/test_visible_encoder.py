"""The Transformer consumes only the immutable player-visible snapshot."""

from __future__ import annotations

from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanatron import Color, Game, RandomPlayer

from catanLabs.Transformer.agent import TransformerAgent, create_agent
from catanLabs.Transformer.model import (
    CHECKPOINT_SCHEMA, FEATURE_SCHEMA, CatanActionTransformer,
    encode_observation, encode_visible_observation,
)
from catanLabs.Transformer.visible_observation import (
    DEVELOPMENT_CARDS, RESOURCES, VisibleAgentAdapter, snapshot,
)


def _game(seed=42):
    return Game([RandomPlayer(color) for color in Color], seed=seed,
                vps_to_win=10, discard_limit=7, friendly_robber=False,
                number_placement="official_spiral")


def _change_hidden(game):
    clone = game.copy()
    actor = clone.state.current_color()
    for color in clone.state.colors:
        if color == actor:
            continue
        prefix = f"P{clone.state.color_to_index[color]}_"
        for names in (RESOURCES, DEVELOPMENT_CARDS):
            counts = [clone.state.player_state[prefix + name + "_IN_HAND"] for name in names]
            for name, count in zip(names, counts[1:] + counts[:1]):
                clone.state.player_state[prefix + name + "_IN_HAND"] = count
    clone.state.development_listdeck.reverse()
    clone.state.random = random.Random(987654)
    clone.random = clone.state.random
    return clone


class VisibleEncoderTests(unittest.TestCase):
    def test_wrapper_matches_snapshot_encoder_across_turns(self):
        game = _game()
        phases = set()
        for _ in range(24):
            legal = tuple(game.playable_actions)
            observation = snapshot(game, legal)
            phases.add(observation.phase)
            direct = encode_visible_observation(observation)
            wrapped = encode_observation(game, list(legal))
            for a, b in zip(direct, wrapped):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(direct[0].shape[0], 150)
            game.execute(legal[0], validate_action=True)
        self.assertTrue({"BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD", "PLAY_TURN"} <= phases)

    def test_opponent_private_cards_and_deck_do_not_change_encoding(self):
        game = _game(77)
        for _ in range(12):
            legal = tuple(game.playable_actions)
            original = encode_visible_observation(snapshot(game, legal))
            perturbed = encode_visible_observation(snapshot(_change_hidden(game), legal))
            for a, b in zip(original, perturbed):
                np.testing.assert_array_equal(a, b)
            game.execute(legal[0], validate_action=True)

    def test_agent_factory_uses_visible_adapter_and_returns_legal_action(self):
        game = _game(91)
        model = CatanActionTransformer(width=16, layers=1, heads=4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "v2.pt"
            torch.save({
                "schema": CHECKPOINT_SCHEMA,
                "feature_schema": FEATURE_SCHEMA,
                "model_config": {"width": 16, "layers": 1, "heads": 4},
                "model_state": model.state_dict(),
            }, path)
            agent = create_agent(game.state.current_color(), path, seed=0)
            self.assertIsInstance(agent, VisibleAgentAdapter)
            self.assertIsInstance(agent.agent, TransformerAgent)
            legal = tuple(game.playable_actions)
            observation = snapshot(game, legal)
            self.assertEqual(agent.agent.score_visible(observation),
                             agent.agent.score_actions(game, legal))
            self.assertIn(agent.decide(game, legal), legal)


if __name__ == "__main__":
    unittest.main()
