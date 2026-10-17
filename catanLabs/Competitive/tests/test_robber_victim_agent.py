"""The victim tie-break must preserve the tile and ignore hidden state."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanatron import Color, Game, RandomPlayer
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import ActionPrompt

from catanLabs.Catanatron.tournament_protocol import install_v5_protocol
from catanLabs.Competitive.check_visibility import perturbed_hidden
from catanLabs.Competitive.fair_value_agent import FairValueAgent
from catanLabs.Competitive.robber_victim_agent import RobberVictimAgent


class RobberVictimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_v5_protocol()
        cls.config = json.loads((WORKSPACE / "catanLabs/Competitive/configs/robber_victim_research.json").read_text())

    def _multi_victim_position(self):
        game = Game([RandomPlayer(color) for color in Color], seed=1,
                    vps_to_win=10, discard_limit=7, friendly_robber=False,
                    number_placement="official_spiral")
        for _ in range(16):
            if game.state.current_prompt.name == "PLAY_TURN":
                break
            game.execute(game.playable_actions[0], validate_action=True)
        actor = game.state.current_color()
        game.state.current_prompt = ActionPrompt.MOVE_ROBBER
        for color in game.state.colors:
            if color != actor:
                prefix = f"P{game.state.color_to_index[color]}_"
                game.state.player_state[prefix + "WOOD_IN_HAND"] = 1
        game.playable_actions = generate_playable_actions(game.state)
        legal = tuple(game.playable_actions)
        for action in legal:
            if action.value[1] is None:
                continue
            others = [other for other in legal if other.value[0] == action.value[0]
                      and other.value[1] is not None and other != action]
            if others:
                return game, actor, legal, action, others[0]
        self.fail("Seeded position has no tile with two legal victims")

    def test_higher_public_vp_wins_only_same_tile_tie(self):
        game, actor, legal, lower, higher = self._multi_victim_position()
        state = game.state
        low_key = f"P{state.color_to_index[lower.value[1]]}_VICTORY_POINTS"
        high_key = f"P{state.color_to_index[higher.value[1]]}_VICTORY_POINTS"
        state.player_state[low_key] = 2
        state.player_state[high_key] = 8
        candidate = RobberVictimAgent(actor, self.config)
        self.assertEqual(candidate.find_override(game, legal, lower), higher)
        with patch.object(FairValueAgent, "decide", return_value=lower):
            selected = candidate.decide(game, legal)
        self.assertEqual(selected, higher)
        self.assertIn(selected, legal)
        self.assertEqual(selected.value[0], lower.value[0])
        state.player_state[high_key] = 2
        self.assertIsNone(candidate.find_override(game, legal, lower))
        different_tile = next(action for action in legal
                              if action.value[0] != lower.value[0])
        self.assertIsNone(candidate.find_override(game, (different_tile,), different_tile))
        game.execute(selected, validate_action=True)
        self.assertEqual(game.state.board.robber_coordinate, selected.value[0])

    def test_hidden_hands_deck_rng_invariance(self):
        game, actor, legal, lower, higher = self._multi_victim_position()
        state = game.state
        state.player_state[f"P{state.color_to_index[lower.value[1]]}_VICTORY_POINTS"] = 2
        state.player_state[f"P{state.color_to_index[higher.value[1]]}_VICTORY_POINTS"] = 8
        candidate = RobberVictimAgent(actor, self.config)
        before = state.random.getstate()
        first = candidate.find_override(game, legal, lower)
        changed = perturbed_hidden(game, actor)
        second = candidate.find_override(changed, tuple(changed.playable_actions), lower)
        self.assertEqual(first, second)
        self.assertEqual(state.random.getstate(), before)
        with patch.object(FairValueAgent, "decide", return_value=lower):
            self.assertEqual(candidate.decide(game, legal),
                             candidate.decide(changed, tuple(changed.playable_actions)))


if __name__ == "__main__":
    unittest.main()
