"""Road Building continuation must stay legal and insensitive to hidden state."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanatron import Color, Game, RandomPlayer
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import ActionType

from catanLabs.Catanatron.tournament_protocol import install_v5_protocol
from catanLabs.Competitive.check_visibility import perturbed_hidden
from catanLabs.Competitive.road_building_agent import RoadBuildingAgent


class RoadBuildingAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_v5_protocol()
        cls.config = json.loads((WORKSPACE / "catanLabs/Competitive/configs/road_building_research.json").read_text())

    def _position(self):
        game = Game([RandomPlayer(color) for color in Color], seed=42,
                    vps_to_win=10, discard_limit=7, friendly_robber=False,
                    number_placement="official_spiral")
        for _ in range(16):
            if game.state.current_prompt.name == "PLAY_TURN":
                break
            game.execute(game.playable_actions[0], validate_action=True)
        self.assertEqual(game.state.current_prompt.name, "PLAY_TURN")
        actor = game.state.current_color()
        prefix = f"P{game.state.color_to_index[actor]}_"
        game.state.player_state[prefix + "ROAD_BUILDING_IN_HAND"] = 1
        game.state.player_state[prefix + "ROAD_BUILDING_OWNED_AT_START"] = True
        game.state.player_state[prefix + "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = False
        game.state.player_state[prefix + "HAS_ROLLED"] = True
        game.playable_actions = generate_playable_actions(game.state)
        return game, actor

    def test_legal_free_road_continuation_and_fair_fallback(self):
        game, actor = self._position()
        legal = tuple(game.playable_actions)
        card = next(action for action in legal if action.action_type == ActionType.PLAY_ROAD_BUILDING)
        baseline = next(action for action in legal if action.action_type == ActionType.END_TURN)
        agent = RoadBuildingAgent(actor, self.config)
        rng_before = game.state.random.getstate()
        proposed = agent.find_override(game, legal, baseline)
        self.assertEqual(proposed, card)
        self.assertGreater(agent.last_scan["first_roads"], 0)
        self.assertGreater(agent.last_scan["second_roads"], 0)
        self.assertIsNotNone(agent.last_scan["best_line"])
        self.assertIsNotNone(agent.last_scan["best_line"]["second_road"])
        self.assertLessEqual(agent.last_scan["first_roads"], 64)
        self.assertLessEqual(agent.last_scan["second_roads"], 256)
        self.assertEqual(game.state.random.getstate(), rng_before)
        self.assertEqual(agent.decide(game, legal), card)
        planned = tuple(agent._pending_roads)
        self.assertEqual(len(planned), 2)
        game.execute(card, validate_action=True)
        for expected in planned:
            legal_road = tuple(game.playable_actions)
            self.assertIn(expected, legal_road)
            selected = agent.decide(game, legal_road)
            self.assertEqual(selected, expected)
            game.execute(selected, validate_action=True)
        self.assertEqual(agent._pending_roads, ())

    def test_hidden_hands_deck_and_rng_do_not_change_choice(self):
        game, actor = self._position()
        agent = RoadBuildingAgent(actor, self.config)
        legal = tuple(game.playable_actions)
        baseline = next(action for action in legal if action.action_type == ActionType.END_TURN)
        original = agent.find_override(game, legal, baseline)
        changed = perturbed_hidden(game, actor)
        hidden = agent.find_override(changed, tuple(changed.playable_actions), baseline)
        self.assertEqual(original, hidden)
        self.assertEqual(agent.decide(game, legal), agent.decide(changed, tuple(changed.playable_actions)))

    def test_unknown_vp_card_purchase_is_conservatively_protected(self):
        game, actor = self._position()
        prefix = f"P{game.state.color_to_index[actor]}_"
        game.state.player_state[prefix + "VICTORY_POINTS"] = 9
        game.state.player_state[prefix + "ACTUAL_VICTORY_POINTS"] = 9
        for resource in ("SHEEP", "WHEAT", "ORE"):
            game.state.player_state[prefix + resource + "_IN_HAND"] = 1
        game.playable_actions = generate_playable_actions(game.state)
        legal = tuple(game.playable_actions)
        purchase = next(action for action in legal if action.action_type == ActionType.BUY_DEVELOPMENT_CARD)
        agent = RoadBuildingAgent(actor, self.config)
        before = game.state.random.getstate()
        self.assertIsNone(agent.find_override(game, legal, purchase))
        self.assertEqual(agent.last_scan["baseline_vp_upper_bound"], 10)
        self.assertEqual(game.state.random.getstate(), before)


if __name__ == "__main__":
    unittest.main()
