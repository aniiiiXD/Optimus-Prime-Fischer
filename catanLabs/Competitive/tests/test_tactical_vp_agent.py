"""Focused legality, tactical-line, and hidden-state checks for the G5 probe."""

from __future__ import annotations

import json
from pathlib import Path
import random
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
from catanLabs.Competitive.tactical_vp_agent import ROOT_TYPES, POINT_TYPES, TacticalVPAgent


class TacticalVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_v5_protocol()
        cls.config = json.loads((WORKSPACE / "catanLabs/Competitive/configs/tactical_vp_research.json").read_text())

    def _late_trade_position(self):
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
        game.state.player_state[prefix + "VICTORY_POINTS"] = 7
        game.state.player_state[prefix + "ACTUAL_VICTORY_POINTS"] = 7
        game.state.player_state[prefix + "HAS_ROLLED"] = True
        for resource, count in {"WOOD": 0, "BRICK": 4, "SHEEP": 0,
                                "WHEAT": 2, "ORE": 2}.items():
            game.state.player_state[prefix + resource + "_IN_HAND"] = count
        game.playable_actions = generate_playable_actions(game.state)
        return game, actor

    def test_trade_then_city_is_a_legal_exact_point_line(self):
        game, actor = self._late_trade_position()
        agent = TacticalVPAgent(actor, self.config)
        legal = tuple(game.playable_actions)
        end_turn = next(action for action in legal if action.action_type == ActionType.END_TURN)
        proposed = agent.find_override(game, legal, end_turn)
        self.assertIn(proposed, legal)
        self.assertEqual(proposed.action_type, ActionType.MARITIME_TRADE)
        self.assertTrue(agent.last_scan["override"])
        self.assertGreater(agent.last_scan["best_line_vp"], agent.last_scan["baseline_vp"])
        self.assertLessEqual(agent.last_scan["roots"], 64)
        self.assertLessEqual(agent.last_scan["followups"], 256)

    def test_live_decision_ignores_hidden_hands_deck_and_rng(self):
        game, actor = self._late_trade_position()
        agent = TacticalVPAgent(actor, self.config)
        legal = tuple(game.playable_actions)
        before = game.state.random.getstate()
        original = agent.decide(game, legal)
        self.assertEqual(game.state.random.getstate(), before)
        changed = perturbed_hidden(game, actor)
        changed.random = random.Random(91823)
        other = agent.decide(changed, tuple(changed.playable_actions))
        self.assertEqual(original, other)
        self.assertIn(original, legal)

    def test_winning_knight_baseline_cannot_be_overridden(self):
        game, actor = self._late_trade_position()
        prefix = f"P{game.state.color_to_index[actor]}_"
        game.state.player_state[prefix + "VICTORY_POINTS"] = 8
        game.state.player_state[prefix + "ACTUAL_VICTORY_POINTS"] = 8
        game.state.player_state[prefix + "PLAYED_KNIGHT"] = 2
        game.state.player_state[prefix + "KNIGHT_IN_HAND"] = 1
        game.state.player_state[prefix + "KNIGHT_OWNED_AT_START"] = True
        game.state.player_state[prefix + "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = False
        game.playable_actions = generate_playable_actions(game.state)
        legal = tuple(game.playable_actions)
        knight = next(action for action in legal if action.action_type == ActionType.PLAY_KNIGHT_CARD)
        agent = TacticalVPAgent(actor, self.config)
        rng_before = game.state.random.getstate()
        self.assertIsNone(agent.find_override(game, legal, knight))
        self.assertEqual(agent.last_scan["baseline_vp"], 10)
        self.assertEqual(agent.last_scan["roots"], 0)
        self.assertEqual(agent.decide(game, legal), knight)
        self.assertEqual(game.state.random.getstate(), rng_before)

    def test_unknown_development_purchase_uses_vp_upper_bound(self):
        game, actor = self._late_trade_position()
        prefix = f"P{game.state.color_to_index[actor]}_"
        game.state.player_state[prefix + "SHEEP_IN_HAND"] = 1
        game.state.player_state[prefix + "WHEAT_IN_HAND"] = 2
        game.state.player_state[prefix + "ORE_IN_HAND"] = 2
        game.playable_actions = generate_playable_actions(game.state)
        legal = tuple(game.playable_actions)
        purchase = next(action for action in legal if action.action_type == ActionType.BUY_DEVELOPMENT_CARD)
        agent = TacticalVPAgent(actor, self.config)
        self.assertIsNone(agent.find_override(game, legal, purchase))
        self.assertEqual(agent.last_scan["baseline_vp"], 8)

    def test_every_available_root_executes_legally_without_live_rng_change(self):
        game, actor = self._late_trade_position()
        prefix = f"P{game.state.color_to_index[actor]}_"
        for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"):
            game.state.player_state[prefix + resource + "_IN_HAND"] = 4
        for card in ("YEAR_OF_PLENTY", "ROAD_BUILDING"):
            game.state.player_state[prefix + card + "_IN_HAND"] = 1
            game.state.player_state[prefix + card + "_OWNED_AT_START"] = True
        game.playable_actions = generate_playable_actions(game.state)
        roots = {action.action_type: action for action in game.playable_actions
                 if action.action_type in ROOT_TYPES}
        self.assertTrue({ActionType.MARITIME_TRADE, ActionType.BUILD_ROAD,
                         ActionType.BUILD_CITY, ActionType.PLAY_YEAR_OF_PLENTY,
                         ActionType.PLAY_ROAD_BUILDING} <= set(roots))
        rng_before = game.state.random.getstate()
        for action in roots.values():
            after = game.copy()
            after.execute(action, validate_action=True)
            self.assertEqual(game.state.random.getstate(), rng_before)
            for second in after.playable_actions:
                if second.action_type in POINT_TYPES:
                    later = after.copy()
                    later.execute(second, validate_action=True)
                    self.assertEqual(game.state.random.getstate(), rng_before)
                    break


if __name__ == "__main__":
    unittest.main()
