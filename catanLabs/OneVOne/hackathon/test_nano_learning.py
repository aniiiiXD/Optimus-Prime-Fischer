"""Nano remains the baseline when its search is exposed to action learning."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE / "vendor/kit"))

import catan_kit
from catan_kit.obs import Obs
from catanatron import Color, Game, RandomPlayer
from catanatron.models.enums import ActionType, RESOURCES
from catanLabs.OneVOne.agent import DEFAULT_CONFIG
from catanLabs.OneVOne.action_planner import ACTION_DIM, ACTION_TYPES
from catanLabs.OneVOne.learning import (
    ADVANTAGE_MLP_VERSION,
    LearningPlayer,
    POSITION_FEATURES,
    choose,
    heuristic_fingerprint,
)
from catanatron.models.actions import generate_playable_actions
from test_action_learning import make_game
from my_team.nano import Agent as NativeNano
from advantage_learning import _sample_nonterminal


# Golden descriptions recorded once from the Nano source at commit 4ae85912.
# The fixtures below use seeds 20261018 (opening) and 20261019 (rich post-roll).
GOLDEN_NANO_ACTIONS = {
    "opening": ["BUILD_SETTLEMENT", "15"],
    "rich_post_roll": ["BUILD_CITY", "6"],
}


def nano_config():
    config = json.loads(DEFAULT_CONFIG.read_text())
    config["planner"] = "nano"
    return config


def initial_game(seed=20261018):
    return Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=seed)


def trade_game(seed=20261020):
    game = make_game(seed=seed, target=10)
    color = game.state.current_color()
    prefix = f"P{game.state.color_to_index[color]}_"
    for resource, target in zip(RESOURCES, (4, 4, 4, 0, 0)):
        key = prefix + resource + "_IN_HAND"
        change = target - game.state.player_state[key]
        game.state.player_state[key] = target
        game.state.resource_freqdeck[RESOURCES.index(resource)] -= change
    game.playable_actions = generate_playable_actions(game.state)
    return game


def action_description(action):
    return [action.action_type.name, repr(action.value)]


def mlp_member(action_type, sign=1.0):
    w1 = np.zeros((ACTION_DIM, 32))
    w1[2 * len(POSITION_FEATURES) + ACTION_TYPES.index(action_type.name), 0] = sign
    w2 = np.zeros(32)
    w2[0] = 1.0
    return {"w1": w1.tolist(), "b1": np.zeros(32).tolist(),
            "w2": w2.tolist(), "b2": 0.0}


def mlp_model(config, action_type, *, enabled=True, signs=None):
    signs = signs or [1.0] * 5
    return {
        "version": ADVANTAGE_MLP_VERSION,
        "feature_scale": np.ones(ACTION_DIM).tolist(),
        "members": [mlp_member(action_type, sign) for sign in signs],
        "heuristic_sha256": heuristic_fingerprint(config),
        "gate": {"enabled": enabled, "z": 1.64, "margin": 0.0},
    }


class NanoLearningTests(unittest.TestCase):
    def test_rollout_rejects_hidden_cards_that_would_already_end_the_game(self):
        valid = make_game()
        invalid = valid.copy()
        perspective = valid.state.current_color()
        enemy = next(c for c in valid.state.colors if c != perspective)
        prefix = f"P{invalid.state.color_to_index[enemy]}_"
        invalid.state.player_state[prefix + "ACTUAL_VICTORY_POINTS"] = valid.vps_to_win
        self.assertIsNotNone(invalid.winning_color())
        with patch("advantage_learning.masked_copy", side_effect=[invalid, valid]) as sample:
            self.assertIs(_sample_nonterminal(valid, perspective, 173), valid)
            self.assertEqual(sample.call_count, 2)
        with patch("advantage_learning.masked_copy", return_value=invalid):
            with self.assertRaisesRegex(ValueError, "live position"):
                _sample_nonterminal(valid, perspective, 173)

    def setUp(self):
        self.config = nano_config()

    def test_learning_player_matches_native_nano_golden_on_opening_and_rich_turn(self):
        fixtures = [
            ("opening", initial_game()),
            ("rich_post_roll", make_game(seed=20261019, target=10)),
        ]
        for name, game in fixtures:
            color = game.state.current_color()
            actions = list(game.playable_actions)
            native = NativeNano()
            self.assertIsNone(native.search_depth)
            native_index = native.decide(Obs(game, color), actions)
            player = LearningPlayer(color, self.config)
            selected = player.decide(game, actions)

            self.assertIn(selected, actions)
            self.assertEqual(action_description(actions[native_index]), GOLDEN_NANO_ACTIONS[name])
            self.assertEqual(action_description(selected), GOLDEN_NANO_ACTIONS[name])
            self.assertEqual(player.last_plan.actions, actions)
            self.assertEqual(player.planner.last_index, native_index)

    def test_all_roots_and_captured_search_trace_survive_zero_ml_then_mlp_can_cross_types(self):
        game = make_game(seed=20261020, target=10)
        color = game.state.current_color()
        actions = list(game.playable_actions)
        player = LearningPlayer(color, self.config)
        heuristic_action = player.decide(game, actions)
        heuristic_plan = player.last_plan
        learned_plan = player.planner.plan(game, actions)

        self.assertEqual(heuristic_plan.actions, actions)
        self.assertEqual(learned_plan.actions, actions)
        self.assertEqual(heuristic_plan.trace, learned_plan.trace)
        self.assertTrue(all(row["search_depth"] == 5 for row in learned_plan.trace))
        self.assertTrue(any(row["captured_candidates"] for row in learned_plan.trace))
        np.testing.assert_array_equal(heuristic_plan.features, learned_plan.features)
        np.testing.assert_array_equal(heuristic_plan.heuristic, learned_plan.heuristic)
        self.assertFalse(heuristic_plan.winning.any())

        zero_model = mlp_model(self.config, ActionType.END_TURN, signs=[0.0] * 3)
        zero_index, zero_diagnostics = choose(learned_plan, model=zero_model)
        self.assertEqual(learned_plan.actions[zero_index], heuristic_action)
        self.assertFalse(zero_diagnostics["gate_passed"])

        heuristic_index = int(np.asarray(learned_plan.heuristic).argmax())
        target_type = next(action.action_type for action in actions
                           if action.action_type != actions[heuristic_index].action_type)
        consensus = mlp_model(self.config, target_type)
        selected, diagnostics = choose(learned_plan, model=consensus)
        self.assertEqual(learned_plan.actions[selected].action_type, target_type)
        self.assertNotEqual(target_type, learned_plan.actions[heuristic_index].action_type)
        self.assertTrue(diagnostics["gate_passed"])
        self.assertTrue(diagnostics["cross_type"])

        shallow_config = dict(self.config, nano={"search_depth": 4})
        shallow = LearningPlayer(color, shallow_config).planner.plan(game, actions)
        self.assertEqual(shallow.actions, actions)
        self.assertTrue(all(row["search_depth"] == 4 for row in shallow.trace))
        captured = [candidate for row in shallow.trace for candidate in row["captured_candidates"]]
        self.assertTrue(captured)
        for row in shallow.trace:
            for candidate in row["captured_candidates"]:
                self.assertLessEqual(candidate["depth"], 4)
                self.assertEqual(candidate["depth"], len(candidate["line"]))
                if candidate["line"]:
                    self.assertEqual(candidate["line"][0], row["root"])

        trade_position = trade_game()
        trade_actions = list(trade_position.playable_actions)
        trade_plan = LearningPlayer(color, shallow_config).planner.plan(trade_position, trade_actions)
        self.assertEqual(trade_plan.actions, trade_actions)
        trade_candidates = [
            (row, candidate) for row in trade_plan.trace
            for candidate in row["captured_candidates"]
            if any(step[0] == "MARITIME_TRADE" for step in candidate["line"])
        ]
        self.assertTrue(trade_candidates)
        for row, candidate in trade_candidates:
            self.assertEqual(candidate["line"][0], row["root"])
            self.assertEqual(candidate["depth"], len(candidate["line"]))
            self.assertLessEqual(candidate["depth"], 4)

    def test_hidden_dev_and_deck_changes_leave_nano_plan_and_verified_win_protected(self):
        first, second = make_game(seed=20261021, target=10), make_game(seed=20261021, target=10)
        color = first.state.current_color()
        enemy = next(c for c in first.state.colors if c != color)
        for game, card in ((first, "KNIGHT"), (second, "VICTORY_POINT")):
            prefix = f"P{game.state.color_to_index[enemy]}_"
            game.state.development_listdeck.remove(card)
            game.state.player_state[prefix + card + "_IN_HAND"] = 1
            game.state.player_state[prefix + "ACTUAL_VICTORY_POINTS"] += int(card == "VICTORY_POINT")
        second.state.development_listdeck.reverse()
        first_actions = list(first.playable_actions)
        second_actions = list(second.playable_actions)
        self.assertEqual(first_actions, second_actions)
        first_obs, second_obs = Obs(first, color), Obs(second, color)
        self.assertEqual(first_obs.opp_dev_count, second_obs.opp_dev_count)
        self.assertEqual(first_obs.opp_public_vps, second_obs.opp_public_vps)

        first_player = LearningPlayer(color, self.config)
        second_player = LearningPlayer(color, self.config)
        first_action = first_player.decide(first, first_actions)
        second_action = second_player.decide(second, second_actions)
        self.assertEqual(first_action, second_action)
        self.assertEqual(first_player.last_plan.actions, second_player.last_plan.actions)
        self.assertEqual(first_player.last_plan.trace, second_player.last_plan.trace)
        np.testing.assert_array_equal(first_player.last_plan.features, second_player.last_plan.features)
        np.testing.assert_array_equal(first_player.last_plan.heuristic, second_player.last_plan.heuristic)
        np.testing.assert_array_equal(first_player.last_plan.winning, second_player.last_plan.winning)

        winning_game = make_game(seed=20261022, target=3)
        winning_color = winning_game.state.current_color()
        winning_actions = list(winning_game.playable_actions)
        winning_player = LearningPlayer(winning_color, self.config)
        winning_plan = winning_player.planner.plan(winning_game, winning_actions)
        self.assertTrue(winning_plan.winning.any())
        override = mlp_model(self.config, ActionType.END_TURN)
        index, diagnostics = choose(winning_plan, model=override)
        self.assertTrue(winning_plan.winning[index])
        self.assertTrue(diagnostics["winning_override"])
        winning_game.execute(winning_plan.actions[index])
        self.assertEqual(winning_game.winning_color(), winning_color)


if __name__ == "__main__":
    unittest.main()
