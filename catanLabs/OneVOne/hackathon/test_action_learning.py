"""Checks that learning can change real legal choices without changing search."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE / "vendor/kit"))

import catan_kit  # Apply the same engine patches as official matches.
from catanatron import Color, Game, RandomPlayer
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import ActionRecord, ActionType, RESOURCES
from catanLabs.OneVOne.agent import DEFAULT_CONFIG
from catanLabs.OneVOne.action_planner import ACTION_DIM, ACTION_TYPES, ActionPlanner
from catanLabs.OneVOne.learning import POSITION_FEATURES, choose, normalize_scores
import action_learning


def make_game(seed=20261017, target=10):
    """A normal post-roll position with builds, trades, buying and ending legal."""
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
                seed=seed, vps_to_win=target)
    while game.state.is_initial_build_phase:
        game.execute(game.playable_actions[0])
    roll = next(a for a in game.playable_actions if a.action_type == ActionType.ROLL)
    game.execute(roll, action_record=ActionRecord(roll, (1, 1)))
    color = game.state.current_color()
    prefix = f"P{game.state.color_to_index[color]}_"
    for i, resource in enumerate(RESOURCES):
        key = prefix + resource + "_IN_HAND"
        transfer = 4 - game.state.player_state[key]
        game.state.player_state[key] += transfer
        game.state.resource_freqdeck[i] -= transfer
    game.playable_actions = generate_playable_actions(game.state)
    return game


def planner(game):
    return ActionPlanner(game.state.current_color(), json.loads(DEFAULT_CONFIG.read_text()))


def prefer(action_type, strength=4.):
    weights = np.zeros(ACTION_DIM)
    weights[2 * len(POSITION_FEATURES) + ACTION_TYPES.index(action_type.name)] = strength
    return weights


class ActionLearningTests(unittest.TestCase):
    def test_replay_updates_after_four_actual_losses_before_next_choice(self):
        game = make_game()
        plan = planner(game).plan(game, game.playable_actions)
        h, _ = choose(plan)
        end = next(i for i, a in enumerate(plan.actions) if a.action_type == ActionType.END_TURN)
        prior = normalize_scores(plan.heuristic)
        pair = {"features_i": plan.features[end].tolist(), "features_j": plan.features[h].tolist(),
                "prior_i": float(prior[end]), "prior_j": float(prior[h]),
                "outcome_i": 1., "outcome_j": 0.}
        replay = [dict(pair, board_seed=101, game_won=True),
                  dict(pair, board_seed=102, game_won=False)]
        weights, losses = np.zeros(ACTION_DIM), 0
        with patch.object(action_learning, "fit", wraps=action_learning.fit) as fit:
            for won, expected_losses in ((False, 1), (True, 1), (None, 1), (False, 2), (False, 3)):
                losses, weights, report = action_learning.update_after_four_losses(
                    losses, won, replay, {102}, weights)
                self.assertEqual(losses, expected_losses)
                self.assertIsNone(report)
                self.assertEqual(fit.call_count, 0)
                self.assertEqual(choose(plan, weights)[0], h)
            losses, weights, report = action_learning.update_after_four_losses(
                losses, False, replay, {102}, weights)
            self.assertEqual(losses, 0)
            self.assertIsNotNone(report)
            self.assertEqual(fit.call_count, 1)
            self.assertIs(fit.call_args.args[0], replay)
            self.assertTrue(any(row["game_won"] for row in fit.call_args.args[0]))
        self.assertGreater(float(np.linalg.norm(weights)), 0.)
        from catanLabs.OneVOne.learning import LearningPlayer
        next_game = make_game()
        player = LearningPlayer(next_game.state.current_color(), json.loads(DEFAULT_CONFIG.read_text()))
        player.weights = weights
        selected = player.decide(next_game, next_game.playable_actions)
        self.assertNotEqual(selected, plan.actions[h])
        self.assertTrue(player.last_decision["ml_changed"])

    def test_default_blend_overturns_across_families_with_identical_search(self):
        game = make_game()
        legal = list(game.playable_actions)
        heuristic_plan = planner(game).plan(game, legal)
        learned_plan = planner(game).plan(game, legal)
        self.assertEqual(heuristic_plan.actions, legal)
        self.assertEqual(learned_plan.actions, legal)
        self.assertEqual(heuristic_plan.trace, learned_plan.trace)
        self.assertTrue(any(t["followups"] for t in heuristic_plan.trace))
        np.testing.assert_array_equal(heuristic_plan.features, learned_plan.features)
        np.testing.assert_array_equal(heuristic_plan.heuristic, learned_plan.heuristic)
        self.assertEqual(heuristic_plan.features.shape, (len(legal), ACTION_DIM))
        prior = normalize_scores(heuristic_plan.heuristic)
        self.assertTrue(np.isfinite(prior).all())
        self.assertTrue((abs(prior) <= 1).all())
        h, no_model = choose(heuristic_plan)
        zero, _ = choose(learned_plan, np.zeros(ACTION_DIM))
        self.assertEqual(h, int(heuristic_plan.heuristic.argmax()))
        self.assertEqual(zero, h)
        self.assertNotEqual(legal[h].action_type, ActionType.END_TURN)
        self.assertFalse(heuristic_plan.winning.any())
        weights = prefer(ActionType.END_TURN)
        selected, diagnostics = choose(learned_plan, weights)
        self.assertEqual(legal[selected].action_type, ActionType.END_TURN)
        self.assertTrue(diagnostics["ml_changed"])
        self.assertTrue(diagnostics["cross_type"])
        self.assertEqual(diagnostics["executed_index"], selected)
        self.assertFalse(diagnostics["winning_override"])
        print(json.dumps({
            "proof": "default_cross_family_override", "seed": 20261017,
            "legal_moves": len(legal), "heuristic_move": legal[h].action_type.name,
            "executed_move": legal[selected].action_type.name,
            "heuristic_prior": float(prior[h]), "chosen_prior": float(prior[selected]),
            "chosen_residual": float(learned_plan.features[selected] @ weights),
            "same_candidates_and_search": True,
        }))

    def test_verified_winning_line_survives_large_opposing_preference(self):
        # Three VP is a supported engine target: one legal city wins from setup.
        game = make_game(target=3)
        plan = planner(game).plan(game, game.playable_actions)
        self.assertTrue(plan.winning.any())
        selected, diagnostics = choose(plan, prefer(ActionType.END_TURN, 1e6))
        self.assertTrue(plan.winning[selected])
        self.assertTrue(diagnostics["winning_override"])
        self.assertNotEqual(plan.actions[selected].action_type, ActionType.END_TURN)
        color = game.state.current_color()
        game.execute(plan.actions[selected])
        if game.winning_color() is None:
            followup = plan.trace[selected]["followup"]
            self.assertIsNotNone(followup)
            game.execute(next(a for a in game.playable_actions
                              if [a.action_type.name, repr(a.value)] == followup))
        self.assertEqual(game.winning_color(), color)

    def test_action_features_ignore_enemy_hidden_devs_and_future_deck_order(self):
        first, second = make_game(), make_game()
        own = first.state.current_color()
        enemy = next(c for c in first.state.colors if c != own)
        for game, card in ((first, "KNIGHT"), (second, "VICTORY_POINT")):
            prefix = f"P{game.state.color_to_index[enemy]}_"
            game.state.development_listdeck.remove(card)
            game.state.player_state[prefix + card + "_IN_HAND"] = 1
            game.state.player_state[prefix + "ACTUAL_VICTORY_POINTS"] += int(card == "VICTORY_POINT")
        second.state.development_listdeck.reverse()
        a = planner(first).plan(first, first.playable_actions)
        b = planner(second).plan(second, second.playable_actions)
        self.assertEqual(a.actions, b.actions)
        self.assertEqual(a.trace, b.trace)
        np.testing.assert_array_equal(a.features, b.features)
        np.testing.assert_array_equal(a.heuristic, b.heuristic)
        np.testing.assert_array_equal(a.winning, b.winning)


if __name__ == "__main__":
    unittest.main()
