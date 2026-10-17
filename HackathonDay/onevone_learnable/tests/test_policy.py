"""Behavioral checks: information boundaries, learning influence and win safety."""
from pathlib import Path
import random
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent/"official-kit")]
import catan_kit
from catanatron.game import Game, generate_playable_actions
from catanatron.models.enums import Action, ActionRecord, ActionType
from catanatron.models.player import RandomPlayer
from catan_kit.match import COLORS
from features import CANDIDATE_DIM, DEVS, RESOURCES, encode_board, prefix, state_features
from policy import LearnableOneVOne, load_config
from runtime import NeuralScorer


def initial():
    return Game([RandomPlayer(c) for c in COLORS], seed=2026800200)


def rich_position():
    game = initial()
    while game.state.is_initial_build_phase:
        game.execute(game.playable_actions[0])
    color = game.state.current_color()
    roll = Action(color, ActionType.ROLL, None)
    game.execute(roll, action_record=ActionRecord(roll, (3, 3)))
    own, ps = prefix(game, color), game.state.player_state
    for index, resource in enumerate(RESOURCES):
        current = ps[own+resource+"_IN_HAND"]
        ps[own+resource+"_IN_HAND"] = 5
        game.state.resource_freqdeck[index] -= 5-current
    game.playable_actions = generate_playable_actions(game.state)
    return game, color


class TargetScorer:
    def __init__(self, target):
        self.target = target

    def predict(self, board, candidates):
        out = np.full(len(candidates), -10., dtype=np.float32)
        out[self.target] = 10
        return out


class PolicyTests(unittest.TestCase):
    def test_every_legal_move_is_exposed_and_source_position_unchanged(self):
        game, color = rich_position()
        policy = LearnableOneVOne(color)
        before = dict(game.state.player_state)
        board_before = dict(game.state.board.buildings)
        _, candidates = policy.prepare_candidates(game, game.playable_actions)
        self.assertEqual([c.action for c in candidates], game.playable_actions)
        self.assertTrue(all(c.features.shape == (CANDIDATE_DIM,) for c in candidates))
        self.assertTrue(all(np.isfinite(c.features).all() for c in candidates))
        self.assertEqual(before, game.state.player_state)
        self.assertEqual(board_before, game.state.board.buildings)

    def test_hidden_assignments_cannot_change_features_scores_or_choice(self):
        game, color = rich_position()
        enemy = next(c for c in COLORS if c != color)
        other = prefix(game, enemy)
        first, second = game.copy(), game.copy()
        for world, card, private_vp in ((first, "KNIGHT", 2), (second, "VICTORY_POINT", 99)):
            for d in DEVS:
                world.state.player_state[other+d+"_IN_HAND"] = int(d == card)
                world.state.player_state[other+d+"_OWNED_AT_START"] = int(d == card)
            world.state.player_state[other+"ACTUAL_VICTORY_POINTS"] = private_vp
        second.state.development_listdeck.reverse()
        np.testing.assert_array_equal(state_features(first, color), state_features(second, color))
        np.testing.assert_array_equal(encode_board(first, color), encode_board(second, color))
        policy = LearnableOneVOne(color)
        _, a = policy.prepare_candidates(first, game.playable_actions)
        _, b = policy.prepare_candidates(second, game.playable_actions)
        np.testing.assert_array_equal(np.stack([c.features for c in a]), np.stack([c.features for c in b]))
        self.assertEqual([c.score for c in a], [c.score for c in b])

    def test_a_learned_score_can_overturn_the_full_expert_order(self):
        game = initial()
        color = game.state.current_color()
        config = load_config()
        config["learning_weight"] = 2
        policy = LearnableOneVOne(color, config)
        _, candidates = policy.prepare_candidates(game, game.playable_actions)
        target = min(candidates, key=lambda c: c.rank).index
        expert = max(candidates, key=lambda c: c.rank).index
        self.assertNotEqual(target, expert)
        policy.scorer = TargetScorer(target)
        self.assertEqual(policy.decide(game, game.playable_actions), target)
        self.assertEqual(policy.stats["model_changes"], 1)

    def test_immediate_guaranteed_win_overrides_a_bad_model(self):
        game, color = rich_position()
        own = prefix(game, color)
        game.state.player_state[own+"ACTUAL_VICTORY_POINTS"] = 9
        game.state.player_state[own+"VICTORY_POINTS"] = 9
        config = load_config()
        config["learning_weight"] = 20
        end = next(i for i,a in enumerate(game.playable_actions) if a.action_type == ActionType.END_TURN)
        policy = LearnableOneVOne(color, config, TargetScorer(end))
        choice = policy.decide(game, game.playable_actions)
        self.assertEqual(policy.last_decision["reason"], "guaranteed_win")
        self.assertEqual(game.playable_actions[choice].action_type, ActionType.BUILD_CITY)

    def test_invalid_model_outputs_use_expert_and_are_reported(self):
        class Bad:
            def predict(self, board, candidates):
                return np.full(len(candidates), np.nan)
        game = initial()
        policy = LearnableOneVOne(game.state.current_color(), scorer=Bad())
        choice = policy.decide(game, game.playable_actions)
        self.assertEqual(choice, policy.last_decision["prior_choice"])
        self.assertEqual(policy.stats["model_errors"], 1)
        self.assertIn("finite", policy.last_decision["model_error"])

    def test_clock_reserve_returns_legal_action_without_neural_call(self):
        game, color = rich_position()
        policy = LearnableOneVOne(color, scorer=TargetScorer(0))
        choice = policy.decide(game, game.playable_actions, cpu_left=.02)
        self.assertIn(choice, range(len(game.playable_actions)))
        self.assertEqual(policy.stats["model_calls"], 0)
        self.assertEqual(policy.last_decision["reason"], "clock_reserve")

    def test_missing_configured_checkpoint_fails_visibly(self):
        config = load_config()
        config.update(scorer="mlp", checkpoint="does-not-exist.npz")
        with self.assertRaises(FileNotFoundError):
            LearnableOneVOne(COLORS[0], config)


if __name__ == "__main__":
    unittest.main()
