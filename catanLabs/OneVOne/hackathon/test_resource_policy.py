"""Small real-engine checks for resource scoring and the warm start."""

import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor/kit"))

import catan_kit
from catan_kit.obs import Obs
from catanatron import Color, Game, RandomPlayer
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import Action, ActionType, RESOURCES
from agent import Agent as Baseline
from resource_policy import Agent, ResourcePlayer, DEFAULT_CONFIG, resource_need
from resource_experiments import paired_difference


class ResourceTests(unittest.TestCase):
    def test_need_tracks_missing_city_resource(self):
        hand = dict.fromkeys(RESOURCES, 0)
        hand.update(WHEAT=2, ORE=2)
        self.assertGreater(resource_need(hand, "ORE"), resource_need(hand, "WOOD"))
        hand.update(ORE=3)
        self.assertEqual(resource_need(hand, "ORE"), 0)

    def test_zero_correction_preserves_baseline_choices(self):
        game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=101)
        baseline, candidate = Baseline(), Agent(exact_monopoly=False, params=[0, 0, 0])
        for _ in range(80):
            if game.winning_color() is not None:
                break
            obs = Obs(game, game.state.current_color())
            legal = game.playable_actions
            expected = baseline.decide(obs, legal)
            actual = candidate.decide(obs, legal)
            self.assertEqual(actual, expected)
            game.execute(legal[actual])

    def test_exact_monopoly_uses_resources_and_preserves_input(self):
        game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=102)
        while game.state.current_prompt.name != "PLAY_TURN":
            game.execute(game.playable_actions[0])
        color = game.state.current_color()
        enemy = next(c for c in game.state.colors if c != color)
        ps = game.state.player_state
        prefix = f"P{game.state.color_to_index[color]}_"
        ep = f"P{game.state.color_to_index[enemy]}_"
        ps[prefix + "MONOPOLY_IN_HAND"] = 1
        ps[prefix + "MONOPOLY_OWNED_AT_START"] = True
        ps[ep + "ORE_IN_HAND"] = 6
        game.playable_actions = generate_playable_actions(game.state)
        action = Action(color, ActionType.PLAY_MONOPOLY, "ORE")
        self.assertIn(action, game.playable_actions)
        player = ResourcePlayer(color, json.loads(DEFAULT_CONFIG.read_text()), True, 0)
        before = dict(ps)
        after = game.copy()
        after.execute(action)
        self.assertEqual(player._monopoly_value(game, "ORE"), player.value_fn(after, color))
        self.assertEqual(ps, before)
        self.assertEqual(after.state.player_state[ep + "ORE_IN_HAND"], 0)

    def test_invalid_parameters_are_rejected(self):
        for params in ([0, 0], [0, float("nan"), 0], [0, 5, 0]):
            with self.assertRaises(ValueError):
                Agent(params=params)

    def test_paired_analysis_checks_schedule(self):
        records = [{"game": i, "board_seed": 1, "winner": "a"} for i in range(2)]
        self.assertEqual(paired_difference({"records": records}, {"records": records})["win_rate_difference"], 0)
        other = [dict(r, board_seed=2) for r in records]
        with self.assertRaises(ValueError):
            paired_difference({"records": records}, {"records": other})


if __name__ == "__main__":
    unittest.main()
