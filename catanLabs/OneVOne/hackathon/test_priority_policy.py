import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor/kit"))

import catan_kit
from catan_kit.obs import Obs
from catanatron import Color, Game, RandomPlayer
from catanatron.features import build_production_features
from catanatron.models.enums import RESOURCES
from agent import Agent as Baseline
from priority_policy import Agent, DEFAULT_CONFIG, PriorityPlayer, normalize_priorities


class PriorityTests(unittest.TestCase):
    def test_normalization_and_invalid_values(self):
        self.assertEqual(normalize_priorities([2] * 5), [1] * 5)
        self.assertAlmostEqual(sum(normalize_priorities([1, 2, 3, 4, 5])), 5)
        for values in ([1, 2], [0, 1, 1, 1, 1], [1, 1, 1, 1, float("inf")]):
            with self.assertRaises(ValueError):
                normalize_priorities(values)

    def test_equal_priorities_preserve_baseline(self):
        game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=201)
        base, candidate = Baseline(), Agent()
        for _ in range(80):
            if game.winning_color() is not None:
                break
            legal = game.playable_actions
            obs = Obs(game, game.state.current_color())
            expected = base.decide(obs, legal)
            self.assertEqual(candidate.decide(obs, legal), expected)
            game.execute(legal[expected])

    def test_correction_matches_engine_resource_production(self):
        game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=202)
        while game.state.current_prompt.name != "PLAY_TURN":
            game.execute(game.playable_actions[0])
        weights = normalize_priorities([1, 1, 1, 2, 3])
        color = game.state.current_color()
        player = PriorityPlayer(color, json.loads(DEFAULT_CONFIG.read_text()), weights)
        features = build_production_features(True)(game, color)
        expected = player.weights["production"] * sum((w - 1) * features[f"EFFECTIVE_P0_{r}_PRODUCTION"] for r, w in zip(RESOURCES, weights))
        self.assertAlmostEqual(player.priority_value(game, color) - player.base_value_fn(game, color), expected, delta=0.2)


if __name__ == "__main__":
    unittest.main()
