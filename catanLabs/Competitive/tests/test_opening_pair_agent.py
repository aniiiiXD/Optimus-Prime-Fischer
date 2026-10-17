"""Last-seat opening plans must be legal, bounded, and hidden-state safe."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanatron import Color, Game, RandomPlayer
from catanatron.players.value import ValueFunctionPlayer

from catanLabs.Catanatron.tournament_protocol import install_v5_protocol
from catanLabs.Competitive.check_visibility import perturbed_hidden
from catanLabs.Competitive.fair_value_agent import FairValueAgent
from catanLabs.Competitive.opening_pair_agent import OpeningPairAgent


class OpeningPairAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_v5_protocol()
        cls.config = json.loads((WORKSPACE / "catanLabs/Competitive/configs/opening_pair_research.json").read_text())

    def _last_seat_position(self, seed):
        game = Game([RandomPlayer(color) for color in Color], seed=seed,
                    vps_to_win=10, discard_limit=7, friendly_robber=False,
                    number_placement="official_spiral")
        last = game.state.colors[-1]
        players = {color: ValueFunctionPlayer(color) for color in game.state.colors[:-1]}
        while game.state.current_color() != last:
            actor = game.state.current_color()
            game.execute(players[actor].decide(game, game.playable_actions),
                         validate_action=True)
        return game, last

    def test_exact_four_action_continuation_is_legal(self):
        for seed in (20272700, 20272701, 20272702, 20272703):
            with self.subTest(seed=seed):
                game, color = self._last_seat_position(seed)
                agent = OpeningPairAgent(color, self.config)
                rng_before = game.state.random.getstate()
                first = agent.decide(game, game.playable_actions)
                line = (first, *agent._pending)
                self.assertEqual(len(line), 4)
                self.assertEqual(game.state.random.getstate(), rng_before)
                self.assertTrue(agent.last_scan["eligible"])
                self.assertLessEqual(agent.last_scan["first_considered"], 12)
                self.assertLessEqual(agent.last_scan["lines_considered"], 432)
                for index, expected in enumerate(line):
                    self.assertIn(expected, game.playable_actions)
                    selected = first if index == 0 else agent.decide(game, game.playable_actions)
                    self.assertEqual(selected, expected)
                    game.execute(selected, validate_action=True)
                self.assertEqual(agent._pending, ())
                self.assertEqual(len(game.state.buildings_by_color[color]["SETTLEMENT"]), 2)

    def test_hidden_deck_and_rng_perturbation_invariant(self):
        game, color = self._last_seat_position(20272704)
        agent = OpeningPairAgent(color, self.config)
        original_rng = game.state.random.getstate()
        first = agent.find_opening_plan(game, game.playable_actions)
        line = agent._candidate_plan
        alternate = perturbed_hidden(game, color)
        hidden_agent = OpeningPairAgent(color, self.config)
        hidden_first = hidden_agent.find_opening_plan(alternate, alternate.playable_actions)
        self.assertEqual(first, hidden_first)
        self.assertEqual(line, hidden_agent._candidate_plan)
        self.assertEqual(game.state.random.getstate(), original_rng)

    def test_non_last_seat_and_later_phase_use_fairvalue(self):
        first_game = Game([RandomPlayer(color) for color in Color], seed=20272705,
                          vps_to_win=10, discard_limit=7, friendly_robber=False,
                          number_placement="official_spiral")
        first_color = first_game.state.colors[0]
        agent = OpeningPairAgent(first_color, self.config)
        fair = FairValueAgent(first_color, self.config)
        self.assertFalse(agent._eligible(first_game, first_game.playable_actions))
        self.assertEqual(agent.decide(first_game, first_game.playable_actions),
                         fair.decide(first_game, first_game.playable_actions))
        players = {color: ValueFunctionPlayer(color) for color in first_game.state.colors}
        while first_game.state.is_initial_build_phase:
            actor = first_game.state.current_color()
            first_game.execute(players[actor].decide(first_game, first_game.playable_actions),
                               validate_action=True)
        self.assertFalse(agent._eligible(first_game, first_game.playable_actions))
        self.assertEqual(agent.decide(first_game, first_game.playable_actions),
                         fair.decide(first_game, first_game.playable_actions))
        # Invalid or stale planned move must fall back to legal FairValue play.
        game, color = self._last_seat_position(20272706)
        agent = OpeningPairAgent(color, self.config)
        agent._pending = (object(),)
        chosen = agent.decide(game, game.playable_actions)
        self.assertIn(chosen, game.playable_actions)

    def test_invalid_search_budget_rejected(self):
        with self.assertRaises(ValueError):
            OpeningPairAgent(Color.BLUE, {**self.config, "opening_max_lines": 0})


if __name__ == "__main__":
    unittest.main()
