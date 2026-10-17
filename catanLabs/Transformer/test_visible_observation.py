"""Focused tests of the player-visible Catanatron decision boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
import random
import sys
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanatron import Color, Game, RandomPlayer  # noqa: E402
from catanatron.players.value import ValueFunctionPlayer  # noqa: E402
from catanLabs.Transformer.visible_observation import (  # noqa: E402
    DEVELOPMENT_CARDS, RESOURCES, VisibleAgentAdapter, snapshot,
)


def perturbed_hidden(game, actor):
    clone = game.copy()
    state = clone.state
    for color in state.colors:
        if color == actor:
            continue
        prefix = f"P{state.color_to_index[color]}_"
        for card_names in (RESOURCES, DEVELOPMENT_CARDS):
            counts = [state.player_state[prefix + card + "_IN_HAND"] for card in card_names]
            for card, count in zip(card_names, counts[1:] + counts[:1]):
                state.player_state[prefix + card + "_IN_HAND"] = count
    state.development_listdeck.reverse()
    state.random = random.Random(99991)
    clone.random = state.random
    return clone


class FirstVisibleAgent:
    def decide_visible(self, observation):
        assert observation.acting_relative_seat == 0
        assert observation.legal_actions
        return 0


class InvalidVisibleAgent:
    def decide_visible(self, observation):
        return len(observation.legal_actions)


def assert_plain_tree(testcase, value):
    """The snapshot must contain no simulator objects, mutable lists or dicts."""
    if is_dataclass(value):
        for field in fields(value):
            assert_plain_tree(testcase, getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            assert_plain_tree(testcase, item)
    else:
        testcase.assertTrue(value is None or type(value) in (bool, int, float, str), type(value))


class VisibleObservationTests(unittest.TestCase):
    def test_four_seats_seeded_games_and_hidden_invariance(self):
        checked = 0
        kinds = set()
        phases = set()
        for seat in range(4):
            game = Game([ValueFunctionPlayer(color) for color in Color], seed=20265170 + seat,
                        vps_to_win=10, discard_limit=7, friendly_robber=False,
                        number_placement="official_spiral")
            actor = game.state.colors[seat]
            adapter = VisibleAgentAdapter(FirstVisibleAgent())
            for _ in range(1500):
                if game.winning_color() is not None:
                    break
                current = game.state.current_color()
                legal = tuple(game.playable_actions)
                if current == actor:
                    before_rng = game.state.random.getstate()
                    observation = snapshot(game, legal)
                    changed = snapshot(perturbed_hidden(game, actor), legal)
                    self.assertEqual(observation, changed)
                    self.assertEqual(game.state.random.getstate(), before_rng)
                    self.assertEqual(len(observation.tiles), 19)
                    self.assertEqual(len(observation.nodes), 54)
                    self.assertEqual(len(observation.edges), 72)
                    self.assertEqual(len(observation.players), 4)
                    assert_plain_tree(self, observation)
                    self.assertTrue(all(p.own_resources is None and p.own_development_cards is None
                                        for p in observation.players[1:]))
                    with self.assertRaises(FrozenInstanceError):
                        observation.turn_count = 999
                    selected = adapter.decide(game, legal)
                    self.assertIn(selected, legal)
                    kinds.update(action.kind for action in observation.legal_actions)
                    phases.add(observation.phase)
                    checked += 1
                # Advance games with the engine's random players. The tested
                # adapter is still exercised at every candidate decision.
                action = game.state.current_player().decide(game, legal)
                game.execute(action, validate_action=True)
            self.assertIsNotNone(game.winning_color(), f"Seat {seat} game hit the 1500-action cap")
        self.assertGreater(checked, 100)
        self.assertTrue({"BUILD_SETTLEMENT", "BUILD_ROAD", "ROLL", "END_TURN"} <= kinds, kinds)
        self.assertTrue({"BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD", "PLAY_TURN"} <= phases, phases)

    def test_adapter_rejects_nonlegal_index(self):
        game = Game([RandomPlayer(color) for color in Color], seed=7, vps_to_win=10)
        with self.assertRaises(ValueError):
            VisibleAgentAdapter(InvalidVisibleAgent()).decide(game, game.playable_actions)


if __name__ == "__main__":
    unittest.main()
