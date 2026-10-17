"""Deterministic checks for teacher collection's tournament-v5 contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanLabs.Catanatron.tournament_protocol import canonical_action_key, install_v5_protocol


class TournamentProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from catanatron import Color, Game, RandomPlayer

        install_v5_protocol()
        cls.colors = list(Color)
        cls.Game = Game
        cls.RandomPlayer = RandomPlayer

    def new_game(self, seed=31415):
        game = self.Game([self.RandomPlayer(color) for color in self.colors], seed=seed,
                         vps_to_win=10, discard_limit=7, friendly_robber=False,
                         number_placement="official_spiral")
        game.playable_actions.sort(key=canonical_action_key)
        return game

    def advance_to_roll(self, game):
        from catanatron.models.enums import ActionType

        for _ in range(32):
            rolls = [action for action in game.playable_actions if action.action_type == ActionType.ROLL]
            if rolls:
                return rolls[0]
            game.execute(game.playable_actions[0], validate_action=True)
        self.fail("Did not reach a legal roll")

    def test_copy_rng_isolated_and_forced_roll_preserves_live_stream(self):
        from catanatron.models.enums import Action

        game = self.new_game()
        roll = self.advance_to_roll(game)
        live_rng_state = game.state.random.getstate()
        clone = game.copy()
        self.assertIsNot(clone.state.random, game.state.random)
        clone.state.random.random()
        self.assertEqual(game.state.random.getstate(), live_rng_state)

        forced = Action(roll.color, roll.action_type, (2, 2))
        record = clone.execute(forced, validate_action=False)
        self.assertEqual(record.result, (2, 2))
        self.assertEqual(game.state.random.getstate(), live_rng_state)
        self.assertEqual(clone.playable_actions, sorted(clone.playable_actions, key=canonical_action_key))

        ordinary_record = game.execute(roll, validate_action=True)
        self.assertIsInstance(ordinary_record.result, tuple)
        self.assertNotEqual(game.state.random.getstate(), live_rng_state)

    def test_teacher_decision_does_not_consume_live_chance(self):
        from catanatron.players.minimax import AlphaBetaPlayer

        game = self.new_game(seed=27182)
        self.advance_to_roll(game)
        teacher = AlphaBetaPlayer(game.state.current_color(), AlphaBetaPlayer.Params(depth=1))
        state_before = game.state.random.getstate()
        legal = list(game.playable_actions)
        first = teacher.decide(game, legal)
        second = teacher.decide(game, legal)
        self.assertIn(first, legal)
        self.assertEqual(first, second)
        self.assertEqual(game.state.random.getstate(), state_before)

    def test_hypothetical_dev_draw_uses_requested_card(self):
        from catanatron.models.enums import Action, ActionType

        game = self.new_game(seed=12345)
        roll = self.advance_to_roll(game)
        game.execute(roll, validate_action=True)
        while game.state.current_prompt.name != "PLAY_TURN":
            game.execute(game.playable_actions[0], validate_action=True)
        clone = game.copy()
        color = clone.state.current_color()
        prefix = f"P{clone.state.color_to_index[color]}_"
        for resource in ("ORE", "WHEAT", "SHEEP"):
            clone.state.player_state[prefix + resource + "_IN_HAND"] += 1
        desired = next(card for card in clone.state.development_listdeck
                       if card != clone.state.development_listdeck[-1])
        original_top = clone.state.development_listdeck[-1]
        rng_state = clone.state.random.getstate()
        record = clone.execute(Action(color, ActionType.BUY_DEVELOPMENT_CARD, desired),
                               validate_action=False)
        self.assertEqual(record.result, desired)
        self.assertEqual(record.action.value, desired)
        self.assertNotEqual(desired, original_top)
        self.assertEqual(clone.state.random.getstate(), rng_state)

    def test_default_fair_teacher_ignores_hidden_hands_deck_and_rng(self):
        from catanatron.players.value import ValueFunctionPlayer
        from catanLabs.Competitive.fair_value_agent import FairValueAgent

        config = json.loads((WORKSPACE / "catanLabs/Competitive/configs/fair_army_finish.json").read_text())
        game = self.Game([ValueFunctionPlayer(color) for color in self.colors], seed=20265220,
                         vps_to_win=10, discard_limit=7, friendly_robber=False,
                         number_placement="official_spiral")
        game.playable_actions.sort(key=canonical_action_key)
        changed_hidden_states = 0
        checked = 0
        for _ in range(120):
            if game.winning_color() is not None:
                break
            actor = game.state.current_color()
            legal = list(game.playable_actions)
            altered = game.copy()
            changed = False
            for color in game.state.colors:
                if color == actor:
                    continue
                prefix = f"P{game.state.color_to_index[color]}_"
                for names in (("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"),
                              ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")):
                    amounts = [altered.state.player_state[prefix + name + "_IN_HAND"] for name in names]
                    rotated = amounts[1:] + amounts[:1]
                    changed |= rotated != amounts
                    for name, amount in zip(names, rotated):
                        altered.state.player_state[prefix + name + "_IN_HAND"] = amount
            altered.state.development_listdeck.reverse()
            altered.state.random = random.Random(99991)
            altered.random = altered.state.random
            chosen = FairValueAgent(actor, config).decide(game, legal)
            changed_choice = FairValueAgent(actor, config).decide(altered, legal)
            self.assertEqual(chosen, changed_choice)
            changed_hidden_states += changed
            checked += 1
            game.execute(game.state.current_player().decide(game, legal), validate_action=True)
        self.assertGreater(checked, 30)
        self.assertGreater(changed_hidden_states, 0)

    def test_collector_replay_has_identical_arrays_and_schema(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            outputs = [Path(directory) / f"sample_{index}.npz" for index in range(2)]
            for output in outputs:
                completed = subprocess.run(
                    [sys.executable, str(WORKSPACE / "catanLabs/Transformer/collect.py"),
                     "--games", "1", "--max-examples", "8", "--wall-seconds", "30",
                     "--seed", "10031", "--output", str(output)],
                    cwd=WORKSPACE, capture_output=True, text=True, timeout=45,
                    env={**os.environ, "PYTHONHASHSEED": "0"}, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

            manifests = [json.loads(output.with_suffix(".json").read_text()) for output in outputs]
            self.assertEqual(manifests[0]["dataset_sha256"], manifests[1]["dataset_sha256"])
            self.assertEqual(manifests[0]["benchmark_protocol"],
                             "catanLabs-four-player-v5-canonical-actions-and-chance")
            with np.load(outputs[0]) as dataset:
                self.assertEqual(dataset["dataset_schema"].item(), manifests[0]["dataset_schema"])
                self.assertEqual(dataset["feature_schema"].item(), manifests[0]["feature_schema"])
                self.assertTrue(np.all(dataset["labels"] < dataset["candidate_counts"]))
                self.assertTrue(np.all(dataset["labels"] >= 0))
                self.assertGreater(len(dataset["labels"]), 0)


if __name__ == "__main__":
    unittest.main()
