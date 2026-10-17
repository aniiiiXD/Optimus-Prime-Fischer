"""A learned leaf preference changes a real root without changing search."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import random

import numpy as np

from test_action_learning import make_game
from catanLabs.OneVOne.action_planner import ACTION_DIM, ACTION_TYPES
from catanLabs.OneVOne.learning import heuristic_fingerprint
from catanLabs.OneVOne.lookahead_value import VERSION, LookaheadPlayer, load_model, select
from lookahead_learning import calibrate
from advantage_learning import _rollout
from catanLabs.OneVOne.reply_search import check_replies, next_turn_value
from my_team.nano import Agent as Nano, AT


CONFIG = {"planner": "nano", "nano": {"search_depth": 4}}


def model_for(action_type):
    w1, w2 = np.zeros((ACTION_DIM, 32)), np.zeros(32)
    w1[40 + ACTION_TYPES.index(action_type), 0] = 1.
    w2[0] = .8
    member = {"w1": w1.tolist(), "b1": np.zeros(32).tolist(), "w2": w2.tolist(), "b2": .1}
    return {"version": VERSION, "config_sha256": heuristic_fingerprint(CONFIG),
            "scale": np.ones(ACTION_DIM).tolist(), "members": [member.copy() for _ in range(3)],
            "gate": {"enabled": True, "z": 1., "margin": 0.}}


class LookaheadValueTests(unittest.TestCase):
    def test_inactive_inference_skips_features_and_matches_raw_nano(self):
        game = make_game()
        color = game.state.current_color()
        player = LookaheadPlayer(color, CONFIG)
        from catan_kit.obs import Obs
        view = Obs(game, color)
        view.cpu_left = .5
        baseline = Nano(CONFIG["nano"])
        expected = game.playable_actions[baseline.decide(view, game.playable_actions)]
        with patch.object(player.planner, "plan", side_effect=AssertionError("unneeded features")):
            self.assertEqual(player.decide(game, game.playable_actions, cpu_left=.5, capture=False), expected)
            player.model = model_for("END_TURN")
            player.model["gate"]["enabled"] = False
            self.assertEqual(player.decide(game, game.playable_actions, cpu_left=.5, capture=False), expected)

    def test_default_gate_can_override_across_types_without_altering_search(self):
        game = make_game(seed=20261020)
        player = LookaheadPlayer(game.state.current_color(), CONFIG)
        actions = list(game.playable_actions)
        heuristic_action = player.decide(game, actions)
        trace = player.plan.trace
        features = [o["features"].copy() for o in player.options]
        self.assertEqual({o["root"] for o in player.options}, set(range(len(actions))))
        self.assertEqual(len(player.options), len(actions))
        self.assertTrue(any(len(o["line"]) > 1 for o in player.options))
        self.assertTrue(all(len(o["line"]) <= 4 for o in player.options))
        self.assertNotEqual(heuristic_action.action_type.name, "END_TURN")
        player.model = model_for("END_TURN")
        learned_action = player.decide(game, actions)
        self.assertEqual(learned_action.action_type.name, "END_TURN")
        self.assertTrue(player.last_decision["ml_changed"])
        self.assertTrue(player.last_decision["cross_type"])
        self.assertEqual(player.plan.trace, trace)
        for before, option in zip(features, player.options):
            np.testing.assert_array_equal(before, option["features"])
        player.model["gate"]["enabled"] = False
        self.assertEqual(player.decide(game, actions), heuristic_action)

    def test_verified_win_and_model_boundaries(self):
        game = make_game(seed=20261022, target=3)
        player = LookaheadPlayer(game.state.current_color(), CONFIG)
        player.model = model_for("END_TURN")
        action = player.decide(game, game.playable_actions)
        self.assertTrue(player.last_decision["winning_override"])
        game.execute(action)
        self.assertEqual(game.winning_color(), player.color)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model = model_for("END_TURN")
            path.write_text(json.dumps(model))
            load_model(path, CONFIG)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_model(path, {"planner": "nano", "nano": {"search_depth": 1}})
            model["members"][0]["w2"] = [float("nan")] * 32
            path.write_text(json.dumps(model))
            with self.assertRaisesRegex(ValueError, "Invalid lookahead w2"):
                load_model(path, CONFIG)

    def test_tied_rollout_labels_cannot_enable_a_gate(self):
        game = make_game()
        player = LookaheadPlayer(game.state.current_color(), CONFIG)
        player.decide(game, game.playable_actions)
        model = model_for("END_TURN")
        _, diagnostic = select(player.plan, player.options, model)
        baseline = int(np.argmax(player.plan.heuristic))
        base_option = next(j for j, o in enumerate(player.options) if o["root"] == baseline)
        position = {"heuristic": player.plan.heuristic.tolist(), "baseline": baseline,
                    "action_types": [a.action_type.name for a in player.plan.actions],
                    "options": [{"root": o["root"], "features": o["features"].tolist()} for o in player.options],
                    "trials": [{"option_index": j, "root": player.options[j]["root"], "target": .5}
                               for j in (base_option, diagnostic["leaf_index"])]}
        rows = [dict(position, board_seed=i//2) for i in range(8)]
        members_before = json.dumps(model["members"])
        report = calibrate(model, rows)
        self.assertFalse(report["accepted"])
        self.assertFalse(model["gate"]["enabled"])
        self.assertEqual(json.dumps(model["members"]), members_before)

    def test_unmeasured_choices_cannot_enable_a_gate(self):
        game = make_game()
        player = LookaheadPlayer(game.state.current_color(), CONFIG)
        player.decide(game, game.playable_actions)
        model = model_for("END_TURN")
        chosen, _ = select(player.plan, player.options, model)
        baseline = int(np.argmax(player.plan.heuristic))
        row = {"heuristic": player.plan.heuristic.tolist(), "baseline": baseline,
               "action_types": [a.action_type.name for a in player.plan.actions],
               "options": [{"root": o["root"], "features": o["features"].tolist()} for o in player.options],
               "trials": [{"root": baseline, "target": 0.}, {"root": chosen, "target": 1.}]}
        rows = [dict(row, board_seed=i) for i in range(8)]
        rows[-1] = dict(rows[-1], trials=[{"root": baseline, "target": 0.}])
        report = calibrate(model, rows)
        self.assertFalse(report["accepted"])
        self.assertTrue(all(r["unmeasured_overrides"] == 1 for r in report["trials"]))

    def test_root_rollouts_use_the_deployed_replanning_player(self):
        game = make_game(target=3)
        action = next(a for a in game.playable_actions if a.action_type == AT.END_TURN)
        with patch("catanLabs.OneVOne.lookahead_value.LookaheadPlayer", wraps=LookaheadPlayer) as player:
            with patch("advantage_learning.LearningPlayer", side_effect=AssertionError("wrong policy")):
                result = _rollout(game, action, game.state.current_color(), CONFIG, 42421, lookahead=True)
        player.assert_called_once()
        self.assertIsNotNone(result["outcome"], result)

    def test_wider_search_is_explicit_and_preserves_legal_roots(self):
        game = make_game()
        color = game.state.current_color()
        default = LookaheadPlayer(color, CONFIG)
        explicit = LookaheadPlayer(color, dict(CONFIG, nano={"search_depth": 4, "beam_width": 4}))
        self.assertEqual(default.decide(game, game.playable_actions), explicit.decide(game, game.playable_actions))
        wide = LookaheadPlayer(color, dict(CONFIG, nano={"search_depth": 4, "beam_width": 8}))
        self.assertIn(wide.decide(game, game.playable_actions), game.playable_actions)
        self.assertEqual(len(wide.options), len(game.playable_actions))
        self.assertTrue(all(len(o["line"]) <= 4 for o in wide.options))
        for value in (0, True, 17, 4.5):
            with self.assertRaises(ValueError):
                Nano({"beam_width": value})

    def test_reply_check_is_bounded_and_respects_clock_reserve(self):
        game = make_game()
        color = game.state.current_color()
        game.state.num_turns = 12
        game.state.player_state[f"P{game.state.color_to_index[color]}_ACTUAL_VICTORY_POINTS"] = 3
        player = LookaheadPlayer(color, dict(CONFIG, reply_check=True))
        with patch("catanLabs.OneVOne.reply_search.check_replies", side_effect=lambda g,p,i,c,cfg: (i, {"reply_checked": True})) as check:
            player.decide(game, game.playable_actions, cpu_left=.15)
            check.assert_not_called()
            player.decide(game, game.playable_actions, cpu_left=.5)
            player.decide(game, game.playable_actions, cpu_left=.5)
            self.assertEqual(check.call_count, 1)

    def test_incomplete_reply_samples_preserve_the_incumbent(self):
        game = make_game()
        player = LookaheadPlayer(game.state.current_color(), CONFIG)
        incumbent = player.decide(game, game.playable_actions)
        index = player.plan.actions.index(incumbent)
        with patch("catanLabs.OneVOne.reply_search.next_turn_value", return_value=None):
            chosen, diagnostic = check_replies(game, player.plan, index, player.color, CONFIG)
        self.assertEqual(chosen, index)
        self.assertFalse(diagnostic["reply_complete"])

    def test_reply_sampling_masks_hidden_cards_and_preserves_randomness(self):
        game = make_game()
        color = game.state.current_color()
        enemy = next(c for c in game.state.colors if c != color)
        action = next(a for a in game.playable_actions if a.action_type == AT.END_TURN)
        results = []
        state = random.getstate()
        for card in ("KNIGHT", "VICTORY_POINT"):
            world = game.copy()
            prefix = f"P{world.state.color_to_index[enemy]}_"
            world.state.development_listdeck.remove(card)
            world.state.player_state[prefix + card + "_IN_HAND"] = 1
            world.state.player_state[prefix + card + "_OWNED_AT_START"] = 1
            world.state.player_state[prefix + "ACTUAL_VICTORY_POINTS"] += int(card == "VICTORY_POINT")
            results.append(next_turn_value(world, action, color, CONFIG, 881))
            self.assertEqual(random.getstate(), state)
        self.assertIsNotNone(results[0])
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
