"""Confidence gates retain the all-legal root planner and reject weak overrides."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE / "vendor/kit"))

from catanLabs.OneVOne.agent import DEFAULT_CONFIG
from catanLabs.OneVOne.action_planner import ACTION_DIM, ACTION_TYPES
from catanLabs.OneVOne.learning import (
    ADVANTAGE_MLP_VERSION,
    ADVANTAGE_MODEL_VERSION,
    LearningPlayer,
    POSITION_FEATURES,
    choose,
    heuristic_fingerprint,
    predict_advantages,
)
from catanatron.models.enums import ActionType
from test_action_learning import make_game, planner, prefer
from advantage_learning import MODEL_VERSION as FITTED_MODEL_VERSION, fit_advantage


def model_for(members, config=None, *, enabled=True, z=1.64, margin=0.0):
    if config is None:
        config = json.loads(DEFAULT_CONFIG.read_text())
    return {
        "version": ADVANTAGE_MLP_VERSION,
        "feature_scale": np.ones(ACTION_DIM).tolist(),
        "members": members,
        "heuristic_sha256": heuristic_fingerprint(config),
        "gate": {"enabled": enabled, "z": z, "margin": margin},
    }


def linear_model_for(weights, config=None, *, enabled=True, z=1.64, margin=0.0):
    if config is None:
        config = json.loads(DEFAULT_CONFIG.read_text())
    return {
        "version": ADVANTAGE_MODEL_VERSION,
        "ensemble_weights": np.asarray(weights, dtype=float).tolist(),
        "heuristic_sha256": heuristic_fingerprint(config),
        "gate": {"enabled": enabled, "z": z, "margin": margin},
    }


def mlp_member(action_type, sign=1.0):
    """A one-unit tanh MLP that gives a signed score to one action feature."""
    w1 = np.zeros((ACTION_DIM, 32))
    w1[2 * len(POSITION_FEATURES) + ACTION_TYPES.index(action_type.name), 0] = sign
    w2 = np.zeros(32)
    w2[0] = 1.0
    return {"w1": w1.tolist(), "b1": np.zeros(32).tolist(),
            "w2": w2.tolist(), "b2": 0.0}


def feature_for(action_type):
    vector = np.zeros(ACTION_DIM)
    vector[2 * len(POSITION_FEATURES) + ACTION_TYPES.index(action_type.name)] = 1.0
    return vector


class AdvantageLearningTests(unittest.TestCase):
    def test_ensemble_uses_same_legal_root_features_and_search(self):
        game = make_game()
        legal = list(game.playable_actions)
        heuristic_plan = planner(game).plan(game, legal)
        ensemble_plan = planner(game).plan(game, legal)

        self.assertEqual(heuristic_plan.actions, legal)
        self.assertEqual(ensemble_plan.actions, legal)
        self.assertEqual(heuristic_plan.trace, ensemble_plan.trace)
        self.assertTrue(any(row["followups"] for row in ensemble_plan.trace))
        self.assertTrue(any(row["followups"] for row in ensemble_plan.trace))
        np.testing.assert_array_equal(heuristic_plan.features, ensemble_plan.features)
        np.testing.assert_array_equal(heuristic_plan.heuristic, ensemble_plan.heuristic)
        np.testing.assert_array_equal(heuristic_plan.winning, ensemble_plan.winning)

        baseline, _ = choose(heuristic_plan)
        zero_members = [mlp_member(ActionType.END_TURN, 0.0) for _ in range(3)]
        selected, diagnostics = choose(
            ensemble_plan, model=model_for(zero_members, enabled=True))
        self.assertEqual(selected, baseline)
        self.assertFalse(diagnostics["gate_passed"])
        self.assertEqual(diagnostics["executed_index"], baseline)
        linear_selected, linear_diagnostics = choose(
            ensemble_plan,
            model=linear_model_for(np.zeros((3, ACTION_DIM)), enabled=True))
        self.assertEqual(linear_selected, baseline)
        self.assertFalse(linear_diagnostics["gate_passed"])

    def test_consensus_override_preserves_wins_and_rejects_wrong_config(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        model = model_for([mlp_member(ActionType.END_TURN) for _ in range(5)],
                          config, enabled=True, z=1.64, margin=0.0)

        game = make_game()
        legal = list(game.playable_actions)
        h_plan = planner(game).plan(game, legal)
        m_plan = planner(game).plan(game, legal)
        self.assertEqual(h_plan.actions, m_plan.actions)
        self.assertEqual(h_plan.trace, m_plan.trace)
        np.testing.assert_array_equal(h_plan.features, m_plan.features)
        heuristic_index = int(np.asarray(h_plan.heuristic).argmax())
        selected, diagnostics = choose(m_plan, model=model)
        self.assertEqual(m_plan.actions[heuristic_index].action_type, ActionType.BUILD_ROAD)
        self.assertEqual(m_plan.actions[selected].action_type.name, "END_TURN")
        self.assertTrue(diagnostics["gate_passed"])
        self.assertGreater(diagnostics["advantage_lcb"], 0.0)
        advantages = predict_advantages(model, m_plan.features, heuristic_index)
        np.testing.assert_array_equal(advantages[heuristic_index], np.zeros(len(model["members"])))
        pair = m_plan.features[[heuristic_index, selected]]
        forward = predict_advantages(model, pair, 0)[1]
        reverse = predict_advantages(model, pair, 1)[0]
        np.testing.assert_array_equal(forward, -reverse)

        linear_weights = prefer(ActionType.END_TURN, strength=20.0)
        linear_model = linear_model_for(np.tile(linear_weights, (5, 1)), config)
        linear_selected, linear_diagnostics = choose(m_plan, model=linear_model)
        self.assertEqual(linear_selected, selected)
        self.assertTrue(linear_diagnostics["gate_passed"])

        winning_game = make_game(target=3)
        winning_plan = planner(winning_game).plan(winning_game, winning_game.playable_actions)
        self.assertTrue(winning_plan.winning.any())
        winning_index, winning_diagnostics = choose(winning_plan, model=model)
        self.assertTrue(winning_plan.winning[winning_index])
        self.assertTrue(winning_diagnostics["winning_override"])
        color = winning_game.state.current_color()
        winning_game.execute(winning_plan.actions[winning_index])
        if winning_game.winning_color() is None:
            followup = winning_plan.trace[winning_index]["followup"]
            self.assertIsNotNone(followup)
            winning_game.execute(next(a for a in winning_game.playable_actions
                                      if [a.action_type.name, repr(a.value)] == followup))
        self.assertEqual(winning_game.winning_color(), color)

        wrong_config = json.loads(DEFAULT_CONFIG.read_text())
        wrong_config["value_weights"]["reachable_production_0"] += 0.01
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "advantage.json"
            path.write_text(json.dumps(model))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                LearningPlayer(winning_game.state.current_color(), wrong_config, model=path)
            invalid = json.loads(json.dumps(model))
            invalid["members"][0]["w1"] = [[0.0] * 32]
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "dimensions"):
                LearningPlayer(winning_game.state.current_color(), config, model=path)
            invalid["members"][0]["w1"] = np.zeros((ACTION_DIM, 32)).tolist()
            invalid["members"][0]["w1"][0][0] = float("nan")
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                LearningPlayer(winning_game.state.current_color(), config, model=path)

    def test_uncertain_disabled_and_small_calibration_keep_heuristic(self):
        game = make_game()
        plan = planner(game).plan(game, game.playable_actions)
        uncertain = [mlp_member(ActionType.END_TURN, sign)
                     for sign in (1.0, -1.0, 1.0, -1.0, 0.0)]
        heuristic, _ = choose(plan)
        selected, diagnostics = choose(plan, model=model_for(uncertain, enabled=True))
        self.assertEqual(selected, heuristic)
        self.assertFalse(diagnostics["gate_passed"])
        self.assertLessEqual(diagnostics["advantage_lcb"], 0.0)
        linear_uncertain = np.stack([prefer(ActionType.END_TURN),
                                     -prefer(ActionType.END_TURN),
                                     prefer(ActionType.END_TURN)])
        selected, diagnostics = choose(
            plan, model=linear_model_for(linear_uncertain, enabled=True))
        self.assertEqual(selected, heuristic)
        self.assertFalse(diagnostics["gate_passed"])

        selected, diagnostics = choose(
            plan, model=model_for([mlp_member(ActionType.END_TURN) for _ in range(5)],
                                  enabled=False))
        self.assertEqual(selected, heuristic)
        self.assertFalse(diagnostics["gate_passed"])
        selected, diagnostics = choose(
            plan, model=linear_model_for(
                np.tile(prefer(ActionType.END_TURN), (5, 1)), enabled=False))
        self.assertEqual(selected, heuristic)
        self.assertFalse(diagnostics["gate_passed"])

        end_feature = feature_for(ActionType.END_TURN)
        zero_feature = np.zeros(ACTION_DIM)
        train_pairs = [
            {"board_seed": seed, "features_i": end_feature.tolist(),
             "features_j": zero_feature.tolist(), "outcome_i": 1.0, "outcome_j": 0.0}
            for seed in (11, 12)
        ]

        def calibration_position(seed, candidate_outcome):
            return {
                "board_seed": seed, "heuristic_index": 0,
                "all_features": np.stack([zero_feature, end_feature]).tolist(),
                "alternatives": [
                    {"index": 0, "features": zero_feature.tolist(),
                     "trials": [{"outcome": 0.0, "rollout_seed": seed * 10 + k}
                                for k in range(3)]},
                    {"index": 1, "features": end_feature.tolist(),
                     "trials": [{"outcome": candidate_outcome, "rollout_seed": seed * 10 + k}
                                for k in range(3)]},
                ],
            }

        calibration_pairs = [dict(row, outcome_i=0.0) for row in train_pairs]
        base_rows = {"pairs": train_pairs, "positions": [
            calibration_position(21, 1.0), calibration_position(22, 0.0)]}
        altered_rows = {"pairs": train_pairs + [
            dict(row, board_seed=seed) for row, seed in zip(calibration_pairs, (21, 22))],
            "positions": [calibration_position(21, 0.0), calibration_position(22, 1.0)]}
        first, report = fit_advantage(base_rows, {11, 12}, {21, 22}, DEFAULT_CONFIG.read_text())
        second, _ = fit_advantage(altered_rows, {11, 12}, {21, 22}, DEFAULT_CONFIG.read_text())
        self.assertEqual(first["version"], FITTED_MODEL_VERSION)
        self.assertEqual(first["version"], ADVANTAGE_MLP_VERSION)
        np.testing.assert_array_equal(first["feature_scale"], second["feature_scale"])
        for member_a, member_b in zip(first["members"], second["members"]):
            for name in ("w1", "b1", "w2", "b2"):
                np.testing.assert_array_equal(member_a[name], member_b[name])
        self.assertFalse(first["gate"]["enabled"])
        self.assertEqual(report["gate_calibration"]["overrides"], 0)
        self.assertEqual(report["gate_calibration"]["reason"], "insufficient calibration evidence")


if __name__ == "__main__":
    unittest.main()
