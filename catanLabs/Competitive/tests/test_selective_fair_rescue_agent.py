"""Focused tests of the three frozen-policy intervention triggers."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanLabs.Competitive.selective_fair_rescue_agent import (
    SelectiveFairRescuePolicy, load_config,
)
from catanLabs.Competitive.selective_rescue_eval import (
    assert_fresh_boards, persist_partial, validate_matched_grid,
)


@dataclass(frozen=True)
class Action:
    action_type: object
    value: object = None


def action(kind, value=None):
    return Action(SimpleNamespace(name=kind), value)


class Model:
    def __init__(self, index):
        self.index = index
        self.received = []

    def decide_visible(self, visible):
        self.received.append(visible)
        return self.index


class Fair:
    def __init__(self, choice):
        self.choice = choice
        self.calls = 0

    def decide(self, game, legal):
        self.calls += 1
        return self.choice


def policy(mode, model_index, fair_choice):
    visible = object()
    model = Model(model_index)
    fair = Fair(fair_choice)
    agent = SelectiveFairRescuePolicy(model, fair, mode,
                                      snapshot_fn=lambda game, legal: visible)
    return agent, model, fair, visible


def game(phase):
    return SimpleNamespace(state=SimpleNamespace(current_prompt=SimpleNamespace(name=phase)))


class SelectiveFairRescueTests(unittest.TestCase):
    def test_opening_and_robber_rescue_only_their_phases(self):
        first = action("BUILD_SETTLEMENT", 1)
        second = action("BUILD_SETTLEMENT", 2)
        for mode, phase in (("opening", "BUILD_INITIAL_SETTLEMENT"),
                            ("opening", "BUILD_INITIAL_ROAD"),
                            ("robber", "MOVE_ROBBER")):
            with self.subTest(mode=mode, phase=phase):
                agent, model, fair, visible = policy(mode, 0, second)
                self.assertEqual(agent.decide(game(phase), (first, second)), second)
                self.assertEqual(model.received, [visible])
                self.assertEqual(fair.calls, 1)
                self.assertEqual(agent.summary()["overrides"], 1)
                self.assertEqual(agent.decide(game("PLAY_TURN"), (first, second)), first)
                self.assertEqual(fair.calls, 1)

    def test_trade_end_switches_only_cross_type_disagreement(self):
        end = action("END_TURN")
        trade_a = action("MARITIME_TRADE", ("SHEEP", "BRICK"))
        trade_b = action("MARITIME_TRADE", ("WOOD", "ORE"))
        road = action("BUILD_ROAD", (1, 2))
        cases = (
            ((end, trade_a), 0, trade_a, trade_a, 1),
            ((trade_a, end), 0, end, end, 1),
            ((trade_a, trade_b), 0, trade_b, trade_a, 0),
            ((road, end), 0, end, road, 0),
        )
        for legal, model_index, fair_choice, expected, overrides in cases:
            with self.subTest(legal=legal):
                agent, _, fair, _ = policy("trade_end", model_index, fair_choice)
                self.assertEqual(agent.decide(game("PLAY_TURN"), legal), expected)
                self.assertEqual(fair.calls, 1)
                self.assertEqual(agent.summary()["overrides"], overrides)
                self.assertEqual(agent.summary()["disagreed"], 1)
                self.assertEqual(agent.decide(game("MOVE_ROBBER"), legal), legal[0])
                self.assertEqual(fair.calls, 1)

    def test_forced_choice_skips_both_policies_and_bad_choices_fail(self):
        only = action("END_TURN")
        agent, model, fair, _ = policy("trade_end", 0, only)
        self.assertEqual(agent.decide(game("PLAY_TURN"), (only,)), only)
        self.assertEqual(model.received, [])
        self.assertEqual(fair.calls, 0)
        self.assertEqual(agent.summary()["forced"], 1)
        bad_model, _, _, _ = policy("opening", 2, only)
        with self.assertRaisesRegex(ValueError, "illegal action index"):
            bad_model.decide(game("BUILD_INITIAL_ROAD"), (only, action("BUILD_ROAD")))
        bad_fair, _, _, _ = policy("opening", 0, action("BUILD_ROAD", (9, 9)))
        with self.assertRaisesRegex(ValueError, "illegal action"):
            bad_fair.decide(game("BUILD_INITIAL_ROAD"), (only, action("BUILD_ROAD")))

    def test_three_configs_pin_same_frozen_assets(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        found = {mode: load_config(root / f"selective_rescue_{mode}.json")
                 for mode in ("opening", "robber", "trade_end")}
        self.assertEqual({config["model_checkpoint_sha256"] for config in found.values()},
                         {"29b0f7418a873f34a02157dbde10b7f28f4d47ef2356d74076b55173efef3bbd"})
        self.assertEqual(len({config["fair_config_sha256"] for config in found.values()}), 1)

    def test_runner_rejects_sealed_dataset_board(self):
        with self.assertRaisesRegex(ValueError, "overlap frozen dataset split"):
            assert_fresh_boards(20269012, 1)
        self.assertEqual(len(assert_fresh_boards(20273500, 8)), 64)

    def test_partial_report_survives_each_game_without_becoming_final(self):
        report = {"protocol": "v5-test", "rules": {"target_vp": 10},
                  "candidate": {"type": "opening"}, "first_seed": 17,
                  "boards_per_mix": 1, "split_file_sha256": "split",
                  "runner_sha256": "runner"}
        row = {"seed": 17, "seat": 0, "status": "completed",
               "won": False, "candidate_vp": 8}
        with TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "partial" / "mixed_diverse__opening.json"
            for count in (1, 2):
                persist_partial(output, report, "opening", "mixed_diverse",
                                [row] * count, [1.0], Counter(overrides=count), 4)
                saved = json.loads(path.read_text())
                self.assertEqual(saved["status"], "unverified_partial")
                self.assertEqual(saved["attempts_recorded"], count)
                self.assertEqual(saved["expected_games"], 4)
                self.assertEqual(saved["tier"]["rescue_summary"]["overrides"], count)
                self.assertFalse(path.with_name(path.name + ".tmp").exists())
                self.assertFalse((output / "opening.json").exists())

    def test_final_grid_rejects_missing_and_mismatched_assignments(self):
        from catanLabs.Competitive.selective_rescue_eval import ARMS, COMPOSITIONS
        rows = [{"seed": 17, "seat": seat, "color": str(seat),
                 "opponent_assignments": {"other": "value"}} for seat in range(4)]
        reports = {arm: {"tiers": {mix: {"games": [dict(row) for row in rows]}
                                   for mix in COMPOSITIONS}} for arm in ARMS}
        validate_matched_grid(reports, 1, 17)
        reports["opening"]["tiers"]["mixed_diverse"]["games"][0]["opponent_assignments"] = {"other": "weighted"}
        with self.assertRaisesRegex(RuntimeError, "assignments differ"):
            validate_matched_grid(reports, 1, 17)
        reports["opening"]["tiers"]["mixed_diverse"]["games"].pop()
        with self.assertRaisesRegex(RuntimeError, "Incomplete board/seat grid"):
            validate_matched_grid(reports, 1, 17)


if __name__ == "__main__":
    unittest.main()
