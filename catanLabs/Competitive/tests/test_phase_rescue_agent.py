"""Focused tests for isolated PLAY_TURN and DISCARD interventions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanLabs.Competitive.phase_rescue_agent import PhaseFairRescuePolicy, load_config
from catanLabs.Competitive.phase_rescue_eval import (
    ARMS, COMPOSITIONS, assert_fresh_boards, persist_partial, validate_assets,
    validate_matched_grid,
)


@dataclass(frozen=True)
class Action:
    kind: str


class Model:
    def __init__(self, index):
        self.index = index
        self.calls = 0

    def decide_visible(self, visible):
        self.calls += 1
        return self.index


class Fair:
    def __init__(self, action):
        self.action = action
        self.calls = 0

    def decide(self, game, legal):
        self.calls += 1
        return self.action


def game(phase):
    return SimpleNamespace(state=SimpleNamespace(current_prompt=SimpleNamespace(name=phase)))


class PhaseRescueTests(unittest.TestCase):
    def test_each_arm_only_overrides_its_selected_phase(self):
        first, second = Action("first"), Action("second")
        for arm, eligible in (("play_turn", "PLAY_TURN"), ("discard", "DISCARD")):
            with self.subTest(arm=arm):
                model, fair = Model(0), Fair(second)
                agent = PhaseFairRescuePolicy(model, fair, arm,
                                              snapshot_fn=lambda game, legal: object())
                for other in ("PLAY_TURN", "DISCARD", "MOVE_ROBBER"):
                    choice = agent.decide(game(other), (first, second))
                    self.assertEqual(choice, second if other == eligible else first)
                self.assertEqual(model.calls, 3)
                self.assertEqual(fair.calls, 1)
                self.assertEqual(agent.summary()["overrides"], 1)
                self.assertEqual(set(agent.summary()["by_phase"]),
                                 {"PLAY_TURN", "DISCARD", "MOVE_ROBBER"})

    def test_legal_choices_and_forced_action(self):
        first, second = Action("first"), Action("second")
        model, fair = Model(0), Fair(second)
        agent = PhaseFairRescuePolicy(model, fair, "discard",
                                      snapshot_fn=lambda game, legal: object())
        self.assertEqual(agent.decide(game("DISCARD"), (first,)), first)
        self.assertEqual((model.calls, fair.calls), (0, 0))
        bad_model = PhaseFairRescuePolicy(Model(2), fair, "discard",
                                          snapshot_fn=lambda game, legal: object())
        with self.assertRaisesRegex(ValueError, "illegal action index"):
            bad_model.decide(game("DISCARD"), (first, second))
        bad_fair = PhaseFairRescuePolicy(Model(0), Fair(Action("missing")), "discard",
                                         snapshot_fn=lambda game, legal: object())
        with self.assertRaisesRegex(ValueError, "illegal action"):
            bad_fair.decide(game("DISCARD"), (first, second))

    def test_frozen_configs_and_seed_gate(self):
        assets = validate_assets()
        self.assertEqual(assets["model_sha256"],
                         "29b0f7418a873f34a02157dbde10b7f28f4d47ef2356d74076b55173efef3bbd")
        self.assertEqual(assets["fair_sha256"],
                         "5f908a453c87556463170378ef92d4ce79f03ecfdf0f1c304daae93fd3222ab3")
        for arm in ("play_turn", "discard"):
            self.assertEqual(load_config(WORKSPACE / "catanLabs/Competitive/configs" /
                                         f"phase_rescue_{arm}.json")["intervention"], arm)
        with self.assertRaisesRegex(ValueError, "overlap frozen dataset split"):
            assert_fresh_boards(20269012, 1)
        self.assertEqual(len(assert_fresh_boards(20273500, 8)), 64)

    def test_partial_is_atomic_and_final_grid_is_matched(self):
        report = {"protocol": "test", "rules": {}, "candidate": {}, "first_seed": 19,
                  "boards_per_mix": 1, "split_file_sha256": "split", "runner_sha256": "runner"}
        row = {"seed": 19, "seat": 0, "status": "completed", "won": False,
               "candidate_vp": 7}
        with TemporaryDirectory() as directory:
            output = Path(directory)
            persist_partial(output, report, "discard", "mixed_diverse", [row], [1.0],
                            Counter(overrides=1), 4)
            path = output / "partial/mixed_diverse__discard.json"
            payload = json.loads(path.read_text())
            self.assertEqual(payload["schema"], "catanlabs-phase-rescue-partial-v1")
            self.assertEqual((payload["attempts_recorded"], payload["expected_games"]), (1, 4))
            self.assertFalse((output / "discard.json").exists())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
        rows = [{"seed": 19, "seat": seat, "color": str(seat),
                 "opponent_assignments": {"other": "value"}} for seat in range(4)]
        reports = {arm: {"tiers": {mix: {"games": [dict(row) for row in rows]}
                                   for mix in COMPOSITIONS}} for arm in ARMS}
        validate_matched_grid(reports, 1, 19)
        reports["discard"]["tiers"]["mixed_strong"]["games"].pop()
        with self.assertRaisesRegex(RuntimeError, "Incomplete board/seat grid"):
            validate_matched_grid(reports, 1, 19)


if __name__ == "__main__":
    unittest.main()
