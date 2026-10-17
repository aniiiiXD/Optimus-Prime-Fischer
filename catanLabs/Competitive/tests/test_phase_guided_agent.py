"""Phase gates must route only their chosen phase and preserve legal actions."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanLabs.Competitive.phase_guided_agent import PhaseGuidedPolicy
from catanLabs.Competitive.phase_guided_compare import summarize


class Stub:
    def __init__(self, choice):
        self.choice = choice
        self.calls = []

    def decide(self, game, legal):
        self.calls.append(game.state.current_prompt.name)
        return self.choice


def game(phase):
    return SimpleNamespace(state=SimpleNamespace(
        current_prompt=SimpleNamespace(name=phase)))


class PhaseGuidedTests(unittest.TestCase):
    def test_opening_only_routes_both_setup_prompts(self):
        fair_choice, guided_choice = object(), object()
        fair, guided = Stub(fair_choice), Stub(guided_choice)
        policy = PhaseGuidedPolicy(guided, fair, "opening")
        for phase in ("BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD"):
            self.assertIs(policy.decide(game(phase), (fair_choice, guided_choice)), guided_choice)
        for phase in ("PLAY_TURN", "DISCARD", "MOVE_ROBBER"):
            self.assertIs(policy.decide(game(phase), (fair_choice, guided_choice)), fair_choice)
        self.assertEqual(guided.calls, ["BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD"])
        self.assertEqual(fair.calls, ["PLAY_TURN", "DISCARD", "MOVE_ROBBER"])

    def test_turn_only_and_all_modes(self):
        fair_choice, guided_choice = object(), object()
        fair, guided = Stub(fair_choice), Stub(guided_choice)
        turn = PhaseGuidedPolicy(guided, fair, "turn")
        self.assertIs(turn.decide(game("PLAY_TURN"), (fair_choice, guided_choice)), guided_choice)
        self.assertIs(turn.decide(game("BUILD_INITIAL_SETTLEMENT"),
                                  (fair_choice, guided_choice)), fair_choice)
        self.assertIs(turn.decide(game("MOVE_ROBBER"), (fair_choice, guided_choice)), fair_choice)
        all_phases = PhaseGuidedPolicy(guided, fair, "all")
        for phase in ("BUILD_INITIAL_ROAD", "PLAY_TURN", "DISCARD", "MOVE_ROBBER"):
            self.assertIs(all_phases.decide(game(phase), (fair_choice, guided_choice)),
                          guided_choice)

    def test_forced_and_illegal_choice_checks(self):
        only, other = object(), object()
        fair, guided = Stub(other), Stub(other)
        policy = PhaseGuidedPolicy(guided, fair, "all")
        self.assertIs(policy.decide(game("PLAY_TURN"), (only,)), only)
        self.assertEqual(guided.calls, [])
        with self.assertRaisesRegex(ValueError, "illegal action"):
            policy.decide(game("PLAY_TURN"), (only, object()))
        with self.assertRaisesRegex(ValueError, "No legal"):
            policy.decide(game("PLAY_TURN"), ())
        with self.assertRaisesRegex(ValueError, "Invalid guided phase mode"):
            PhaseGuidedPolicy(guided, fair, "unknown")

    def test_configs_share_frozen_fair_and_guided_parameters(self):
        configs = WORKSPACE / "catanLabs/Competitive/configs"
        rows = {mode: json.loads((configs / f"phase_guided_{mode}.json").read_text())
                for mode in ("opening", "turn", "all")}
        self.assertEqual({row["fair_config"] for row in rows.values()},
                         {"fair_army_finish.json"})
        self.assertEqual({row["guided_config"] for row in rows.values()},
                         {"guided_search_3.json"})
        self.assertEqual({row["mode"] for row in rows.values()}, set(rows))

    def test_comparison_rejects_mismatched_opponents(self):
        def report():
            rows = [{"seed": 17, "seat": seat, "color": str(seat),
                     "opponent_assignments": {"RED": "value"},
                     "status": "completed", "won": False, "candidate_vp": 8}
                    for seat in range(4)]
            return {"protocol": "v5", "rules": {"target_vp": 10},
                    "boards_per_mix": 1,
                    "tiers": {"mixed_strong": {"games": rows,
                            "summary": {"wins": 0, "win_rate_all_attempts": 0.0,
                                        "mean_final_vp": 8.0, "decision_latency_ms": {}}}}}
        reports = {name: report() for name in ("opening", "turn", "all", "fair")}
        reports["turn"]["tiers"]["mixed_strong"]["games"][0][
            "opponent_assignments"] = {"RED": "alphabeta1"}
        with self.assertRaisesRegex(ValueError, "Unmatched opponent assignment"):
            summarize(reports)


if __name__ == "__main__":
    unittest.main()
