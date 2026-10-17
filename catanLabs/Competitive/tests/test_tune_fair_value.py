"""Guard the fixed policy-search design before any tuning games are run."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanatron.players.value import DEFAULT_WEIGHTS
from catanLabs.Competitive import tune_fair_value as tuner


def report(wins: tuple[bool, bool], vp: tuple[int, int], trace=("setup", ("BUILD", "1", "RED"))):
    return {"tiers": {mix: {"games": [
        {"seed": 10, "seat": seat, "status": "completed", "color": "RED",
         "opponent_assignments": {"BLUE": "value"}, "won": wins[seat],
         "candidate_vp": vp[seat], "decision_trace": [trace]}
        for seat in range(2)]}
        for mix in tuner.COMPOSITIONS}}


class FixedPolicySearchTests(unittest.TestCase):
    def test_fixed_variants_leave_incumbent_unchanged_and_change_four_weights(self):
        base = json.loads(tuner.BASE_CONFIG.read_text())
        original = tuner.canonical_bytes(base)
        configs = tuner.configs_from_base(base, DEFAULT_WEIGHTS)
        self.assertEqual(tuner.canonical_bytes(base), original)
        self.assertEqual(configs["fair"], base)
        self.assertEqual(len(configs), 7)
        self.assertEqual(len({tuner.canonical_bytes(config) for config in configs.values()}), 7)
        self.assertEqual(configs["reach_army_up"]["value_weights"]["reachable_production_0"], 150000)
        self.assertEqual(configs["reach_army_down"]["value_weights"]["army_size"], 15000)
        self.assertEqual(configs["reach_down_sites_up"]["value_weights"]["buildable_nodes"], 15000)
        self.assertEqual(configs["hand_up_army_down"]["value_weights"]["hand_synergy"], 1500)

    def test_wins_are_primary_and_incomplete_games_cannot_rank(self):
        fair = report((False, False), (9, 9))
        more_wins = report((True, False), (3, 3))
        more_vp = report((False, False), (10, 10))
        ranked = tuner.rank_variants({"fair": fair, "reach_army_up": more_wins,
                                      "reach_army_down": more_vp})
        self.assertEqual(ranked[0]["name"], "reach_army_up")
        self.assertEqual(ranked[0]["mean_win_rate_difference"], 0.5)
        more_wins["tiers"]["mixed_strong"]["games"][0]["status"] = "capped"
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            tuner.objective(more_wins, fair)

    def test_divergence_reports_first_shared_trajectory_change(self):
        fair = report((False, False), (5, 5))
        changed = report((False, False), (5, 5), ("early", ("ROLL", "2", "RED")))
        result = tuner.divergence_by_phase(changed, fair)
        self.assertEqual(result["early"], 4)
        self.assertEqual(result["total_games"], 4)
        self.assertEqual(result["first_action_divergence_fraction"], 1.0)

    def test_cpu_or_wall_budget_stops_and_atomic_partial_replaces(self):
        with self.assertRaises(tuner.BudgetExceeded):
            tuner.check_budget({"work_deadline": time.perf_counter() + 10,
                                "cpu_deadline": time.process_time() - 1})
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "partial.json"
            tuner.write_json_atomic(file, {"games": [1]})
            tuner.write_json_atomic(file, {"games": [1, 2]})
            self.assertEqual(json.loads(file.read_text()), {"games": [1, 2]})
            self.assertFalse(file.with_name(file.name + ".tmp").exists())

    def test_tuning_and_smoke_seeds_are_disjoint_from_frozen_split(self):
        digest = tuner.assert_fresh_seeds((tuner.SMOKE, tuner.STAGE1, tuner.STAGE2))
        self.assertEqual(len(digest), 64)
        with self.assertRaisesRegex(ValueError, "overlap"):
            tuner.assert_fresh_seeds((tuner.STAGE1, tuner.STAGE1))


if __name__ == "__main__":
    unittest.main()
