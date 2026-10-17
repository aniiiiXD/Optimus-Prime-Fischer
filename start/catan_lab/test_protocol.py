"""Checks for misleading evaluation accounting and observation leakage."""
import importlib.util
from pathlib import Path
import sys
import time
import unittest

import numpy as np

from catan_lab.worker import ROOT, finish, wilson


class AccountingTests(unittest.TestCase):
    def test_capped_games_remain_in_all_attempts_denominator(self):
        games = [{"seat": 0, "winner": 0, "student_won": True},
                 {"seat": 1, "winner": None, "student_won": False}]
        report = finish({}, games, [], time.perf_counter() - 0.01, time.process_time())
        self.assertEqual(report["win_rate_completed"], 1.0)
        self.assertEqual(report["win_rate_all_attempts"], 0.5)
        self.assertEqual(report["capped_games"], 1)
        self.assertEqual(report["by_seat"]["1"]["completed"], 0)

    def test_all_wins_do_not_imply_certainty(self):
        lower, upper = wilson(16, 16)
        self.assertLess(lower, 0.81)
        self.assertAlmostEqual(upper, 1.0)
        self.assertIsNone(wilson(0, 0))


class NativeObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "artifacts/native"))
        if importlib.util.find_spec("catan_py") is None:
            raise unittest.SkipTest("Run python -m catan_lab.setup_runtime first")
        import catan_py
        cls.engine = catan_py

    def test_realistic_view_redacts_only_the_opponent_private_block(self):
        self.assertEqual(self.engine.OBS_VERSION, 1, "Update audit offsets for a new schema")
        public = np.ones(self.engine.OBS_DIM, dtype=bool)
        public[1279:1309] = False
        perfect = self.engine.Env(seed=944, visibility="perfect")
        honest = self.engine.Env(seed=944, visibility="realistic")
        rng = np.random.default_rng(944)
        witnessed_hidden_cards = False
        for _ in range(300):
            a, b = perfect.obs(), honest.obs()
            np.testing.assert_array_equal(a[public], b[public])
            np.testing.assert_array_equal(b[~public], 0)
            witnessed_hidden_cards |= bool(a[~public].any())
            pm, hm = perfect.mask(), honest.mask()
            np.testing.assert_array_equal(pm, hm)
            legal = np.flatnonzero(pm)
            self.assertGreater(len(legal), 0)
            action = int(rng.choice(legal))
            result_a, result_b = perfect.step(action), honest.step(action)
            self.assertEqual(result_a, result_b)
            if result_a[2]:
                break
        self.assertTrue(witnessed_hidden_cards, "Audit must visit nonempty opponent hands")


if __name__ == "__main__":
    unittest.main()
