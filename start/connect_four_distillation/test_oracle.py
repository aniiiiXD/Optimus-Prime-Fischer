"""Quick integration checks for the external solver and our game rules."""

import unittest

from connect_four_distillation.oracle import PonsOracle, SOLVER, validate_history


FIXTURE = "135567125554756746421621316174"


class OracleTest(unittest.TestCase):
    @unittest.skipUnless(SOLVER.is_file(), "external Connect Four solver is not installed")
    def test_exact_analysis_matches_local_legality(self) -> None:
        with PonsOracle(timeout_seconds=5) as oracle:
            result = oracle.analyze(FIXTURE)
        self.assertEqual(result["ply"], 30)
        self.assertEqual(result["best_columns"], [3])
        self.assertEqual(result["scores"], [-1000, 0, 6, 5, -1000, -1, 5])
        self.assertEqual(result["legal_columns"], [2, 3, 4, 6, 7])

    def test_bad_history_is_rejected_before_solver(self) -> None:
        with self.assertRaises(ValueError):
            validate_history("1238")


if __name__ == "__main__":
    unittest.main()
