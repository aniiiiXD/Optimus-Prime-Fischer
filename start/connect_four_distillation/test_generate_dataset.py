"""Small deterministic checks for the CPU dataset ordering path."""

from __future__ import annotations

import unittest

from .generate_dataset import balance_phases


class PhaseBalanceTests(unittest.TestCase):
    def test_interleaves_phases_without_dropping_candidates(self) -> None:
        candidates = [
            {"ply": ply, "id": index}
            for index, ply in enumerate((12, 13, 14, 20, 21, 30, 31))
        ]
        ordered = balance_phases(candidates)
        self.assertEqual({row["id"] for row in ordered}, set(range(7)))
        self.assertEqual(len(ordered), len(candidates))
        self.assertEqual(
            ["early" if row["ply"] < 20 else "middle" if row["ply"] < 30 else "late" for row in ordered[:6]],
            ["early", "middle", "late", "early", "middle", "late"],
        )
        self.assertEqual([row["id"] for row in candidates], list(range(7)))


if __name__ == "__main__":
    unittest.main()
