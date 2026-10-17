"""Frozen split and supplemental-corpus merge rules."""

from __future__ import annotations

import unittest

from catanLabs.Transformer.merge_on_policy import (
    _assert_compatible, choose_on_policy_validation, merged_split,
)


class MergeSplitTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "schema": "catanlabs-board-seed-split-v1",
            "board_seeds": {"train": [100, 101], "validation": [102], "final": [103]},
            "game_ids": {"train": [0, 1], "validation": [2], "final": [3]},
            "membership_sha256": "frozen-base-membership",
        }

    def test_reserve_requires_ten_independent_boards(self):
        self.assertEqual(choose_on_policy_validation(set(range(9))), set())
        first = choose_on_policy_validation(set(range(10, 20)))
        self.assertEqual(len(first), 1)
        self.assertEqual(first, choose_on_policy_validation(set(reversed(range(10, 20)))))
        self.assertEqual(len(choose_on_policy_validation(set(range(100)))), 10)

    def test_base_final_is_sealed_and_shifted_games_partition(self):
        result = merged_split(self.base, {4: (104, "train"), 5: (105, "validation")}, "a" * 64)
        self.assertEqual(result["game_ids"]["final"], [3])
        self.assertEqual(result["board_seeds"]["final"], [103])
        self.assertEqual(result["game_ids"]["train"], [0, 1, 4])
        self.assertEqual(result["game_ids"]["validation"], [2, 5])
        self.assertEqual(result["base_membership_sha256"], "frozen-base-membership")
        with self.assertRaisesRegex(ValueError, "cannot enter final"):
            merged_split(self.base, {4: (104, "final")}, "a" * 64)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            merged_split(self.base, {4: (103, "train")}, "a" * 64)

    def test_supplemental_role_needs_ten_completed_games(self):
        shared = {
            "benchmark_protocol": "v5", "rules_manifest_sha256": "rules",
            "raw_teacher_config_sha256": "config",
            "raw_teacher_source_sha256": "teacher",
            "engine": {"commit": "engine"},
        }
        base = {"manifest": shared}
        supplemental = {"manifest": {
            **shared, "behavior": "transformer_selfplay", "smoke_only": False,
            "behavior_checkpoint_used": True,
            "collector_schema": "catanlabs-on-policy-fair-labels-v1",
            "games": [{"status": "completed"} for _ in range(9)],
        }}
        with self.assertRaisesRegex(ValueError, "at least 10"):
            _assert_compatible(base, supplemental)
        supplemental["manifest"]["games"].append({"status": "completed"})
        _assert_compatible(base, supplemental)


if __name__ == "__main__":
    unittest.main()
