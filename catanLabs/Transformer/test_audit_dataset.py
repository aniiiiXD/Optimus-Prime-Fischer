"""Tests for immutable board splits and honest Gate 2 audit failures."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from catanLabs.Transformer.audit_dataset import (
    audit_dataset, build_split, freeze_split, sha256,
)
from catanLabs.Transformer.model import DATASET_SCHEMA, FEATURE_SCHEMA


class DatasetAuditTests(unittest.TestCase):
    def test_board_split_deterministic_disjoint_and_frozen(self):
        seeds = set(range(20000, 20100))
        completed = {index: seed for index, seed in enumerate(sorted(seeds))}
        first = build_split(completed, "a" * 64)
        second = build_split(dict(reversed(list(completed.items()))), "a" * 64)
        self.assertEqual(first, second)
        groups = [set(group) for group in first["board_seeds"].values()]
        self.assertEqual([len(group) for group in groups], [80, 10, 10])
        self.assertEqual(set.union(*groups), seeds)
        self.assertTrue(all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)))
        game_groups = [set(group) for group in first["game_ids"].values()]
        self.assertEqual([len(group) for group in game_groups], [80, 10, 10])
        self.assertEqual(set.union(*game_groups), set(completed))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            freeze_split(path, first)
            original = path.read_bytes()
            freeze_split(path, second)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                freeze_split(path, build_split(completed | {100: 20100}, "b" * 64))

    def test_collector_raw_rows_reconstruct_and_are_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.npz"
            subprocess.run([
                sys.executable, str(WORKSPACE / "catanLabs/Transformer/collect.py"),
                "--games", "1", "--max-examples", "1000", "--wall-seconds", "25",
                "--seed", "20266501", "--output", str(dataset),
            ], cwd=WORKSPACE, check=True, capture_output=True, text=True, timeout=35)
            report, split = audit_dataset(
                dataset, dataset.with_suffix(".json"),
                dataset.with_name("dataset.raw.jsonl.gz"),
                WORKSPACE / "catanLabs/rules_manifest.json", sample_size=16,
            )
            self.assertFalse(report["gate2_pass"])  # Deliberately only one completed game.
            self.assertEqual(report["completed_games"], 1)
            self.assertIsNone(split)
            self.assertEqual(report["raw"]["status"], "checked")
            self.assertEqual(report["raw"]["duplicates"], 0)
            self.assertEqual(report["raw"]["metadata_errors"], 0)
            self.assertEqual(report["raw"]["hidden_field_errors"], 0)
            self.assertEqual(report["raw"]["reconstructed_samples"], 16)

    def test_npz_without_raw_cannot_pass_even_with_valid_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dataset = directory / "dataset.npz"
            manifest_path = directory / "dataset.json"
            np.savez_compressed(dataset,
                                dataset_schema=np.asarray(DATASET_SCHEMA),
                                feature_schema=np.asarray(FEATURE_SCHEMA),
                                labels=np.asarray([0, 0], dtype=np.int64),
                                candidate_counts=np.asarray([2, 2], dtype=np.int64),
                                game_ids=np.asarray([0, 1], dtype=np.int32),
                                phases=np.asarray(["BUILD_INITIAL_SETTLEMENT", "PLAY_TURN"]),
                                behaviors=np.asarray(["fair", "value"]))
            rules_path = WORKSPACE / "catanLabs/rules_manifest.json"
            rules = json.loads(rules_path.read_text())
            manifest_path.write_text(json.dumps({
                "dataset_sha256": sha256(dataset),
                "rules_manifest_sha256": sha256(rules_path),
                "dataset_schema": DATASET_SCHEMA,
                "feature_schema": FEATURE_SCHEMA,
                "benchmark_protocol": rules["local_protocol"],
                "engine": {"commit": rules["engine"]["commit"]},
                "unique_examples": 2,
                "games": [{"game_id": index, "seed": 500 + index, "status": "completed"}
                          for index in range(2)],
            }))
            report, split = audit_dataset(dataset, manifest_path,
                                          directory / "missing.raw.jsonl.gz", rules_path,
                                          sample_size=2)
            self.assertFalse(report["gate2_pass"])
            self.assertIsNone(split)
            self.assertEqual(report["raw"]["status"], "missing")
            self.assertTrue(any("raw visible snapshots missing" in reason
                                for reason in report["failures"]))


if __name__ == "__main__":
    unittest.main()
