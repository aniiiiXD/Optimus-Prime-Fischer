"""Provenance guards for re-encoding the audited native Catan corpus."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from catanLabs.Transformer.migrate_v3 import checked_inputs


class MigrationProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.base = WORKSPACE / "catanLabs/Transformer/runs/gate2_native_v2"
        self.dataset = self.base / "dataset.npz"
        self.manifest = self.base / "dataset.json"
        self.audit = self.base / "dataset.audit.json"
        self.split = self.base / "dataset.split.json"
        self.raw = self.base / "dataset.raw.jsonl.gz"

    def test_frozen_source_is_bound_to_passed_audit_and_split(self):
        manifest, audit, split, dataset_hash = checked_inputs(
            self.dataset, self.manifest, self.audit, self.split, self.raw,
        )
        self.assertEqual(audit["completed_games"], 100)
        self.assertEqual(audit["unique_decisions"], manifest["unique_examples"])
        self.assertEqual(split["dataset_sha256"], dataset_hash)
        self.assertEqual(split["membership_sha256"], audit["split_membership_sha256"])

    def test_changed_source_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "dataset.json"
            data = json.loads(self.manifest.read_text())
            data["dataset_sha256"] = "0" * 64
            tampered.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "source dataset hash"):
                checked_inputs(self.dataset, tampered, self.audit, self.split, self.raw)


if __name__ == "__main__":
    unittest.main()
