"""Frozen v2 policy remains replayable after the v3 feature correction."""

from pathlib import Path
import unittest

import numpy as np

from catanLabs.Transformer.legacy_v2_agent import LegacyV2TransformerAgent
from catanLabs.Transformer.raw_storage import read_raw_records


ROOT = Path(__file__).resolve().parents[1]


class LegacyV2Tests(unittest.TestCase):
    def test_first_audited_v2_record_encodes_identically(self):
        raw = ROOT / "runs/gate2_native_v2/dataset.raw.jsonl.gz"
        dataset = ROOT / "runs/gate2_native_v2/dataset.npz"
        checkpoint = ROOT / "runs/gate3_matched/transformer/best.pt"
        if not all(path.exists() for path in (raw, dataset, checkpoint)):
            self.skipTest("Historical v2 artifacts are unavailable")
        record = next(read_raw_records(raw))
        agent = LegacyV2TransformerAgent(None, checkpoint)
        encoded = agent._encode(record.observation)
        with np.load(dataset, allow_pickle=False) as stored:
            for key, actual in zip(
                ("state_features", "state_kinds", "state_ids", "action_features",
                 "action_types", "target_groups", "target_ids"), encoded,
            ):
                np.testing.assert_array_equal(actual, stored[key][0, :len(actual)]
                                              if key.startswith("action_") or key in ("target_groups", "target_ids")
                                              else stored[key][0])
        self.assertEqual(len(agent.score_visible(record.observation)), len(record.observation.legal_actions))


if __name__ == "__main__":
    unittest.main()
