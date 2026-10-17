"""Focused action MLP forward, mask, and checkpoint checks."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest

import torch

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanatron import Color, Game, RandomPlayer

from catanLabs.Transformer.mlp_agent import create_agent
from catanLabs.Transformer.mlp import CHECKPOINT_SCHEMA, CatanActionMLP, parameter_count
from catanLabs.Transformer.model import (
    DATASET_SCHEMA, FEATURE_SCHEMA, CatanActionTransformer, STATE_TOKEN_COUNT,
)
from catanLabs.Transformer.visible_observation import VisibleAgentAdapter


def sample_batch():
    torch.manual_seed(31)
    state_features = torch.randn(2, STATE_TOKEN_COUNT, 40)
    state_kinds = torch.zeros((2, STATE_TOKEN_COUNT), dtype=torch.long)
    state_ids = torch.arange(STATE_TOKEN_COUNT).remainder(73).repeat(2, 1)
    action_features = torch.randn(2, 4, 20)
    action_types = torch.tensor([[3, 4, 0, 0], [0, 3, 5, 0]])
    target_groups = torch.tensor([[3, 2, 0, 0], [0, 3, 1, 0]])
    target_ids = torch.tensor([[5, 16, 0, 0], [0, 8, 4, 0]])
    action_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    return (state_features, state_kinds, state_ids, action_features,
            action_types, target_groups, target_ids, action_mask)


class ActionMLPTests(unittest.TestCase):
    def test_parameter_match_forward_and_padding_mask(self):
        model = CatanActionMLP(width=32).eval()
        transformer = CatanActionTransformer(width=32, layers=1, heads=4)
        self.assertEqual(parameter_count(32), 13729)
        self.assertEqual(sum(p.numel() for p in transformer.parameters()), 13857)
        self.assertLess(abs(parameter_count(32) - 13857) / 13857, 0.01)
        batch = sample_batch()
        with torch.inference_mode():
            scores = model(*batch)
            self.assertEqual(tuple(scores.shape), (2, 4))
            self.assertTrue(torch.isfinite(scores[batch[-1]]).all())
            self.assertTrue((scores[~batch[-1]] == -1e9).all())

            changed = list(batch)
            changed[3] = batch[3].clone()
            changed[3][~batch[-1]] = 1000.0
            altered = model(*changed)
            torch.testing.assert_close(scores[batch[-1]], altered[batch[-1]])

    def test_checkpoint_round_trip_preserves_scores(self):
        model = CatanActionMLP(width=32).eval()
        batch = sample_batch()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.pt"
            torch.save({
                "schema": CHECKPOINT_SCHEMA,
                "dataset_schema": DATASET_SCHEMA,
                "feature_schema": FEATURE_SCHEMA,
                "model_config": {"width": 32},
                "model_state": model.state_dict(),
            }, path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["schema"], CHECKPOINT_SCHEMA)
            restored = CatanActionMLP(**checkpoint["model_config"]).eval()
            restored.load_state_dict(checkpoint["model_state"])
            with torch.inference_mode():
                torch.testing.assert_close(model(*batch), restored(*batch))

    def test_benchmark_adapter_legal_action_and_hash_check(self):
        model = CatanActionMLP(width=32).eval()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.pt"
            dataset_hash = "a" * 64
            torch.save({
                "schema": CHECKPOINT_SCHEMA,
                "dataset_schema": DATASET_SCHEMA,
                "feature_schema": FEATURE_SCHEMA,
                "model_config": {"width": 32},
                "model_state": model.state_dict(),
                "dataset_sha256": dataset_hash,
                "split_file": None,
                "split_file_sha256": None,
            }, path)
            summary = {"dataset_sha256": dataset_hash,
                       "split_file_sha256": None, "feature_schema": FEATURE_SCHEMA}
            path.with_name("training.json").write_text(json.dumps(summary))
            game = Game([RandomPlayer(color) for color in Color], seed=19, vps_to_win=10)
            agent = create_agent(game.state.current_color(), path, seed=0)
            self.assertIsInstance(agent, VisibleAgentAdapter)
            self.assertIn(agent.decide(game, game.playable_actions), game.playable_actions)
            summary["dataset_sha256"] = "b" * 64
            path.with_name("training.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "hashes disagree"):
                create_agent(game.state.current_color(), path, seed=0)


if __name__ == "__main__":
    unittest.main()
