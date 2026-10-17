"""Bounded visible-policy collection smoke with fair labels."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from catanLabs.Transformer.agent import TransformerAgent
from catanLabs.Transformer.collect_on_policy import collect
from catanLabs.Transformer.model import (
    CHECKPOINT_SCHEMA, DATASET_SCHEMA, FEATURE_SCHEMA, CatanActionTransformer,
)
from catanLabs.Transformer.raw_storage import read_raw_records


class VisibleStubPolicy:
    """Small resource-building policy that sees only visible snapshots."""

    def __init__(self):
        self.rng = np.random.default_rng(12)

    def decide_visible(self, observation):
        priority = {
            "ROLL": 100, "BUILD_CITY": 90, "BUILD_SETTLEMENT": 80,
            "BUY_DEVELOPMENT_CARD": 70, "PLAY_KNIGHT_CARD": 65,
            "PLAY_YEAR_OF_PLENTY": 60, "PLAY_MONOPOLY": 60,
            "PLAY_ROAD_BUILDING": 60, "BUILD_ROAD": 50, "END_TURN": -100,
        }
        nodes = {node.id: node for node in observation.nodes}
        scores = []
        for action in observation.legal_actions:
            score = priority.get(action.kind, 0)
            if action.kind == "BUILD_SETTLEMENT":
                score += sum(nodes[action.value].production)
            scores.append(score + float(self.rng.random()) * 0.01)
        return max(range(len(scores)), key=scores.__getitem__)


class EndlessStubPolicy:
    """Roll and end turns forever after the mandatory initial placements."""

    def decide_visible(self, observation):
        for kind in ("ROLL", "END_TURN"):
            for index, action in enumerate(observation.legal_actions):
                if action.kind == kind:
                    return index
        return 0


class OnPolicyCollectorTests(unittest.TestCase):
    @staticmethod
    def checkpoint(path):
        model = CatanActionTransformer(width=16, layers=1, heads=4)
        torch.save({
            "schema": CHECKPOINT_SCHEMA,
            "dataset_schema": DATASET_SCHEMA,
            "feature_schema": FEATURE_SCHEMA,
            "model_config": {"width": 16, "layers": 1, "heads": 4},
            "model_state": model.state_dict(),
        }, path)

    def test_complete_game_has_visible_behavior_and_fair_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint = folder / "stub.pt"
            self.checkpoint(checkpoint)
            output = folder / "dataset.npz"
            result = collect(
                checkpoint, output, games=1, seed=20274000,
                wall_seconds=15, max_examples=1000,
                fair_config_path=WORKSPACE / "catanLabs/Competitive/configs/fair_army_finish.json",
                behavior_agent=VisibleStubPolicy(),
            )
            self.assertEqual(result["games"][0]["status"], "completed")
            self.assertGreater(result["unique_examples"], 100)
            self.assertEqual(result["behavior"], "smoke_visible_stub")
            self.assertTrue(result["smoke_only"])
            self.assertFalse(result["behavior_checkpoint_used"])
            self.assertEqual(result["teacher"], "fair")
            self.assertEqual(result["dataset_schema"], DATASET_SCHEMA)
            self.assertEqual(json.loads(output.with_suffix(".json").read_text()), result)
            raw = list(read_raw_records(folder / "dataset.raw.jsonl.gz"))
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(len(raw), len(data["labels"]))
                self.assertTrue(np.all(data["outcomes"] >= 0))
                for index, record in enumerate(raw):
                    self.assertEqual(record.target_action_index, int(data["labels"][index]))
                    self.assertEqual(record.game_id, int(data["game_ids"][index]))
                    self.assertEqual(record.feature_schema, FEATURE_SCHEMA)
                    self.assertEqual(record.behavior, "smoke_visible_stub")
                    self.assertEqual(record.teacher, "fair")
                    self.assertTrue(all(player.own_resources is None
                                        and player.own_development_cards is None
                                        for player in record.observation.players[1:]))
            real_agent = TransformerAgent(None, checkpoint)
            decision = real_agent.decide_visible(raw[0].observation)
            self.assertTrue(0 <= decision < len(raw[0].observation.legal_actions))

    def test_incomplete_game_contributes_no_dataset_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint = folder / "stub.pt"
            self.checkpoint(checkpoint)
            output = folder / "dataset.npz"
            with self.assertRaisesRegex(RuntimeError, "No complete game"):
                collect(
                    checkpoint, output, games=1, seed=20274000,
                    wall_seconds=15, max_examples=1000,
                    fair_config_path=WORKSPACE / "catanLabs/Competitive/configs/fair_army_finish.json",
                    behavior_agent=EndlessStubPolicy(),
                )
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".json").exists())
            self.assertEqual(list(read_raw_records(folder / "dataset.raw.jsonl.gz")), [])


if __name__ == "__main__":
    unittest.main()
