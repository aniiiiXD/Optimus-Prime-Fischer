"""Validation must score only actions the game actually permits."""

import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from catanLabs.Transformer.train import (
    _metrics, choice_training_indices, split_indices, teacher_from_manifest,
)


class FixedLogits(torch.nn.Module):
    def forward(self, *_args):
        return torch.tensor([[0.0, 10.0], [0.0, 1.0]])


class TrainMetricsTests(unittest.TestCase):
    def test_padded_action_cannot_win_accuracy_argmax(self):
        data = {
            "counts": np.asarray([1, 2]),
            "labels": torch.tensor([0, 1]),
        }
        for key in (
            "state_features", "state_kinds", "state_ids", "action_features",
            "action_types", "target_groups", "target_ids",
        ):
            data[key] = torch.zeros(2, 2)
        result = _metrics(FixedLogits(), data, np.asarray([0, 1]), batch_size=2)
        self.assertEqual(result["accuracy"], 1.0)

    def test_audited_split_excludes_final_games(self):
        game_ids = np.asarray([0, 0, 1, 1, 2, 2])
        split = {
            "schema": "catanlabs-board-seed-split-v1",
            "dataset_sha256": "dataset-hash",
            "game_ids": {"train": [0], "validation": [1], "final": [2]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(split))
            train, validation, heldout, final, _, _ = split_indices(
                game_ids, 5, path, "dataset-hash"
            )
            self.assertEqual(train.tolist(), [0, 1])
            self.assertEqual(validation.tolist(), [2, 3])
            self.assertEqual(heldout, {1})
            self.assertEqual(final, {2})
            with self.assertRaises(ValueError):
                split_indices(game_ids, 5, path, "other-dataset")

    def test_checkpoint_teacher_comes_from_matching_dataset_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.npz"
            manifest = dataset.with_suffix(".json")
            manifest.write_text(json.dumps({"dataset_sha256": "matching-hash", "teacher": "fair"}))
            self.assertEqual(teacher_from_manifest(dataset, "matching-hash"), "fair")
            with self.assertRaises(ValueError):
                teacher_from_manifest(dataset, "different-hash")

    def test_forced_rows_are_removed_from_optimizer_split(self):
        rows = np.asarray([0, 1, 2, 3])
        counts = np.asarray([1, 3, 1, 2])
        self.assertEqual(choice_training_indices(rows, counts).tolist(), [1, 3])
        self.assertEqual(choice_training_indices(rows, counts, True).tolist(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
