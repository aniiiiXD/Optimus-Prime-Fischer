"""Dataset and one-step training checks without downloading a teacher."""

from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from gamebench.games.othello import OthelloAdapter
from .dataset import Example, canonical_state_key, deduplicate_examples, load_examples, split_examples, transform_action, transform_board
from .model import ModelConfig, OthelloTransformer
from .train import make_batch, run_epoch
from .train import main as train_main


class OthelloPipelineTests(unittest.TestCase):
    @staticmethod
    def examples() -> list[Example]:
        first = OthelloAdapter().reset()
        legal_first = first.legal_actions()
        second = first.clone()
        second.apply_action(legal_first[0])
        legal_second = second.legal_actions()
        return [
            Example(tuple(first.board), first.current_player, legal_first, tuple(float(i) for i in range(len(legal_first))), None, "a", "test-teacher"),
            Example(tuple(second.board), second.current_player, legal_second, tuple(float(i) for i in range(len(legal_second))), 1.0, "b", "test-teacher"),
        ]

    def test_group_split_and_batch(self) -> None:
        train, validation, removed = split_examples(self.examples(), 0.5, 42)
        self.assertEqual((len(train), len(validation), removed), (1, 1, 0))
        self.assertNotEqual(canonical_state_key(train[0]), canonical_state_key(validation[0]))
        batch = make_batch(self.examples(), 2.0, 16.0)
        self.assertEqual(tuple(batch["board"].shape), (2, 64))
        self.assertEqual(tuple(batch["teacher_probs"].shape), (2, 65))
        self.assertTrue(torch.allclose(batch["teacher_probs"].sum(-1), torch.ones(2)))
        self.assertEqual(int(batch["legal"].sum()), sum(len(x.legal) for x in self.examples()))

    def test_canonical_deduplication(self) -> None:
        first = self.examples()[0]
        unique, removed = deduplicate_examples([first, first])
        self.assertEqual((len(unique), removed), (1, 1))

    def test_symmetry_preserves_legal_moves(self) -> None:
        rng = random.Random(14)
        state = OthelloAdapter().reset()
        for _ in range(12):
            state.apply_action(rng.choice(state.legal_actions()))
        for rotation in range(4):
            for reflect in (False, True):
                board = transform_board(tuple(state.board), rotation, reflect)
                transformed = type(state)(list(board), state.current_player)
                expected = {transform_action(action, rotation, reflect) for action in state.legal_actions()}
                self.assertEqual(set(transformed.legal_actions()), expected)

    def test_loader_rejects_wrong_legal_actions(self) -> None:
        example = self.examples()[0]
        row = {
            "board": list(example.board),
            "player": example.player,
            "legal": [0],
            "teacher_scores": [0.0],
            "group": "a",
            "teacher_id": "test-teacher",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "legal actions"):
                load_examples(path)

    def test_training_step(self) -> None:
        torch.manual_seed(3)
        model = OthelloTransformer(ModelConfig(width=32, heads=4, blocks=1, mlp_width=64))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = run_epoch(model, self.examples(), 2, 2.0, 16.0, optimizer, random.Random(3))
        self.assertGreaterEqual(metrics["teacher_regret"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

    def test_training_cli_creates_checkpoint_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "toy.jsonl"
            output = Path(directory) / "student.pt"
            rows = [
                {
                    "board": list(example.board),
                    "player": example.player,
                    "legal": list(example.legal),
                    "teacher_scores": list(example.teacher_scores),
                    "result": example.result,
                    "group": example.group,
                    "teacher_id": example.teacher_id,
                }
                for example in self.examples()
            ]
            dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            argv = [
                "train", "--dataset", str(dataset), "--output", str(output),
                "--epochs", "1", "--batch-size", "2", "--threads", "1",
                "--width", "32", "--heads", "4", "--blocks", "1", "--mlp-width", "64",
            ]
            with patch("sys.argv", argv):
                train_main()
            self.assertTrue(output.is_file())
            report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(report["train_examples"], 1)
            self.assertEqual(report["validation_examples"], 1)
            self.assertEqual(len(report["epochs"]), 1)


if __name__ == "__main__":
    unittest.main()
