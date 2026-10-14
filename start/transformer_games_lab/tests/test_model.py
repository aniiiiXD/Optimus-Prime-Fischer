from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from gamebench.agents.transformer import TransformerAgent
from gamebench.models.tiny_transformer import TinyGameTransformer, TinyTransformerConfig


class ModelTests(unittest.TestCase):
    def test_policy_shape_and_padding(self) -> None:
        config = TinyTransformerConfig(vocab_size=12, action_size=9, context_length=10)
        model = TinyGameTransformer(config)
        tokens = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
        lengths = torch.tensor([3, 4])
        logits = model(tokens, lengths)
        self.assertEqual(tuple(logits.shape), (2, 9))
        self.assertTrue(torch.all(model.token_embedding.weight[0] == 0))

    def test_checkpoint_agent_loads_and_checks_game(self) -> None:
        config = TinyTransformerConfig(vocab_size=9, action_size=7, context_length=43)
        model = TinyGameTransformer(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({
                "game": "connect4",
                "model_config": config.as_dict(),
                "state_dict": model.state_dict(),
            }, path)
            agent = TransformerAgent.from_checkpoint(path, expected_game="connect4")
            self.assertEqual(agent.model.parameter_count, model.parameter_count)
            with self.assertRaises(ValueError):
                TransformerAgent.from_checkpoint(path, expected_game="othello")


if __name__ == "__main__":
    unittest.main()
