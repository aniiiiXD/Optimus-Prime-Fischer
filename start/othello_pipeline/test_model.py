"""Correctness checks before any teacher data or training is trusted."""

from __future__ import annotations

import unittest

import torch

from gamebench.games.othello import OthelloAdapter, PASS_ACTION
from .model import ModelConfig, OthelloTransformer, PAD_ACTION, encode_board


class OthelloModelTests(unittest.TestCase):
    def test_opening_shapes_and_mask(self) -> None:
        state = OthelloAdapter().reset()
        model = OthelloTransformer(ModelConfig()).eval()
        board = encode_board(state.board, state.current_player).unsqueeze(0)
        side = torch.tensor([state.current_player])
        candidates = torch.tensor([[*state.legal_actions(), PAD_ACTION]])
        mask = candidates.ne(PAD_ACTION)
        with torch.inference_mode():
            output = model(board, side, candidates, mask)
        self.assertEqual(model.parameter_count, sum(p.numel() for p in model.parameters()))
        self.assertEqual(tuple(output.policy_logits.shape), (1, 5))
        self.assertEqual(tuple(output.action_values.shape), (1, 5))
        self.assertEqual(tuple(output.state_value.shape), (1,))
        self.assertTrue(torch.isfinite(output.policy_logits[0, :4]).all())
        self.assertLess(output.policy_logits[0, 4].item(), -1e20)
        self.assertTrue(torch.isfinite(output.state_value).all())

    def test_pass_is_distinct_from_padding(self) -> None:
        state = OthelloAdapter().reset()
        model = OthelloTransformer().eval()
        board = encode_board(state.board, state.current_player).unsqueeze(0)
        with torch.inference_mode():
            output = model(board, torch.tensor([0]), torch.tensor([[PASS_ACTION, PAD_ACTION]]), torch.tensor([[True, False]]))
        self.assertTrue(torch.isfinite(output.policy_logits[0, 0]))
        self.assertLess(output.policy_logits[0, 1].item(), -1e20)

    def test_backward(self) -> None:
        state = OthelloAdapter().reset()
        model = OthelloTransformer(ModelConfig(width=32, heads=4, blocks=1, mlp_width=64))
        board = encode_board(state.board, state.current_player).unsqueeze(0)
        actions = torch.tensor([list(state.legal_actions())])
        output = model(board, torch.tensor([0]), actions, torch.ones_like(actions, dtype=torch.bool))
        (-torch.log_softmax(output.policy_logits, dim=-1)[0, 0] + output.state_value.square().sum()).backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
