"""Focused tactical and budget tests for the isolated hybrid agent."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from connect_four_engine import ConnectFour
from connect_four_transformer.hybrid_search import TransformerGuidedSearchAgent
from connect_four_transformer.model import ModelOutput


class LeftBiasedModel(nn.Module):
    """A deliberately bad mock network: always prefers the leftmost column."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        board_tokens: torch.Tensor,
        candidate_ids: torch.Tensor,
        landing_rows: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> ModelOutput:
        self.calls += 1
        logits = -candidate_ids.to(torch.float32) * 10.0
        logits = logits.masked_fill(~candidate_mask, torch.finfo(torch.float32).min)
        q_values = -candidate_ids.to(torch.float32) / 10.0
        q_values = q_values.masked_fill(~candidate_mask, 0.0)
        state_value = torch.zeros(board_tokens.shape[0], dtype=torch.float32)
        return ModelOutput(logits, q_values, state_value)


class HybridSearchTests(unittest.TestCase):
    def make_agent(self, **kwargs: object) -> TransformerGuidedSearchAgent:
        return TransformerGuidedSearchAgent(LeftBiasedModel(), **kwargs)  # type: ignore[arg-type]

    def test_takes_immediate_win_even_when_model_disagrees(self) -> None:
        # X has bottom-row pieces in columns 1, 2, 3; column 4 wins.
        game = ConnectFour.from_moves([0, 6, 1, 6, 2, 5])
        before = game.snapshot()
        result = self.make_agent().evaluate(game)

        self.assertEqual(result.best_action, 3)
        self.assertEqual(result.reason, "take_immediate_win")
        self.assertEqual(game.snapshot(), before)

    def test_forced_block_overrides_neural_preference(self) -> None:
        # O to move; X threatens a bottom-row four in columns 1–4.
        game = ConnectFour.from_moves([0, 6, 1, 4, 2])
        before = game.snapshot()
        result = self.make_agent(max_depth=4, node_limit=500).evaluate(game)

        self.assertEqual(result.best_action, 3)
        self.assertEqual(result.reason, "forced_one_ply_block")
        self.assertEqual(game.snapshot(), before)

    def test_tiny_node_budget_returns_a_legal_fallback_and_restores_board(self) -> None:
        game = ConnectFour.from_moves([3, 2, 4, 3, 2, 4, 1, 5])
        before = game.snapshot()
        result = self.make_agent(max_depth=6, node_limit=1).evaluate(game)

        self.assertIn(result.best_action, before and game.legal_actions())
        self.assertEqual(result.stats.completed_depth, 0)
        self.assertGreaterEqual(result.stats.model_evaluations, 1)
        self.assertEqual(game.snapshot(), before)

    def test_zero_model_value_skips_leaf_forward_passes(self) -> None:
        model = LeftBiasedModel()
        agent = TransformerGuidedSearchAgent(
            model, max_depth=3, node_limit=1_000,
            model_value_weight=0.0, max_model_evaluations=32,
        )

        result = agent.evaluate(ConnectFour())

        self.assertGreater(result.stats.completed_depth, 0)
        self.assertEqual(result.stats.model_evaluations, 1)
        self.assertEqual(model.calls, 1)
        self.assertFalse(result.stats.model_budget_exhausted)

    def test_terminal_position_is_rejected(self) -> None:
        game = ConnectFour.from_moves([0, 6, 1, 6, 2, 5, 3])
        with self.assertRaises(ValueError):
            self.make_agent().evaluate(game)


if __name__ == "__main__":
    unittest.main()
