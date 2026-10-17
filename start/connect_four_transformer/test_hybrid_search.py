"""Focused tactical and budget tests for the isolated hybrid agent."""

from __future__ import annotations

import unittest
import random

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

    def test_fast_tactical_check_matches_engine_on_random_positions(self) -> None:
        rng = random.Random(5172)
        for _ in range(200):
            game = ConnectFour()
            for _ in range(rng.randrange(1, 34)):
                if game.is_terminal:
                    break
                game.step(rng.choice(game.legal_actions()))
            if game.is_terminal:
                continue
            before = game.snapshot()
            for player in (1, -1):
                for action in game.legal_actions():
                    trial = game.copy()
                    trial.current_player = player
                    expected = trial.step(action).winner == player
                    actual = TransformerGuidedSearchAgent._winning_drop(game, action, player)
                    self.assertEqual(actual, expected)
            self.assertEqual(game.snapshot(), before)

    def test_fast_heuristic_matches_reference_windows(self) -> None:
        rng = random.Random(2074)

        def reference(game: ConnectFour) -> int:
            board = game.board
            player = game.current_player
            score = 3 * sum(board[row * 7 + 3] == player for row in range(6))
            score -= 3 * sum(board[row * 7 + 3] == -player for row in range(6))
            weights = (0, 1, 8, 70, 1_000)
            for row in range(6):
                for column in range(7):
                    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                        end_row, end_column = row + 3 * dr, column + 3 * dc
                        if not (0 <= end_row < 6 and 0 <= end_column < 7):
                            continue
                        cells = [board[(row + i * dr) * 7 + column + i * dc] for i in range(4)]
                        own, opponent = cells.count(player), cells.count(-player)
                        if not opponent:
                            score += weights[own]
                        elif not own:
                            score -= weights[opponent]
            return max(-5_000, min(5_000, score))

        for _ in range(200):
            game = ConnectFour()
            for _ in range(rng.randrange(0, 34)):
                if game.is_terminal:
                    break
                game.step(rng.choice(game.legal_actions()))
            self.assertEqual(TransformerGuidedSearchAgent._heuristic(game), reference(game))


if __name__ == "__main__":
    unittest.main()
