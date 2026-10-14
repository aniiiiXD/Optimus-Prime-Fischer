"""Small correctness tests; run with ``python -m unittest ...``."""

from __future__ import annotations

import unittest

import torch

from tic_tac_toe_transformer.input import tokenize_position

from tic_tac_toe_shared_transformer.data import grouped_split, load_rows
from tic_tac_toe_shared_transformer.model import SharedActionTransformer


class SharedTransformerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = SharedActionTransformer().eval()
        self.tokens = torch.tensor(
            [tokenize_position(["X", " ", " ", " ", "O", " ", " ", " ", " "], "X")],
            dtype=torch.long,
        )

    def test_variable_candidate_count(self) -> None:
        candidates = torch.tensor([[1, 2, 5, 8]], dtype=torch.long)
        output = self.model(self.tokens, candidates)
        self.assertEqual(tuple(output.policy_logits.shape), (1, 4))
        self.assertEqual(tuple(output.action_value_logits.shape), (1, 4, 3))
        self.assertEqual(tuple(output.state_value_logits.shape), (1, 3))

    def test_candidate_order_only_reorders_outputs(self) -> None:
        first = torch.tensor([[1, 2, 5]], dtype=torch.long)
        second = torch.tensor([[5, 1, 2]], dtype=torch.long)
        with torch.inference_mode():
            first_logits = self.model(self.tokens, first).policy_logits
            second_logits = self.model(self.tokens, second).policy_logits
        torch.testing.assert_close(first_logits[:, [2, 0, 1]], second_logits)

    def test_symmetry_families_do_not_cross_splits(self) -> None:
        rows = load_rows()
        splits = grouped_split(rows)
        membership: dict[tuple[str, str], str] = {}
        from tic_tac_toe_transformer.data.symmetry import canonical_board

        for split, indices in splits.items():
            for index in indices:
                row = rows[index]
                key = (canonical_board(row["board"]), row["current_player"])
                if key in membership:
                    self.assertEqual(membership[key], split)
                else:
                    membership[key] = split


if __name__ == "__main__":
    unittest.main()
