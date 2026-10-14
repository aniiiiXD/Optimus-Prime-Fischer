"""Regression tests for move-based tactical benchmark labels."""

from __future__ import annotations

import unittest

from connect_four_engine import ConnectFour

from .arena_cpu import category_and_critical, immediate_wins


class TacticalCategoryTests(unittest.TestCase):
    def test_unique_real_block_is_the_critical_action(self) -> None:
        # X threatens the bottom row in column 3; O has exactly one safe reply.
        game = ConnectFour.from_moves([0, 6, 1, 6, 2])

        category, critical_actions = category_and_critical(game)

        self.assertEqual(category, "forced_block")
        self.assertEqual(critical_actions, [3])

    def test_unblockable_threat_is_not_scored_as_a_missed_block(self) -> None:
        # This held-out solver position was previously mislabeled forced_block:
        # dropping in the threatened column gives the opponent a new immediate
        # win, so every legal response loses on the next move.
        history = "2534314677333213262772415"
        game = ConnectFour.from_moves([int(column) - 1 for column in history])

        self.assertEqual(immediate_wins(game, -game.current_player), [4])
        category, critical_actions = category_and_critical(game)
        self.assertEqual(category, "unavoidable_loss")
        self.assertEqual(critical_actions, [])


if __name__ == "__main__":
    unittest.main()
