"""Rule, gravity, symmetry, and search-operation tests."""

from __future__ import annotations

import unittest

from .engine import (
    COLUMNS,
    PLAYER_ONE,
    PLAYER_TWO,
    ROWS,
    ConnectFour,
    IllegalMoveError,
    find_winner,
    mirror_action,
    mirror_board,
)


class ConnectFourTests(unittest.TestCase):
    def test_empty_state(self) -> None:
        game = ConnectFour()
        self.assertEqual(game.legal_actions(), tuple(range(COLUMNS)))
        self.assertEqual(game.current_player, PLAYER_ONE)
        self.assertFalse(game.is_terminal)
        game.assert_invariants()

    def test_gravity_and_turns(self) -> None:
        game = ConnectFour()
        first = game.step(3)
        second = game.step(3)
        self.assertEqual((first.move.row, first.move.player), (ROWS - 1, PLAYER_ONE))
        self.assertEqual((second.move.row, second.move.player), (ROWS - 2, PLAYER_TWO))
        self.assertEqual(game.current_player, PLAYER_ONE)
        game.assert_invariants()

    def test_full_column_is_removed_and_rejected(self) -> None:
        game = ConnectFour()
        for _ in range(ROWS):
            game.step(0)
        self.assertNotIn(0, game.legal_actions())
        with self.assertRaises(IllegalMoveError):
            game.step(0)

    def test_vertical_win(self) -> None:
        game = ConnectFour.from_moves([0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(game.winner, PLAYER_ONE)
        self.assertEqual(len(game.winning_line), 4)
        self.assertEqual(game.legal_actions(), ())
        game.assert_invariants()

    def test_horizontal_win(self) -> None:
        game = ConnectFour.from_moves([0, 0, 1, 1, 2, 2, 3])
        self.assertEqual(game.winner, PLAYER_ONE)
        self.assertEqual({row for row, _ in game.winning_line}, {ROWS - 1})
        game.assert_invariants()

    def test_both_diagonal_directions(self) -> None:
        rising = ConnectFour.from_moves([0, 1, 1, 2, 4, 2, 2, 3, 4, 3, 5, 3, 3])
        falling = ConnectFour.from_moves([3, 2, 2, 1, 6, 1, 1, 0, 6, 0, 5, 0, 0])
        self.assertEqual(rising.winner, PLAYER_ONE)
        self.assertEqual(falling.winner, PLAYER_ONE)
        rising.assert_invariants()
        falling.assert_invariants()

    def test_no_moves_after_win(self) -> None:
        game = ConnectFour.from_moves([0, 1, 0, 1, 0, 1, 0])
        with self.assertRaises(IllegalMoveError):
            game.step(2)

    def test_undo_restores_exact_state(self) -> None:
        game = ConnectFour.from_moves([3, 2, 3, 2, 3, 2])
        before = game.snapshot()
        game.step(3)
        self.assertTrue(game.is_terminal)
        undone = game.undo()
        self.assertEqual(undone.column, 3)
        self.assertEqual(game.snapshot(), before)
        game.assert_invariants()

    def test_copy_is_independent(self) -> None:
        original = ConnectFour.from_moves([3, 2])
        clone = original.copy()
        clone.step(3)
        self.assertNotEqual(clone.snapshot(), original.snapshot())
        original.assert_invariants()
        clone.assert_invariants()

    def test_relative_encoding(self) -> None:
        game = ConnectFour.from_moves([3, 2])
        absolute = tuple(game.board)
        self.assertEqual(game.encode(PLAYER_ONE), absolute)
        self.assertEqual(game.encode(PLAYER_TWO), tuple(-cell for cell in absolute))

    def test_mirror_is_an_involution(self) -> None:
        game = ConnectFour.from_moves([0, 1, 0, 6])
        self.assertEqual(mirror_board(mirror_board(game.board)), tuple(game.board))
        self.assertEqual(mirror_action(mirror_action(1)), 1)
        mirrored_moves = [mirror_action(move.column) for move in game.history]
        mirrored_game = ConnectFour.from_moves(mirrored_moves)
        self.assertEqual(tuple(mirrored_game.board), mirror_board(game.board))
        self.assertEqual(mirrored_game.canonical_key(), game.canonical_key())

    def test_find_winner_matches_incremental_result(self) -> None:
        game = ConnectFour.from_moves([0, 0, 1, 1, 2, 2, 3])
        self.assertEqual(find_winner(game.board), game.winner)


if __name__ == "__main__":
    unittest.main()
