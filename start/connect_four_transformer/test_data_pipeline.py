"""Small regression tests for canonical identity and persistent teacher labels."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from connect_four_engine import ConnectFour, mirror_board

from .data import canonical_position, tactical_target
from .generate_curriculum import canonicalize_candidate
from .generate_dataset import Candidate
from .teacher_cache import TeacherLabelCache


class DataPipelineTests(unittest.TestCase):
    @staticmethod
    def tactical_row(game: ConnectFour) -> dict[str, object]:
        legal = list(game.legal_actions())
        return {
            "board": game.board.copy(),
            "current_player": game.current_player,
            "legal_actions": legal,
            "landing_rows": [5 - game.column_heights[action] for action in legal],
        }

    def test_position_identity_merges_mirrors_but_keeps_side_to_move(self) -> None:
        board = [0] * 42
        board[35 + 1] = 1
        board[35 + 4] = -1
        mirrored = mirror_board(tuple(board))
        self.assertEqual(
            canonical_position({"board": board, "current_player": 1}),
            canonical_position({"board": mirrored, "current_player": 1}),
        )
        self.assertNotEqual(
            canonical_position({"board": board, "current_player": 1}),
            canonical_position({"board": board, "current_player": -1}),
        )

    def test_candidate_move_history_is_mirrored_with_its_board(self) -> None:
        moves = (1, 4)
        game = ConnectFour.from_moves(moves)
        candidate = Candidate(
            moves=moves,
            board=tuple(game.board),
            current_player=game.current_player,
            source_policy="test",
            game_id=0,
            opening_group="test",
        )
        canonical = canonicalize_candidate(candidate)
        reconstructed = ConnectFour.from_moves(canonical.moves)
        self.assertEqual(tuple(reconstructed.board), tuple(canonical.board))
        self.assertEqual(
            canonical_position({"board": canonical.board, "current_player": canonical.current_player}),
            canonical_position({"board": game.board, "current_player": game.current_player}),
        )

    def test_cache_persists_and_search_settings_invalidate_key(self) -> None:
        board = tuple([0] * 42)
        row = {"board": list(board), "policy_target": [1.0] + [0.0] * 6}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.sqlite3"
            with TeacherLabelCache(path) as cache:
                self.assertIsNone(
                    cache.get(board, 1, depth=5, node_limit=5000, temperature=0.18)
                )
                cache.put(
                    row,
                    board,
                    1,
                    depth=5,
                    node_limit=5000,
                    temperature=0.18,
                )
            with TeacherLabelCache(path) as cache:
                self.assertEqual(
                    cache.get(board, 1, depth=5, node_limit=5000, temperature=0.18), row
                )
                self.assertIsNone(
                    cache.get(board, 1, depth=6, node_limit=5000, temperature=0.18)
                )
                self.assertIsNone(
                    cache.get(board, -1, depth=5, node_limit=5000, temperature=0.18)
                )

    def test_tactical_target_labels_only_a_real_forced_block(self) -> None:
        game = ConnectFour.from_moves([0, 6, 1, 6, 2])
        self.assertEqual(tactical_target(self.tactical_row(game)), [3])

    def test_tactical_target_skips_an_unavoidable_immediate_loss(self) -> None:
        # Blocking column 4 raises the column and gives X a different winning
        # drop; no move actually stops the loss on the following turn.
        history = "2534314677333213262772415"
        game = ConnectFour.from_moves([int(column) - 1 for column in history])
        self.assertEqual(tactical_target(self.tactical_row(game)), [])


if __name__ == "__main__":
    unittest.main()
