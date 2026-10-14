from __future__ import annotations

import random
import unittest

from gamebench.games.base import BOS_TOKEN
from gamebench.games.connect4 import ConnectFourAdapter
from gamebench.games.othello import OthelloAdapter


class OthelloTests(unittest.TestCase):
    def test_initial_legal_moves_and_flip(self) -> None:
        state = OthelloAdapter().reset()
        self.assertEqual({state.action_name(action) for action in state.legal_actions()}, {"c4", "d3", "e6", "f5"})
        action = next(action for action in state.legal_actions() if state.action_name(action) == "c4")
        state.apply_action(action)
        self.assertEqual(state.board.count(0), 4)
        self.assertEqual(state.board.count(1), 1)
        self.assertEqual(state.history_tokens(), (BOS_TOKEN, action + 2))

    def test_random_game_terminates(self) -> None:
        state = OthelloAdapter().reset()
        rng = random.Random(7)
        while not state.is_terminal():
            state.apply_action(rng.choice(state.legal_actions()))
        self.assertIn(state.returns(), ((1.0, -1.0), (-1.0, 1.0), (0.0, 0.0)))
        self.assertLessEqual(len(state.history), state.spec.max_game_length)

    def test_clone_is_independent(self) -> None:
        state = OthelloAdapter().reset()
        clone = state.clone()
        clone.apply_action(clone.legal_actions()[0])
        self.assertNotEqual(state.board, clone.board)
        self.assertEqual(state.history, ())


class ConnectFourTests(unittest.TestCase):
    def test_vertical_win(self) -> None:
        state = ConnectFourAdapter().reset()
        for action in (0, 1, 0, 1, 0, 1, 0):
            state.apply_action(action)
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.returns(), (1.0, -1.0))

    def test_observation_shape(self) -> None:
        state = ConnectFourAdapter().reset()
        self.assertEqual(len(state.observation_tensor()), 3 * 6 * 7)


if __name__ == "__main__":
    unittest.main()
