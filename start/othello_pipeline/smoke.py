"""Fast, dependency-free check of the shared Othello adapter."""

from __future__ import annotations

import random

from gamebench.games.othello import OthelloAdapter, PASS_ACTION
from .encoding import action_from_gtp, action_to_gtp, board_to_obf


def main() -> None:
    game = OthelloAdapter().reset()
    opening = set(game.legal_actions())
    expected = {2 * 8 + 3, 3 * 8 + 2, 4 * 8 + 5, 5 * 8 + 4}
    assert opening == expected, (opening, expected)
    assert {action_to_gtp(action) for action in opening} == {"d3", "c4", "f5", "e6"}
    assert all(action_from_gtp(action_to_gtp(action)) == action for action in (*opening, PASS_ACTION))
    initial_obf = board_to_obf(game)
    assert len(initial_obf) == 66 and initial_obf.endswith(" X")
    assert initial_obf[3 * 8 : 4 * 8] == "---OX---"
    assert initial_obf[4 * 8 : 5 * 8] == "---XO---"

    rng = random.Random(7)
    moves = 0
    passes = 0
    while not game.is_terminal():
        legal = game.legal_actions()
        assert legal
        action = rng.choice(legal)
        if action == PASS_ACTION:
            passes += 1
        game.apply_action(action)
        moves += 1
        assert moves <= game.spec.max_game_length

    assert game.returns() in ((1.0, -1.0), (-1.0, 1.0), (0.0, 0.0))
    print(f"Othello adapter OK: 4 opening moves; {moves} actions, {passes} passes; result={game.returns()}")


if __name__ == "__main__":
    main()
