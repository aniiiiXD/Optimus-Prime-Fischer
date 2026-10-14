"""Tokenize one Tic-Tac-Toe board for a future Transformer input layer.

Only input encoding lives here. This module does not define a neural network,
training procedure, game engine, or action-selection policy.
"""

from __future__ import annotations


PAD = 0
EMPTY = 1
X = 2
O = 3
TURN_X = 4
TURN_O = 5

VOCABULARY_SIZE = 6
BOARD_SIZE = 9
SEQUENCE_LENGTH = BOARD_SIZE + 1

_CELL_TOKENS = {" ": EMPTY, "X": X, "O": O}


def tokenize_position(board: list[str] | tuple[str, ...], current_player: str) -> list[int]:
    """Return 9 row-major cell tokens followed by the side-to-move token.

    The board uses the same representation as ``tic_tac_toe_lab``:
    ``"X"``, ``"O"``, and ``" "`` for an empty square.
    """
    if len(board) != BOARD_SIZE:
        raise ValueError(f"board must contain exactly {BOARD_SIZE} squares")
    if current_player not in {"X", "O"}:
        raise ValueError("current_player must be 'X' or 'O'")

    try:
        cell_tokens = [_CELL_TOKENS[cell] for cell in board]
    except (KeyError, TypeError) as error:
        raise ValueError("board squares must be 'X', 'O', or ' '") from error

    turn_token = TURN_X if current_player == "X" else TURN_O
    return [*cell_tokens, turn_token]
