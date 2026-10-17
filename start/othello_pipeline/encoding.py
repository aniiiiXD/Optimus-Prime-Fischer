"""Translate our 0..64 Othello actions to Neural Reversi's CLI formats."""

from __future__ import annotations

from gamebench.games.othello import OthelloState, PASS_ACTION


def action_to_gtp(action: int) -> str:
    """Use conventional column-letter + row-number squares, e.g. index 19 -> d3."""
    if action == PASS_ACTION:
        return "pass"
    if not 0 <= action < 64:
        raise ValueError(f"invalid Othello action: {action}")
    row, col = divmod(action, 8)
    return f"{'abcdefgh'[col]}{row + 1}"


def action_from_gtp(move: str) -> int:
    square = move.strip().lower()
    if square == "pass":
        return PASS_ACTION
    if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
        raise ValueError(f"invalid GTP square: {move!r}")
    return (int(square[1]) - 1) * 8 + "abcdefgh".index(square[0])


def board_to_obf(state: OthelloState) -> str:
    """Create one `solve` input line: 64 cells plus X/O side-to-move."""
    cells = "".join({-1: "-", 0: "X", 1: "O"}[cell] for cell in state.board)
    side = "X" if state.current_player == 0 else "O"
    return f"{cells} {side}"
