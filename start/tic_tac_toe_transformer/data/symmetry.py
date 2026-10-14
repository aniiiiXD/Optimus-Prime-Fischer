"""The eight rotations and reflections of a 3x3 board."""

from __future__ import annotations

from collections.abc import Callable, Sequence


CoordinateTransform = Callable[[int, int], tuple[int, int]]


def _identity(row: int, column: int) -> tuple[int, int]:
    return row, column


def _rotate_90(row: int, column: int) -> tuple[int, int]:
    return column, 2 - row


def _rotate_180(row: int, column: int) -> tuple[int, int]:
    return 2 - row, 2 - column


def _rotate_270(row: int, column: int) -> tuple[int, int]:
    return 2 - column, row


def _reflect(row: int, column: int) -> tuple[int, int]:
    return row, 2 - column


def _reflect_rotate_90(row: int, column: int) -> tuple[int, int]:
    return _rotate_90(*_reflect(row, column))


def _reflect_rotate_180(row: int, column: int) -> tuple[int, int]:
    return _rotate_180(*_reflect(row, column))


def _reflect_rotate_270(row: int, column: int) -> tuple[int, int]:
    return _rotate_270(*_reflect(row, column))


TRANSFORMS: tuple[CoordinateTransform, ...] = (
    _identity,
    _rotate_90,
    _rotate_180,
    _rotate_270,
    _reflect,
    _reflect_rotate_90,
    _reflect_rotate_180,
    _reflect_rotate_270,
)


def transform_board(board: Sequence[str], transform: CoordinateTransform) -> str:
    if len(board) != 9:
        raise ValueError("board must contain exactly nine cells")
    transformed = [" "] * 9
    for source, cell in enumerate(board):
        row, column = divmod(source, 3)
        new_row, new_column = transform(row, column)
        transformed[3 * new_row + new_column] = cell
    return "".join(transformed)


def transform_square(square: int, transform: CoordinateTransform) -> int:
    if square not in range(1, 10):
        raise ValueError("square must be between 1 and 9")
    row, column = divmod(square - 1, 3)
    new_row, new_column = transform(row, column)
    return 3 * new_row + new_column + 1


def symmetry_orbit(board: Sequence[str]) -> set[str]:
    """Return all unique symmetric forms of a board."""
    return {transform_board(board, transform) for transform in TRANSFORMS}


def canonical_board(board: Sequence[str]) -> str:
    """Return one stable key shared by all symmetric forms of a board."""
    return min(symmetry_orbit(board))


__all__ = [
    "TRANSFORMS",
    "canonical_board",
    "symmetry_orbit",
    "transform_board",
    "transform_square",
]
