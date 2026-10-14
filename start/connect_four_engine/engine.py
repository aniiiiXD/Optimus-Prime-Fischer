"""Fast, inspectable Connect Four state transitions and rule checks.

Representation
--------------
The board is a flat 42-value list in row-major order. Row 0 is the top and row
5 is the bottom. Values are ``0`` (empty), ``1`` (Player 1 / X), and ``-1``
(Player 2 / O). Actions are zero-based columns 0..6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


ROWS = 6
COLUMNS = 7
CONNECT = 4
BOARD_CELLS = ROWS * COLUMNS

EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = -1
PLAYERS = (PLAYER_ONE, PLAYER_TWO)

DIRECTIONS = (
    (0, 1),   # horizontal
    (1, 0),   # vertical
    (1, 1),   # top-left to bottom-right
    (1, -1),  # top-right to bottom-left
)


class IllegalMoveError(ValueError):
    """Raised when an action cannot be applied to the current state."""


@dataclass(frozen=True)
class Move:
    """One applied action and its landing coordinate."""

    column: int
    row: int
    player: int


@dataclass(frozen=True)
class MoveResult:
    """Information returned by :meth:`ConnectFour.step`.

    ``reward`` is from the acting player's perspective: 1 for a win and 0 for
    an unfinished game or draw. In zero-sum training, the opponent receives
    the negated terminal reward.
    """

    move: Move
    reward: int
    winner: int | None
    is_terminal: bool
    is_draw: bool
    winning_line: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ConnectFourState:
    """Immutable snapshot suitable for logging, datasets, and cache keys."""

    board: tuple[int, ...]
    current_player: int
    winner: int | None
    move_count: int
    is_terminal: bool

    @property
    def is_draw(self) -> bool:
        return self.is_terminal and self.winner is None


def _index(row: int, column: int) -> int:
    return row * COLUMNS + column


def _validate_board(board: Sequence[int]) -> None:
    if len(board) != BOARD_CELLS:
        raise ValueError(f"board must contain exactly {BOARD_CELLS} cells")
    if any(cell not in (EMPTY, PLAYER_ONE, PLAYER_TWO) for cell in board):
        raise ValueError("board cells must be -1, 0, or 1")


def find_winner(board: Sequence[int]) -> int | None:
    """Return the first player with four connected pieces, otherwise ``None``."""
    _validate_board(board)
    for row in range(ROWS):
        for column in range(COLUMNS):
            player = board[_index(row, column)]
            if player == EMPTY:
                continue
            for row_step, column_step in DIRECTIONS:
                end_row = row + (CONNECT - 1) * row_step
                end_column = column + (CONNECT - 1) * column_step
                if not (0 <= end_row < ROWS and 0 <= end_column < COLUMNS):
                    continue
                if all(
                    board[_index(row + offset * row_step, column + offset * column_step)]
                    == player
                    for offset in range(1, CONNECT)
                ):
                    return player
    return None


def mirror_action(column: int) -> int:
    """Map a column to its left-right reflection."""
    if isinstance(column, bool) or not isinstance(column, int) or column not in range(COLUMNS):
        raise ValueError("column must be an integer from 0 to 6")
    return COLUMNS - 1 - column


def mirror_board(board: Sequence[int]) -> tuple[int, ...]:
    """Return the board reflected across its vertical center line."""
    _validate_board(board)
    return tuple(
        board[_index(row, COLUMNS - 1 - column)]
        for row in range(ROWS)
        for column in range(COLUMNS)
    )


class ConnectFour:
    """Mutable Connect Four engine optimized for step/undo search.

    The object changes in place. Use :meth:`snapshot` for an immutable record or
    :meth:`copy` when a separate mutable branch is required.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> ConnectFourState:
        """Return the engine to the empty starting position."""
        self.board = [EMPTY] * BOARD_CELLS
        self.current_player = PLAYER_ONE
        self.winner: int | None = None
        self.move_count = 0
        self.column_heights = [0] * COLUMNS
        self.history: list[Move] = []
        self.winning_line: tuple[tuple[int, int], ...] = ()
        return self.snapshot()

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None or self.move_count == BOARD_CELLS

    @property
    def is_draw(self) -> bool:
        return self.move_count == BOARD_CELLS and self.winner is None

    def legal_actions(self) -> tuple[int, ...]:
        """Return playable zero-based columns in left-to-right order."""
        if self.is_terminal:
            return ()
        return tuple(
            column for column, height in enumerate(self.column_heights) if height < ROWS
        )

    def step(self, column: int) -> MoveResult:
        """Drop the current player's piece into ``column`` and return the transition."""
        if self.is_terminal:
            raise IllegalMoveError("the game is already over")
        if isinstance(column, bool) or not isinstance(column, int):
            raise IllegalMoveError("column must be an integer from 0 to 6")
        if column not in range(COLUMNS):
            raise IllegalMoveError("column must be between 0 and 6")
        if self.column_heights[column] == ROWS:
            raise IllegalMoveError(f"column {column} is full")

        player = self.current_player
        row = ROWS - 1 - self.column_heights[column]
        move = Move(column=column, row=row, player=player)
        self.board[_index(row, column)] = player
        self.column_heights[column] += 1
        self.move_count += 1
        self.history.append(move)

        line = self._winning_line_from(row, column, player)
        if line:
            self.winner = player
            self.winning_line = line
        elif self.move_count < BOARD_CELLS:
            self.current_player = -player

        return MoveResult(
            move=move,
            reward=1 if self.winner == player else 0,
            winner=self.winner,
            is_terminal=self.is_terminal,
            is_draw=self.is_draw,
            winning_line=self.winning_line,
        )

    def undo(self) -> Move:
        """Undo and return the latest move in O(1), enabling recursive search."""
        if not self.history:
            raise IllegalMoveError("there is no move to undo")
        move = self.history.pop()
        self.board[_index(move.row, move.column)] = EMPTY
        self.column_heights[move.column] -= 1
        self.move_count -= 1
        self.current_player = move.player
        self.winner = None
        self.winning_line = ()
        return move

    def snapshot(self) -> ConnectFourState:
        """Return an immutable copy of the current logical state."""
        return ConnectFourState(
            board=tuple(self.board),
            current_player=self.current_player,
            winner=self.winner,
            move_count=self.move_count,
            is_terminal=self.is_terminal,
        )

    def state_key(self) -> tuple[tuple[int, ...], int]:
        """Return a hashable key for transposition tables and memoization."""
        return tuple(self.board), self.current_player

    def canonical_key(self) -> tuple[tuple[int, ...], int]:
        """Return one cache key shared by a board and its mirror image."""
        board = tuple(self.board)
        mirrored = mirror_board(board)
        return min(board, mirrored), self.current_player

    def encode(self, perspective: int | None = None) -> tuple[int, ...]:
        """Encode cells relative to ``perspective``: own=1, opponent=-1, empty=0."""
        player = self.current_player if perspective is None else perspective
        if player not in PLAYERS:
            raise ValueError("perspective must be 1 or -1")
        return tuple(cell * player for cell in self.board)

    def copy(self) -> ConnectFour:
        """Return an independent mutable engine with the same state."""
        clone = ConnectFour()
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.winner = self.winner
        clone.move_count = self.move_count
        clone.column_heights = self.column_heights.copy()
        clone.history = self.history.copy()
        clone.winning_line = self.winning_line
        return clone

    @classmethod
    def from_moves(cls, columns: Sequence[int]) -> ConnectFour:
        """Build a state by replaying a sequence of zero-based columns."""
        game = cls()
        for column in columns:
            game.step(column)
        return game

    def _winning_line_from(
        self, row: int, column: int, player: int
    ) -> tuple[tuple[int, int], ...]:
        for row_step, column_step in DIRECTIONS:
            coordinates = [(row, column)]
            for sign in (-1, 1):
                distance = 1
                branch: list[tuple[int, int]] = []
                while True:
                    candidate_row = row + sign * distance * row_step
                    candidate_column = column + sign * distance * column_step
                    if not (
                        0 <= candidate_row < ROWS and 0 <= candidate_column < COLUMNS
                    ):
                        break
                    if self.board[_index(candidate_row, candidate_column)] != player:
                        break
                    branch.append((candidate_row, candidate_column))
                    distance += 1
                if sign < 0:
                    coordinates = list(reversed(branch)) + coordinates
                else:
                    coordinates.extend(branch)
            if len(coordinates) >= CONNECT:
                move_index = coordinates.index((row, column))
                start = min(max(move_index - (CONNECT - 1), 0), len(coordinates) - CONNECT)
                return tuple(coordinates[start : start + CONNECT])
        return ()

    def assert_invariants(self) -> None:
        """Raise ``AssertionError`` if internal gravity/count caches disagree."""
        assert len(self.board) == BOARD_CELLS
        assert self.move_count == sum(cell != EMPTY for cell in self.board)
        assert self.move_count == len(self.history)
        for column in range(COLUMNS):
            occupied = sum(
                self.board[_index(row, column)] != EMPTY for row in range(ROWS)
            )
            assert occupied == self.column_heights[column]
            for row in range(ROWS - occupied):
                assert self.board[_index(row, column)] == EMPTY
            for row in range(ROWS - occupied, ROWS):
                assert self.board[_index(row, column)] != EMPTY
        assert self.winner == find_winner(self.board)


__all__ = [
    "BOARD_CELLS",
    "COLUMNS",
    "CONNECT",
    "EMPTY",
    "PLAYER_ONE",
    "PLAYER_TWO",
    "ROWS",
    "ConnectFour",
    "ConnectFourState",
    "IllegalMoveError",
    "Move",
    "MoveResult",
    "find_winner",
    "mirror_action",
    "mirror_board",
]
