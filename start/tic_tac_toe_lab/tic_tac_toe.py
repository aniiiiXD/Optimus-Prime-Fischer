"""Reusable Tic-Tac-Toe module with a terminal interface and unbeatable AI.

Example:
    from tic_tac_toe import TicTacToe, best_move

    game = TicTacToe()
    game.move(1)  # X takes the top-left square
    computer_square = best_move(game.board, ai_mark="O")
    game.move(computer_square)
"""

from __future__ import annotations


WINNING_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)

__all__ = ["TicTacToe", "best_move", "available_moves", "winner"]


def winner(board: list[str]) -> str | None:
    """Return the winner (``'X'`` or ``'O'``), or ``None`` if there is none."""
    for first, second, third in WINNING_LINES:
        if board[first] != " " and board[first] == board[second] == board[third]:
            return board[first]
    return None


def available_moves(board: list[str]) -> list[int]:
    """Return available squares using human-friendly numbering (1-9)."""
    return [index + 1 for index, mark in enumerate(board) if mark == " "]


def best_move(board: list[str], ai_mark: str = "O") -> int:
    """Return the best available square (1-9) for ``ai_mark`` using minimax.

    The function does not mutate ``board``. It raises ``ValueError`` if the
    game has already ended, no squares are open, or the mark is invalid.
    """
    if ai_mark not in {"X", "O"}:
        raise ValueError("ai_mark must be 'X' or 'O'.")
    if winner(board) or not available_moves(board):
        raise ValueError("There is no valid move: the game is over.")

    human_mark = "O" if ai_mark == "X" else "X"
    best_score = float("-inf")
    selected_move = available_moves(board)[0]

    for move in available_moves(board):
        board[move - 1] = ai_mark
        score = _minimax(board, False, ai_mark, human_mark, depth=0)
        board[move - 1] = " "
        if score > best_score:
            best_score = score
            selected_move = move

    return selected_move


def _minimax(
    board: list[str],
    is_maximizing: bool,
    ai_mark: str,
    human_mark: str,
    depth: int,
) -> int:
    result = winner(board)
    if result == ai_mark:
        return 10 - depth
    if result == human_mark:
        return depth - 10
    if not available_moves(board):
        return 0

    mark = ai_mark if is_maximizing else human_mark
    scores = []
    for move in available_moves(board):
        board[move - 1] = mark
        scores.append(_minimax(board, not is_maximizing, ai_mark, human_mark, depth + 1))
        board[move - 1] = " "

    return max(scores) if is_maximizing else min(scores)


class TicTacToe:
    """State and rules for a two-player Tic-Tac-Toe game."""

    def __init__(self) -> None:
        self.board = [" "] * 9
        self.current_player = "X"

    @property
    def is_over(self) -> bool:
        """Whether the game has a winner or is a draw."""
        return winner(self.board) is not None or not available_moves(self.board)

    @property
    def is_draw(self) -> bool:
        """Whether the board is full without a winner."""
        return winner(self.board) is None and not available_moves(self.board)

    def move(self, square: int) -> None:
        """Place the current player's mark in square 1-9 and switch turns."""
        if self.is_over:
            raise ValueError("The game is already over.")
        if square not in range(1, 10):
            raise ValueError("Choose a square from 1 to 9.")
        if self.board[square - 1] != " ":
            raise ValueError("That square is already occupied.")

        self.board[square - 1] = self.current_player
        if not self.is_over:
            self.current_player = "O" if self.current_player == "X" else "X"

    def reset(self) -> None:
        """Start a fresh game."""
        self.board = [" "] * 9
        self.current_player = "X"


def show_board(board: list[str]) -> None:
    """Print a board, using numbers for available squares."""
    cells = [mark if mark != " " else str(index + 1) for index, mark in enumerate(board)]
    print(f"\n {cells[0]} | {cells[1]} | {cells[2]}")
    print("---+---+---")
    print(f" {cells[3]} | {cells[4]} | {cells[5]}")
    print("---+---+---")
    print(f" {cells[6]} | {cells[7]} | {cells[8]}\n")


def _human_move(game: TicTacToe) -> None:
    while True:
        choice = input(f"Player {game.current_player}, choose a square (1-9): ").strip()
        try:
            game.move(int(choice))
            return
        except ValueError as error:
            print(error)


def play_game(vs_computer: bool = True) -> str | None:
    """Play one terminal game and return the winner, or ``None`` for a draw."""
    game = TicTacToe()
    ai_mark = "O"

    while not game.is_over:
        show_board(game.board)
        if vs_computer and game.current_player == ai_mark:
            move = best_move(game.board, ai_mark)
            print(f"Computer chooses square {move}.")
            game.move(move)
        else:
            _human_move(game)

    show_board(game.board)
    result = winner(game.board)
    print(f"Player {result} wins!" if result else "It's a draw!")
    return result


def main() -> None:
    """Run the terminal version of the game."""
    print("Welcome to Tic-Tac-Toe!")
    while True:
        mode = input("Play against the computer? (y/n): ").strip().lower() == "y"
        play_game(vs_computer=mode)
        if input("Play again? (y/n): ").strip().lower() != "y":
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
