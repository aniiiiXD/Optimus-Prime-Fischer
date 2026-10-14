"""Two-player terminal interface for validating the Connect Four engine."""

from __future__ import annotations

from .engine import COLUMNS, EMPTY, PLAYER_ONE, ConnectFour, IllegalMoveError


SYMBOLS = {EMPTY: "·", PLAYER_ONE: "X", -PLAYER_ONE: "O"}


def render(game: ConnectFour) -> None:
    highlighted = set(game.winning_line)
    print()
    for row in range(6):
        cells = []
        for column in range(COLUMNS):
            symbol = SYMBOLS[game.board[row * COLUMNS + column]]
            cells.append(f"[{symbol}]" if (row, column) in highlighted else f" {symbol} ")
        print("".join(cells))
    print(" " + "  ".join(str(column) for column in range(1, COLUMNS + 1)))
    print()


def main() -> None:
    game = ConnectFour()
    print("Connect Four — Player 1 is X, Player 2 is O.")
    while not game.is_terminal:
        render(game)
        label = "X" if game.current_player == PLAYER_ONE else "O"
        try:
            human_column = int(input(f"Player {label}, choose column 1–7: ").strip())
            game.step(human_column - 1)
        except (ValueError, IllegalMoveError) as error:
            print(f"Invalid move: {error}")
    render(game)
    if game.is_draw:
        print("Draw.")
    else:
        label = "X" if game.winner == PLAYER_ONE else "O"
        print(f"Player {label} wins.")


if __name__ == "__main__":
    main()
