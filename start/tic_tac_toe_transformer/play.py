"""Play Tic-Tac-Toe against the trained Transformer in the terminal."""

from __future__ import annotations

import argparse
from pathlib import Path

from tic_tac_toe_lab.tic_tac_toe import (
    TicTacToe,
    show_board,
    winner,
)
from tic_tac_toe_transformer.agents import TransformerAgent


DEFAULT_CHECKPOINT = (
    Path(__file__).parent / "artifacts" / "checkpoints" / "optimal_policy.pt"
)


def transformer_move(
    agent: TransformerAgent, game: TicTacToe
) -> tuple[int, list[tuple[int, float]]]:
    top_moves = agent.top_moves(game.board, game.current_player, count=3)
    return top_moves[0][0], top_moves


def play_one_game(agent: TransformerAgent, human_mark: str) -> None:
    game = TicTacToe()
    ai_mark = "O" if human_mark == "X" else "X"
    print(f"\nYou are {human_mark}; Transformer is {ai_mark}.")
    print("Choose a numbered empty square. Enter q to quit.\n")

    while not game.is_over:
        show_board(game.board)
        if game.current_player == human_mark:
            try:
                response = input(f"Your move ({human_mark}) [1-9]: ").strip().lower()
            except EOFError:
                print("\nGame ended.")
                return
            if response in {"q", "quit", "exit"}:
                print("Game ended.")
                return
            try:
                game.move(int(response))
            except ValueError as error:
                print(f"{error} Try again.")
        else:
            move, top_moves = transformer_move(agent, game)
            picks = ", ".join(f"{square} ({prob:.0%})" for square, prob in top_moves)
            print(f"Transformer's top picks: {picks}")
            print(f"Transformer plays square {move}.")
            game.move(move)

    show_board(game.board)
    result = winner(game.board)
    if result == human_mark:
        print("You win!")
    elif result == ai_mark:
        print("Transformer wins!")
    else:
        print("It's a draw!")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--human", choices=("X", "O"), default="X", help="your mark; default: X")
    args = parser.parse_args()

    agent = TransformerAgent(args.checkpoint)
    print("Loaded trained Tic-Tac-Toe Transformer on CPU.")
    while True:
        play_one_game(agent, args.human)
        try:
            again = input("Play again? [y/N]: ").strip().lower()
        except EOFError:
            break
        if again not in {"y", "yes"}:
            break


if __name__ == "__main__":
    main()
