"""Play in a terminal while seeing the shared model's top candidate decisions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tic_tac_toe_lab.tic_tac_toe import TicTacToe, available_moves, winner

from tic_tac_toe_shared_transformer.agent import SharedTransformerAgent


DEFAULT_CHECKPOINT = (
    Path(__file__).parent / "artifacts" / "checkpoints" / "shared_multitask_seed13.pt"
)


def render(board: list[str]) -> None:
    cells = [cell if cell != " " else str(index + 1) for index, cell in enumerate(board)]
    print(f"\n {cells[0]} | {cells[1]} | {cells[2]}")
    print("---+---+---")
    print(f" {cells[3]} | {cells[4]} | {cells[5]}")
    print("---+---+---")
    print(f" {cells[6]} | {cells[7]} | {cells[8]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--human", choices=("X", "O"), default="X")
    args = parser.parse_args()

    agent = SharedTransformerAgent(args.checkpoint)
    game = TicTacToe()
    while not game.is_over:
        render(game.board)
        mark = game.current_player
        if mark == args.human:
            legal = available_moves(game.board)
            while True:
                try:
                    move = int(input(f"Your move ({', '.join(map(str, legal))}): "))
                except (ValueError, EOFError):
                    print("Enter one of the displayed square numbers.")
                    continue
                if move in legal:
                    break
                print("That square is not legal.")
        else:
            decision = agent.inspect(game.board, mark)
            top = decision["choices"][:3]
            print(
                "Model board value: "
                f"{decision['predicted_state_outcome']} | top candidates: "
                + ", ".join(
                    f"{choice['move']}={choice['policy_probability']:.1%}"
                    f"/{choice['predicted_outcome']}"
                    for choice in top
                )
            )
            move = int(top[0]["move"])
            print(f"Model plays {move}.")
        game.move(move)

    render(game.board)
    result = winner(game.board)
    print("Draw." if result is None else f"{result} wins.")


if __name__ == "__main__":
    main()
