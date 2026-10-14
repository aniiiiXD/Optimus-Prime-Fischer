"""Play against the CPU-trained neural-network Tic-Tac-Toe model."""

from pathlib import Path

from mlp_agent import MLPAgent
from tic_tac_toe import TicTacToe, show_board, winner


MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "models" / "tic_tac_toe_mlp.npz"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first: python3 train_mlp.py")
    agent = MLPAgent.load(MODEL_PATH)
    game = TicTacToe()
    print("You are X. The CPU MLP is O.")

    while not game.is_over:
        show_board(game.board)
        if game.current_player == "X":
            try:
                game.move(int(input("Your move (1-9): ").strip()))
            except ValueError as error:
                print(error)
        else:
            top_three = agent.top_moves(game.board, "O")
            print("MLP top 3: " + ", ".join(f"{square} ({score:.1%})" for square, score in top_three))
            move = agent.choose_move(game.board, "O")
            print(f"MLP chooses square {move}.")
            game.move(move)

    show_board(game.board)
    result = winner(game.board)
    print("Draw!" if result is None else f"{'You' if result == 'X' else 'MLP'} win!")


if __name__ == "__main__":
    main()
