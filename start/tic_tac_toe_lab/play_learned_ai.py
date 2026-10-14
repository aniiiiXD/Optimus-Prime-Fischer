"""Play Tic-Tac-Toe against the Q-learning agent trained by this project."""

from pathlib import Path

from q_learning import QLearningAgent
from tic_tac_toe import TicTacToe, show_board, winner
from visualization import format_move_decision


MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "models" / "o_q_table.json"


def human_move(game: TicTacToe) -> None:
    """Ask the human player for a legal move."""
    while True:
        try:
            game.move(int(input("Your move, choose a square (1-9): ").strip()))
            return
        except ValueError as error:
            print(error)


def play() -> None:
    """Run one game: human X versus learned AI O."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run: python3 train_and_visualize.py")

    ai = QLearningAgent.load(MODEL_PATH)
    game = TicTacToe()
    print("You are X. The learning AI is O.")

    while not game.is_over:
        show_board(game.board)
        if game.current_player == "X":
            human_move(game)
        else:
            move = ai.choose_move(game.board, explore=False)
            print(format_move_decision(ai, game.board, selected_move=move))
            print(f"AI chooses square {move}.")
            game.move(move)

    show_board(game.board)
    result = winner(game.board)
    print("It's a draw!" if result is None else f"{'You' if result == 'X' else 'AI'} win!")


if __name__ == "__main__":
    play()
