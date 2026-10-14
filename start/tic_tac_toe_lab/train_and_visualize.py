"""Train Tic-Tac-Toe agents, then display their learning progress."""

from pathlib import Path

from q_learning import train_self_play
from visualization import plot_move_values, plot_training


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "artifacts" / "models"
CHARTS = ROOT / "artifacts" / "charts"


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    x_agent, o_agent, history = train_self_play(games=1_000_000, random_opening=True)
    x_agent.save(MODELS / "x_q_table.json")
    o_agent.save(MODELS / "o_q_table.json")
    print(f"Training complete: {len(history.outcomes):,} games with randomized legal openings.")
    print(f"Saved trained agents to {MODELS.relative_to(ROOT)}/.")
    training_figure = plot_training(history, show=False)
    training_figure.savefig(CHARTS / "training_progress.png", dpi=160)
    values_figure = plot_move_values(x_agent, [" "] * 9, show=False)
    values_figure.savefig(CHARTS / "empty_board_values.png", dpi=160)
    print(f"Saved charts to {CHARTS.relative_to(ROOT)}/.")

    # Display both charts for local interactive use.
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
