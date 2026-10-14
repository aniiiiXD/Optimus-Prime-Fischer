"""Charts for inspecting Tic-Tac-Toe Q-learning training."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from q_learning import QLearningAgent, TrainingHistory


def format_move_decision(
    agent: QLearningAgent, board: list[str], selected_move: int | None = None
) -> str:
    """Return the AI's top three learned move choices for the current board."""
    from q_learning import encode_state
    from tic_tac_toe import available_moves

    state = encode_state(board)
    legal_moves = available_moves(board)
    values = agent.q_table[state]
    updates = agent.updates[state]
    selected = selected_move if selected_move is not None else agent.choose_move(board, explore=False)
    ranked_moves = sorted(legal_moves, key=lambda square: values[square - 1], reverse=True)[:3]
    lines = ["\nAI's top 3 learned choices", "rank | square | Q-score | training updates | choice"]
    for rank, square in enumerate(ranked_moves, start=1):
        choice = "SELECTED" if square == selected else "alternative"
        lines.append(
            f" {rank}   |   {square}    | {values[square - 1]:>7.3f} |"
            f" {updates[square - 1]:>8}         | {choice}"
        )
    lines.append("Higher Q-score means a better learned expected outcome.")
    return "\n".join(lines)


def plot_training(history: TrainingHistory, window: int = 250, show: bool = True):
    """Plot rolling outcomes, exploration, and game length.

    Returns the Matplotlib figure so callers can save it with
    ``figure.savefig('training.png')``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Install plotting support with: python3 -m pip install matplotlib") from error

    if not history.outcomes:
        raise ValueError("Training history is empty.")

    episodes = list(range(1, len(history.outcomes) + 1))
    x_wins = _rolling_rate(history.outcomes, "X", window)
    o_wins = _rolling_rate(history.outcomes, "O", window)
    draws = _rolling_rate(history.outcomes, "draw", window)
    moves = _rolling_mean(history.moves_per_game, window)

    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    outcome_axis, epsilon_axis, moves_axis, summary_axis = axes.flat

    outcome_axis.plot(episodes, x_wins, label="X wins", color="#2563eb")
    outcome_axis.plot(episodes, o_wins, label="O wins", color="#dc2626")
    outcome_axis.plot(episodes, draws, label="Draws", color="#16a34a")
    outcome_axis.set(title=f"Rolling outcome rate ({window} games)", xlabel="Game", ylabel="Rate", ylim=(0, 1))
    outcome_axis.legend()
    outcome_axis.grid(alpha=0.25)

    epsilon_axis.plot(episodes, history.epsilon, color="#7c3aed")
    epsilon_axis.set(title="Exploration rate", xlabel="Game", ylabel="Epsilon", ylim=(0, 1))
    epsilon_axis.grid(alpha=0.25)

    moves_axis.plot(episodes, moves, color="#ea580c")
    moves_axis.set(title=f"Rolling game length ({window} games)", xlabel="Game", ylabel="Moves")
    moves_axis.grid(alpha=0.25)

    totals = {label: history.outcomes.count(label) for label in ("X", "O", "draw")}
    summary_axis.bar(totals.keys(), totals.values(), color=["#2563eb", "#dc2626", "#16a34a"])
    summary_axis.set(title="Total training outcomes", ylabel="Games")

    if show:
        plt.show()
    return figure


def plot_move_values(agent: QLearningAgent, board: list[str], show: bool = True):
    """Show the agent's learned move values overlaid on a board."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise ImportError("Install plotting support with: python3 -m pip install matplotlib numpy") from error

    from q_learning import encode_state

    values = agent.q_table[encode_state(board)]
    data = np.array(values, dtype=float).reshape(3, 3)
    figure, axis = plt.subplots(figsize=(5, 5))
    image = axis.imshow(data, cmap="RdYlGn")
    figure.colorbar(image, ax=axis, label="Learned Q-value")
    axis.set(xticks=range(3), yticks=range(3), xticklabels=(1, 2, 3), yticklabels=(1, 2, 3))
    axis.set_title(f"{agent.mark}'s learned move values")
    for index, value in enumerate(values):
        row, column = divmod(index, 3)
        label = board[index] if board[index] != " " else f"{value:.2f}"
        axis.text(column, row, label, ha="center", va="center", fontsize=13, fontweight="bold")

    if show:
        plt.show()
    return figure


def _rolling_rate(outcomes: list[str], target: str, window: int) -> list[float]:
    values = [1.0 if item == target else 0.0 for item in outcomes]
    return _rolling_mean(values, window)


def _rolling_mean(values: list[float | int], window: int) -> list[float]:
    result = []
    running_total = 0.0
    for index, value in enumerate(values):
        running_total += value
        if index >= window:
            running_total -= values[index - window]
        result.append(running_total / min(index + 1, window))
    return result
