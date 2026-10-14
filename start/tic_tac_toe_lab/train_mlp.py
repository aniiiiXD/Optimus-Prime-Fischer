"""Train the CPU MLP on the exact minimax policy for every reachable board."""

from __future__ import annotations

from functools import lru_cache
import time
from pathlib import Path

import numpy as np

from mlp_agent import MLPAgent, encode_board
from tic_tac_toe import available_moves, winner


# Exact, reproducible training settings.
SEED = 7
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 128
EPOCHS = 1_200
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "models" / "tic_tac_toe_mlp.npz"


def other(player: str) -> str:
    return "O" if player == "X" else "X"


@lru_cache(maxsize=None)
def minimax_value(board_key: str, player: str) -> int:
    """Return -1, 0, or 1 for the current player under perfect play."""
    board = list(board_key)
    result = winner(board)
    if result is not None:
        return -1  # The prior player made the winning move.
    legal_moves = available_moves(board)
    if not legal_moves:
        return 0

    values = []
    for move in legal_moves:
        board[move - 1] = player
        values.append(-minimax_value("".join(board), other(player)))
        board[move - 1] = " "
    return max(values)


def optimal_target(board: list[str], player: str) -> np.ndarray:
    """Return equal target probability for every minimax-optimal legal move."""
    legal_moves = available_moves(board)
    scores = []
    for move in legal_moves:
        board[move - 1] = player
        scores.append(-minimax_value("".join(board), other(player)))
        board[move - 1] = " "
    best_score = max(scores)
    best_moves = [move for move, score in zip(legal_moves, scores) if score == best_score]
    target = np.zeros(9, dtype=np.float32)
    target[[move - 1 for move in best_moves]] = 1 / len(best_moves)
    return target


def reachable_positions() -> list[tuple[list[str], str]]:
    """Enumerate every non-terminal legal position and its player to move."""
    positions: list[tuple[list[str], str]] = []
    seen: set[tuple[str, str]] = set()

    def visit(board: list[str], player: str) -> None:
        if winner(board) or not available_moves(board):
            return
        key = ("".join(board), player)
        if key in seen:
            return
        seen.add(key)
        positions.append((board.copy(), player))
        for move in available_moves(board):
            board[move - 1] = player
            visit(board, other(player))
            board[move - 1] = " "

    visit([" "] * 9, "X")
    return positions


def make_dataset() -> tuple[np.ndarray, np.ndarray, list[tuple[list[str], str]]]:
    positions = reachable_positions()
    inputs = np.stack([encode_board(board, player) for board, player in positions])
    targets = np.stack([optimal_target(board.copy(), player) for board, player in positions])
    return inputs, targets, positions


def evaluate(agent: MLPAgent, inputs: np.ndarray, targets: np.ndarray) -> float:
    probabilities = agent.probabilities(inputs)
    predicted = probabilities.argmax(axis=1)
    return float(np.mean(targets[np.arange(len(targets)), predicted] > 0))


def main() -> None:
    started = time.perf_counter()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    inputs, targets, _ = make_dataset()
    rng = np.random.default_rng(SEED)
    agent = MLPAgent(hidden_size=HIDDEN_SIZE, seed=SEED)

    for epoch in range(1, EPOCHS + 1):
        indices = rng.permutation(len(inputs))
        losses = []
        for start in range(0, len(inputs), BATCH_SIZE):
            batch = indices[start:start + BATCH_SIZE]
            losses.append(agent.train_batch(
                inputs[batch], targets[batch], LEARNING_RATE, WEIGHT_DECAY,
                ADAM_BETA_1, ADAM_BETA_2,
            ))
        if epoch in {1, 100, 300, 600, EPOCHS}:
            accuracy = evaluate(agent, inputs, targets)
            print(f"epoch {epoch:4}: loss={np.mean(losses):.4f}, optimal-move accuracy={accuracy:.2%}")

    agent.save(MODEL_PATH)
    print(f"Saved {MODEL_PATH.relative_to(ROOT)} after {time.perf_counter() - started:.2f}s.")


if __name__ == "__main__":
    main()
