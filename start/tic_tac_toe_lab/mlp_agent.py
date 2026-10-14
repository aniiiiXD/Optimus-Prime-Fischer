"""A small CPU-only NumPy multilayer perceptron for Tic-Tac-Toe."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from tic_tac_toe import available_moves


def encode_board(board: list[str], player: str) -> np.ndarray:
    """Encode a board from the current player's viewpoint: self=1, foe=-1."""
    opponent = "O" if player == "X" else "X"
    return np.array(
        [1.0 if square == player else -1.0 if square == opponent else 0.0 for square in board],
        dtype=np.float32,
    )


class MLPAgent:
    """9 → 64 → 64 → 9 ReLU network with Adam training on CPU."""

    def __init__(self, hidden_size: int = 64, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.hidden_size = hidden_size
        self.params = {
            "W1": (rng.standard_normal((9, hidden_size)) * np.sqrt(2 / 9)).astype(np.float32),
            "b1": np.zeros(hidden_size, dtype=np.float32),
            "W2": (rng.standard_normal((hidden_size, hidden_size)) * np.sqrt(2 / hidden_size)).astype(np.float32),
            "b2": np.zeros(hidden_size, dtype=np.float32),
            "W3": (rng.standard_normal((hidden_size, 9)) * np.sqrt(2 / hidden_size)).astype(np.float32),
            "b3": np.zeros(9, dtype=np.float32),
        }
        self.adam_m = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.adam_v = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.step = 0

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Return output logits and intermediate values needed for backprop."""
        z1 = inputs @ self.params["W1"] + self.params["b1"]
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.params["W2"] + self.params["b2"]
        h2 = np.maximum(z2, 0)
        logits = h2 @ self.params["W3"] + self.params["b3"]
        return logits, (inputs, z1, h1, z2, h2)

    def probabilities(self, inputs: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(inputs)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=-1, keepdims=True)

    def train_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
        weight_decay: float = 1e-5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
    ) -> float:
        """Run one Adam update using soft-target cross-entropy loss."""
        logits, (x, z1, h1, z2, h2) = self.forward(inputs)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = -np.mean(np.sum(targets * np.log(probabilities + 1e-8), axis=1))

        batch_size = inputs.shape[0]
        gradient_logits = (probabilities - targets) / batch_size
        gradients = {
            "W3": h2.T @ gradient_logits + weight_decay * self.params["W3"],
            "b3": gradient_logits.sum(axis=0),
        }
        gradient_h2 = gradient_logits @ self.params["W3"].T
        gradient_z2 = gradient_h2 * (z2 > 0)
        gradients["W2"] = h1.T @ gradient_z2 + weight_decay * self.params["W2"]
        gradients["b2"] = gradient_z2.sum(axis=0)
        gradient_h1 = gradient_z2 @ self.params["W2"].T
        gradient_z1 = gradient_h1 * (z1 > 0)
        gradients["W1"] = x.T @ gradient_z1 + weight_decay * self.params["W1"]
        gradients["b1"] = gradient_z1.sum(axis=0)

        self.step += 1
        for name, gradient in gradients.items():
            self.adam_m[name] = beta1 * self.adam_m[name] + (1 - beta1) * gradient
            self.adam_v[name] = beta2 * self.adam_v[name] + (1 - beta2) * gradient**2
            corrected_m = self.adam_m[name] / (1 - beta1**self.step)
            corrected_v = self.adam_v[name] / (1 - beta2**self.step)
            self.params[name] -= learning_rate * corrected_m / (np.sqrt(corrected_v) + adam_epsilon)
        return float(loss)

    def choose_move(self, board: list[str], player: str) -> int:
        """Choose the legal square with the highest neural-network probability."""
        legal_moves = available_moves(board)
        if not legal_moves:
            raise ValueError("No legal moves are available.")
        probabilities = self.probabilities(encode_board(board, player)[None, :])[0]
        return max(legal_moves, key=lambda square: probabilities[square - 1])

    def top_moves(self, board: list[str], player: str, limit: int = 3) -> list[tuple[int, float]]:
        """Return the most likely legal moves and their network probabilities."""
        probabilities = self.probabilities(encode_board(board, player)[None, :])[0]
        legal_moves = available_moves(board)
        ranked = sorted(legal_moves, key=lambda square: probabilities[square - 1], reverse=True)
        return [(square, float(probabilities[square - 1])) for square in ranked[:limit]]

    def save(self, path: str | Path) -> None:
        """Save network parameters in NumPy's compact binary format."""
        np.savez(path, hidden_size=self.hidden_size, **self.params)

    @classmethod
    def load(cls, path: str | Path) -> "MLPAgent":
        """Load saved network parameters."""
        data = np.load(path)
        agent = cls(hidden_size=int(data["hidden_size"]))
        for name in agent.params:
            agent.params[name] = data[name]
        return agent
