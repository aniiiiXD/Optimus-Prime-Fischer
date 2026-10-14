"""Self-play Q-learning for :mod:`tic_tac_toe`."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import random

from tic_tac_toe import TicTacToe, available_moves, winner


State = str


def encode_state(board: list[str]) -> State:
    """Create a compact, hashable representation of a board."""
    return "".join(board).replace(" ", "-")


class QLearningAgent:
    """A Q-learning Tic-Tac-Toe player with epsilon-greedy exploration."""

    def __init__(
        self,
        mark: str,
        learning_rate: float = 0.2,
        discount: float = 0.95,
        epsilon: float = 1.0,
        random_seed: int | None = None,
    ) -> None:
        if mark not in {"X", "O"}:
            raise ValueError("mark must be 'X' or 'O'.")
        self.mark = mark
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.q_table: dict[State, list[float]] = defaultdict(lambda: [0.0] * 9)
        self.updates: dict[State, list[int]] = defaultdict(lambda: [0] * 9)
        self._random = random.Random(random_seed)

    def choose_move(self, board: list[str], explore: bool = True) -> int:
        """Choose a legal square, optionally using exploration."""
        state = encode_state(board)
        legal_moves = available_moves(board)
        if not legal_moves:
            raise ValueError("No legal moves are available.")
        if explore and self._random.random() < self.epsilon:
            return self._random.choice(legal_moves)

        values = self.q_table[state]
        best_value = max(values[move - 1] for move in legal_moves)
        best_moves = [move for move in legal_moves if values[move - 1] == best_value]
        return self._random.choice(best_moves)

    def learn(
        self,
        state: State,
        move: int,
        reward: float,
        next_board: list[str] | None = None,
        terminal: bool = False,
    ) -> None:
        """Update one state-action value using the Q-learning rule."""
        old_value = self.q_table[state][move - 1]
        future_value = 0.0
        if not terminal and next_board is not None:
            next_moves = available_moves(next_board)
            if next_moves:
                next_values = self.q_table[encode_state(next_board)]
                future_value = max(next_values[next_move - 1] for next_move in next_moves)

        target = reward + self.discount * future_value
        self.q_table[state][move - 1] = old_value + self.learning_rate * (target - old_value)
        self.updates[state][move - 1] += 1

    def decay_exploration(self, factor: float, minimum: float) -> None:
        """Reduce random exploration, without dropping below ``minimum``."""
        self.epsilon = max(minimum, self.epsilon * factor)

    def save(self, path: str | Path) -> None:
        """Save learned values and settings to a JSON file."""
        data = {
            "mark": self.mark,
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "epsilon": self.epsilon,
            "q_table": dict(self.q_table),
            "updates": dict(self.updates),
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        """Load a previously saved agent from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            mark=data["mark"],
            learning_rate=data["learning_rate"],
            discount=data["discount"],
            epsilon=data["epsilon"],
        )
        agent.q_table.update(data["q_table"])
        agent.updates.update(data.get("updates", {}))
        return agent


@dataclass
class TrainingHistory:
    """Per-game measurements for charts and training inspection."""

    outcomes: list[str] = field(default_factory=list)
    moves_per_game: list[int] = field(default_factory=list)
    epsilon: list[float] = field(default_factory=list)

    def record(self, outcome: str, moves: int, exploration_rate: float) -> None:
        self.outcomes.append(outcome)
        self.moves_per_game.append(moves)
        self.epsilon.append(exploration_rate)


def train_self_play(
    games: int,
    x_agent: QLearningAgent | None = None,
    o_agent: QLearningAgent | None = None,
    decay: float = 0.9995,
    minimum_epsilon: float = 0.02,
    random_opening: bool = True,
    random_seed: int | None = None,
) -> tuple[QLearningAgent, QLearningAgent, TrainingHistory]:
    """Train two agents through self-play with varied, legal opening boards.

    When ``random_opening`` is enabled, games begin after 0-4 random legal
    moves. Empty boards remain most common, while unusual corner/fork setups
    are deliberately encountered often enough to be learned.
    """
    x_agent = x_agent or QLearningAgent("X")
    o_agent = o_agent or QLearningAgent("O")
    if x_agent.mark != "X" or o_agent.mark != "O":
        raise ValueError("x_agent must use X and o_agent must use O.")

    history = TrainingHistory()
    agents = {"X": x_agent, "O": o_agent}
    opening_random = random.Random(random_seed)
    opening_depths = [0] * 40 + [1] * 15 + [2] * 15 + [3] * 15 + [4] * 15

    for _ in range(games):
        game = TicTacToe()
        last_action: dict[str, tuple[State, int] | None] = {"X": None, "O": None}

        # Random, valid opening positions broaden coverage beyond one opening.
        if random_opening:
            for _ in range(opening_random.choice(opening_depths)):
                game.move(opening_random.choice(available_moves(game.board)))

        while not game.is_over:
            agent = agents[game.current_player]

            # The agent's previous move now has a resulting next state.
            previous = last_action[agent.mark]
            if previous is not None:
                state, move = previous
                agent.learn(state, move, reward=0.0, next_board=game.board)

            state = encode_state(game.board)
            move = agent.choose_move(game.board)
            last_action[agent.mark] = (state, move)
            game.move(move)

        outcome = winner(game.board) or "draw"
        # Tic-Tac-Toe is zero-sum: draws should not receive a win-like bonus.
        rewards = {"X": 0.0 if outcome == "draw" else (1.0 if outcome == "X" else -1.0),
                   "O": 0.0 if outcome == "draw" else (1.0 if outcome == "O" else -1.0)}
        for mark, agent in agents.items():
            previous = last_action[mark]
            if previous is not None:
                state, move = previous
                agent.learn(state, move, reward=rewards[mark], terminal=True)

        history.record(outcome, 9 - len(available_moves(game.board)), (x_agent.epsilon + o_agent.epsilon) / 2)
        x_agent.decay_exploration(decay, minimum_epsilon)
        o_agent.decay_exploration(decay, minimum_epsilon)

    return x_agent, o_agent, history
