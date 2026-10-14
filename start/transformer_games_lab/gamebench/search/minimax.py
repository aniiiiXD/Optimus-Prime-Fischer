"""Depth-limited negamax with alpha-beta pruning."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from gamebench.agents.base import Agent
from gamebench.games.base import GameState


def _default_evaluation(state: GameState, root_player: int) -> float:
    if state.is_terminal():
        return state.returns()[root_player]
    # Generic mobility heuristic. A game adapter can later inject a stronger
    # evaluator without changing the search algorithm.
    sign = 1.0 if state.current_player == root_player else -1.0
    return 0.01 * sign * len(state.legal_actions())


def negamax(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    root_player: int,
) -> float:
    if depth == 0 or state.is_terminal():
        return _default_evaluation(state, root_player)
    maximizing = state.current_player == root_player
    if maximizing:
        value = -math.inf
        for action in state.legal_actions():
            child = state.clone()
            child.apply_action(action)
            value = max(value, negamax(child, depth - 1, alpha, beta, root_player))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    value = math.inf
    for action in state.legal_actions():
        child = state.clone()
        child.apply_action(action)
        value = min(value, negamax(child, depth - 1, alpha, beta, root_player))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


@dataclass
class MinimaxAgent(Agent):
    depth: int = 3
    name: str = "minimax"

    def select_action(self, state: GameState, rng: random.Random) -> int:
        root_player = state.current_player
        scored: list[tuple[float, int]] = []
        for action in state.legal_actions():
            child = state.clone()
            child.apply_action(action)
            scored.append((negamax(child, self.depth - 1, -math.inf, math.inf, root_player), action))
        best = max(score for score, _ in scored)
        return rng.choice([action for score, action in scored if score == best])

    def metadata(self) -> dict[str, int | str]:
        return {**super().metadata(), "depth": self.depth}
