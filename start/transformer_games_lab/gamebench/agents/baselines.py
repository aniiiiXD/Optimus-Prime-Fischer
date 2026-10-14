from __future__ import annotations

import random

from gamebench.agents.base import Agent
from gamebench.games.base import GameState


class RandomAgent(Agent):
    name = "random"

    def select_action(self, state: GameState, rng: random.Random) -> int:
        legal = state.legal_actions()
        if not legal:
            raise ValueError("RandomAgent received a terminal state")
        return rng.choice(legal)


class GreedyAgent(Agent):
    """Game-independent one-ply baseline.

    It takes an immediate win when available. Otherwise it minimizes the next
    player's legal options, with deterministic center preference when actions
    are ordered spatially. This is a baseline, not a claim of strong play.
    """

    name = "greedy"

    def select_action(self, state: GameState, rng: random.Random) -> int:
        player = state.current_player
        scored: list[tuple[float, int]] = []
        middle = (state.spec.action_space_size - 1) / 2
        for action in state.legal_actions():
            child = state.clone()
            child.apply_action(action)
            if child.is_terminal():
                score = 10_000 * child.returns()[player]
            else:
                score = -len(child.legal_actions()) - 0.001 * abs(action - middle)
            scored.append((score, action))
        best = max(score for score, _ in scored)
        choices = [action for score, action in scored if score == best]
        return rng.choice(choices)
