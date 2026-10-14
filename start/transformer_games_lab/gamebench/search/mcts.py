"""Small generic UCT search baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from gamebench.agents.base import Agent
from gamebench.games.base import GameState


@dataclass
class Node:
    state: GameState
    parent: "Node | None" = None
    action: int | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: list["Node"] = field(default_factory=list)
    unexpanded: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.unexpanded and not self.state.is_terminal():
            self.unexpanded = list(self.state.legal_actions())


@dataclass
class MCTSAgent(Agent):
    simulations: int = 100
    exploration: float = math.sqrt(2.0)
    name: str = "mcts"

    def select_action(self, state: GameState, rng: random.Random) -> int:
        root_player = state.current_player
        root = Node(state.clone())
        for _ in range(self.simulations):
            node = root
            while not node.unexpanded and node.children and not node.state.is_terminal():
                log_parent = math.log(max(1, node.visits))
                direction = 1.0 if node.state.current_player == root_player else -1.0
                node = max(
                    node.children,
                    key=lambda child: direction * child.value_sum / max(1, child.visits)
                    + self.exploration * math.sqrt(log_parent / max(1, child.visits)),
                )
            if node.unexpanded:
                action = rng.choice(node.unexpanded)
                node.unexpanded.remove(action)
                child_state = node.state.clone()
                child_state.apply_action(action)
                child = Node(child_state, node, action)
                node.children.append(child)
                node = child
            rollout = node.state.clone()
            while not rollout.is_terminal():
                rollout.apply_action(rng.choice(rollout.legal_actions()))
            value = rollout.returns()[root_player]
            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent
        if not root.children:
            raise ValueError("MCTSAgent received a terminal state")
        return max(root.children, key=lambda child: child.visits).action  # type: ignore[return-value]

    def metadata(self) -> dict[str, float | int | str]:
        return {**super().metadata(), "simulations": self.simulations, "exploration": self.exploration}
