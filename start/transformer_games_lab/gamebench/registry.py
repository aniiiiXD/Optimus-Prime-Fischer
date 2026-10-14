"""Central factories: CLI strings become replaceable game and agent plugins."""

from __future__ import annotations

from collections.abc import Callable

from gamebench.agents.base import Agent
from gamebench.agents.baselines import GreedyAgent, RandomAgent
from gamebench.agents.transformer import TransformerAgent
from gamebench.games.base import GameAdapter
from gamebench.games.connect4 import ConnectFourAdapter
from gamebench.games.othello import OthelloAdapter
from gamebench.search.mcts import MCTSAgent
from gamebench.search.minimax import MinimaxAgent


GAME_FACTORIES: dict[str, Callable[[], GameAdapter]] = {
    "othello": OthelloAdapter,
    "connect4": ConnectFourAdapter,
}


def make_game(name: str) -> GameAdapter:
    try:
        return GAME_FACTORIES[name.lower()]()
    except KeyError as error:
        raise ValueError(f"unknown game {name!r}; choose from {sorted(GAME_FACTORIES)}") from error


def make_agent(name: str, expected_game: str | None = None) -> Agent:
    normalized = name.lower()
    if normalized == "random":
        return RandomAgent()
    if normalized == "greedy":
        return GreedyAgent()
    if normalized.startswith("minimax"):
        depth = int(normalized.split(":", 1)[1]) if ":" in normalized else 3
        return MinimaxAgent(depth=depth, name=f"minimax:{depth}")
    if normalized.startswith("mcts"):
        simulations = int(normalized.split(":", 1)[1]) if ":" in normalized else 100
        return MCTSAgent(simulations=simulations, name=f"mcts:{simulations}")
    if normalized.startswith("transformer:"):
        checkpoint = name.split(":", 1)[1]
        if not checkpoint:
            raise ValueError("transformer agent needs a checkpoint path")
        return TransformerAgent.from_checkpoint(checkpoint, expected_game=expected_game)
    raise ValueError(
        "unknown agent; use random, greedy, minimax[:depth], mcts[:simulations], "
        "or transformer:/checkpoint.pt"
    )
