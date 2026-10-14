"""Run complete games between agents using the real rules engine."""

from __future__ import annotations

from collections import Counter
import time
from typing import Any

from tic_tac_toe_lab.tic_tac_toe import TicTacToe, available_moves, winner

from ..agents.base import Agent
from .metrics import summarize_latencies


def play_game(x_agent: Agent, o_agent: Agent) -> dict[str, Any]:
    game = TicTacToe()
    moves: list[int] = []
    latency_by_mark: dict[str, list[float]] = {"X": [], "O": []}
    agents = {"X": x_agent, "O": o_agent}

    while not game.is_over:
        mark = game.current_player
        started = time.perf_counter()
        move = agents[mark].select_move(game.board.copy(), mark)
        latency_by_mark[mark].append((time.perf_counter() - started) * 1_000)
        if move not in available_moves(game.board):
            raise RuntimeError(f"{agents[mark].name} selected illegal square {move}")
        moves.append(move)
        game.move(move)

    return {
        "winner": winner(game.board),
        "moves": moves,
        "latency_ms": latency_by_mark,
    }


def run_matchup(x_agent: Agent, o_agent: Agent, games: int) -> dict[str, Any]:
    outcomes: Counter[str] = Counter()
    x_latencies: list[float] = []
    o_latencies: list[float] = []
    move_counts: list[int] = []
    for _ in range(games):
        result = play_game(x_agent, o_agent)
        outcome = result["winner"] or "draw"
        outcomes[outcome] += 1
        x_latencies.extend(result["latency_ms"]["X"])
        o_latencies.extend(result["latency_ms"]["O"])
        move_counts.append(len(result["moves"]))

    return {
        "x_agent": x_agent.name,
        "o_agent": o_agent.name,
        "games": games,
        "x_wins": outcomes["X"],
        "o_wins": outcomes["O"],
        "draws": outcomes["draw"],
        "mean_game_length": sum(move_counts) / len(move_counts),
        "x_move_latency": summarize_latencies(x_latencies),
        "o_move_latency": summarize_latencies(o_latencies),
    }
