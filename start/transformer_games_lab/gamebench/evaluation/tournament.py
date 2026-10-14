"""Deterministic round-robin evaluation with seat swapping and Elo."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import statistics
import time
from typing import Any

from gamebench.agents.base import Agent
from gamebench.games.base import GameAdapter


@dataclass(frozen=True)
class TournamentConfig:
    games_per_pair: int = 20
    seed: int = 0
    elo_k: float = 24.0
    record_games: bool = False


def play_game(
    game: GameAdapter,
    seats: tuple[Agent, Agent],
    seed: int,
) -> dict[str, Any]:
    state = game.reset(seed=seed)
    rng = random.Random(seed)
    decisions: list[dict[str, Any]] = []
    for ply in range(game.spec.max_game_length):
        if state.is_terminal():
            break
        player = state.current_player
        agent = seats[player]
        visible_state = state.clone()
        started = time.perf_counter_ns()
        error: str | None = None
        try:
            action = agent.select_action(visible_state, rng)
        except Exception as exc:  # evaluator converts agent crashes into forfeits
            action = -1
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter_ns() - started) / 1e6
        legal = state.legal_actions()
        if action not in legal:
            returns = [-1.0, -1.0]
            returns[player] = -1.0
            returns[1 - player] = 1.0
            return {
                "returns": tuple(returns),
                "plies": ply,
                "forfeit_player": player,
                "error": error or f"illegal action {action}; legal={legal}",
                "history": state.history,
                "decisions": decisions,
            }
        decisions.append({"player": player, "action": action, "latency_ms": latency_ms})
        state.apply_action(action)
    if not state.is_terminal():
        raise RuntimeError(f"{game.spec.name} exceeded declared max_game_length={game.spec.max_game_length}")
    return {
        "returns": state.returns(),
        "plies": len(state.history),
        "forfeit_player": None,
        "error": None,
        "history": state.history,
        "decisions": decisions,
    }


def _update_elo(ratings: dict[str, float], first: str, second: str, score_first: float, k: float) -> None:
    expected_first = 1.0 / (1.0 + 10 ** ((ratings[second] - ratings[first]) / 400))
    change = k * (score_first - expected_first)
    ratings[first] += change
    ratings[second] -= change


def run_tournament(
    game: GameAdapter,
    agents: list[Agent],
    config: TournamentConfig | None = None,
) -> dict[str, Any]:
    config = config or TournamentConfig()
    if len(agents) < 2:
        raise ValueError("a tournament needs at least two agents")
    names = [agent.name for agent in agents]
    if len(set(names)) != len(names):
        raise ValueError("agent names must be unique")
    stats: dict[str, dict[str, Any]] = {
        name: {"wins": 0, "draws": 0, "losses": 0, "points": 0.0, "decision_ms": []}
        for name in names
    }
    ratings = {name: 1000.0 for name in names}
    games: list[dict[str, Any]] = []
    game_index = 0
    for first_index in range(len(agents)):
        for second_index in range(first_index + 1, len(agents)):
            pair = (agents[first_index], agents[second_index])
            for repetition in range(config.games_per_pair):
                seats = pair if repetition % 2 == 0 else (pair[1], pair[0])
                result = play_game(game, seats, config.seed + game_index)
                game_index += 1
                returns = result["returns"]
                for seat_index, agent in enumerate(seats):
                    name = agent.name
                    value = returns[seat_index]
                    stats[name]["wins" if value > 0 else "losses" if value < 0 else "draws"] += 1
                    stats[name]["points"] += (value + 1) / 2
                for decision in result["decisions"]:
                    stats[seats[decision["player"]].name]["decision_ms"].append(decision["latency_ms"])
                score_first = (returns[0] + 1) / 2
                _update_elo(ratings, seats[0].name, seats[1].name, score_first, config.elo_k)
                if config.record_games:
                    games.append({
                        "seats": [seats[0].name, seats[1].name],
                        **{key: value for key, value in result.items() if key != "decisions"},
                    })
    summary: dict[str, Any] = {}
    for name, values in stats.items():
        latencies = values.pop("decision_ms")
        played = values["wins"] + values["draws"] + values["losses"]
        summary[name] = {
            **values,
            "games": played,
            "score_rate": values["points"] / played if played else 0.0,
            "elo": ratings[name],
            "decision_latency_ms": {
                "mean": statistics.fmean(latencies) if latencies else None,
                "median": statistics.median(latencies) if latencies else None,
                "moves": len(latencies),
            },
        }
    return {
        "game": game.metadata(),
        "config": asdict(config),
        "agents": [agent.metadata() for agent in agents],
        "standings": dict(sorted(summary.items(), key=lambda item: item[1]["elo"], reverse=True)),
        "games": games,
    }
