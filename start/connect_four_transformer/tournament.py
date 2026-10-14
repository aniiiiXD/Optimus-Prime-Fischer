"""Run fixed-opening games against random, tactical, and search opponents."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import time
from typing import Protocol

from connect_four_engine import PLAYER_ONE, ConnectFour

from .agent import TransformerAgent
from .search import AlphaBetaTeacher, CENTER_FIRST


DEFAULT_CHECKPOINT = Path(__file__).parent / "artifacts" / "checkpoints" / "candidate_v0.pt"
DEFAULT_REPORT = Path(__file__).parent / "artifacts" / "reports" / "tournament_v0.json"


class Agent(Protocol):
    name: str

    def choose(self, game: ConnectFour) -> int: ...


class RandomAgent:
    name = "random"

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def choose(self, game: ConnectFour) -> int:
        return self.rng.choice(game.legal_actions())


class TacticalAgent:
    name = "tactical"

    @staticmethod
    def _wins(game: ConnectFour, player: int) -> list[int]:
        actions = []
        for action in game.legal_actions():
            branch = game.copy()
            branch.current_player = player
            if branch.step(action).winner == player:
                actions.append(action)
        return actions

    def choose(self, game: ConnectFour) -> int:
        wins = self._wins(game, game.current_player)
        if wins:
            return min(wins, key=CENTER_FIRST.index)
        blocks = self._wins(game, -game.current_player)
        if blocks:
            return min(blocks, key=CENTER_FIRST.index)
        return next(action for action in CENTER_FIRST if action in game.legal_actions())


class SearchAgent:
    name = "alpha_beta_depth3"

    def __init__(self) -> None:
        self.teacher = AlphaBetaTeacher()

    def choose(self, game: ConnectFour) -> int:
        return self.teacher.evaluate(game, max_depth=3, node_limit=5_000).best_action


class ModelAgent:
    name = "transformer_v0"

    def __init__(self, checkpoint: Path) -> None:
        self.agent = TransformerAgent(checkpoint)

    def choose(self, game: ConnectFour) -> int:
        return self.agent.choose(game)


def make_openings(count: int, seed: int) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    openings = []
    while len(openings) < count:
        game = ConnectFour()
        moves = []
        for _ in range(rng.randint(0, 4)):
            if game.is_terminal:
                break
            action = rng.choice(game.legal_actions())
            game.step(action)
            moves.append(action)
        if not game.is_terminal:
            openings.append(tuple(moves))
    return openings


def play_game(
    x_agent: Agent, o_agent: Agent, opening: tuple[int, ...]
) -> tuple[int | None, dict[str, list[float]]]:
    game = ConnectFour.from_moves(opening)
    latency = {x_agent.name: [], o_agent.name: []}
    while not game.is_terminal:
        agent = x_agent if game.current_player == PLAYER_ONE else o_agent
        started = time.perf_counter_ns()
        action = agent.choose(game)
        latency[agent.name].append((time.perf_counter_ns() - started) / 1_000_000)
        if action not in game.legal_actions():
            raise RuntimeError(f"{agent.name} chose illegal column {action}")
        game.step(action)
    return game.winner, latency


def matchup(
    model: ModelAgent,
    opponent: Agent,
    openings: list[tuple[int, ...]],
    *,
    model_player: int,
) -> dict[str, float | int | str]:
    outcomes: Counter[str] = Counter()
    model_latencies = []
    opponent_latencies = []
    for opening in openings:
        x_agent, o_agent = (model, opponent) if model_player == PLAYER_ONE else (opponent, model)
        winner, latency = play_game(x_agent, o_agent, opening)
        if winner == model_player:
            outcomes["win"] += 1
        elif winner is None:
            outcomes["draw"] += 1
        else:
            outcomes["loss"] += 1
        model_latencies.extend(latency[model.name])
        opponent_latencies.extend(latency[opponent.name])
    return {
        "opponent": opponent.name,
        "model_side": "X" if model_player == PLAYER_ONE else "O",
        "games": len(openings),
        "wins": outcomes["win"],
        "draws": outcomes["draw"],
        "losses": outcomes["loss"],
        "model_move_median_ms": statistics.median(model_latencies),
        "opponent_move_median_ms": statistics.median(opponent_latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--games-per-side", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.games_per_side <= 0:
        parser.error("games-per-side must be positive")

    model = ModelAgent(args.checkpoint)
    openings = make_openings(args.games_per_side, args.seed)
    opponents: list[Agent] = [
        RandomAgent(args.seed + 1),
        TacticalAgent(),
        SearchAgent(),
    ]
    results = []
    for opponent in opponents:
        for side in (PLAYER_ONE, -PLAYER_ONE):
            result = matchup(model, opponent, openings, model_player=side)
            results.append(result)
            print(
                f"vs {opponent.name:<18} as {result['model_side']}: "
                f"{result['wins']}W {result['draws']}D {result['losses']}L"
            )
    report = {
        "checkpoint": str(args.checkpoint),
        "games_per_side": args.games_per_side,
        "opening_seed": args.seed,
        "opening_length_range": [0, 4],
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved report: {args.report}")


if __name__ == "__main__":
    main()
