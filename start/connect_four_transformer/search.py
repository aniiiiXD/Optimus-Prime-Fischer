"""Measured negamax alpha-beta teacher for Connect Four datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

from connect_four_engine import COLUMNS, ConnectFour


WIN_SCORE = 1_000_000
CENTER_FIRST = (3, 2, 4, 1, 5, 0, 6)


class NodeBudgetExceeded(RuntimeError):
    pass


@dataclass
class SearchStats:
    nodes: int = 0
    cache_hits: int = 0
    cutoffs: int = 0
    completed_depth: int = 0
    elapsed_seconds: float = 0.0

    @property
    def nodes_per_second(self) -> float:
        return self.nodes / self.elapsed_seconds if self.elapsed_seconds else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {**asdict(self), "nodes_per_second": self.nodes_per_second}


@dataclass(frozen=True)
class SearchResult:
    best_action: int
    action_scores: dict[int, int]
    action_values: dict[int, float]
    policy: dict[int, float]
    state_value: float
    stats: SearchStats


@dataclass(frozen=True)
class _CacheEntry:
    value: int
    depth: int
    flag: str


class AlphaBetaTeacher:
    """Iterative-deepening search with mirror-aware transposition caching."""

    def __init__(self, *, max_table_entries: int = 500_000) -> None:
        self.max_table_entries = max_table_entries
        self.table: dict[tuple[tuple[int, ...], int, int], _CacheEntry] = {}
        self.stats = SearchStats()
        self.node_limit: int | None = None

    def clear_cache(self) -> None:
        self.table.clear()

    def evaluate(
        self,
        game: ConnectFour,
        *,
        max_depth: int = 5,
        node_limit: int | None = None,
        temperature: float = 0.18,
    ) -> SearchResult:
        if game.is_terminal:
            raise ValueError("cannot search a terminal state")
        if max_depth <= 0 or (node_limit is not None and node_limit <= 0):
            raise ValueError("max_depth and node_limit must be positive")

        original = game.snapshot()
        self.stats = SearchStats()
        self.node_limit = node_limit
        started = time.perf_counter()
        completed_scores: dict[int, int] | None = None

        for depth in range(1, max_depth + 1):
            try:
                scores = self._root_scores(game, depth)
            except NodeBudgetExceeded:
                break
            completed_scores = scores
            self.stats.completed_depth = depth

        self.stats.elapsed_seconds = time.perf_counter() - started
        if game.snapshot() != original:
            raise RuntimeError("search failed to restore the supplied game state")
        if completed_scores is None:
            completed_scores = self._fallback_scores(game)

        values = {
            action: self.score_to_value(score) for action, score in completed_scores.items()
        }
        policy = self._soft_policy(values, temperature)
        best_action = max(
            CENTER_FIRST,
            key=lambda action: completed_scores.get(action, -math.inf),
        )
        return SearchResult(
            best_action=best_action,
            action_scores=completed_scores,
            action_values=values,
            policy=policy,
            state_value=max(values.values()),
            stats=self.stats,
        )

    def _root_scores(self, game: ConnectFour, depth: int) -> dict[int, int]:
        scores: dict[int, int] = {}
        for action in self._ordered_actions(game):
            actor = game.current_player
            result = game.step(action)
            try:
                if result.winner == actor:
                    value = WIN_SCORE
                elif result.is_draw:
                    value = 0
                else:
                    # Every root action gets a full search window because these
                    # scores become training labels, not merely move ordering.
                    value = -self._negamax(game, depth - 1, -WIN_SCORE, WIN_SCORE, ply=1)
            finally:
                game.undo()
            scores[action] = value
        return scores

    def _negamax(
        self, game: ConnectFour, depth: int, alpha: int, beta: int, *, ply: int
    ) -> int:
        self.stats.nodes += 1
        if self.node_limit is not None and self.stats.nodes > self.node_limit:
            raise NodeBudgetExceeded
        if depth == 0:
            return self._heuristic(game)

        key = (*game.canonical_key(), depth)
        original_alpha = alpha
        original_beta = beta
        entry = self.table.get(key)
        if entry is not None and entry.depth >= depth:
            self.stats.cache_hits += 1
            if entry.flag == "exact":
                return entry.value
            if entry.flag == "lower":
                alpha = max(alpha, entry.value)
            elif entry.flag == "upper":
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value

        best = -WIN_SCORE
        for action in self._ordered_actions(game):
            actor = game.current_player
            result = game.step(action)
            try:
                if result.winner == actor:
                    # Keep cached values independent of distance from the
                    # current search root; the same board can be queried later
                    # from a different root position.
                    value = WIN_SCORE
                elif result.is_draw:
                    value = 0
                else:
                    value = -self._negamax(game, depth - 1, -beta, -alpha, ply=ply + 1)
            finally:
                game.undo()
            best = max(best, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                self.stats.cutoffs += 1
                break

        flag = "exact"
        if best <= original_alpha:
            flag = "upper"
        elif best >= original_beta:
            flag = "lower"
        if len(self.table) >= self.max_table_entries:
            self.table.clear()
        self.table[key] = _CacheEntry(best, depth, flag)
        return best

    def _ordered_actions(self, game: ConnectFour) -> list[int]:
        legal = set(game.legal_actions())
        wins: list[int] = []
        remaining: list[int] = []
        for action in CENTER_FIRST:
            if action not in legal:
                continue
            actor = game.current_player
            result = game.step(action)
            is_win = result.winner == actor
            game.undo()
            (wins if is_win else remaining).append(action)
        return wins + remaining

    def _heuristic(self, game: ConnectFour) -> int:
        board = game.board
        player = game.current_player
        score = 3 * sum(board[row * COLUMNS + 3] == player for row in range(6))
        score -= 3 * sum(board[row * COLUMNS + 3] == -player for row in range(6))
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        weights = (0, 1, 8, 70, WIN_SCORE)
        for row in range(6):
            for column in range(7):
                for dr, dc in directions:
                    end_row = row + 3 * dr
                    end_column = column + 3 * dc
                    if not (0 <= end_row < 6 and 0 <= end_column < 7):
                        continue
                    window = [board[(row + i * dr) * 7 + column + i * dc] for i in range(4)]
                    own = window.count(player)
                    opponent = window.count(-player)
                    if opponent == 0:
                        score += weights[own]
                    elif own == 0:
                        score -= weights[opponent]
        return max(-WIN_SCORE + 100, min(WIN_SCORE - 100, score))

    def _fallback_scores(self, game: ConnectFour) -> dict[int, int]:
        scores: dict[int, int] = {}
        for action in self._ordered_actions(game):
            actor = game.current_player
            result = game.step(action)
            scores[action] = WIN_SCORE if result.winner == actor else -self._heuristic(game)
            game.undo()
        return scores

    @staticmethod
    def score_to_value(score: int) -> float:
        if score >= WIN_SCORE - 100:
            return 1.0
        if score <= -WIN_SCORE + 100:
            return -1.0
        return math.tanh(score / 80.0)

    @staticmethod
    def _soft_policy(values: dict[int, float], temperature: float) -> dict[int, float]:
        if temperature <= 0:
            best = max(values.values())
            winners = [action for action, value in values.items() if value == best]
            return {action: (1.0 / len(winners) if action in winners else 0.0) for action in values}
        maximum = max(values.values())
        weights = {
            action: math.exp((value - maximum) / temperature)
            for action, value in values.items()
        }
        total = sum(weights.values())
        return {action: weight / total for action, weight in weights.items()}


__all__ = ["AlphaBetaTeacher", "SearchResult", "SearchStats"]
