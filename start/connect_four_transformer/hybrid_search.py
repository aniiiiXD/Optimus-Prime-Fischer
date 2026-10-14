"""Experimental Transformer-guided alpha-beta agent for Connect Four.

The ordinary :class:`TransformerAgent` and :class:`AlphaBetaTeacher` are left
unchanged. This module uses Transformer scores to order root moves, while
alpha-beta handles adversarial replies. Learned leaf values are optional and
disabled by default; immediate wins and forced one-ply blocks are resolved by
the game rules, never by the net.

Example::

    from pathlib import Path
    from connect_four_transformer.hybrid_search import TransformerGuidedSearchAgent

    agent = TransformerGuidedSearchAgent.from_checkpoint(
        Path("connect_four_distillation/artifacts/checkpoints/"
             "candidate_pons_phases_v3_5min.pt"),
        max_depth=5,
        node_limit=5_000,
        model_value_weight=0.0,
        max_model_evaluations=1,
    )
    result = agent.evaluate(game)
    print(result.best_action, result.stats)

CPU note: the default uses one Transformer inference for root priors/Q and no
leaf inferences. Set ``model_value_weight`` above zero and raise
``max_model_evaluations`` to blend in neural leaf values. The node budget bounds
search work. Depth is iterative, so an interrupted iteration falls back to the
last completed depth. This remains a game-specific agent, not a general game
engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any

import torch

from connect_four_engine import ConnectFour

from .config import ModelConfig
from .data import board_tokens
from .model import ConnectFourTransformer
from .search import CENTER_FIRST, WIN_SCORE, NodeBudgetExceeded


@dataclass
class HybridSearchStats:
    nodes: int = 0
    cache_hits: int = 0
    cutoffs: int = 0
    completed_depth: int = 0
    model_evaluations: int = 0
    model_cache_hits: int = 0
    model_budget_exhausted: bool = False
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class HybridSearchResult:
    best_action: int
    action_scores: dict[int, int]
    model_state_value: float
    stats: HybridSearchStats
    reason: str


@dataclass(frozen=True)
class _Entry:
    value: int
    depth: int
    flag: str


class TransformerGuidedSearchAgent:
    """Alpha-beta search guided by a trained Connect Four Transformer.

    ``model_value_weight`` optionally blends the network's leaf state-value
    estimate with a small handcrafted Connect Four window heuristic. It
    defaults to zero to avoid repeated tiny-model calls in the tree. Root
    actions are ordered by Transformer policy probability and action Q, with
    center preference as the final deterministic tie-breaker.
    """

    def __init__(
        self,
        model: ConnectFourTransformer,
        *,
        max_depth: int = 5,
        node_limit: int = 5_000,
        model_value_weight: float = 0.0,
        max_model_evaluations: int = 1,
        max_table_entries: int = 250_000,
    ) -> None:
        if max_depth <= 0 or node_limit <= 0:
            raise ValueError("max_depth and node_limit must be positive")
        if not 0.0 <= model_value_weight <= 1.0:
            raise ValueError("model_value_weight must be in [0, 1]")
        if max_model_evaluations < 1 or max_table_entries < 1:
            raise ValueError("model/table budgets must be positive")
        self.model = model.cpu().eval()
        self.max_depth = max_depth
        self.node_limit = node_limit
        self.model_value_weight = model_value_weight
        self.max_model_evaluations = max_model_evaluations
        self.max_table_entries = max_table_entries
        self._table: dict[tuple[tuple[int, ...], int, int], _Entry] = {}
        self._model_cache: dict[tuple[tuple[int, ...], int], tuple[dict[int, float], dict[int, float], float]] = {}
        self._stats = HybridSearchStats()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        **kwargs: Any,
    ) -> TransformerGuidedSearchAgent:
        """Load a standard project checkpoint without changing its weights."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = ConnectFourTransformer(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        agent = cls(model, **kwargs)
        agent.metadata = {
            key: value for key, value in checkpoint.items() if key != "model_state_dict"
        }
        agent.checkpoint_path = Path(checkpoint_path)
        return agent

    @torch.inference_mode()
    def _model_scores(
        self, game: ConnectFour, *, enforce_budget: bool
    ) -> tuple[dict[int, float], dict[int, float], float] | None:
        # Policy/Q columns are directional, so a mirror-canonical cache key
        # would require mirroring action ids on lookup. Use the exact key here;
        # the separate alpha-beta table is safely mirror-canonical.
        key = game.state_key()
        cached = self._model_cache.get(key)
        if cached is not None:
            self._stats.model_cache_hits += 1
            return cached
        if enforce_budget and self._stats.model_evaluations >= self.max_model_evaluations:
            self._stats.model_budget_exhausted = True
            return None

        legal = list(game.legal_actions())
        count = len(legal)
        candidate_ids = legal + [7] * (7 - count)
        landing_rows = [5 - game.column_heights[action] for action in legal] + [6] * (7 - count)
        candidate_mask = [True] * count + [False] * (7 - count)
        output = self.model(
            torch.tensor([board_tokens(game.board, game.current_player)], dtype=torch.long),
            torch.tensor([candidate_ids], dtype=torch.long),
            torch.tensor([landing_rows], dtype=torch.long),
            torch.tensor([candidate_mask], dtype=torch.bool),
        )
        probabilities = torch.softmax(output.policy_logits[0, :count], dim=-1)
        policy = {action: float(probabilities[i].item()) for i, action in enumerate(legal)}
        q_values = {
            action: float(output.action_values[0, i].item()) for i, action in enumerate(legal)
        }
        value = float(output.state_value[0].item())
        answer = (policy, q_values, value)
        self._model_cache[key] = answer
        self._stats.model_evaluations += 1
        return answer

    @staticmethod
    def _immediate_wins(game: ConnectFour, player: int) -> list[int]:
        """Find winning drops in place, undoing each probe instead of cloning.

        This routine sits at most alpha-beta leaves, where copies multiply
        quickly; ``ConnectFour.step/undo`` keeps the tactical check exact with
        much less allocation overhead.
        """
        wins: list[int] = []
        original_player = game.current_player
        for action in game.legal_actions():
            game.current_player = player
            result = game.step(action)
            game.undo()
            game.current_player = original_player
            if result.winner == player:
                wins.append(action)
        return wins

    @classmethod
    def _safe_root_actions(cls, game: ConnectFour, actions: list[int]) -> list[int]:
        """Moves that do not hand the opponent an immediate winning drop."""
        actor = game.current_player
        safe: list[int] = []
        for action in actions:
            game.step(action)
            try:
                if game.winner == actor or game.is_terminal:
                    safe.append(action)
                elif not cls._immediate_wins(game, game.current_player):
                    safe.append(action)
            finally:
                game.undo()
        return safe

    @staticmethod
    def _heuristic(game: ConnectFour) -> int:
        """Small symmetric window heuristic, from the side-to-move view."""
        board = game.board
        player = game.current_player
        score = 3 * sum(board[row * 7 + 3] == player for row in range(6))
        score -= 3 * sum(board[row * 7 + 3] == -player for row in range(6))
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        weights = (0, 1, 8, 70, 1_000)
        for row in range(6):
            for column in range(7):
                for dr, dc in directions:
                    end_row = row + 3 * dr
                    end_column = column + 3 * dc
                    if not (0 <= end_row < 6 and 0 <= end_column < 7):
                        continue
                    window = [board[(row + i * dr) * 7 + column + i * dc] for i in range(4)]
                    own, opponent = window.count(player), window.count(-player)
                    if opponent == 0:
                        score += weights[own]
                    elif own == 0:
                        score -= weights[opponent]
        return max(-5_000, min(5_000, score))

    @staticmethod
    def _ordered_actions(game: ConnectFour) -> list[int]:
        legal = set(game.legal_actions())
        wins: list[int] = []
        remaining: list[int] = []
        for action in CENTER_FIRST:
            if action not in legal:
                continue
            actor = game.current_player
            result = game.step(action)
            game.undo()
            (wins if result.winner == actor else remaining).append(action)
        return wins + remaining

    @torch.inference_mode()
    def _leaf_value(self, game: ConnectFour) -> int:
        """Evaluate a leaf with exact one-ply tactical checks before estimates."""
        player = game.current_player
        if self._immediate_wins(game, player):
            return WIN_SCORE - 1

        # A leaf is a loss if every move permits an immediate opposing win.
        safe = False
        saw_reply_win = False
        for action in game.legal_actions():
            result = game.step(action)
            try:
                if result.winner == player:
                    return WIN_SCORE - 1
                replies = self._immediate_wins(game, game.current_player)
                if replies:
                    saw_reply_win = True
                else:
                    safe = True
            finally:
                game.undo()
        if saw_reply_win and not safe:
            return -WIN_SCORE + 2

        heuristic_value = math.tanh(self._heuristic(game) / 80.0)
        if self.model_value_weight == 0.0:
            # Keep the Transformer for root move ordering, but do not spend
            # CPU evaluating leaves when its value is explicitly ablated.
            value = heuristic_value
        else:
            model_result = self._model_scores(game, enforce_budget=True)
            if model_result is None:
                value = heuristic_value
            else:
                model_value = max(-1.0, min(1.0, model_result[2]))
                weight = self.model_value_weight
                value = weight * model_value + (1.0 - weight) * heuristic_value
        # If there's a forced block at the horizon, favor the side that still
        # has a safe continuation without treating it as a guaranteed win.
        if saw_reply_win and safe:
            value = max(value, 0.1)
        return int(max(-WIN_SCORE + 100, min(WIN_SCORE - 100, round(value * 10_000))))

    def _negamax(self, game: ConnectFour, depth: int, alpha: int, beta: int) -> int:
        self._stats.nodes += 1
        if self._stats.nodes > self.node_limit:
            raise NodeBudgetExceeded
        if depth <= 0:
            return self._leaf_value(game)

        key = (*game.canonical_key(), depth)
        alpha_original, beta_original = alpha, beta
        entry = self._table.get(key)
        if entry is not None and entry.depth >= depth:
            self._stats.cache_hits += 1
            if entry.flag == "exact":
                return entry.value
            if entry.flag == "lower":
                alpha = max(alpha, entry.value)
            else:
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value

        best = -WIN_SCORE
        for action in self._ordered_actions(game):
            actor = game.current_player
            result = game.step(action)
            try:
                if result.winner == actor:
                    value = WIN_SCORE - 1
                elif result.is_draw:
                    value = 0
                else:
                    value = -self._negamax(game, depth - 1, -beta, -alpha)
            finally:
                game.undo()
            best = max(best, value)
            alpha = max(alpha, best)
            if alpha >= beta:
                self._stats.cutoffs += 1
                break

        if len(self._table) >= self.max_table_entries:
            self._table.clear()
        flag = "exact"
        if best <= alpha_original:
            flag = "upper"
        elif best >= beta_original:
            flag = "lower"
        self._table[key] = _Entry(best, depth, flag)
        return best

    def evaluate(self, game: ConnectFour) -> HybridSearchResult:
        """Return the selected legal action plus bounded-search diagnostics."""
        if game.is_terminal:
            raise ValueError("cannot search a terminal state")
        original = game.snapshot()
        self._stats = HybridSearchStats()
        self._model_cache.clear()
        self._table.clear()
        started = time.perf_counter()

        model_result = self._model_scores(game, enforce_budget=False)
        assert model_result is not None
        policy, q_values, state_value = model_result
        legal = list(game.legal_actions())
        center_rank = {action: rank for rank, action in enumerate(CENTER_FIRST)}
        root_order = sorted(
            legal,
            key=lambda action: (
                0.55 * policy[action] + 0.45 * (q_values[action] + 1.0) / 2.0,
                -center_rank[action],
            ),
            reverse=True,
        )

        wins = self._immediate_wins(game, game.current_player)
        if wins:
            best_action = max(
                wins,
                key=lambda action: (policy[action], q_values[action], -center_rank[action]),
            )
            scores = {
                action: (WIN_SCORE - 1 if action in wins else -WIN_SCORE + 1)
                for action in legal
            }
            reason = "take_immediate_win"
            completed_depth = 0
        else:
            opponent_wins = self._immediate_wins(game, -game.current_player)
            if opponent_wins:
                safe = self._safe_root_actions(game, root_order)
                if safe:
                    safe_set = set(safe)
                    root_order = [action for action in root_order if action in safe_set]
                else:
                    safe = []

            # If a unique safe response exists, it is a forced tactical block;
            # restrict search to safe moves because every other move loses now.
            restricted = bool(opponent_wins and root_order and len(root_order) < len(legal))
            scores: dict[int, int] = {
                action: -WIN_SCORE + 1 for action in legal if action not in root_order
            }
            best_action = root_order[0]
            best_score = -WIN_SCORE
            completed_depth = 0

            for depth in range(1, self.max_depth + 1):
                current_scores: dict[int, int] = dict(scores)
                alpha = -WIN_SCORE
                iteration_best_action: int | None = None
                iteration_best_score = -WIN_SCORE
                try:
                    for action in root_order:
                        actor = game.current_player
                        result = game.step(action)
                        try:
                            if result.winner == actor:
                                score = WIN_SCORE - 1
                            elif result.is_draw:
                                score = 0
                            else:
                                score = -self._negamax(game, depth - 1, -WIN_SCORE, -alpha)
                        finally:
                            game.undo()
                        current_scores[action] = score
                        if score > iteration_best_score:
                            iteration_best_action, iteration_best_score = action, score
                        alpha = max(alpha, score)
                except NodeBudgetExceeded:
                    break
                scores = current_scores
                completed_depth = depth
                best_score = max(scores[action] for action in root_order)
                # Stable model-informed tie-break among equal search scores.
                tied = [action for action in root_order if scores[action] == best_score]
                best_action = (
                    tied[0]
                    if tied
                    else (iteration_best_action if iteration_best_action is not None else root_order[0])
                )

            if completed_depth == 0:
                # Node budget was too small for even depth one; root Transformer
                # ranking remains the bounded, legal fallback.
                viable = root_order
                best_action = max(
                    viable,
                    key=lambda action: (
                        0.55 * policy[action] + 0.45 * (q_values[action] + 1.0) / 2.0,
                        -center_rank[action],
                    ),
                )
                scores.update({action: int(round(q_values[action] * 10_000)) for action in viable})

            reason = (
                "forced_one_ply_block"
                if restricted
                else ("alpha_beta_search" if completed_depth else "transformer_budget_fallback")
            )

        self._stats.completed_depth = completed_depth
        self._stats.elapsed_seconds = time.perf_counter() - started
        if game.snapshot() != original:
            raise RuntimeError("hybrid search failed to restore the supplied game state")
        return HybridSearchResult(
            best_action=best_action,
            action_scores=scores,
            model_state_value=state_value,
            stats=self._stats,
            reason=reason,
        )

    def choose(self, game: ConnectFour) -> int:
        """Select one legal column, compatible with tournament Agent protocol."""
        return self.evaluate(game).best_action


__all__ = [
    "HybridSearchResult",
    "HybridSearchStats",
    "TransformerGuidedSearchAgent",
]
