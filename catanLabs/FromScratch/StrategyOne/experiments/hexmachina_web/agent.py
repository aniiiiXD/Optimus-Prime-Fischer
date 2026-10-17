"""Approximate port of the published HexMachina Appendix A.1 policy.

The paper does not publish its adapters.py, so this uses Catanatron's public
engine API. It preserves the published strategy components, not an exact agent.
"""

from __future__ import annotations

import random

from catanatron.features import build_production_features
from catanatron.models.enums import CITY, SETTLEMENT, ActionType
from catanatron.models.map import number_probability
from catanatron.players.tree_search_utils import execute_spectrum
from catanatron.players.value import base_fn, value_production
from catanatron.state_functions import player_key, player_num_resource_cards


_BASE = base_fn()
_PRODUCTION = build_production_features(True)
_IMPORTANT = {
    ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD,
    ActionType.BUILD_CITY, ActionType.BUY_DEVELOPMENT_CARD,
    ActionType.PLAY_KNIGHT_CARD, ActionType.MOVE_ROBBER,
}


class HexMachinaApprox:
    """Phase-aware filter, 85/15 blended evaluator, two-ply greedy rollout."""

    def __init__(self, color, seed=0):
        self.color = color
        self.random = random.Random(seed)
        self.decisions = 0
        self.simulated_branches = 0

    def phase(self, game):
        vp = game.state.player_state[f"{player_key(game.state, self.color)}_VICTORY_POINTS"]
        if game.state.num_turns < 20 and vp < 5:
            return "early"
        if game.state.num_turns < 45 and vp < 7:
            return "mid"
        return "late"

    @staticmethod
    def production(game, color):
        sample = _PRODUCTION(game, color)
        return value_production(sample, "P0"), value_production(sample, "P1", False)

    def _node_quality(self, game, node):
        tiles = game.state.board.map.adjacent_tiles[node]
        return sum(number_probability(tile.number) for tile in tiles if tile.number)

    def _robber_impact(self, game, action):
        tile = game.state.board.map.tiles[action.value[0]]
        probability = number_probability(tile.number) if tile.number else 0.0
        own = enemy = 0
        for node in tile.nodes.values():
            building = game.state.board.buildings.get(node)
            if building is None:
                continue
            owner, kind = building
            amount = 2 if kind == CITY else 1
            if owner == self.color:
                own += amount
            else:
                enemy += amount
        victim = action.value[1]
        theft = min(player_num_resource_cards(game.state, victim), 8) if victim is not None else 0
        return 85.0 * probability * (enemy - 1.2 * own) + 1.5 * theft

    def pre_score(self, game, action):
        kind = action.action_type
        phase = self.phase(game)
        mult = {
            "early": {"settlement": 2.0, "road": 1.8, "city": 0.8, "dev": 1.2},
            "mid": {"settlement": 1.0, "road": 1.0, "city": 1.25, "dev": 1.0},
            "late": {"settlement": 0.8, "road": 0.9, "city": 1.5, "dev": 1.0},
        }[phase]
        if kind == ActionType.BUILD_SETTLEMENT:
            return 90 * mult["settlement"] + 150 * self._node_quality(game, action.value)
        if kind == ActionType.BUILD_CITY:
            return 100 * mult["city"] + 70 * self._node_quality(game, action.value)
        if kind == ActionType.BUILD_ROAD:
            edge = action.value
            return 20 * mult["road"] + 8 * sum(
                bool(game.state.board.map.adjacent_tiles[node]) for node in edge
            )
        if kind == ActionType.BUY_DEVELOPMENT_CARD:
            return 45 * mult["dev"]
        if kind == ActionType.MOVE_ROBBER:
            return 50 + self._robber_impact(game, action)
        if kind == ActionType.PLAY_KNIGHT_CARD:
            return 70
        if kind == ActionType.ROLL:
            return 100
        if kind == ActionType.END_TURN:
            return 0
        if kind == ActionType.MARITIME_TRADE:
            return 12
        if kind == ActionType.OFFER_TRADE:
            return -15
        return 5

    def candidates(self, game, actions):
        """Top-K plus an action from every major type, capped at 12."""
        ordered = sorted(actions, key=lambda a: self.pre_score(game, a), reverse=True)
        picked = ordered[:8]
        for kind in _IMPORTANT:
            representative = next((a for a in ordered if a.action_type == kind), None)
            if representative is not None and representative not in picked:
                picked.append(representative)
        return picked[:12]

    def evaluate(self, game):
        state = game.state
        key = player_key(state, self.color)
        vp = state.player_state[f"{key}_VICTORY_POINTS"]
        if game.winning_color() == self.color:
            return 1e6
        if game.winning_color() is not None:
            return -1e6
        # The published 85/15 raw blend is ineffective with Catanatron's
        # 3e14 VP weight. Rescale base_fn to VP~=100 and production~=1e2.
        raw = _BASE(game, self.color)
        normalized_base = raw / 1e8 - 3e6 * vp + 100 * vp
        own_prod, enemy_prod = self.production(game, self.color)
        settlements = len(state.buildings_by_color[self.color][SETTLEMENT])
        cities = len(state.buildings_by_color[self.color][CITY])
        roads = len(state.board.get_player_roads(self.color)) if hasattr(state.board, "get_player_roads") else 0
        phase = self.phase(game)
        phase_mult = {
            "early": (2.0, 0.8, 1.8, 80),
            "mid": (1.0, 1.25, 1.0, 45),
            "late": (0.8, 1.5, 0.9, 30),
        }[phase]
        custom = (
            100 * vp + 25 * phase_mult[0] * settlements
            + 60 * phase_mult[1] * cities + 6 * phase_mult[2] * roads
            + phase_mult[3] * (own_prod - enemy_prod)
            + player_num_resource_cards(state, self.color)
        )
        return 0.85 * normalized_base + 0.15 * custom

    def _branches(self, game, action, cap=3):
        try:
            branches = execute_spectrum(game, action)
        except Exception:
            clone = game.copy()
            clone.execute(action)
            branches = [(clone, 1.0)]
        if len(branches) <= cap:
            return branches
        branches = sorted(branches, key=lambda item: item[1], reverse=True)[:cap]
        weight = sum(prob for _, prob in branches)
        return [(child, prob / weight) for child, prob in branches]

    def _rollout(self, game):
        if game.winning_color() is not None or not game.playable_actions:
            return self.evaluate(game)
        actor = game.state.current_color()
        top = sorted(game.playable_actions, key=lambda a: self.pre_score(game, a), reverse=True)[:3]
        # A single greedy response is the paper's light two-ply rollout.
        best = -float("inf") if actor == self.color else float("inf")
        for action in top:
            try:
                branches = self._branches(game, action, cap=1)
                score = self.evaluate(branches[0][0])
            except Exception:
                continue
            best = max(best, score) if actor == self.color else min(best, score)
        return self.evaluate(game) if best in (float("inf"), -float("inf")) else best

    def decide(self, game, playable_actions):
        self.decisions += 1
        if len(playable_actions) == 1:
            return playable_actions[0]
        best_action = playable_actions[0]
        best_score = -float("inf")
        for action in self.candidates(game, playable_actions):
            try:
                outcomes = self._branches(game, action, cap=3)
                score = 0.0
                for child, probability in outcomes:
                    self.simulated_branches += 1
                    score += probability * self._rollout(child)
                if action.action_type == ActionType.MOVE_ROBBER:
                    score += 0.15 * self._robber_impact(game, action)
            except Exception:
                continue
            if score > best_score:
                best_score, best_action = score, action
        return best_action


def create_agent(color, checkpoint=None, seed=0):
    return HexMachinaApprox(color, seed)
