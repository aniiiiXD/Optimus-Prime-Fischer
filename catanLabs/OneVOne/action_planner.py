"""One shared, visible-outcome planner for heuristic and learned action scoring."""

from dataclasses import dataclass

import numpy as np

from catanatron.features import build_production_features, reachability_features
from catanatron.models.enums import Action, ActionRecord, ActionType, DEVELOPMENT_CARDS, RESOURCES
from catanatron.models.map import number_probability
from catanatron.players.value import value_production
from catanLabs.Competitive.tactical_vp_agent import ROOT_TYPES, POINT_TYPES
from catanLabs.OneVOne.agent import OneVOneAgent
from catanLabs.OneVOne.learning import (
    POSITION_FEATURES, PositionFeatures, HEURISTIC_WEIGHTS, normalize_scores,
)
from catanLabs.OneVOne.policy_features import ACTION_TYPES


ACTION_DIM = 2 * len(POSITION_FEATURES) + len(ACTION_TYPES) + 1


@dataclass
class Plan:
    actions: list
    heuristic: np.ndarray
    features: np.ndarray
    winning: np.ndarray
    trace: list


class _CachedValue:
    """Cache board components while preserving the original base_fn sum order."""

    def __init__(self, color, weights):
        self.color, self.weights, self.cache = color, weights, {}

    def __call__(self, game, color):
        state, board, w = game.state, game.state.board, self.weights
        key = (frozenset(board.buildings.items()), frozenset(board.roads.items()), board.robber_coordinate)
        if key not in self.cache:
            production = build_production_features(True)(game, color)
            own = value_production(production, "P0")
            enemy = value_production(production, "P1", False)
            reach = reachability_features(game, color, 2)
            r0, r1 = (sum(reach[f"P0_{level}_ROAD_REACHABLE_{r}"] for r in RESOURCES) for level in (0, 1))
            sites = len(board.buildable_node_ids(color))
            tiles = {tile for node, (owner, _) in board.buildings.items() if owner == color for tile in board.map.adjacent_tiles[node]}
            self.cache[key] = own, enemy, r0, r1, sites, len(tiles)
        own, enemy, r0, r1, sites, tiles = self.cache[key]
        prefix = f"P{state.color_to_index[color]}_"
        ps = state.player_state
        hand = {r: ps[prefix + r + "_IN_HAND"] for r in RESOURCES}
        city = (max(2 - hand["WHEAT"], 0) + max(3 - hand["ORE"], 0)) / 5
        settlement = sum(max(1 - hand[r], 0) for r in ("WHEAT", "SHEEP", "BRICK", "WOOD")) / 4
        count = sum(hand.values())
        return float(
            ps[prefix + "VICTORY_POINTS"] * w["public_vps"]
            + own * w["production"] + enemy * w["enemy_production"]
            + r0 * w["reachable_production_0"] + r1 * w["reachable_production_1"]
            + (2 - city - settlement) / 2 * w["hand_synergy"]
            + sites * w["buildable_nodes"] + tiles * w["num_tiles"]
            + count * w["hand_resources"] + (w["discard_penalty"] if count > 7 else 0)
            + ps[prefix + "LONGEST_ROAD_LENGTH"] * (w["longest_road"] if sites == 0 else .1)
            + sum(ps[prefix + d + "_IN_HAND"] for d in DEVELOPMENT_CARDS) * w["hand_devs"]
            + ps[prefix + "PLAYED_KNIGHT"] * w["army_size"]
        )


def _description(action):
    return [action.action_type.name, repr(action.value)]


class ActionPlanner:
    def __init__(self, color, config):
        self.color = color
        self.original = OneVOneAgent(color, config)
        self.base = _CachedValue(color, self.original.weights)
        self.position = PositionFeatures(color)

    def _evaluate(self, game):
        features = self.position(game)
        score = self.base(game, self.color) + 1e7 * float(features @ HEURISTIC_WEIGHTS)
        return score, features

    def _outcome(self, game, action):
        """Return public expected value/features and a safe deterministic state."""
        kind = action.action_type
        if kind == ActionType.ROLL:
            score, features = 0., np.zeros(len(POSITION_FEATURES))
            for number in range(2, 13):
                after = game.copy()
                dice = (number // 2, number - number // 2)
                after.execute(action, validate_action=False, action_record=ActionRecord(action, dice))
                value, vector = self._evaluate(after)
                probability = number_probability(number)
                score += probability * value
                features += probability * vector
            return score, features, None
        if kind == ActionType.BUY_DEVELOPMENT_CARD:
            after = game.copy()
            ps = after.state.player_state
            prefix = f"P{after.state.color_to_index[self.color]}_"
            for resource in ("SHEEP", "WHEAT", "ORE"):
                ps[prefix + resource + "_IN_HAND"] -= 1
            # Match the original unknown-card count proxy, never the deck's top card.
            ps[prefix + "KNIGHT_IN_HAND"] += 1
            score, features = self._evaluate(after)
            if self.original._vp(game) >= self.original.config.get("dev_buy_vp_threshold", 7):
                score += self.original.config.get("dev_buy_late_bonus", 0.)
            return score, features, None
        if kind == ActionType.MOVE_ROBBER:
            coordinate, victim = action.value
            outcomes = [(None, 1.)]
            if victim is not None:
                prefix = f"P{game.state.color_to_index[victim]}_"
                hand = {r: game.state.player_state[prefix + r + "_IN_HAND"] for r in RESOURCES}
                total = sum(hand.values())
                outcomes = [(r, n / total) for r, n in hand.items() if n] if total else [(None, 1.)]
            score, features = 0., np.zeros(len(POSITION_FEATURES))
            for resource, probability in outcomes:
                after = game.copy()
                if resource is None:
                    no_theft = Action(action.color, kind, (coordinate, None))
                    after.execute(no_theft, validate_action=False)
                else:
                    after.execute(action, validate_action=False, action_record=ActionRecord(action, resource))
                value, vector = self._evaluate(after)
                score += probability * value
                features += probability * vector
            score += 1e7 * self.original.robber._robber_score(game, action)
            return score, features, None
        after = game.copy()
        # Monopoly is deterministic here: official observations expose resources.
        after.execute(action, validate_action=False)
        score, features = self._evaluate(after)
        return score, features, after

    def plan(self, game, actions):
        actions = list(actions)
        if not actions:
            raise ValueError("Cannot plan without legal actions")
        self.base.cache.clear()
        self.position.clear()
        current = self.position(game)
        raw, immediate, leaves, winning, trace = [], [], [], [], []
        roots, followups = 0, 0
        eligible = self.original._eligible(game)
        current_vp = self.original._vp(game)
        for action in actions:
            score, vector, after = self._outcome(game, action)
            leaf = vector
            verified = after is not None and self.original._vp(after) >= game.vps_to_win
            detail = {"root": _description(action), "followup": None, "followups": 0,
                      "deterministic": after is not None, "lookahead": False, "visited": []}
            # Root caps only limit extra search; every legal move retains its score.
            if (eligible and action.action_type in ROOT_TYPES
                    and roots < self.original.max_roots and followups < self.original.max_followups):
                roots += 1
                detail["lookahead"] = True
                if (after is not None and after.state.current_color() == self.color
                        and after.winning_color() is None):
                    seconds = sorted((a for a in after.playable_actions if a.action_type in POINT_TYPES),
                                     key=lambda a: (a.action_type.name, repr(a.value)))
                    for second in seconds:
                        if followups >= self.original.max_followups:
                            break
                        later = after.copy()
                        later.execute(second, validate_action=False)
                        followups += 1
                        detail["followups"] += 1
                        detail["visited"].append(_description(second))
                        later_vp = self.original._vp(later)
                        later_win = later_vp >= game.vps_to_win
                        # Original tactical search only evaluates lines that gain VP.
                        if later_vp <= current_vp and not later_win:
                            continue
                        later_score, later_vector = self._evaluate(later)
                        if later_score > score or (later_win and not verified):
                            score, leaf = later_score, later_vector
                            detail["followup"] = _description(second)
                        verified = verified or later_win
            raw.append(score)
            immediate.append(vector - current)
            leaves.append(leaf - current)
            winning.append(verified)
            trace.append(detail)
        raw = np.asarray(raw, dtype=float)
        one_hot = np.zeros((len(actions), len(ACTION_TYPES)))
        for i, action in enumerate(actions):
            if action.action_type.name in ACTION_TYPES:
                one_hot[i, ACTION_TYPES.index(action.action_type.name)] = 1.
        features = np.column_stack((immediate, leaves, one_hot, normalize_scores(raw)))
        if features.shape != (len(actions), ACTION_DIM) or not np.isfinite(features).all():
            raise ValueError("Invalid action planner features")
        return Plan(actions, raw, features, np.asarray(winning, dtype=bool), trace)
