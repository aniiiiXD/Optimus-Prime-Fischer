"""Explainable prior over every legal action, with deterministic outcome features.

The original OneVOne values are an expert prior. They are compressed to ranks
before mixing with learning, so 3e14 VP weights cannot silence a neural score.
No strategic choice is removed from the legal candidate set.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from catanatron.models.enums import Action, ActionRecord, ActionType
from catanatron.models.map import number_probability
from catanatron.players.value import DEFAULT_WEIGHTS
from features import COMPONENT_NAMES, RESOURCES, candidate_features, prefix, state_features

SAFE_NAMES = frozenset({"END_TURN", "BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY",
                        "MARITIME_TRADE", "PLAY_YEAR_OF_PLENTY", "PLAY_ROAD_BUILDING",
                        "PLAY_KNIGHT_CARD", "PLAY_MONOPOLY", "DISCARD_RESOURCE"})


@dataclass
class Candidate:
    index: int
    action: object
    score: float
    state: np.ndarray
    successor: object | None
    rank: float = 0
    tactical_gain: float = 0
    robber_utility: float = 0
    unknown_card: bool = False
    stochastic: bool = False
    features: np.ndarray | None = None
    tactical_preferred: bool = False


def ranks(scores):
    """Equal expert scores get equal ranks; deterministic ties follow legal order."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 1 or np.ptp(scores) == 0:
        return np.zeros(len(scores), dtype=np.float32)
    # Rank distinct values to avoid diluting a rare strategic action with many
    # equivalent road/trade variants. No arbitrary absolute-score scale remains.
    unique = np.unique(scores)
    return (np.searchsorted(unique, scores) / (len(unique) - 1)).astype(np.float32)


class HeuristicExpert:
    def __init__(self, color, config):
        self.color, self.config = color, config
        self.weights = {**DEFAULT_WEIGHTS, **config.get("value_weights", {})}

    def value(self, vector):
        values = dict(zip(COMPONENT_NAMES, vector[:len(COMPONENT_NAMES)]))
        # Preserve the original expert's score as a prior, exposing each part.
        values["discard_penalty"] = values.pop("discard_risk")
        values["public_vps"] = values.pop("public_vp")
        road = values.pop("longest_road")
        return (sum(self.weights[name] * float(value) for name, value in values.items())
                + road * (self.weights["longest_road"] if values["buildable_nodes"] == 0 else 0.1))

    def robber_value(self, game, action):
        board, ps = game.state.board, game.state.player_state
        coordinate, victim = action.value
        tile = board.map.land_tiles[coordinate]
        if tile.resource is None:
            return -2.0
        opponent = next(c for c in game.state.colors if c != self.color)
        own, other = prefix(game, self.color), prefix(game, opponent)
        probability = (6-abs(7-tile.number))/36
        pressure = max(.8, min(1.45, 1 + .12*(ps[other+"VICTORY_POINTS"]-ps[own+"VICTORY_POINTS"])))
        def resource_value(color, public_vp):
            production = dict.fromkeys(RESOURCES, 0.)
            blocked_tile = board.map.land_tiles[board.robber_coordinate]
            for node, (owner, kind) in board.buildings.items():
                if owner != color:
                    continue
                for adjacent in board.map.adjacent_tiles[node]:
                    if adjacent.resource is not None and adjacent is not blocked_tile:
                        production[adjacent.resource] += (6-abs(7-adjacent.number))/36*(2 if kind == "CITY" else 1)
            index = RESOURCES.index(tile.resource)
            value = (1.15, 1.15, .85, 1.45, 1.35)[index]*(1+2.2*max(0., .11-production[tile.resource]))
            return value*(.75 if public_vp >= 6 and tile.resource in ("WOOD", "BRICK") else 1)
        own_value = resource_value(self.color, ps[own+"VICTORY_POINTS"])
        enemy_value = resource_value(opponent, ps[other+"VICTORY_POINTS"])
        blocked = 0.0
        for node in tile.nodes.values():
            if node in board.buildings:
                owner, kind = board.buildings[node]
                blocked += probability * (2 if kind == "CITY" else 1) * (pressure*enemy_value if owner == opponent else -1.2*own_value)
        hand = sum(ps[other+r+"_IN_HAND"] for r in RESOURCES)
        return 9 * blocked + ((.2 + .08*min(8, hand)) if victim == opponent and hand else 0)

    def annotate_tactics(self, game, before, candidates):
        """Keep the original two-action VP rule as an overridable prior.

        All candidates remain available. The preferred root gets the highest
        expert rank, while its prospective VP gain is exposed to the model.
        """
        state = game.state
        if (state.is_initial_build_phase or state.current_color() != self.color
                or state.current_prompt.name != "PLAY_TURN"
                or not state.player_state[prefix(game, self.color)+"HAS_ROLLED"]):
            return
        baseline = max(candidates, key=lambda c: c.score)
        current = before[13]
        baseline_vp = current
        if baseline.successor is not None:
            baseline_vp = baseline.state[13]
        elif baseline.unknown_card:
            baseline_vp += 1  # conservative prior; the separate win guard is exact
        if baseline_vp >= game.vps_to_win:
            return
        roots = {"MARITIME_TRADE", "BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY",
                 "PLAY_YEAR_OF_PLENTY", "PLAY_ROAD_BUILDING", "PLAY_MONOPOLY"}
        points = {"BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY"}
        limit = self.config.get("heuristic_plan_nodes", 128)
        nodes, best, best_key = 0, None, (False, baseline_vp, float("-inf"))

        def consider(candidate, vector):
            nonlocal best, best_key
            vp = vector[13]
            candidate.tactical_gain = max(candidate.tactical_gain, vp-current)
            key = (vp >= game.vps_to_win, vp, self.value(vector))
            if vp > current and key > best_key:
                best, best_key = candidate, key

        for c in candidates:
            if nodes >= limit:
                break
            if c.successor is None or c.action.action_type.name not in roots:
                continue
            consider(c, c.state)
            after = c.successor
            if after.state.current_color() != self.color:
                continue
            for second in sorted((a for a in after.playable_actions if a.action_type.name in points),
                                 key=lambda a: (a.action_type.name, repr(a.value))):
                if nodes >= limit:
                    break
                later = after.copy()
                later.execute(second, validate_action=False)
                nodes += 1
                consider(c, state_features(later, self.color))
        if best is not None and best_key[1] > baseline_vp:
            best.tactical_preferred = True

    def prepare(self, game, actions):
        before = state_features(game, self.color)
        candidates = []
        for index, action in enumerate(actions):
            name, after = action.action_type.name, None
            unknown = stochastic = False
            robber = 0.0
            if name in SAFE_NAMES:
                after = game.copy()
                after.execute(action, validate_action=False)
                vector = state_features(after, self.color)
                score = self.value(vector)
            elif name == "ROLL":
                # Enumerate public dice sums without consulting the future RNG.
                vectors = []
                for number in range(2, 13):
                    world = game.copy()
                    world.execute(action, validate_action=False,
                                  action_record=ActionRecord(action, (number//2, number-number//2)))
                    vectors.append(number_probability(number)*state_features(world, self.color))
                vector = sum(vectors)
                score = self.value(vector)
                stochastic = True
            elif name == "BUY_DEVELOPMENT_CARD":
                # Count an unknown acquired card, pay its known cost. Its type,
                # potential VP, and future effects stay unknown to the scorer.
                world = game.copy()
                ps, own = world.state.player_state, prefix(world, self.color)
                for resource in ("SHEEP", "WHEAT", "ORE"):
                    ps[own + resource + "_IN_HAND"] -= 1
                vector = state_features(world, self.color)
                vector[COMPONENT_NAMES.index("hand_devs")] += 1
                score = self.value(vector)
                if before[13] >= self.config.get("dev_buy_vp_threshold", 7):
                    score += self.config.get("dev_buy_late_bonus", 30000)
                unknown = stochastic = True
            elif name == "MOVE_ROBBER":
                # Move without resolving random theft; its expected utility is
                # a feature rather than a sampled resource identity.
                world = game.copy()
                world.state.board.robber_coordinate = action.value[0]
                vector = state_features(world, self.color)
                robber = self.robber_value(game, action)
                score = robber
                stochastic = True
            else:
                raise ValueError(f"Unimplemented legal action: {name}")
            gain = max(0, float(vector[13]-before[13]))
            candidates.append(Candidate(index, action, score, vector, after,
                                        tactical_gain=gain, robber_utility=robber,
                                        unknown_card=unknown, stochastic=stochastic))
        self.annotate_tactics(game, before, candidates)
        expert_ranks = ranks([c.score for c in candidates])
        if any(c.tactical_preferred for c in candidates):
            # Strictly preserve the expert preference without censoring moves
            # or introducing its astronomical VP coefficient into neural logits.
            expert_ranks *= len(candidates)/(len(candidates)+1)
            for c in candidates:
                if c.tactical_preferred:
                    expert_ranks[c.index] = 1.
        for c, rank in zip(candidates, expert_ranks):
            c.rank = float(rank)
            c.features = candidate_features(before, c.state, c.action, c.rank,
                                            c.tactical_gain, c.robber_utility,
                                            c.unknown_card, c.stochastic)
        return before, candidates
