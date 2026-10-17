"""Experimental resource-aware scoring, warm-started from OneVOne."""

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from catanatron.models.enums import Action, ActionType, RESOURCES
from catanLabs.OneVOne.agent import DEFAULT_CONFIG, OneVOneAgent, OneVOneRobber


RECIPES = (
    ({"WHEAT": 2, "ORE": 3}, 1.0),
    ({"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1}, 1.0),
    ({"SHEEP": 1, "WHEAT": 1, "ORE": 1}, 0.5),
)


def resource_need(hand, resource):
    """Marginal progress toward the opponent's next city, settlement, or card."""
    score = 0.0
    for cost, priority in RECIPES:
        missing = sum(max(0, count - hand[r]) for r, count in cost.items())
        if hand[resource] < cost.get(resource, 0):
            score += priority * (1.0 / missing - 1.0 / (missing + 1))
    return score


class ResourceRobber(OneVOneRobber):
    def __init__(self, color, config, need_weight):
        super().__init__(color, config.get("robber_weights"), config.get("onevone_robber"))
        self.need_weight = need_weight

    def _robber_score(self, game, action):
        score = super()._robber_score(game, action)
        tile = game.state.board.map.land_tiles[action.value[0]]
        if not self.need_weight or tile.resource is None:
            return score
        enemy = next(c for c in game.state.colors if c != self.color)
        prefix = f"P{game.state.color_to_index[enemy]}_"
        hand = {r: game.state.player_state[prefix + r + "_IN_HAND"] for r in RESOURCES}
        blocked = 0
        for node in tile.nodes.values():
            building = game.state.board.buildings.get(node)
            if building is not None and building[0] == enemy:
                blocked += 2 if building[1] == "CITY" else 1
        probability = (6 - abs(7 - tile.number)) / 36.0
        return score + self.weights["robber"] * self.need_weight * probability * blocked * resource_need(hand, tile.resource)


class ResourcePlayer(OneVOneAgent):
    def __init__(self, color, config, exact_monopoly, need_weight):
        super().__init__(color, config)
        self.exact_monopoly = exact_monopoly
        self.robber = ResourceRobber(color, config, need_weight)

    def _monopoly_value(self, game, resource):
        if not self.exact_monopoly:
            return super()._monopoly_value(game, resource)
        # Both resource hands are exposed by the kit; this move has no hidden draw.
        after = game.copy()
        after.execute(Action(self.color, ActionType.PLAY_MONOPOLY, resource))
        return self.value_fn(after, self.color)


class Agent:
    def __init__(self, exact_monopoly=True, params=(0.0, 0.0, 1.0)):
        if len(params) != 3 or any(not math.isfinite(x) or abs(x) > 4 for x in params):
            raise ValueError("Expected three finite scoring parameters in [-4, 4]")
        self.config = json.loads(DEFAULT_CONFIG.read_text())
        weights = self.config["value_weights"]
        weights["hand_synergy"] *= math.exp(params[0])
        for key in ("reachable_production_0", "reachable_production_1", "buildable_nodes"):
            weights[key] *= math.exp(params[1])
        self.exact_monopoly, self.need_weight = exact_monopoly, params[2]
        self.rng = random.Random(0)
        self.players = {}

    def decide(self, obs, playable_actions):
        if obs.me not in self.players:
            self.players[obs.me] = ResourcePlayer(obs.me, self.config, self.exact_monopoly, self.need_weight)
        world = obs.sample_world(self.rng)
        return playable_actions.index(self.players[obs.me].decide(world, playable_actions))
