"""Visible-information Catan policy with a one-action planning horizon.

The evaluator passes the whole Catanatron Game, which contains hidden state.
This policy reads only its own hand/cards, public board and scores, and legal
actions. It never executes a random draw or theft inside a hypothetical copy.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path


DEFAULT = {
    "vp": 20.0,
    "production": 19.0,
    "expansion": 9.0,
    "hand": 0.82,
    "readiness": 1.6,
    "dev_buy": 2.5,
    "army": 1.2,
    "robber": 9.0,
    "road_length": 0.33,
    "diversity": 0.13,
    "initial_value": False,
    "dev_early_penalty": 0.0,
}
RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
BASE_RESOURCE_VALUES = (1.15, 1.15, 0.85, 1.45, 1.35)
DEV_VALUES = {"KNIGHT": 1.5, "MONOPOLY": 1.4,
              "YEAR_OF_PLENTY": 1.3, "ROAD_BUILDING": 1.0}
DEV_COST = {"SHEEP": 1, "WHEAT": 1, "ORE": 1}
DETERMINISTIC = {
    "BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY", "MARITIME_TRADE",
    "DISCARD_RESOURCE", "PLAY_YEAR_OF_PLENTY", "PLAY_ROAD_BUILDING",
    "PLAY_KNIGHT_CARD", "END_TURN",
}


class CompetitiveAgent:
    def __init__(self, color, weights=None):
        self.color = color
        self.weights = {**DEFAULT, **(weights or {})}
        self._graph = None
        self._map_id = None

    def _network(self, map_obj):
        if self._map_id != id(map_obj):
            graph = {node: set() for node in map_obj.land_nodes}
            for tile in map_obj.land_tiles.values():
                for a, b in tile.edges.values():
                    graph[a].add(b)
                    graph[b].add(a)
            self._graph = graph
            self._map_id = id(map_obj)
        return self._graph

    def _production(self, game, color):
        state = game.state
        board = state.board
        robber_tile = board.map.land_tiles[board.robber_coordinate]
        result = dict.fromkeys(RESOURCES, 0.0)
        for node, (owner, kind) in board.buildings.items():
            if owner != color:
                continue
            for tile in board.map.adjacent_tiles[node]:
                if tile.resource is None or tile is robber_tile:
                    continue
                n = tile.number
                pips = (6 - abs(7 - n)) / 36.0
                result[tile.resource] += pips * (2 if kind == "CITY" else 1)
        return result

    def _resource_values(self, production, vp):
        weights = {}
        for index, resource in enumerate(RESOURCES):
            scarcity = max(0.0, 0.11 - production[resource])
            value = BASE_RESOURCE_VALUES[index] * (1.0 + 2.2 * scarcity)
            if vp >= 6 and resource in ("WOOD", "BRICK"):
                value *= 0.75
            weights[resource] = value
        return weights

    def _node_quality(self, game, node, resource_values):
        board = game.state.board
        quality = 0.0
        for tile in board.map.adjacent_tiles[node]:
            if tile.resource is None:
                continue
            pips = (6 - abs(7 - tile.number)) / 36.0
            quality += pips * resource_values[tile.resource]
        ports = board.map.port_nodes
        if node in ports.get(None, ()):
            quality += 0.035
        for resource in RESOURCES:
            if node in ports.get(resource, ()):
                quality += 0.045
        return quality

    def _expansion(self, game, resource_values):
        board = game.state.board
        color = self.color
        graph = self._network(board.map)
        # A 0-1 path counts additional roads needed to reach a legal site.
        owned = set(game.state.buildings_by_color[color]["SETTLEMENT"])
        owned.update(game.state.buildings_by_color[color]["CITY"])
        for edge, owner in board.roads.items():
            if owner == color:
                owned.update(edge)
        if not owned:
            return 0.0
        distances = {node: 0 for node in owned}
        agenda = [(0, node) for node in owned]
        heapq.heapify(agenda)
        while agenda:
            distance, node = heapq.heappop(agenda)
            if distance != distances[node] or distance >= 3:
                continue
            if board.is_enemy_node(node, color):
                continue
            for neighbor in graph[node]:
                edge_color = board.get_edge_color((node, neighbor))
                if edge_color is not None and edge_color != color:
                    continue
                next_distance = distance + (edge_color is None)
                if next_distance < distances.get(neighbor, 4):
                    distances[neighbor] = next_distance
                    heapq.heappush(agenda, (next_distance, neighbor))
        available = board.board_buildable_ids
        prospects = sorted(
            (self._node_quality(game, node, resource_values) / (0.75 + 1.25 * distance)
             for node in available if (distance := distances.get(node, 4)) <= 2),
            reverse=True,
        )
        return (prospects[0] if prospects else 0.0) + (0.35 * prospects[1] if len(prospects) > 1 else 0.0)

    def _hand_value(self, game, resource_values):
        state = game.state
        prefix = f"P{state.color_to_index[self.color]}_"
        hand = {resource: state.player_state[prefix + resource + "_IN_HAND"] for resource in RESOURCES}
        unit_value = sum(resource_values[resource] * (min(amount, 3) + 0.2 * max(0, amount - 3))
                         for resource, amount in hand.items())
        settlement = min(hand[r] for r in ("WOOD", "BRICK", "SHEEP", "WHEAT"))
        city = min(hand["WHEAT"] / 2, hand["ORE"] / 3)
        dev = min(hand[r] for r in ("SHEEP", "WHEAT", "ORE"))
        readiness = min(1.0, settlement) * 1.0 + min(1.0, city) * 1.2 + min(1.0, dev) * 0.35
        return self.weights["hand"] * unit_value + self.weights["readiness"] * readiness

    def _evaluate(self, game):
        state = game.state
        prefix = f"P{state.color_to_index[self.color]}_"
        vp = state.player_state[prefix + "ACTUAL_VICTORY_POINTS"]
        if vp >= game.vps_to_win:
            return 10000.0
        production = self._production(game, self.color)
        resource_values = self._resource_values(production, vp)
        production_score = sum(
            resource_values[r] * (production[r] + self.weights["diversity"] * production[r] / (production[r] + 0.07))
            for r in RESOURCES
        )
        expansion = self._expansion(game, resource_values)
        phase = 1.0 if vp <= 6 else 0.55
        score = (self.weights["vp"] * vp
                 + self.weights["production"] * phase * production_score
                 + self.weights["expansion"] * phase * expansion
                 + self._hand_value(game, resource_values)
                 + self.weights["road_length"] * state.player_state[prefix + "LONGEST_ROAD_LENGTH"])
        played_knights = state.player_state[prefix + "PLAYED_KNIGHT"]
        score += self.weights["army"] * min(played_knights, 3)
        for card, value in DEV_VALUES.items():
            score += value * state.player_state[prefix + card + "_IN_HAND"]
        return score

    def _robber_score(self, game, action):
        state = game.state
        board = state.board
        coord, victim = action.value
        tile = board.map.land_tiles[coord]
        if tile.resource is None:
            return -2.0
        pips = (6 - abs(7 - tile.number)) / 36.0
        impact = 0.0
        for node in tile.nodes.values():
            building = board.buildings.get(node)
            if building is None:
                continue
            owner, kind = building
            scale = 2 if kind == "CITY" else 1
            if owner == self.color:
                impact -= 2.2 * scale
            else:
                owner_prefix = f"P{state.color_to_index[owner]}_"
                enemy_vp = state.player_state[owner_prefix + "VICTORY_POINTS"]
                impact += scale * (0.8 + 0.12 * enemy_vp)
        if victim is not None:
            impact += 0.45
        return self.weights["robber"] * pips * impact

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        if game.state.is_initial_build_phase and self.weights["initial_value"]:
            from catanatron.players.value import ValueFunctionPlayer
            return ValueFunctionPlayer(self.color).decide(game, playable_actions)
        base = self._evaluate(game)
        state = game.state
        prefix = f"P{state.color_to_index[self.color]}_"
        own_vp = state.player_state[prefix + "ACTUAL_VICTORY_POINTS"]
        best_action, best_score = None, float("-inf")
        for action in playable_actions:
            kind = action.action_type.name
            if kind == "ROLL":
                score = base + 0.5
            elif kind == "MOVE_ROBBER":
                score = base + self._robber_score(game, action)
            elif kind == "BUY_DEVELOPMENT_CARD":
                production = self._production(game, self.color)
                values = self._resource_values(production, own_vp)
                cost = sum(values[r] for r in DEV_COST)
                score = base + self.weights["dev_buy"] - 0.5 * cost
                if own_vp < 5:
                    score -= self.weights["dev_early_penalty"]
                if own_vp >= 8:
                    score += 1.4
            elif kind == "PLAY_MONOPOLY":
                production = self._production(game, self.color)
                need = self._resource_values(production, own_vp)[action.value]
                score = base + (1.8 * need if state.num_turns > 25 else 0.0)
            elif kind in DETERMINISTIC:
                after = game.copy()
                after.execute(action, validate_action=False)
                score = self._evaluate(after)
                if kind == "END_TURN":
                    score += 0.08
            else:
                score = base - 100.0
            if score > best_score:
                best_score, best_action = score, action
        return best_action


def create_agent(color, checkpoint, seed):
    del seed
    weights = json.loads(Path(checkpoint).read_text()) if checkpoint is not None else None
    return CompetitiveAgent(color, weights)
