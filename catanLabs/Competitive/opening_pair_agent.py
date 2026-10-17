"""Unpromoted, bounded joint opening planner for the last setup seat.

The fourth player makes both setup placements consecutively. We plan that
deterministic four-action sequence, then use FairValue for every other decision.
Only public board data, own pieces, and validated legal actions are read.
"""

from __future__ import annotations

import json
from pathlib import Path

from catanatron.models.enums import ActionPrompt, ActionType, SETTLEMENT

from catanLabs.Competitive.fair_value_agent import FairValueAgent


RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
PIP_WEIGHTS = {"WOOD": 1.0, "BRICK": 1.0, "SHEEP": 0.90,
               "WHEAT": 1.15, "ORE": 1.10}


class OpeningPairAgent(FairValueAgent):
    def __init__(self, color, config):
        super().__init__(color, config)
        self.max_first = int(config.get("opening_max_first", 12))
        self.max_second_per_road = int(config.get("opening_max_second_per_road", 4))
        self.max_lines = int(config.get("opening_max_lines", 432))
        if min(self.max_first, self.max_second_per_road, self.max_lines) < 1:
            raise ValueError("Opening search budgets must be positive")
        self._pending = ()
        self._candidate_plan = ()
        self.last_scan = {}

    def _eligible(self, game, actions):
        state = game.state
        if (not state.is_initial_build_phase
                or state.current_prompt != ActionPrompt.BUILD_INITIAL_SETTLEMENT
                or state.current_color() != self.color
                or state.colors[-1] != self.color
                or len(state.buildings_by_color[self.color][SETTLEMENT]) != 0
                or not actions
                or any(a.action_type != ActionType.BUILD_SETTLEMENT for a in actions)):
            return False
        return sum(len(state.buildings_by_color[c][SETTLEMENT]) for c in state.colors) == 3

    @staticmethod
    def _neighbors(game):
        graph = {node: set() for node in game.state.board.map.land_nodes}
        for tile in game.state.board.map.land_tiles.values():
            for a, b in tile.edges.values():
                graph[a].add(b)
                graph[b].add(a)
        return graph

    @staticmethod
    def _node_data(game):
        board = game.state.board
        production = {}
        starting = {}
        for node in board.map.land_nodes:
            rates = dict.fromkeys(RESOURCES, 0)
            hand = dict.fromkeys(RESOURCES, 0)
            for tile in board.map.adjacent_tiles[node]:
                if tile.resource in rates:
                    rates[tile.resource] += 6 - abs(7 - tile.number)
                    hand[tile.resource] += 1
            production[node] = rates
            starting[node] = hand
        return production, starting

    def _pair_score(self, game, first, second, production, starting):
        """Cheap public opening proxy. Units are weighted dice pips."""
        rates = {r: production[first][r] + production[second][r] for r in RESOURCES}
        score = sum(PIP_WEIGHTS[r] * rates[r] for r in RESOURCES)
        # Reward usable build cycles; do not force all five resources.
        road_cycle = min(rates["WOOD"], rates["BRICK"])
        settlement_cycle = min(rates[r] for r in ("WOOD", "BRICK", "SHEEP", "WHEAT"))
        city_cycle = min(rates["WHEAT"] / 2, rates["ORE"] / 3)
        score += 0.38 * road_cycle + 0.55 * settlement_cycle + 0.45 * city_cycle
        # Only the second settlement yields starting cards.
        score += 0.38 * sum(PIP_WEIGHTS[r] * starting[second][r] for r in RESOURCES)
        score += 0.12 * sum(min(rates[r], 4) for r in RESOURCES)
        ports = game.state.board.map.port_nodes
        for node in (first, second):
            if node in ports.get(None, ()):
                score += 0.25
            for resource in RESOURCES:
                if node in ports.get(resource, ()):
                    score += min(0.50, 0.06 * rates[resource])
        return score

    @staticmethod
    def _road_score(game, edge, occupied, graph, production):
        board = game.state.board
        start = next((node for node in edge if node in occupied[-2:]), None)
        if start is None:
            return 0.0
        far = edge[1] if edge[0] == start else edge[0]
        blocked = set(occupied)
        for node in occupied:
            blocked.update(graph[node])
        prospects = []
        for node in graph[far]:
            if (node in blocked
                    or board.get_edge_color(tuple(sorted((far, node)))) is not None):
                continue
            prospects.append(sum(PIP_WEIGHTS[r] * production[node][r] for r in RESOURCES))
        return 0.12 * max(prospects, default=0.0)

    def find_opening_plan(self, game, playable_actions):
        """Return first move and store the exact legal continuation for decide()."""
        info = {"eligible": self._eligible(game, playable_actions),
                "first_considered": 0, "second_considered": 0,
                "lines_considered": 0, "truncated": False}
        self._candidate_plan = ()
        if not info["eligible"]:
            self.last_scan = info
            return None

        graph = self._neighbors(game)
        production, starting = self._node_data(game)
        available = {a.value for a in playable_actions}
        # Optimistic pair quality orders roots before any copied-state work.
        ranked = []
        for first in playable_actions:
            possible = (second for second in available
                        if second != first.value and second not in graph[first.value])
            upper = max((self._pair_score(game, first.value, second, production, starting)
                         for second in possible), default=float("-inf"))
            ranked.append((upper, first))
        ranked.sort(key=lambda item: (-item[0], repr(item[1].value)))
        if len(ranked) > self.max_first:
            info["truncated"] = True

        best_score = float("-inf")
        best_line = ()
        existing_occupied = tuple(game.state.board.buildings)
        for _, first in ranked[:self.max_first]:
            after_first = game.copy()
            after_first.execute(first, validate_action=True)
            info["first_considered"] += 1
            first_roads = [a for a in after_first.playable_actions
                           if a.action_type == ActionType.BUILD_ROAD]
            for road1 in first_roads:
                after_road1 = after_first.copy()
                after_road1.execute(road1, validate_action=True)
                if after_road1.state.current_color() != self.color:
                    continue
                seconds = [a for a in after_road1.playable_actions
                           if a.action_type == ActionType.BUILD_SETTLEMENT]
                seconds.sort(key=lambda a: (-self._pair_score(
                    game, first.value, a.value, production, starting), repr(a.value)))
                if len(seconds) > self.max_second_per_road:
                    info["truncated"] = True
                for second in seconds[:self.max_second_per_road]:
                    if info["lines_considered"] >= self.max_lines:
                        info["truncated"] = True
                        break
                    after_second = after_road1.copy()
                    after_second.execute(second, validate_action=True)
                    info["second_considered"] += 1
                    pair_score = self._pair_score(
                        game, first.value, second.value, production, starting)
                    occupied = (*existing_occupied, first.value, second.value)
                    road1_score = self._road_score(game, road1.value,
                                                   occupied, graph, production)
                    for road2 in after_second.playable_actions:
                        if road2.action_type != ActionType.BUILD_ROAD:
                            continue
                        if info["lines_considered"] >= self.max_lines:
                            info["truncated"] = True
                            break
                        info["lines_considered"] += 1
                        score = pair_score + road1_score + self._road_score(
                            game, road2.value, occupied, graph, production)
                        line = (first, road1, second, road2)
                        # Canonical action order gives stable tie breaking.
                        if score > best_score:
                            best_score = score
                            best_line = line
                    if info["lines_considered"] >= self.max_lines:
                        break
                if info["lines_considered"] >= self.max_lines:
                    break
            if info["lines_considered"] >= self.max_lines:
                break

        if best_line:
            # Validate the entire line once before it can be replayed live.
            branch = game.copy()
            for action in best_line:
                if action not in branch.playable_actions:
                    raise AssertionError("Opening planner produced an illegal continuation")
                branch.execute(action, validate_action=True)
            self._candidate_plan = best_line
            info["best_score"] = best_score
            info["best_line"] = [(a.action_type.name, repr(a.value)) for a in best_line]
        self.last_scan = info
        return best_line[0] if best_line else None

    def decide(self, game, playable_actions):
        if self._pending:
            expected = self._pending[0]
            if (game.state.is_initial_build_phase
                    and game.state.current_color() == self.color
                    and expected in playable_actions):
                self._pending = self._pending[1:]
                return expected
            self._pending = ()
        if self._eligible(game, playable_actions):
            first = self.find_opening_plan(game, playable_actions)
            if first is not None:
                self._pending = self._candidate_plan[1:]
                return first
        return super().decide(game, playable_actions)


def create_agent(color, checkpoint, seed):
    del seed
    config = json.loads(Path(checkpoint).read_text()) if checkpoint is not None else {}
    return OpeningPairAgent(color, config)
