"""Deterministic planner for the pinned, patched catan_kit engine.

Uses legal engine transitions, a bounded search over complete purchase plans,
exact conditional development-card probabilities, and a transparent economic
evaluation. No runtime learning, external files, network, or private information.
The evaluator is hand designed, not a pretrained neural network.
"""

from collections import deque
from functools import lru_cache
from math import comb, exp, log
import random

import catan_kit  # Installs the supplied determinism patches.
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import ActionRecord, ActionType as AT


RES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEVS = ("KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY")
DECK = (14, 5, 2, 2, 2)
RID = {r: i for i, r in enumerate(RES)}
COSTS = ((1, 1, 0, 0, 0), (1, 1, 1, 1, 0), (0, 0, 0, 2, 3), (0, 0, 1, 1, 1))
PIPS = (0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1)
NEIGHBORS = tuple(tuple(sorted(v for v in STATIC_GRAPH.neighbors(n) if v < 54)) for n in range(54))


def action_key(action):
    return action.action_type.value, repr(action.value)


def hand(ps, key):
    return tuple(ps[key + r + "_IN_HAND"] for r in RES)


@lru_cache(maxsize=512)
def hidden_belief(unseen, n, public):
    """Count posterior conditioned on the fact that the live game has not ended.

    Declining to play a card is not taken as proof of its absence. Unknown card
    ages are not assumed known. The posterior is a count prior, not an inference
    that a sampled opponent hand is the real one.
    """
    holdings = []

    def visit(t, left, counts, weight):
        if t == 4:
            if 0 <= left <= unseen[t]:
                h = counts + (left,)
                if public + h[1] < 10:
                    holdings.append((h, weight * comb(unseen[t], left)))
            return
        lo = max(0, left - sum(unseen[t + 1:]))
        hi = min(left, unseen[t], 9 - public if t == 1 else left)
        for count in range(lo, hi + 1):
            visit(t + 1, left - count, counts + (count,), weight * comb(unseen[t], count))

    visit(0, n, (), 1)
    total = sum(w for _, w in holdings)
    if not total:
        raise ValueError("No nonterminal hidden hand agrees with the observation")
    posterior = tuple((h, w / total) for h, w in holdings)
    expected = tuple(sum(h[t] * p for h, p in posterior) for t in range(5))
    remaining = sum(unseen) - n
    draws = tuple((unseen[t] - expected[t]) / remaining if remaining else 0.0 for t in range(5))
    vp = tuple(sum(p for h, p in posterior if h[1] == k) for k in range(6))
    present = tuple(sum(p for h, p in posterior if h[t]) for t in range(5))
    representative = max(posterior, key=lambda hp: (hp[1], hp[0]))[0]
    return posterior, expected, draws, vp, present, representative


class Agent:
    """Fixed-work planner; returns an index in the supplied legal action list."""

    def __init__(self, config=None, plan_observer=None):
        config = dict(config or {})
        resource_weights = config.get("resource_weights", (0.88, 0.83, 0.78, 1.20, 1.18))
        if isinstance(resource_weights, (str, bytes)):
            raise ValueError("nano.resource_weights must contain five positive finite values")
        try:
            resource_weights = tuple(resource_weights)
        except TypeError as exc:
            raise ValueError("nano.resource_weights must contain five positive finite values") from exc
        if any(isinstance(value, bool) for value in resource_weights):
            raise ValueError("nano.resource_weights must contain five positive finite values")
        try:
            resource_weights = tuple(float(value) for value in resource_weights)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("nano.resource_weights must contain five positive finite values") from exc
        if len(resource_weights) != 5 or any(not (0 < value < float("inf")) for value in resource_weights):
            raise ValueError("nano.resource_weights must contain five positive finite values")
        self.resource_weights = resource_weights
        self.production_scale = self._positive_scale(config, "production_scale")
        self.expansion_scale = self._positive_scale(config, "expansion_scale")
        self.army_scale = self._positive_scale(config, "army_scale")
        self.vp_scale = self._positive_scale(config, "vp_scale")
        self.search_depth = config.get("search_depth")
        if self.search_depth is not None and (
                isinstance(self.search_depth, bool) or not isinstance(self.search_depth, int)
                or not 1 <= self.search_depth <= 8):
            raise ValueError("nano.search_depth must be an integer from 1 through 8")
        self.plan_observer = plan_observer
        self.beam_width = config.get("beam_width", 4)
        if type(self.beam_width) is not int or not 1 <= self.beam_width <= 16:
            raise ValueError("nano.beam_width must be an integer from 1 through 16")
        self.map_signature = None
        self.layout_cache = {}
        self.opening_scores = {}
        self.robber_cache = {}
        self.calls = 0
        self.last_history_length = -1
        self.distances = []
        for start in range(54):
            distance = [99] * 54
            distance[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for nxt in NEIGHBORS[node]:
                    if distance[nxt] == 99:
                        distance[nxt] = distance[node] + 1
                        queue.append(nxt)
            self.distances.append(distance)

    @staticmethod
    def _positive_scale(config, name):
        value = config.get(name, 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"nano.{name} must be positive and finite")
        try:
            value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"nano.{name} must be positive and finite") from exc
        if not (0 < value < float("inf")):
            raise ValueError(f"nano.{name} must be positive and finite")
        return value

    def _init_map(self, board):
        signature = (tuple((c, t.resource, t.number) for c, t in sorted(board.map.land_tiles.items())),
                     tuple((str(r), tuple(sorted(ns))) for r, ns in
                           sorted(board.map.port_nodes.items(), key=lambda x: str(x[0]))))
        if signature == self.map_signature:
            return
        self.map_signature = signature
        self.layout_cache.clear()
        self.opening_scores.clear()
        self.robber_cache.clear()
        self.node_prod, self.node_rolls, self.node_ports = [], [], []
        for node in range(54):
            prod = [0.0] * 5
            rolls = [[0] * 5 for _ in range(13)]
            for tile in board.map.adjacent_tiles[node]:
                if tile.resource is not None:
                    r = RID[tile.resource]
                    prod[r] += PIPS[tile.number] / 36.0
                    rolls[tile.number][r] += 1
            self.node_prod.append(tuple(prod))
            self.node_rolls.append(tuple(tuple(x) for x in rolls))
            self.node_ports.append(tuple(r for r, nodes in board.map.port_nodes.items() if node in nodes))

    def _set_belief(self, obs):
        self.unseen = tuple(DECK[t] - obs.my_devs[d] - obs.played_devs["me"][d]
                            - obs.played_devs["opp"][d] for t, d in enumerate(DEVS))
        self.belief = hidden_belief(self.unseen, obs.opp_dev_count, obs.opp_public_vps)

    def _assign_hidden(self, game, counts):
        """Install a possible hand in a simulation without reading masked cards."""
        ps = game.state.player_state
        for t, d in enumerate(DEVS):
            ps[self.ek + d + "_IN_HAND"] = counts[t]
            if d != "VICTORY_POINT":
                ps[self.ek + d + "_OWNED_AT_START"] = False
        ps[self.ek + "ACTUAL_VICTORY_POINTS"] = ps[self.ek + "VICTORY_POINTS"] + counts[1]
        game.state.development_listdeck = [d for t, d in enumerate(DEVS)
                                           for _ in range(self.unseen[t] - counts[t])]

    def _layout(self, board):
        """Cache spatial features separately from hands and robber placement."""
        key = (tuple(sorted(board.buildings.items())),
               tuple((e, c) for e, c in sorted(board.roads.items()) if e[0] < e[1]))
        cached = self.layout_cache.get(key)
        if cached is not None:
            return cached
        result = {}
        for color in (self.me, self.enemy):
            prod = [0.0] * 5
            rolls = [[0] * 5 for _ in range(13)]
            settlements, cities, sources = [], [], set()
            rates = [4] * 5
            for node, (owner, kind) in board.buildings.items():
                if owner != color:
                    continue
                sources.add(node)
                (settlements if kind == "SETTLEMENT" else cities).append(node)
                mult = 1 if kind == "SETTLEMENT" else 2
                for r in range(5):
                    prod[r] += mult * self.node_prod[node][r]
                for number in range(2, 13):
                    for r in range(5):
                        rolls[number][r] += mult * self.node_rolls[node][number][r]
                for port in self.node_ports[node]:
                    if port is None:
                        rates = [min(x, 3) for x in rates]
                    else:
                        rates[RID[port]] = 2
            for edge, owner in board.roads.items():
                if owner == color:
                    sources.update(n for n in edge if board.buildings.get(n, (color,))[0] == color)
            dist = [99] * 54
            todo = deque()
            for node in sorted(sources):
                dist[node] = 0
                todo.append(node)
            while todo:
                node = todo.popleft()
                if board.buildings.get(node, (color,))[0] != color:
                    continue
                for nxt in NEIGHBORS[node]:
                    road = board.roads.get((node, nxt))
                    if road is not None and road != color:
                        continue
                    weight = 0 if road == color else 1
                    distance = dist[node] + weight
                    if distance < dist[nxt] and distance <= 4:
                        dist[nxt] = distance
                        if weight:
                            todo.append(nxt)
                        else:
                            todo.appendleft(nxt)
            sites = []
            for node in sorted(board.board_buildable_ids):
                d = dist[node]
                if d <= 4:
                    p = self.node_prod[node]
                    quality = sum(p) + 0.45 * sum(min(0.12, p[r]) / (0.10 + prod[r]) for r in range(5))
                    sites.append((d, quality, node))
            sites.sort(key=lambda x: (-x[1] / (1 + x[0]), x[2]))
            result[color] = (tuple(prod), tuple(tuple(x) for x in rolls), tuple(rates),
                             tuple(sorted(settlements)), tuple(sorted(cities)), tuple(sites), tuple(dist))
        if len(self.layout_cache) >= 512:
            self.layout_cache.clear()
        self.layout_cache[key] = result
        return result

    def _effective_prod(self, board, color, spatial):
        prod = list(spatial[0])
        tile = board.map.land_tiles[board.robber_coordinate]
        if tile.resource is not None:
            blocked = sum((1 if kind == "SETTLEMENT" else 2) for node in tile.nodes.values()
                          for owner, kind in [board.buildings.get(node, (None, None))] if owner == color)
            # The robber will move again; current blockage is not permanent.
            prod[RID[tile.resource]] -= 0.60 * blocked * PIPS[tile.number] / 36.0
        return prod

    def _economy(self, prod, rates):
        weights = self.resource_weights
        value = 0.0
        surplus = sum(max(0.0, prod[j] - 0.10) / rates[j] * 0.20 for j in range(5))
        for r in range(5):
            effective = prod[r] + surplus - max(0.0, prod[r] - 0.10) / rates[r] * 0.20
            value += weights[r] * (2.8 * effective + 0.28 * (1 - exp(-8 * effective)))
        return value + sum(max(0.0, prod[r] - 0.17) * (4 - rates[r]) * 0.45 for r in range(5))

    @staticmethod
    def _deficit(h, cost, rates, bank):
        missing = tuple(max(0, cost[r] - h[r]) for r in range(5))
        trades = sum(max(0, h[r] - cost[r]) // rates[r] for r in range(5))
        possible = sum(min(missing[r], bank[r]) for r in range(5))
        return sum(missing) - min(trades, possible)

    def _hand_value(self, h, spatial, bank):
        prod, _, rates, settlements, cities, sites, _ = spatial
        value = sum((0.10 + (0.035 if r in (3, 4) else 0.0)) / (1 + 1.5 * prod[r])
                    * min(h[r], 5) + 0.032 * max(0, h[r] - 5) for r in range(5))
        goals = []
        if settlements and len(cities) < 4:
            goals.append((COSTS[2], 0.66))
        if len(settlements) < 5 and any(d == 0 for d, _, _ in sites):
            goals.append((COSTS[1], 0.73))
        elif sites and len(settlements) < 5:
            d = min(x[0] for x in sites)
            goals.append(((1 + d, 1 + d, 1, 1, 0), 0.64 / (1 + 0.18 * d)))
        if self.deck_left:
            goals.append((COSTS[3], 0.37))
        if goals:
            readiness = sorted((b * exp(-0.85 * self._deficit(h, c, rates, bank)) for c, b in goals), reverse=True)
            value += readiness[0] + 0.18 * sum(readiness[1:])
        if sum(h) > 7:
            value -= (sum(h) // 2) * 0.024
        return value

    def _threat(self, game):
        """Integrate next-roll purchase threats over dice and hidden VP counts.

        An opponent may select either an expansion, city, or development route.
        Knight presence is a conservative bound, not knowledge of card age.
        """
        st, ps = game.state, game.state.player_state
        public = ps[self.ek + "VICTORY_POINTS"]
        if public < 5:
            return 0.0
        need_one = sum(p for v, p in enumerate(self.belief[3]) if public + v == 9)
        need_two = sum(p for v, p in enumerate(self.belief[3]) if public + v == 8)
        if need_one + need_two == 0:
            return 0.0
        layout = self._layout(st.board)
        enemy = layout[self.enemy]
        h = hand(ps, self.ek)
        city_ok = bool(enemy[3]) and ps[self.ek + "CITIES_AVAILABLE"] > 0
        settle_ok = ps[self.ek + "SETTLEMENTS_AVAILABLE"] > 0 and any(x[0] == 0 for x in enemy[5])
        tile = st.board.map.land_tiles[st.board.robber_coordinate]
        blocked = {self.me: [0] * 5, self.enemy: [0] * 5}
        if tile.resource is not None:
            for node in tile.nodes.values():
                owner, kind = st.board.buildings.get(node, (None, None))
                if owner is not None:
                    blocked[owner][RID[tile.resource]] += 1 if kind == "SETTLEMENT" else 2
        danger = 0.0
        for number in range(2, 13):
            if number == 7:
                continue
            after = list(h)
            bank = list(st.resource_freqdeck)
            for r in range(5):
                ours = layout[self.me][1][number][r] - (blocked[self.me][r] if tile.number == number else 0)
                theirs = enemy[1][number][r] - (blocked[self.enemy][r] if tile.number == number else 0)
                if ours + theirs <= bank[r]:
                    after[r] += theirs
                    bank[r] -= ours + theirs
            city = city_ok and self._deficit(after, COSTS[2], enemy[2], bank) == 0
            settle = settle_ok and self._deficit(after, COSTS[1], enemy[2], bank) == 0
            dev = self.deck_left and self._deficit(after, COSTS[3], enemy[2], bank) == 0
            danger += PIPS[number] / 36.0 * need_one * max(float(city or settle), self.belief[2][1] if dev else 0.0)
        army_target = ps[self.mk + "PLAYED_KNIGHT"] if ps[self.mk + "HAS_ARMY"] else 2
        if ps[self.ek + "PLAYED_KNIGHT"] >= army_target and not ps[self.ek + "HAS_ARMY"]:
            danger = max(danger, (need_one + need_two) * self.belief[4][0] * 0.65)
        return danger

    def _value(self, game):
        st, ps = game.state, game.state.player_state
        points = ps[self.mk + "ACTUAL_VICTORY_POINTS"]
        if points >= game.vps_to_win:
            return 10000.0
        spatial = self._layout(st.board)
        own, enemy = spatial[self.me], spatial[self.enemy]
        prod = self._effective_prod(st.board, self.me, own)
        eprod = self._effective_prod(st.board, self.enemy, enemy)
        ep = ps[self.ek + "VICTORY_POINTS"] + self.belief[1][1]
        point_value = 1.45 + 0.14 * max(points, ep)
        economy_scale = max(0.65, 1.25 - 0.055 * max(points, ep))
        score = self.vp_scale * (point_value * points - (1.10 + 0.12 * ep) * ep)
        score += economy_scale * self.production_scale * (self._economy(prod, own[2]) - 0.82 * self._economy(eprod, enemy[2]))
        score += self._hand_value(hand(ps, self.mk), own, st.resource_freqdeck)
        score -= 0.55 * self._hand_value(hand(ps, self.ek), enemy, st.resource_freqdeck)
        if ps[self.mk + "SETTLEMENTS_AVAILABLE"] > 0:
            options = sorted((q * exp(-0.73 * d) for d, q, _ in own[5]), reverse=True)
            score += self.expansion_scale * 0.28 * sum(q * w for q, w in zip(options[:3], (1.0, 0.40, 0.15)))
        if enemy[5]:
            score -= self.expansion_scale * 0.13 * max(q * exp(-0.73 * d) for d, q, _ in enemy[5])
        knights, played = ps[self.mk + "KNIGHT_IN_HAND"], ps[self.mk + "PLAYED_KNIGHT"]
        enemy_army = ps[self.ek + "PLAYED_KNIGHT"] if ps[self.ek + "HAS_ARMY"] else 2
        needed = max(0, enemy_army + 1 - played)
        if not ps[self.mk + "HAS_ARMY"]:
            score += self.army_scale * (0.28 * min(played, 2) + 0.32 * min(knights, needed))
            score += self.army_scale * 0.42 * min(1.0, (played + min(knights, 2)) / max(3, enemy_army + 1))
        else:
            score += self.army_scale * 0.13 * knights
        score += 0.30 * ps[self.mk + "YEAR_OF_PLENTY_IN_HAND"]
        score += 0.30 * ps[self.mk + "ROAD_BUILDING_IN_HAND"]
        eh = hand(ps, self.ek)
        score += (0.24 + 0.055 * min(8, max(eh))) * ps[self.mk + "MONOPOLY_IN_HAND"]
        length, enemy_length = ps[self.mk + "LONGEST_ROAD_LENGTH"], ps[self.ek + "LONGEST_ROAD_LENGTH"]
        target = max(5, enemy_length + 1) if ps[self.ek + "HAS_ROAD"] else 5
        if not ps[self.mk + "HAS_ROAD"]:
            score += 0.07 * min(length, target) + 0.18 * exp(-max(0, target - length))
        else:
            score += 0.04 * min(3, max(0, length - enemy_length))
        return score

    def _opening_score(self, nodes):
        if nodes in self.opening_scores:
            return self.opening_scores[nodes]
        prod = tuple(sum(self.node_prod[n][r] for n in nodes) for r in range(5))
        rates = [4] * 5
        for n in nodes:
            for port in self.node_ports[n]:
                if port is None:
                    rates = [min(x, 3) for x in rates]
                else:
                    rates[RID[port]] = 2
        score = self._economy(prod, rates)
        score += 0.65 * min(prod[3], prod[4] * 2 / 3)
        score += 0.30 * min(prod[0], prod[1], prod[2], prod[3])
        rolls = [sum(sum(self.node_rolls[n][number]) for n in nodes) for number in range(2, 13)]
        score -= 0.012 * sum(max(0, x - 2) ** 2 for x in rolls)
        if len(nodes) == 2:
            start = [sum(self.node_rolls[nodes[-1]][number][r] for number in range(2, 13)) for r in range(5)]
            score += 0.045 * (start[3] + start[4]) + 0.018 * sum(start)
        self.opening_scores[nodes] = score
        return score

    def _opening(self, game, actions):
        board = game.state.board
        owned = tuple(sorted(n for n, (c, _) in board.buildings.items() if c == self.me))
        if actions[0].action_type == AT.BUILD_ROAD:
            def road_score(action):
                g = game.copy()
                g.execute(action)
                sites = self._layout(g.state.board)[self.me][5]
                return max((q * exp(-0.65 * d) for d, q, _ in sites), default=0.0)
            return max(actions, key=lambda a: (road_score(a), action_key(a)))
        if owned:
            return max(actions, key=lambda a: (self._opening_score(owned + (a.value,)), -a.value))
        candidates = sorted((a.value for a in actions), key=lambda n: (-self._opening_score((n,)), n))[:10]
        all_free = set(board.board_buildable_ids)
        # Cache pair scores: setup minimax otherwise repeats the same arithmetic.
        pairs = {}
        def pair(a, b):
            key = a, b
            if key not in pairs:
                pairs[key] = self._opening_score(key)
            return pairs[key]
        def score_first(node):
            free = all_free - {node} - set(NEIGHBORS[node])
            future = sorted(free, key=lambda n: (-pair(node, n), n))
            if not board.buildings:
                replies = sorted(free, key=lambda n: (-self._opening_score((n,)), n))[:4]
                values = []
                for reply in replies:
                    after = free - {reply} - set(NEIGHBORS[reply])
                    second_replies = sorted(after, key=lambda n: (-pair(reply, n), n))[:3]
                    for second in second_replies:
                        available = after - {second} - set(NEIGHBORS[second])
                        ours = max((pair(node, n) for n in future if n in available), default=0.0)
                        values.append(ours - 0.30 * pair(reply, second))
                return 0.6 * min(values) + 0.4 * sum(values) / len(values) if values else 0.0
            return max((pair(node, n) for n in future), default=self._opening_score((node,)))
        selected = max(candidates, key=lambda n: (score_first(n), -n))
        return next(a for a in actions if a.value == selected)

    def _robber(self, game, actions):
        cache_key = self._signature(game), self.belief[1:5], self.me
        if cache_key in self.robber_cache:
            return self.robber_cache[cache_key]
        best, best_value = actions[0], -float("inf")
        for action in actions:
            h = hand(game.state.player_state, self.ek) if action.value[1] is not None else (0,) * 5
            total = sum(h)
            outcomes = [(r, amount / total) for r, amount in enumerate(h) if amount] if total else [(None, 1.0)]
            score = 0.0
            for r, probability in outcomes:
                g = game.copy()
                g.execute(action, action_record=ActionRecord(action, RES[r] if r is not None else None))
                score += probability * (self._value(g) - 5.5 * self._threat(g))
            if score > best_value:
                best_value, best = score, action
        if len(self.robber_cache) >= 64:
            self.robber_cache.clear()
        self.robber_cache[cache_key] = best, best_value
        return best, best_value

    def _discard(self, game, actions):
        ps = game.state.player_state
        h = hand(ps, self.mk)
        spatial = self._layout(game.state.board)[self.me]
        remaining = game.state.discard_counts[game.state.color_to_index[self.me]]
        keep = sum(h) - remaining
        # Construct final kept hands around purchases, then fill by marginal value.
        protected = [(0,) * 5] + list(COSTS[1:])
        protected += [tuple(a[r] + b[r] for r in range(5)) for a in COSTS[1:] for b in COSTS[1:]]
        candidates = []
        bank = game.state.resource_freqdeck
        for goal in protected:
            target = [min(h[r], goal[r]) for r in range(5)]
            if sum(target) > keep:
                continue
            while sum(target) < keep:
                options = []
                for r in range(5):
                    if target[r] < h[r]:
                        after = target.copy()
                        after[r] += 1
                        options.append((self._hand_value(after, spatial, bank), -r, after))
                target = max(options, key=lambda x: (x[0], x[1]))[2]
            candidates.append((self._hand_value(target, spatial, bank), tuple(target)))
        target = max(candidates)[1]
        return next(a for a in actions if h[RID[a.value]] > target[RID[a.value]])

    def _buy_value(self, game, action):
        """Integrate draws in bounded utility, not a huge terminal sentinel.

        A small chance of drawing a winning VP must not automatically outweigh
        every strong nonterminal plan. The logistic scale is a heuristic value
        calibration, not a claim of an empirically calibrated win probability.
        """
        utility = 0.0
        for t, probability in enumerate(self.belief[2]):
            if probability <= 0:
                continue
            h = next(h for h, _ in self.belief[0] if h[t] < self.unseen[t])
            g = game.copy()
            self._assign_hidden(g, h)
            g.execute(action, action_record=ActionRecord(action, DEVS[t]))
            value = self._value(g)
            utility += probability * (1.0 if value >= 10000 else 1 / (1 + exp(-value / 4)))
        utility = min(1 - 1e-12, max(1e-12, utility))
        return 4 * log(utility / (1 - utility))

    def _signature(self, game):
        st = game.state
        return (tuple(sorted(st.board.buildings.items())),
                tuple((e, c) for e, c in sorted(st.board.roads.items()) if e[0] < e[1]),
                st.board.robber_coordinate, tuple(st.resource_freqdeck),
                tuple(sorted(st.player_state.items())), st.is_road_building,
                st.free_roads_available, st.current_prompt, st.current_turn_index)

    def _trade_plans(self, game, root_actions):
        """Fund purchases by converting surplus cards directly into deficits.

        At fixed ports, an intermediate currency conversion cannot improve a
        direct trade. Do not trade away cards needed by the target purchase.
        This removes irrelevant exchanges from the resource search entirely.
        Promising paths are still validated action by action by the engine.
        """
        st, ps = game.state, game.state.player_state
        spatial = self._layout(st.board)[self.me]
        rates = spatial[2]
        h0, bank0 = hand(ps, self.mk), tuple(st.resource_freqdeck)
        goals = []
        if spatial[3] and ps[self.mk + "CITIES_AVAILABLE"]:
            goals.append(COSTS[2])
        if ps[self.mk + "SETTLEMENTS_AVAILABLE"] and spatial[5]:
            d = min(x[0] for x in spatial[5])
            goals.append((1 + d, 1 + d, 1, 1, 0))
        if self.deck_left:
            goals.append(COSTS[3])
        root_trades = {a.value for a in root_actions if a.action_type == AT.MARITIME_TRADE}
        if not goals or not root_trades:
            return []
        chosen, used = [], set()
        marginal = tuple((0.10 + (0.035 if r in (3, 4) else 0.0)) /
                         (1 + 1.5 * spatial[0][r]) for r in range(5))
        for cost in goals:
            missing = tuple(max(0, cost[r] - h0[r]) for r in range(5))
            count = sum(missing)
            if not 0 < count <= 6:
                continue
            if any(missing[r] > bank0[r] for r in range(5)):
                continue
            if sum(max(0, h0[r] - cost[r]) // rates[r] for r in range(5)) < count:
                continue
            takes = tuple(r for r in range(5) for _ in range(missing[r]))
            queue = [(h0, ())]
            for take in takes:
                layer, seen = [], set()
                for h, path in queue:
                    for give in range(5):
                        if give == take or h[give] - rates[give] < cost[give]:
                            continue
                        offer = (RES[give],) * rates[give] + (None,) * (4 - rates[give]) + (RES[take],)
                        if not path and offer not in root_trades:
                            continue
                        after = list(h)
                        after[give] -= rates[give]
                        after[take] += 1
                        after = tuple(after)
                        if after in seen:
                            continue
                        seen.add(after)
                        rank = sum(marginal[r] * min(after[r], 5) + 0.032 * max(0, after[r] - 5)
                                   for r in range(5))
                        layer.append((rank, after, path + (offer,)))
                layer.sort(key=lambda x: (-x[0], repr(x[2])))
                queue = [(h, path) for _, h, path in layer[:4]]
                if not queue:
                    break
            if queue:
                path = queue[0][1]
                if path not in used:
                    used.add(path)
                    chosen.append(path)
        return chosen

    def _turn_search(self, game, actions):
        # Fixed quotas decrease in long games, where the clock stops granting
        # increments after 96 decisions. This depends on decision count, not time.
        quota = 32 if self.calls < 64 else (20 if self.calls < 96 else 12)
        if game.state.player_state[self.mk + "ACTUAL_VICTORY_POINTS"] >= 8:
            for action in actions:
                if action.action_type in (AT.BUILD_CITY, AT.BUILD_SETTLEMENT,
                                           AT.BUILD_ROAD, AT.PLAY_KNIGHT_CARD):
                    g = game.copy()
                    g.execute(action)
                    if g.state.player_state[self.mk + "ACTUAL_VICTORY_POINTS"] >= 10:
                        if self.plan_observer is not None:
                            self.plan_observer(10000.0, g, action)
                        return action
        end = next((a for a in actions if a.action_type == AT.END_TURN), None)
        baseline = self._value(game)
        candidates = [(baseline, game, end)] if end is not None else []
        frontier, seen, expanded = [], set(), 0
        for path in self._trade_plans(game, actions):
            if self.search_depth is not None and len(path) > self.search_depth:
                continue
            g = game.copy()
            first = next(a for a in actions if a.action_type == AT.MARITIME_TRADE and a.value == path[0])
            for offer in path:
                trade = next(a for a in g.playable_actions if a.action_type == AT.MARITIME_TRADE and a.value == offer)
                g.execute(trade)
            score = self._value(g) - (0.0001 * len(path) if self.search_depth is not None else 0.0)
            frontier.append((score, g, first, len(path)))
            if self.search_depth is not None:
                candidates.append((score, g, first))
        frontier.insert(0, (baseline, game, None, 0))
        for _ in range(5 if self.search_depth is None else self.search_depth):
            children = []
            for _, state, first, depth in frontier:
                if self.search_depth is not None and depth >= self.search_depth:
                    continue
                available = actions if first is None else sorted(state.playable_actions, key=action_key)
                ranked = []
                spatial = self._layout(state.state.board)[self.me]
                h = hand(state.state.player_state, self.mk)
                for action in available:
                    kind = action.action_type
                    if kind in (AT.END_TURN, AT.ROLL):
                        continue
                    if kind == AT.MARITIME_TRADE:
                        # Multitrade paths are seeded separately; suppress detours.
                        continue
                    if kind == AT.PLAY_YEAR_OF_PLENTY:
                        after = list(h)
                        for r in action.value:
                            after[RID[r]] += 1
                        rank = 8 + self._hand_value(after, spatial, state.state.resource_freqdeck)
                    elif kind == AT.PLAY_MONOPOLY:
                        rank = 10 + state.state.player_state[self.ek + action.value + "_IN_HAND"]
                    elif kind == AT.BUILD_CITY:
                        rank = 30 + sum(self.node_prod[action.value])
                    elif kind == AT.BUILD_SETTLEMENT:
                        rank = 31 + sum(self.node_prod[action.value])
                    elif kind == AT.BUILD_ROAD:
                        a, b = action.value
                        # Prefer the endpoint that extends, rather than retraces,
                        # our current network toward a useful open intersection.
                        endpoint = b if spatial[6][a] == 0 else a
                        rank = 5 + max((q / (1 + self.distances[endpoint][n]) for d, q, n in spatial[5]), default=0)
                    else:
                        rank = 12
                    ranked.append((rank, action))
                ranked.sort(key=lambda x: (-x[0], action_key(x[1])))
                selected, kinds = [], {}
                for _, action in ranked:
                    kind = action.action_type
                    cap = 4 if kind == AT.PLAY_YEAR_OF_PLENTY else 8
                    if kinds.get(kind, 0) < cap:
                        selected.append(action)
                        kinds[kind] = kinds.get(kind, 0) + 1
                if first is not None:
                    selected = selected[:5]
                for action in selected:
                    if expanded >= quota:
                        break
                    expanded += 1
                    root = action if first is None else first
                    if action.action_type == AT.BUY_DEVELOPMENT_CARD:
                        candidates.append((self._buy_value(state, action) - 0.0001 * (depth + 1), state, root))
                        continue
                    successor = state.copy()
                    successor.execute(action)
                    if successor.state.player_state[self.mk + "ACTUAL_VICTORY_POINTS"] >= 10:
                        if self.plan_observer is not None:
                            for score, candidate_state, candidate_root in candidates:
                                self.plan_observer(score, candidate_state, candidate_root)
                            self.plan_observer(10000.0, successor, root)
                        return root
                    if action.action_type == AT.PLAY_KNIGHT_CARD:
                        _, score = self._robber(successor, sorted(successor.playable_actions, key=action_key))
                        # _robber already integrated the threat penalty.
                        candidates.append((score + 5.5 * self._threat(successor), successor, root))
                        continue
                    signature = self._signature(successor)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    score = self._value(successor) - 0.0001 * (depth + 1)
                    candidates.append((score, successor, root))
                    children.append((score, successor, root, depth + 1))
            children.sort(key=lambda x: (-x[0], action_key(x[2])))
            frontier, counts = [], {}
            for item in children:
                key = action_key(item[2])
                if counts.get(key, 0) < 2:
                    frontier.append(item)
                    counts[key] = counts.get(key, 0) + 1
                if len(frontier) == self.beam_width:
                    break
            if not frontier or expanded >= quota:
                break
        # Compare robust purchase threats on a small set of finalists, including
        # saving. This avoids charging an expensive dice integration at every node.
        if self.plan_observer is not None:
            for score, state, root in candidates:
                self.plan_observer(score, state, root)
        candidates.sort(key=lambda x: (-x[0], action_key(x[2])))
        finalists, roots = [], set()
        for item in candidates:
            key = action_key(item[2])
            if key not in roots:
                roots.add(key)
                finalists.append(item)
            if len(finalists) == 5:
                break
        if end is not None and action_key(end) not in roots:
            finalists.append((baseline, game, end))
        if not finalists:
            return actions[0]
        return max(finalists, key=lambda x: (x[0] - 5.5 * self._threat(x[1]), action_key(x[2])))[2]

    @staticmethod
    def _budget_move(obs, actions):
        """Cheap legal policy when the harness reports little CPU remaining.

        It is deterministic in the complete observation, including cpu_left.
        The reserve protects against slower evaluation hardware and IPC costs.
        """
        def readiness(h):
            return max(-sum(max(0, cost[r] - h[r]) for r in range(5))
                       for cost in COSTS[1:])

        h = tuple(obs.my_resources[r] for r in RES)
        enemy = next((c for c, _ in obs.board.buildings.values() if c != obs.me), None)
        def score(action):
            kind = action.action_type
            if kind in (AT.BUILD_CITY, AT.BUILD_SETTLEMENT):
                production = sum(PIPS[t.number] for t in obs.board.map.adjacent_tiles[action.value]
                                 if t.resource is not None)
                return 100 + production
            if kind == AT.MOVE_ROBBER:
                tile = obs.board.map.land_tiles[action.value[0]]
                denial = 0.0
                for node in tile.nodes.values():
                    owner, building = obs.board.buildings.get(node, (None, None))
                    if owner is not None:
                        denial += (1 if owner == enemy else -1.1) * (1 if building == "SETTLEMENT" else 2)
                return (PIPS[tile.number] if tile.number is not None else 0) * denial + (1 if action.value[1] else 0)
            if kind == AT.DISCARD_RESOURCE:
                r = RID[action.value]
                return h[r] - (0.4 if r in (3, 4) else 0)
            if kind == AT.PLAY_MONOPOLY:
                return 8 + obs.opp_resources[action.value]
            if kind == AT.PLAY_YEAR_OF_PLENTY:
                after = list(h)
                for r in action.value:
                    after[RID[r]] += 1
                return 5 + readiness(after) - readiness(h)
            if kind == AT.MARITIME_TRADE:
                after = list(h)
                for r in action.value[:-1]:
                    if r is not None:
                        after[RID[r]] -= 1
                after[RID[action.value[-1]]] += 1
                return 4 if readiness(after) > readiness(h) else -2
            return {AT.PLAY_KNIGHT_CARD: 9, AT.ROLL: 7,
                    AT.BUY_DEVELOPMENT_CARD: 3, AT.PLAY_ROAD_BUILDING: 1,
                    AT.END_TURN: 0, AT.BUILD_ROAD: -1}.get(kind, -3)

        return max(actions, key=lambda a: (score(a), action_key(a)))

    def decide(self, obs, playable_actions):
        if not playable_actions:
            raise ValueError("The engine supplied no legal action")
        if len(playable_actions) == 1:
            return 0
        if obs.cpu_left is not None and obs.cpu_left < 0.100:
            chosen = self._budget_move(obs, playable_actions)
            return int(playable_actions.index(chosen))
        actions = sorted(playable_actions, key=action_key)
        self.me = obs.me
        self._init_map(obs.board)
        self._set_belief(obs)
        game = obs.sample_world(random.Random(0))
        st = game.state
        history_length = len(st.action_records)
        if history_length < self.last_history_length:
            self.calls = 0
            self.robber_cache.clear()
        self.last_history_length = history_length
        # Public history makes the work schedule independent of call repetition,
        # fresh instances, and process timing. Counting forced moves is deliberately
        # conservative because those moves do not earn clock increments.
        self.calls = 1 + sum(record.action.color == self.me for record in st.action_records)
        self.enemy = next(c for c in st.colors if c != self.me)
        self.mk = "P%d_" % st.color_to_index[self.me]
        self.ek = "P%d_" % st.color_to_index[self.enemy]
        self.deck_left = len(st.development_listdeck)
        self._assign_hidden(game, self.belief[5])
        if st.is_initial_build_phase:
            chosen = self._opening(game, actions)
        elif actions[0].action_type == AT.DISCARD_RESOURCE:
            chosen = self._discard(game, actions)
        elif actions[0].action_type == AT.MOVE_ROBBER:
            chosen = self._robber(game, actions)[0]
        elif not st.player_state[self.mk + "HAS_ROLLED"] and not st.is_road_building:
            chosen = next(a for a in actions if a.action_type == AT.ROLL)
            knight = next((a for a in actions if a.action_type == AT.PLAY_KNIGHT_CARD), None)
            if knight is not None:
                g = game.copy()
                g.execute(knight)
                if g.state.player_state[self.mk + "ACTUAL_VICTORY_POINTS"] >= 10:
                    chosen = knight
                else:
                    _, moved = self._robber(g, sorted(g.playable_actions, key=action_key))
                    if moved > self._value(game) - 5.5 * self._threat(game) + 0.12:
                        chosen = knight
        else:
            chosen = self._turn_search(game, actions)
        return int(playable_actions.index(chosen))
