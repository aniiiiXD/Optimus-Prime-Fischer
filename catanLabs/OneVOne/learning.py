"""Economy features and a learned residual over a shared all-legal-action planner."""

import json
import hashlib
from pathlib import Path

import numpy as np

from catanLabs.Competitive.agent import CompetitiveAgent, RESOURCES


MODEL_VERSION = "onevone-action-residual-v1"
POSITION_FEATURES = tuple(f"{phase}_{r.lower()}" for phase in ("early", "late") for r in RESOURCES) + (
    "early_balance", "early_expansion", "contested_sites", "settlement_readiness",
    "city_readiness", "early_roads", "discard_risk", "late_army", "public_vp_gap", "production_gap",
)
HEURISTIC_WEIGHTS = np.array([
    .2, .2, .1, .3, .3, .05, .05, .1, .35, .35,
    .25, .7, .4, .2, .25, -.15, -.1, .1, 0., 0.,
])


def checked_vector(values):
    result = np.asarray(values, dtype=float)
    if result.shape != (len(POSITION_FEATURES),) or not np.isfinite(result).all():
        raise ValueError("Invalid economy feature/weight vector")
    return result


def load_model(path):
    from catanLabs.OneVOne.action_planner import ACTION_DIM

    model = json.loads(Path(path).read_text())
    weights = np.asarray(model.get("weights"), dtype=float)
    if model.get("version") != MODEL_VERSION or weights.shape != (ACTION_DIM,) or not np.isfinite(weights).all():
        raise ValueError("Invalid action residual model version, dimensions or weights")
    model["weights"] = weights
    return model


def normalize_scores(raw):
    """A finite heuristic prior in logit units, preserving ordering across VP scales."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Expected finite scores for every legal action")
    delta = raw - np.median(raw)
    scale = max(1., float(np.median(np.abs(delta))))
    compressed = np.sign(delta) * np.log1p(np.abs(delta) / scale)
    return compressed / max(1., float(np.max(np.abs(compressed))))


def choose(plan, weights=None):
    """Only proven wins bypass scoring; learned logits can overturn any ordinary prior."""
    raw = np.asarray(plan.heuristic, dtype=float)
    prior = normalize_scores(raw)
    matrix = np.asarray(plan.features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(plan.actions) or not np.isfinite(matrix).all():
        raise ValueError("Invalid legal-action features")
    residual = np.zeros(len(raw))
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (matrix.shape[1],) or not np.isfinite(weights).all():
            raise ValueError("Invalid learned action weights")
        residual = matrix @ weights
        if not np.isfinite(residual).all():
            raise ValueError("Non-finite learned action scores")
    heuristic_index = int(raw.argmax())
    model_index = int((prior + residual).argmax()) if np.any(residual) else heuristic_index
    winners = np.flatnonzero(plan.winning)
    executed = int(winners[raw[winners].argmax()]) if len(winners) else model_index
    changed = executed != heuristic_index and not len(winners)
    description = lambda i: f"{plan.actions[i].action_type.name}:{plan.actions[i].value!r}"
    diagnostics = {
        "heuristic_index": heuristic_index, "model_index": model_index, "executed_index": executed,
        "winning_override": bool(len(winners)), "ml_changed": bool(changed),
        "cross_type": bool(changed and plan.actions[executed].action_type != plan.actions[heuristic_index].action_type),
        "legal_count": len(plan.actions), "heuristic_action": description(heuristic_index),
        "model_action": description(model_index), "executed_action": description(executed),
        "heuristic_prior": float(prior[heuristic_index]), "executed_prior": float(prior[executed]),
        "heuristic_residual": float(residual[heuristic_index]), "executed_residual": float(residual[executed]),
        "search_trace_hash": hashlib.sha256(json.dumps(plan.trace, sort_keys=True).encode()).hexdigest(),
        "followups": sum(row["followups"] for row in plan.trace),
    }
    return executed, diagnostics


class PositionFeatures:
    def __init__(self, color):
        self.color = color
        self.geometry = CompetitiveAgent(color)
        self.cache = {}

    def clear(self):
        self.cache.clear()

    def __call__(self, game):
        state, board, color = game.state, game.state.board, self.color
        if len(state.colors) != 2:
            raise ValueError("Economy features require two players")
        prefix = f"P{state.color_to_index[color]}_"
        ps = state.player_state
        key = (frozenset(board.buildings.items()), frozenset(board.roads.items()), board.robber_coordinate, ps[prefix + "VICTORY_POINTS"])
        if key not in self.cache:
            production = self.geometry._production(game, color)
            vp = ps[prefix + "VICTORY_POINTS"]
            resource_values = self.geometry._resource_values(production, vp)
            # Expansion only counts legal sites at most two additional roads away.
            expansion = self.geometry._expansion(game, resource_values)
            sites = set(board.buildable_node_ids(color))
            enemy = next(c for c in state.colors if c != color)
            enemy_production = sum(self.geometry._production(game, enemy).values())
            contested = sites.intersection(board.buildable_node_ids(enemy))
            urgency = max((self.geometry._node_quality(game, n, resource_values) for n in contested), default=0.)
            ports = board.get_player_port_resources(color)
            rates = {r: 2 if r in ports else 3 if None in ports else 4 for r in RESOURCES}
            roads = len({tuple(sorted(edge)) for edge, owner in board.roads.items() if owner == color})
            self.cache[key] = production, expansion, urgency, bool(sites), rates, roads, enemy_production
        production, expansion, urgency, sites, rates, roads, enemy_production = self.cache[key]
        hand = {r: ps[prefix + r + "_IN_HAND"] for r in RESOURCES}

        def readiness(cost, available):
            if not available:
                return 0.
            missing = sum(max(0, amount - hand[r]) for r, amount in cost.items())
            credits = sum(max(0, hand[r] - cost.get(r, 0)) // rates[r] for r in RESOURCES)
            # A feature estimate; the legal-action planner still validates every trade.
            return 1 / (1 + max(0, missing - credits))

        early = max(0., min(1., (7 - ps[prefix + "VICTORY_POINTS"]) / 5))
        enemy = next(c for c in state.colors if c != color)
        other = f"P{state.color_to_index[enemy]}_"
        prod = [production[r] for r in RESOURCES]
        settlements = len(state.buildings_by_color[color]["SETTLEMENT"])
        values = [early * p for p in prod] + [(1 - early) * p for p in prod]
        values += [
            early * sum(p / (p + .07) for p in prod) / 5,
            early * expansion, early * urgency,
            early * readiness(dict.fromkeys(("WOOD", "BRICK", "SHEEP", "WHEAT"), 1), sites and ps[prefix + "SETTLEMENTS_AVAILABLE"] > 0),
            readiness({"WHEAT": 2, "ORE": 3}, settlements and ps[prefix + "CITIES_AVAILABLE"] > 0),
            early * roads / 15, max(0, sum(hand.values()) - 7) / 10,
            (1 - early) * min(3, ps[prefix + "PLAYED_KNIGHT"]) / 3,
            (ps[prefix + "VICTORY_POINTS"] - ps[other + "VICTORY_POINTS"]) / game.vps_to_win,
            (sum(prod) - enemy_production) / 2,
        ]
        return checked_vector(values)


class LearningPlayer:
    def __init__(self, color, config, model=None):
        from catanLabs.OneVOne.action_planner import ActionPlanner

        self.color = color
        self.planner = ActionPlanner(color, config)
        self.weights = load_model(model)["weights"] if model is not None else None
        self.last_decision = None

    def decide(self, game, playable_actions):
        plan = self.planner.plan(game, playable_actions)
        index, self.last_decision = choose(plan, self.weights)
        return plan.actions[index]
