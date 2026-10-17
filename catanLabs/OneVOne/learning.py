"""Economy features and a learned residual over a shared all-legal-action planner."""

import json
import hashlib
from pathlib import Path

import numpy as np

from catanLabs.Competitive.agent import CompetitiveAgent, RESOURCES


MODEL_VERSION = "onevone-action-residual-v1"
ADVANTAGE_MODEL_VERSION = "onevone-advantage-ensemble-v1"
ADVANTAGE_MLP_VERSION = "onevone-advantage-mlp-v1"
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


def heuristic_fingerprint(config):
    checked_vector(config.get("economy_weights", HEURISTIC_WEIGHTS))
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_model(path):
    from catanLabs.OneVOne.action_planner import ACTION_DIM

    model = json.loads(Path(path).read_text())
    if not isinstance(model, dict):
        raise ValueError("Invalid action model")
    if model.get("version") == MODEL_VERSION:
        try:
            weights = np.asarray(model.get("weights"), dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid action residual model weights") from exc
        if weights.shape != (ACTION_DIM,) or not np.isfinite(weights).all():
            raise ValueError("Invalid action residual model dimensions or weights")
        model["weights"] = weights
    elif model.get("version") in (ADVANTAGE_MODEL_VERSION, ADVANTAGE_MLP_VERSION):
        if model["version"] == ADVANTAGE_MLP_VERSION:
            try:
                scale = np.asarray(model.get("feature_scale"), dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Invalid advantage MLP feature scale") from exc
            members = model.get("members")
            if (scale.shape != (ACTION_DIM,) or not np.isfinite(scale).all() or np.any(scale <= 0)
                    or not isinstance(members, list) or len(members) < 3):
                raise ValueError("Invalid advantage MLP feature scale or members")
            checked_members = []
            for member in members:
                if not isinstance(member, dict) or set(member) != {"w1", "b1", "w2", "b2"}:
                    raise ValueError("Invalid advantage MLP member")
                try:
                    w1 = np.asarray(member["w1"], dtype=float)
                    b1 = np.asarray(member["b1"], dtype=float)
                    w2 = np.asarray(member["w2"], dtype=float)
                    b2 = member["b2"]
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("Invalid advantage MLP member weights") from exc
                if (w1.shape != (ACTION_DIM, 32) or b1.shape != (32,) or w2.shape != (32,)
                        or isinstance(b2, bool) or not isinstance(b2, (int, float))):
                    raise ValueError("Invalid advantage MLP member dimensions")
                try:
                    b2 = float(b2)
                except (OverflowError, ValueError) as exc:
                    raise ValueError("Invalid advantage MLP member bias") from exc
                if not (np.isfinite(w1).all() and np.isfinite(b1).all()
                        and np.isfinite(w2).all() and np.isfinite(b2)):
                    raise ValueError("Non-finite advantage MLP member weights")
                checked_members.append({"w1": w1, "b1": b1, "w2": w2, "b2": b2})
            model["feature_scale"] = scale
            model["members"] = checked_members
        else:
            try:
                ensemble = np.asarray(model.get("ensemble_weights"), dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Invalid advantage ensemble weights") from exc
            if (ensemble.ndim != 2 or ensemble.shape[0] < 3 or ensemble.shape[1] != ACTION_DIM
                    or not np.isfinite(ensemble).all()):
                raise ValueError("Invalid advantage ensemble dimensions or weights")
            model["ensemble_weights"] = ensemble
        gate = model.get("gate")
        fingerprint = model.get("heuristic_sha256")
        if (not isinstance(gate, dict) or set(gate) != {"enabled", "z", "margin"}
                or type(gate["enabled"]) is not bool):
            raise ValueError("Invalid advantage model gate")
        for name in ("z", "margin"):
            value = gate[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Invalid advantage model gate")
            try:
                value = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError("Invalid advantage model gate") from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError("Invalid advantage model gate")
            gate[name] = value
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(c not in "0123456789abcdef" for c in fingerprint)):
            raise ValueError("Invalid advantage model heuristic fingerprint")
        model["gate"] = {"enabled": gate["enabled"], "z": float(gate["z"]), "margin": float(gate["margin"])}
    else:
        raise ValueError("Invalid action model version")
    return model


def predict_advantages(model, features, heuristic_index):
    """Predict each root's paired advantage over the heuristic root for each member."""
    features = np.asarray(features, dtype=float)
    if (features.ndim != 2 or not np.isfinite(features).all()
            or isinstance(heuristic_index, bool) or not isinstance(heuristic_index, (int, np.integer))
            or not 0 <= heuristic_index < len(features)):
        raise ValueError("Invalid action features or heuristic index")
    if model.get("version") == ADVANTAGE_MODEL_VERSION:
        ensemble = np.asarray(model["ensemble_weights"], dtype=float)
        if ensemble.ndim != 2 or ensemble.shape[1] != features.shape[1] or not np.isfinite(ensemble).all():
            raise ValueError("Invalid advantage ensemble dimensions or weights")
        scores = features @ ensemble.T
        advantages = scores - scores[heuristic_index]
        if not np.isfinite(advantages).all():
            raise ValueError("Non-finite advantage predictions")
        return advantages
    if model.get("version") != ADVANTAGE_MLP_VERSION:
        raise ValueError("Unsupported advantage model version")
    scale = np.asarray(model["feature_scale"], dtype=float)
    if scale.shape != (features.shape[1],) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid advantage MLP feature scale")
    differences = (features - features[heuristic_index]) / scale
    predictions = []
    for member in model["members"]:
        def network(rows):
            return np.tanh(rows @ member["w1"] + member["b1"]) @ member["w2"] + member["b2"]

        predictions.append(0.5 * (network(differences) - network(-differences)))
    advantages = np.column_stack(predictions)
    if not np.isfinite(advantages).all():
        raise ValueError("Non-finite advantage predictions")
    return advantages


def normalize_scores(raw):
    """A finite heuristic prior in logit units, preserving ordering across VP scales."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Expected finite scores for every legal action")
    delta = raw - np.median(raw)
    scale = max(1., float(np.median(np.abs(delta))))
    compressed = np.sign(delta) * np.log1p(np.abs(delta) / scale)
    return compressed / max(1., float(np.max(np.abs(compressed))))


def choose(plan, weights=None, model=None):
    """Only proven wins bypass scoring; learned logits can overturn any ordinary prior."""
    raw = np.asarray(plan.heuristic, dtype=float)
    prior = normalize_scores(raw)
    matrix = np.asarray(plan.features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(plan.actions) or not np.isfinite(matrix).all():
        raise ValueError("Invalid legal-action features")
    residual = np.zeros(len(raw))
    gated = model is not None and model.get("version") in (ADVANTAGE_MODEL_VERSION, ADVANTAGE_MLP_VERSION)
    predicted_advantage = advantage_std = advantage_lcb = None
    gated_candidate_index = None
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (matrix.shape[1],) or not np.isfinite(weights).all():
            raise ValueError("Invalid learned action weights")
        residual = matrix @ weights
        if not np.isfinite(residual).all():
            raise ValueError("Non-finite learned action scores")
    heuristic_index = int(raw.argmax())
    if gated:
        advantages = predict_advantages(model, matrix, heuristic_index)
        if advantages.ndim != 2 or advantages.shape[1] < 3 or not np.isfinite(advantages).all():
            raise ValueError("Invalid advantage predictions")
        gate = model["gate"]
        means = advantages.mean(axis=1)
        deviations = advantages.std(axis=1)
        lower_bounds = means - gate["z"] * deviations - gate["margin"]
        gated_candidate_index = int(lower_bounds.argmax())
        predicted_advantage = float(means[gated_candidate_index])
        advantage_std = float(deviations[gated_candidate_index])
        advantage_lcb = float(lower_bounds[gated_candidate_index])
        gate_passed = bool(gate["enabled"] and advantage_lcb > 0.)
        model_index = gated_candidate_index if gate_passed else heuristic_index
    else:
        gate_passed = False
        model_index = int((prior + residual).argmax()) if np.any(residual) else heuristic_index
    winners = np.flatnonzero(plan.winning)
    executed = int(winners[raw[winners].argmax()]) if len(winners) else model_index
    changed = executed != heuristic_index and not len(winners)
    description = lambda i: f"{plan.actions[i].action_type.name}:{plan.actions[i].value!r}"
    diagnostics = {
        "heuristic_index": heuristic_index, "model_index": model_index, "executed_index": executed,
        "gated": bool(gated), "gate_passed": gate_passed,
        "predicted_advantage": predicted_advantage, "advantage_std": advantage_std,
        "advantage_lcb": advantage_lcb, "gated_candidate_index": gated_candidate_index,
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
        config = dict(config or {})
        self.color = color
        if config.get("planner") == "nano":
            from catanLabs.OneVOne.nano_planner import NanoPlanner
            self.planner = NanoPlanner(color, config)
        else:
            from catanLabs.OneVOne.action_planner import ActionPlanner
            self.planner = ActionPlanner(color, config)
        self.model = load_model(model) if model is not None else None
        if self.model is not None and self.model["version"] in (ADVANTAGE_MODEL_VERSION, ADVANTAGE_MLP_VERSION):
            if self.model["heuristic_sha256"] != heuristic_fingerprint(config):
                raise ValueError("Advantage model heuristic fingerprint does not match config")
            self.loaded_weights = None
        else:
            self.loaded_weights = self.model["weights"] if self.model is not None else None
        self.weights = self.loaded_weights
        self.last_decision = None
        self.last_plan = None

    def decide(self, game, playable_actions):
        plan = self.planner.plan(game, playable_actions)
        self.last_plan = plan
        index, self.last_decision = choose(plan, self.weights, self.model)
        return plan.actions[index]
