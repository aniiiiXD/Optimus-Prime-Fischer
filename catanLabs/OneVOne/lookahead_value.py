"""Score Nano's existing search leaves with an optional learned value ensemble."""

import json
from pathlib import Path

import numpy as np
from catan_kit.obs import Obs

from catanLabs.OneVOne.action_planner import ACTION_DIM
from catanLabs.OneVOne.learning import heuristic_fingerprint
from catanLabs.OneVOne.nano_planner import NanoPlanner
from my_team.nano import RES, action_key

VERSION = "nano-lookahead-value-v2"
HIDDEN = 32


def load_model(path, config):
    model = json.loads(Path(path).read_text())
    if not isinstance(model, dict) or model.get("version") != VERSION or model.get("config_sha256") != heuristic_fingerprint(config):
        raise ValueError("Lookahead model version or config fingerprint mismatch")
    scale = np.asarray(model.get("scale"), dtype=float)
    if scale.shape != (ACTION_DIM,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid lookahead feature scale")
    members = model.get("members")
    if not isinstance(members, list) or len(members) < 3:
        raise ValueError("Lookahead value needs at least three ensemble members")
    for member in members:
        if not isinstance(member, dict):
            raise ValueError("Invalid lookahead member")
        for name, shape in (("w1", (ACTION_DIM, HIDDEN)), ("b1", (HIDDEN,)),
                            ("w2", (HIDDEN,)), ("b2", ())):
            value = np.asarray(member.get(name), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid lookahead {name}")
            member[name] = value
    gate = model.get("gate", {})
    if not isinstance(gate, dict) or type(gate.get("enabled")) is not bool:
        raise ValueError("Invalid lookahead gate")
    for name in ("z", "margin"):
        value = gate.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError("Invalid lookahead gate")
    model["scale"] = scale
    return model


def predict(model, features):
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != ACTION_DIM or not np.isfinite(matrix).all():
        raise ValueError("Invalid lookahead features")
    normalized = matrix / np.asarray(model["scale"])
    values = np.column_stack([
        np.tanh(normalized @ np.asarray(m["w1"]) + m["b1"]) @ np.asarray(m["w2"]) + m["b2"]
        for m in model["members"]])
    if not np.isfinite(values).all():
        raise ValueError("Non-finite lookahead value")
    return np.clip(values, 0., 1.)


def leaf_options(planner, plan, game):
    """Every legal root gets an immediate leaf; searched continuations are additional leaves."""
    color, state = planner.color, game.state
    enemy = next(c for c in state.colors if c != color)
    own, other = (f"P{state.color_to_index[c]}_" for c in (color, enemy))
    ps = state.player_state
    current = planner.current_features
    production = current[:5] + current[5:10]
    context = [ps[prefix + r + "_IN_HAND"] / 5 for prefix in (own, other) for r in RES]
    context += [ps[own + "ACTUAL_VICTORY_POINTS"] / 10,
                ps[other + "VICTORY_POINTS"] / 10, state.num_turns / 100,
                sum(ps[other + d + "_IN_HAND"] for d in
                    ("KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY")) / 5]
    context += production.tolist() + [float(sum(production) - 2 * current[-1])]
    assert len(context) == 20
    options = []
    keyed = {action_key(action): i for i, action in enumerate(plan.actions)}

    def add(index, position, line, score):
        features = np.concatenate((context, position, plan.features[index, 40:]))
        if features.shape != (ACTION_DIM,) or not np.isfinite(features).all():
            raise ValueError("Invalid search-leaf features")
        options.append({"root": index, "features": features, "line": tuple(line), "score": float(score)})

    for i, action in enumerate(plan.actions):
        add(i, current + plan.features[i, :20], (action,), plan.trace[i]["direct_score"])
    history = len(state.action_records)
    for score, leaf, root in planner.captured:
        i = keyed.get(action_key(root))
        line = tuple(record.action for record in leaf.state.action_records[history:])
        if i is not None and line and action_key(line[0]) == action_key(root):
            add(i, planner.economy(leaf), line, score)
    # One root means one deployed decision: later actions are replanned.
    return [max((o for o in options if o["root"] == i), key=lambda o: o["score"])
            for i in range(len(plan.actions))]


def select(plan, options, model=None, eligible=True):
    baseline = int(np.argmax(plan.heuristic))
    chosen, leaf_index, advantage, passed = baseline, None, 0., False
    if model is not None and eligible:
        values = predict(model, [option["features"] for option in options])
        gate = model["gate"]
        robust = values.mean(axis=1) - gate["z"] * values.std(axis=1)
        best = [max((j for j, option in enumerate(options) if option["root"] == i),
                    key=lambda j: robust[j]) for i in range(len(plan.actions))]
        differences = values[best] - values[best[baseline]]
        bounds = differences.mean(axis=1) - gate["z"] * differences.std(axis=1) - gate["margin"]
        candidate = int(bounds.argmax())
        advantage = float(bounds[candidate])
        passed = bool(gate["enabled"] and advantage > 0)
        chosen = candidate if passed else baseline
        leaf_index = best[chosen]
    winners = np.flatnonzero(plan.winning)
    if len(winners):
        chosen = int(winners[np.asarray(plan.heuristic)[winners].argmax()])
        leaf_index = None
    changed = chosen != baseline and not len(winners)
    return chosen, {"heuristic_index": baseline, "executed_index": chosen,
                    "leaf_index": leaf_index, "advantage_lcb": advantage,
                    "gate_passed": passed, "ml_changed": bool(changed),
                    "cross_type": bool(changed and plan.actions[chosen].action_type != plan.actions[baseline].action_type),
                    "winning_override": bool(len(winners)), "legal_count": len(plan.actions),
                    "leaf_count": len(options), "scope_eligible": bool(eligible)}


class LookaheadPlayer:
    def __init__(self, color, config, model=None):
        self.color, self.config = color, config
        self.planner = NanoPlanner(color, config)
        self.model = load_model(model, config) if model is not None else None
        self.reply_used = False
        self.last_turn = -1

    def decide(self, game, actions, cpu_left=None, capture=True):
        if game.state.num_turns < self.last_turn:
            self.reply_used = False
        self.last_turn = game.state.num_turns
        prefix = f"P{game.state.color_to_index[self.color]}_"
        eligible = (game.state.current_prompt.name == "PLAY_TURN"
                    and game.state.player_state[prefix + "HAS_ROLLED"]
                    and game.state.num_turns <= 100)
        needs_reply = (self.config.get("reply_check", False) and not self.reply_used and eligible
                and game.state.num_turns >= 12
                and game.state.player_state[prefix + "ACTUAL_VICTORY_POINTS"] >= 3
                and sum(game.state.player_state[prefix + r + "_IN_HAND"] for r in RES) >= 4
                and len({a.action_type for a in actions}) > 1
                and (cpu_left is None or cpu_left >= .250))
        needs_model = self.model is not None and self.model["gate"]["enabled"] and eligible
        if not capture and not needs_model and not needs_reply:
            view = Obs(game, self.color)
            view.cpu_left = cpu_left
            observer = self.planner.nano.plan_observer
            self.planner.nano.plan_observer = None
            try:
                index = self.planner.nano.decide(view, actions)
            finally:
                self.planner.nano.plan_observer = observer
            self.last_decision = {"heuristic_index": index, "executed_index": index, "leaf_index": None,
                                  "advantage_lcb": 0., "gate_passed": False, "ml_changed": False,
                                  "cross_type": False, "winning_override": False, "leaf_count": 0,
                                  "legal_count": len(actions), "scope_eligible": bool(eligible), "reply_checked": False}
            self.last_decision["executed_action"] = f"{actions[index].action_type.name}:{actions[index].value!r}"
            return actions[index]
        self.plan = self.planner.plan(game, actions, cpu_left)
        self.options = leaf_options(self.planner, self.plan, game)
        index, self.last_decision = select(self.plan, self.options, self.model, eligible)
        model_index = index
        self.last_decision["reply_checked"] = False
        if needs_reply and not self.plan.winning.any():
            from catanLabs.OneVOne.reply_search import check_replies
            self.reply_used = True
            index, diagnostics = check_replies(game, self.plan, index, self.color, self.config)
            self.last_decision.update(diagnostics)
        self.last_decision["executed_index"] = index
        self.last_decision["ml_changed"] = bool(self.last_decision["ml_changed"] and index == model_index)
        self.last_decision["cross_type"] = bool(self.last_decision["ml_changed"] and
            self.plan.actions[index].action_type != self.plan.actions[self.last_decision["heuristic_index"]].action_type)
        action = self.plan.actions[index]
        self.last_decision["executed_action"] = f"{action.action_type.name}:{action.value!r}"
        return self.plan.actions[index]
