"""Expose Nano's complete legal-action search to the shared advantage scorer."""

import hashlib
import random

import numpy as np
from catanatron.models.enums import ActionRecord, ActionType
from catanatron.models.map import number_probability
from catan_kit.obs import Obs

from catanLabs.OneVOne.action_planner import ACTION_DIM, Plan, _description
from catanLabs.OneVOne.learning import POSITION_FEATURES, PositionFeatures, normalize_scores
from catanLabs.OneVOne.policy_features import ACTION_TYPES
from my_team.nano import Agent as NanoAgent, RES, action_key


class NanoPlanner:
    def __init__(self, color, config):
        self.color = color
        self.config = config
        self.economy = PositionFeatures(color)
        self.last_index = None
        self.captured = []
        self.nano = NanoAgent(self.config.get("nano", {}), plan_observer=self._observe)

    def _observe(self, score, state, root):
        self.captured.append((float(score), state, root))

    def plan(self, game, actions, cpu_left=None):
        actions = list(actions)
        if not actions:
            raise ValueError("Cannot plan without legal actions")
        obs = Obs(game, self.color)
        obs.cpu_left = cpu_left
        self.captured = []
        nano = self.nano
        self.last_index = nano.decide(obs, actions)

        world = obs.sample_world(random.Random(0))
        st = world.state
        nano.me = self.color
        nano.enemy = next(color for color in st.colors if color != self.color)
        nano.mk = f"P{st.color_to_index[nano.me]}_"
        nano.ek = f"P{st.color_to_index[nano.enemy]}_"
        nano._init_map(st.board)
        nano._set_belief(obs)
        nano.deck_left = len(st.development_listdeck)
        nano._assign_hidden(world, nano.belief[5])

        self.economy.clear()
        current = self.economy(world)
        self.current_features = current.copy()
        keyed = {action_key(action): i for i, action in enumerate(actions)}
        by_root = [[] for _ in actions]
        for score, state, root in self.captured:
            index = keyed.get(action_key(root))
            if index is not None:
                by_root[index].append((score, state))

        scores, immediate, leaves, winning, trace = [], [], [], [], []
        for i, action in enumerate(actions):
            score, vector, after = self._immediate(nano, world, action)
            direct_score = score
            leaf = vector
            verified = after is not None and self._actual_vp(nano, after) >= game.vps_to_win
            plans = sorted(by_root[i], key=lambda row: row[0], reverse=True)
            if plans and action.action_type != ActionType.BUY_DEVELOPMENT_CARD:
                plan_score, best_state = plans[0]
                score = plan_score - 5.5 * nano._threat(best_state)
                leaf = self.economy(best_state)
            plan_rows = []
            for candidate_score, candidate_state in plans:
                candidate_features = self.economy(candidate_state)
                candidate_win = self._actual_vp(nano, candidate_state) >= game.vps_to_win
                verified = verified or candidate_win
                line = candidate_state.state.action_records[len(world.state.action_records):]
                described_line = [_description(record.action) for record in line]
                plan_rows.append({
                    "score": float(candidate_score),
                    "features": candidate_features.tolist(),
                    "feature_hash": hashlib.sha256(candidate_features.tobytes()).hexdigest(),
                    "winning": bool(candidate_win),
                    "line": described_line,
                    "depth": len(described_line),
                })
            scores.append(float(score))
            immediate.append(vector - current)
            leaves.append(leaf - current)
            winning.append(bool(verified))
            trace.append({
                "root": _description(action), "direct_score": float(direct_score),
                "captured_candidates": plan_rows, "captured_count": len(plan_rows),
                "followups": len(plan_rows),
                "search_depth": nano.search_depth or 5,
            })

        heuristic_index = int(self.last_index)
        prior = np.minimum(normalize_scores(scores), 0.99)
        prior[heuristic_index] = 1.0
        one_hot = np.zeros((len(actions), len(ACTION_TYPES)))
        for i, action in enumerate(actions):
            if action.action_type.name in ACTION_TYPES:
                one_hot[i, ACTION_TYPES.index(action.action_type.name)] = 1.0
        features = np.column_stack((np.asarray(immediate), np.asarray(leaves), one_hot, prior))
        if features.shape != (len(actions), ACTION_DIM) or not np.isfinite(features).all():
            raise ValueError("Invalid Nano planner features")
        return Plan(actions, prior, features, np.asarray(winning, dtype=bool), trace)

    @staticmethod
    def _actual_vp(nano, game):
        return game.state.player_state[nano.mk + "ACTUAL_VICTORY_POINTS"]

    def _immediate(self, nano, game, action):
        kind = action.action_type
        if kind == ActionType.BUY_DEVELOPMENT_CARD:
            after = game.copy()
            ps = after.state.player_state
            for resource in ("SHEEP", "WHEAT", "ORE"):
                ps[nano.mk + resource + "_IN_HAND"] -= 1
            return nano._buy_value(game, action), self.economy(after), None
        if kind == ActionType.ROLL:
            score = 0.0
            vector = np.zeros(len(POSITION_FEATURES))
            for number in range(2, 13):
                probability = number_probability(number)
                if not probability:
                    continue
                after = game.copy()
                dice = (number // 2, number - number // 2)
                after.execute(action, action_record=ActionRecord(action, dice))
                score += probability * (nano._value(after) - 5.5 * nano._threat(after))
                vector += probability * self.economy(after)
            return score, vector, None
        if kind == ActionType.MOVE_ROBBER:
            h = tuple(game.state.player_state[nano.ek + r + "_IN_HAND"] for r in RES) if action.value[1] is not None else (0,) * 5
            total = sum(h)
            outcomes = [(r, amount / total) for r, amount in enumerate(h) if amount] if total else [(None, 1.0)]
            score = 0.0
            vector = np.zeros(len(POSITION_FEATURES))
            for resource, probability in outcomes:
                after = game.copy()
                after.execute(action, action_record=ActionRecord(action, RES[resource] if resource is not None else None))
                score += probability * (nano._value(after) - 5.5 * nano._threat(after))
                vector += probability * self.economy(after)
            return score, vector, None
        after = game.copy()
        after.execute(action)
        return nano._value(after) - 5.5 * nano._threat(after), self.economy(after), after
