"""Clock-bounded OneVOne search with an optional learned early-position critic."""

import json
import math
import random
import time

from agent import Agent as BaseAgent
from catanLabs.OneVOne.agent import DEFAULT_CONFIG, OneVOneAgent
from catanLabs.Competitive.fair_value_agent import FairValueAgent
from catanLabs.Competitive.tactical_vp_agent import ROOT_TYPES, POINT_TYPES
from catanatron.features import build_production_features, reachability_features
from catanatron.models.enums import ActionType, RESOURCES, DEVELOPMENT_CARDS
from value_critic import Critic, encode


class CachedValue:
    def __init__(self, color, weights):
        self.color, self.w = color, weights
        self.cache = {}

    def __call__(self, game, color):
        state, board, w = game.state, game.state.board, self.w
        key = (frozenset(board.buildings.items()), frozenset(board.roads.items()), board.robber_coordinate)
        if key not in self.cache:
            prod = build_production_features(True)(game, color)
            own_prod = sum(prod[f"EFFECTIVE_P0_{r}_PRODUCTION"] for r in RESOURCES)
            own_prod += sum(prod[f"EFFECTIVE_P0_{r}_PRODUCTION"] != 0 for r in RESOURCES) * 4 * 0.02778
            enemy_prod = sum(prod[f"EFFECTIVE_P1_{r}_PRODUCTION"] for r in RESOURCES)
            reach = reachability_features(game, color, 2)
            r0, r1 = (sum(reach[f"P0_{level}_ROAD_REACHABLE_{r}"] for r in RESOURCES) for level in (0, 1))
            buildable = len(board.buildable_node_ids(color))
            tiles = {tile for node, (owner, _) in board.buildings.items() if owner == color for tile in board.map.adjacent_tiles[node]}
            self.cache[key] = (own_prod, enemy_prod, r0, r1, buildable, len(tiles))
        production, enemy_production, r0, r1, buildable, tiles = self.cache[key]
        prefix = f"P{state.color_to_index[color]}_"
        ps = state.player_state
        hand = {r: ps[prefix + r + "_IN_HAND"] for r in RESOURCES}
        city = (max(2 - hand["WHEAT"], 0) + max(3 - hand["ORE"], 0)) / 5
        settlement = sum(max(1 - hand[r], 0) for r in ("WHEAT", "SHEEP", "BRICK", "WOOD")) / 4
        count = sum(hand.values())
        return float(ps[prefix + "VICTORY_POINTS"] * w["public_vps"]
                     + production * w["production"] + enemy_production * w["enemy_production"]
                     + r0 * w["reachable_production_0"] + r1 * w["reachable_production_1"]
                     + (2 - city - settlement) / 2 * w["hand_synergy"]
                     + buildable * w["buildable_nodes"] + tiles * w["num_tiles"]
                     + count * w["hand_resources"] + (w["discard_penalty"] if count > 7 else 0)
                     + ps[prefix + "LONGEST_ROAD_LENGTH"] * (w["longest_road"] if buildable == 0 else 0.1)
                     + sum(ps[prefix + d + "_IN_HAND"] for d in DEVELOPMENT_CARDS) * w["hand_devs"]
                     + ps[prefix + "PLAYED_KNIGHT"] * w["army_size"])


class ClockPlayer(OneVOneAgent):
    def __init__(self, color, config):
        super().__init__(color, config)
        self.value_fn = CachedValue(color, self.weights)

    def choose(self, game, actions, deadline):
        if len(actions) == 1:
            return actions[0]
        if actions[0].action_type == ActionType.MOVE_ROBBER:
            return max(actions, key=lambda a: self.robber._robber_score(game, a))
        best, best_score = actions[0], float("-inf")
        for action in actions:
            if best_score != float("-inf") and time.process_time() >= deadline:
                break
            if action.action_type == ActionType.ROLL:
                score = self._roll_value(game)
            elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                score = self._buy_value(game)
            elif action.action_type == ActionType.PLAY_MONOPOLY:
                score = self._monopoly_value(game, action.value)
            else:
                after = game.copy()
                after.execute(action, validate_action=False)
                score = self.value_fn(after, self.color)
            if score > best_score:
                best, best_score = action, score
        return best


class Agent(BaseAgent):
    def __init__(self, checkpoint=None, exploration=0.0, blend=0.25, value_weights=None):
        super().__init__()
        self.config = json.loads(DEFAULT_CONFIG.read_text())
        if value_weights is not None:
            allowed = {"production", "reachable_production_0", "reachable_production_1", "buildable_nodes", "army_size", "public_vps"}
            if set(value_weights) - allowed or any(not math.isfinite(v) or v <= 0 for v in value_weights.values()):
                raise ValueError("Invalid heuristic priority weights")
            self.config["value_weights"].update(value_weights)
        self.config.update(tactical_max_roots=8, tactical_max_followups=16)
        self.critic = Critic(checkpoint) if checkpoint else None
        self.exploration, self.blend = exploration, blend

    def decide(self, obs, playable_actions):
        started = time.process_time()
        if obs.me not in self.players:
            self.players[obs.me] = ClockPlayer(obs.me, self.config)
        player = self.players[obs.me]
        player.value_fn.cache.clear()
        world = obs.sample_world(self.rng)
        # Reserve clock for serializing the response; skip the extra scan near exhaustion.
        allowance = obs.cpu_left if obs.cpu_left is not None else 1.0
        initial = world.state.current_prompt.name in {"BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD"}
        deadline = started + min(0.12 if initial else 0.01, max(0.001, allowance * 0.5))
        baseline = player.choose(world, playable_actions, deadline)
        if allowance - (time.process_time() - started) > 0.08:
            baseline = player.find_override(world, playable_actions, baseline) or baseline
        early = world.state.num_turns < 50 and player._vp(world) < 6
        choices = ROOT_TYPES | {ActionType.END_TURN}
        if not early or baseline.action_type not in choices:
            return playable_actions.index(baseline)
        if self.exploration and random.random() < self.exploration:
            same = [a for a in playable_actions if a.action_type == baseline.action_type]
            if len(same) > 1:
                scored = []
                for action in same:
                    after = world.copy()
                    after.execute(action)
                    scored.append((player.value_fn(after, obs.me), action))
                top = sorted(scored, key=lambda pair: pair[0], reverse=True)[:3]
                baseline = random.choice(top)[1]
        if self.critic is None or world.state.current_prompt.name != "PLAY_TURN" or baseline.action_type == ActionType.PLAY_ROAD_BUILDING:
            return playable_actions.index(baseline)
        deadline = started + min(0.035, max(0.0, allowance * 0.65))
        if time.process_time() >= deadline:
            return playable_actions.index(baseline)
        # A small root set retains legal trades/builds; unknown draws are never simulated.
        root_types = (ROOT_TYPES | {ActionType.END_TURN}) - {ActionType.PLAY_ROAD_BUILDING}
        roots = [baseline] + [a for a in playable_actions if a != baseline and a.action_type in root_types]
        roots[1:] = sorted(roots[1:], key=lambda a: a.action_type not in POINT_TYPES)
        leaves, moves, scores = [], [], []
        for root in roots[:12]:
            if time.process_time() >= deadline:
                break
            after = world.copy()
            after.execute(root)
            if player._vp(after) >= world.vps_to_win:
                return playable_actions.index(root)
            leaves.append(encode(after, obs.me)); moves.append(root); scores.append(player.value_fn(after, obs.me))
            if after.state.current_color() == obs.me and root.action_type != ActionType.END_TURN:
                followups = [a for a in after.playable_actions if a.action_type in POINT_TYPES]
                for followup in followups[:3]:
                    if time.process_time() >= deadline:
                        break
                    later = after.copy()
                    later.execute(followup)
                    if player._vp(later) >= world.vps_to_win:
                        return playable_actions.index(root)
                    leaves.append(encode(later, obs.me)); moves.append(root); scores.append(player.value_fn(later, obs.me))
        if not leaves:
            return playable_actions.index(baseline)
        # Bound the critic's influence around the incumbent ranking during bootstrapping.
        import numpy as np
        scores = np.asarray(scores)
        maximum = scores.max()
        eligible = scores >= maximum - 2e7
        preferences = np.clip((scores - maximum) / 2e7, -10, 0) + self.blend * self.critic.predict(leaves)
        preferences[~eligible] = -np.inf
        return playable_actions.index(moves[int(preferences.argmax())])
