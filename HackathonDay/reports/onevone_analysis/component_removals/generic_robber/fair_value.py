"""Value search over public outcomes without sampling hidden dice or cards."""

from __future__ import annotations

import json
from pathlib import Path

from catanatron.models.enums import Action, ActionRecord, ActionType
from catanatron.models.map import number_probability
from catanatron.players.value import DEFAULT_WEIGHTS, base_fn
from competitive import CompetitiveAgent


class FairValueAgent:
    def __init__(self, color, config):
        self.color = color
        self.config = config
        self.weights = {**DEFAULT_WEIGHTS, **config.get("value_weights", {})}
        self.value_fn = base_fn(self.weights)
        self.robber = CompetitiveAgent(color, config.get("robber_weights", {}))

    def _roll_value(self, game):
        total = 0.0
        for number in range(2, 13):
            # A forced sum avoids reading the simulator's next hidden roll.
            dice = (number // 2, number - number // 2)
            forced = Action(self.color, ActionType.ROLL, None)
            after = game.copy()
            after.execute(forced, validate_action=False,
                          action_record=ActionRecord(forced, dice))
            total += number_probability(number) * self.value_fn(after, self.color)
        return total

    def _buy_value(self, game):
        after = game.copy()
        state = after.state
        prefix = f"P{state.color_to_index[self.color]}_"
        for resource in ("SHEEP", "WHEAT", "ORE"):
            state.player_state[prefix + resource + "_IN_HAND"] -= 1
        state.player_state[prefix + "KNIGHT_IN_HAND"] += 1
        # base_fn values all unplayed dev cards through their count, so
        # incrementing a Knight only represents an unknown card's count.
        score = self.value_fn(after, self.color)
        own_vp = game.state.player_state[prefix + "ACTUAL_VICTORY_POINTS"]
        if own_vp >= self.config.get("dev_buy_vp_threshold", 7):
            score += self.config.get("dev_buy_late_bonus", 0.0)
        return score

    def _monopoly_value(self, game, resource):
        after = game.copy()
        state = after.state
        prefix = f"P{state.color_to_index[self.color]}_"
        estimate = 0.0
        for enemy in state.colors:
            if enemy == self.color:
                continue
            production = self.robber._production(game, enemy)
            estimate += min(2.0, 0.25 + 6.0 * production[resource])
        state.player_state[prefix + "MONOPOLY_IN_HAND"] -= 1
        state.player_state[prefix + resource + "_IN_HAND"] += estimate
        return self.value_fn(after, self.color)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        if playable_actions[0].action_type == ActionType.MOVE_ROBBER:
            return max(playable_actions, key=lambda a: self.robber._robber_score(game, a))
        best, best_score = None, float("-inf")
        for action in playable_actions:
            kind = action.action_type
            if kind == ActionType.ROLL:
                score = self._roll_value(game)
            elif kind == ActionType.BUY_DEVELOPMENT_CARD:
                score = self._buy_value(game)
            elif kind == ActionType.PLAY_MONOPOLY:
                score = self._monopoly_value(game, action.value)
            else:
                after = game.copy()
                after.execute(action, validate_action=False)
                score = self.value_fn(after, self.color)
            if score > best_score:
                best, best_score = action, score
        return best


def create_agent(color, checkpoint, seed):
    del seed
    config = json.loads(Path(checkpoint).read_text()) if checkpoint is not None else {}
    return FairValueAgent(color, config)
