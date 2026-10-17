"""Four-player robber selection added to the established Value policy."""

from __future__ import annotations

import json
from pathlib import Path

from catanatron.players.value import DEFAULT_WEIGHTS, ValueFunctionPlayer
from catanLabs.Competitive.agent import CompetitiveAgent


class TacticalValueAgent:
    def __init__(self, color, config):
        weights = {**DEFAULT_WEIGHTS, **config.get("value_weights", {})}
        self.value = ValueFunctionPlayer(
            color, ValueFunctionPlayer.Params(value_fn="contender", weights=weights)
        )
        self.robber = CompetitiveAgent(color, config.get("robber_weights", {}))
        self.own_color = color

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        if playable_actions[0].action_type.name == "MOVE_ROBBER":
            return max(playable_actions, key=lambda action: self.robber._robber_score(game, action))
        return self.value.decide(game, playable_actions)


def create_agent(color, checkpoint, seed):
    del seed
    config = json.loads(Path(checkpoint).read_text()) if checkpoint is not None else {}
    return TacticalValueAgent(color, config)
