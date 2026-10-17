"""Catanatron value policy with parameters supplied by our CPU sweep."""

from __future__ import annotations

import json
from pathlib import Path

from catanatron.players.value import DEFAULT_WEIGHTS, ValueFunctionPlayer


class TunedValueAgent:
    def __init__(self, color, weights):
        self.policy = ValueFunctionPlayer(
            color, ValueFunctionPlayer.Params(value_fn="contender", weights=weights)
        )

    def decide(self, game, playable_actions):
        return self.policy.decide(game, playable_actions)


def create_agent(color, checkpoint, seed):
    del seed
    overrides = json.loads(Path(checkpoint).read_text()) if checkpoint is not None else {}
    return TunedValueAgent(color, {**DEFAULT_WEIGHTS, **overrides})
