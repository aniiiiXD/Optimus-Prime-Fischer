"""Resource-specific production priorities for the existing OneVOne scorer."""

import json
import math

from agent import Agent as BaselineAgent
from catanLabs.OneVOne.agent import DEFAULT_CONFIG, OneVOneAgent
from catanatron.models.enums import RESOURCES


def normalize_priorities(values):
    if len(values) != 5 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Need five finite positive priorities: wood, brick, sheep, wheat, ore")
    mean = sum(values) / 5
    return [v / mean for v in values]


class PriorityPlayer(OneVOneAgent):
    def __init__(self, color, config, priorities):
        super().__init__(color, config)
        self.priorities = dict(zip(RESOURCES, priorities))
        self.base_value_fn = self.value_fn
        if any(v != 1.0 for v in priorities):
            self.value_fn = self.priority_value

    def priority_value(self, game, color):
        production = self.robber._production(game, color)
        correction = sum((self.priorities[r] - 1) * production[r] for r in RESOURCES)
        return self.base_value_fn(game, color) + self.weights["production"] * correction


class Agent(BaselineAgent):
    def __init__(self, priorities=(1, 1, 1, 1, 1)):
        super().__init__()
        self.priorities = normalize_priorities(priorities)
        self.config = json.loads(DEFAULT_CONFIG.read_text())

    def decide(self, obs, playable_actions):
        if obs.me not in self.players:
            self.players[obs.me] = PriorityPlayer(obs.me, self.config, self.priorities)
        return super().decide(obs, playable_actions)
