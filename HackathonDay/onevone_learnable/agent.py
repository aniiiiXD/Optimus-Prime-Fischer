"""Official Agent.decide(obs, legal_actions) -> legal action index entry point."""
from __future__ import annotations

import random
from policy import LearnableOneVOne, load_config


class Agent:
    def __init__(self, config_path=None, scorer=None):
        self.config = load_config(config_path)
        self.scorer = scorer
        self.rng = random.Random(0)
        self.policy = None

    def decide(self, obs, playable_actions):
        if len(playable_actions) == 1:
            return 0
        if self.policy is None or self.policy.color != obs.me:
            self.policy = LearnableOneVOne(obs.me, self.config, self.scorer)
        world = obs.sample_world(self.rng)
        return self.policy.decide(world, playable_actions, obs.cpu_left)
