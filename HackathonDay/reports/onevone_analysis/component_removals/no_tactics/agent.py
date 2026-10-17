"""Official observation/index adapter for the copied OneVOne baseline.

Policy weights and search limits are unchanged. The sampled world contains
only information permitted by the official observation contract.
"""

from __future__ import annotations

import random

from onevone import create_agent


class Agent:
    def __init__(self):
        self.rng = random.Random(0)
        self.policy = None

    def decide(self, obs, playable_actions) -> int:
        if self.policy is None or self.policy.color != obs.me:
            self.policy = create_agent(obs.me)
        world = obs.sample_world(self.rng)
        chosen = self.policy.decide(world, playable_actions)
        # Fail visibly if the adapter returns an action outside the supplied
        # list; the official harness records such faults in its results.
        return playable_actions.index(chosen)
