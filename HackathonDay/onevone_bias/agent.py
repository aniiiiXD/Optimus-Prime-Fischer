"""Official observation/index adapter for the copied OneVOne baseline.

Policy weights and search limits are unchanged. The sampled world contains
only information permitted by the official observation contract.
"""

from __future__ import annotations

import random

from catanatron.models.enums import ActionType
from bias_exploit import guaranteed_win
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
        # F's one-step score misses deterministic turn chains. Our copied
        # baseline can also choose an uncertain dev draw at 9 VP instead of a
        # certain trade/build win. Check that narrow case without looking at
        # a hidden card or exceeding the official CPU clock.
        if (obs.my_actual_vps >= 8
                and chosen.action_type in (ActionType.END_TURN,
                                           ActionType.BUY_DEVELOPMENT_CARD)):
            winner = guaranteed_win(world, obs.me, obs.cpu_left)
            if winner is not None:
                chosen = winner
        # Fail visibly if the adapter returns an action outside the supplied
        # list; the official harness records such faults in its results.
        return playable_actions.index(chosen)
