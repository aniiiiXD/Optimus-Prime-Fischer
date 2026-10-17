"""Official observation/index adapter for the copied OneVOne baseline.

Policy weights and search limits are unchanged. The sampled world contains
only information permitted by the official observation contract.
"""

from __future__ import annotations

import random

from depth_three import depth_three_override
from onevone import create_agent


class Agent:
    def __init__(self):
        self.rng = random.Random(0)
        self.policy = None
        self.last_depth_turn = None

    def decide(self, obs, playable_actions) -> int:
        if self.policy is None or self.policy.color != obs.me:
            self.policy = create_agent(obs.me)
        world = obs.sample_world(self.rng)
        chosen = self.policy.decide(world, playable_actions)
        # The initial post-roll choice is the useful place to plan a sequence.
        # Subsequent choices in this turn are handled by the existing two-step
        # policy; rerunning depth three after every trade wastes the game clock.
        if (obs.turn != self.last_depth_turn
                and world.state.current_prompt.name == "PLAY_TURN"
                and world.state.player_state[
                    f"P{world.state.color_to_index[obs.me]}_HAS_ROLLED"
                ]):
            self.last_depth_turn = obs.turn
            deeper = depth_three_override(
                world, obs.me, playable_actions, chosen, obs.cpu_left
            )
            if deeper is not None:
                chosen = deeper
        # Fail visibly if the adapter returns an action outside the supplied
        # list; the official harness records such faults in its results.
        return playable_actions.index(chosen)
