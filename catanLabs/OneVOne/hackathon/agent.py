"""Local OneVOne adapter for the official observation/index interface."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from catanLabs.OneVOne.agent import create_agent


class Agent:
    def __init__(self):
        self.rng = random.Random(0)
        self.players = {}

    def decide(self, obs, playable_actions):
        if obs.me not in self.players:
            self.players[obs.me] = create_agent(obs.me)
        world = obs.sample_world(self.rng)
        action = self.players[obs.me].decide(world, playable_actions)
        return playable_actions.index(action)
