"""Minimal custom-agent example; replace its decision with our trained model."""


class ExampleAgent:
    def __init__(self, color, seed):
        self.color = color
        self.seed = seed

    def decide(self, game, playable_actions):
        # The future model should encode only player-visible information and
        # score these legal actions. This placeholder simply picks the first.
        return playable_actions[0]


def create_agent(color, checkpoint, seed):
    return ExampleAgent(color, seed)
