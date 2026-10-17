"""Ablation: Value-based robber placement using only public board outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from catanatron.models.enums import Action, ActionType
from catanLabs.Competitive.fair_value_agent import FairValueAgent


class ValueRobberAgent(FairValueAgent):
    def decide(self, game, playable_actions):
        if len(playable_actions) > 1 and playable_actions[0].action_type == ActionType.MOVE_ROBBER:
            scores = []
            for action in playable_actions:
                after = game.copy()
                # Move to the chosen public tile. Theft is unknown, so omit it
                # from the hypothetical outcome and add a fixed victim bonus.
                shadow = Action(self.color, ActionType.MOVE_ROBBER, (action.value[0], None))
                after.execute(shadow, validate_action=False)
                score = self.value_fn(after, self.color)
                if action.value[1] is not None:
                    score += 100000.0
                scores.append(score)
            return playable_actions[max(range(len(scores)), key=scores.__getitem__)]
        return super().decide(game, playable_actions)


def create_agent(color, checkpoint, seed):
    del seed
    config = json.loads(Path(checkpoint).read_text()) if checkpoint is not None else {}
    return ValueRobberAgent(color, config)
