"""Research candidate: choose the stronger public opponent when a robber tile ties."""

from __future__ import annotations

import json
from pathlib import Path

from catanatron.models.enums import ActionType
from catanLabs.Competitive.fair_value_agent import FairValueAgent


class RobberVictimAgent(FairValueAgent):
    """Preserve FairValue's tile choice and change only a tied victim choice."""

    def find_override(self, game, playable_actions, baseline):
        if (baseline.action_type != ActionType.MOVE_ROBBER
                or baseline.value[1] is None):
            return None
        coordinate, baseline_victim = baseline.value
        same_tile = [action for action in playable_actions
                     if action.action_type == ActionType.MOVE_ROBBER
                     and action.value[0] == coordinate
                     and action.value[1] is not None]
        if len(same_tile) < 2:
            return None
        state = game.state

        def public_vp(action):
            victim = action.value[1]
            return state.player_state[f"P{state.color_to_index[victim]}_VICTORY_POINTS"]

        baseline_vp = state.player_state[
            f"P{state.color_to_index[baseline_victim]}_VICTORY_POINTS"]
        leader_vp = max(map(public_vp, same_tile))
        if leader_vp <= baseline_vp:
            return None
        return next(action for action in same_tile if public_vp(action) == leader_vp)

    def decide(self, game, playable_actions):
        baseline = super().decide(game, playable_actions)
        return self.find_override(game, playable_actions, baseline) or baseline


def create_agent(color, checkpoint, seed):
    del seed
    config = json.loads(Path(checkpoint).read_text()) if checkpoint is not None else {}
    return RobberVictimAgent(color, config)
