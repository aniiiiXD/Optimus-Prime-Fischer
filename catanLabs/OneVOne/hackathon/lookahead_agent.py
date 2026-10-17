"""Official-interface adapter for the optional Nano search-leaf value model."""

import json
import os
from pathlib import Path

from agent import Agent as BaseAgent
from catanLabs.OneVOne.lookahead_value import LookaheadPlayer


class Agent(BaseAgent):
    def __init__(self, config_path, model=None, trace_path=None):
        super().__init__()
        self.config = json.loads(Path(config_path).read_text())
        self.model, self.requests = model, 0
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path:
            self.trace_path.mkdir(parents=True, exist_ok=True)

    def decide(self, obs, playable_actions):
        self.requests += 1
        if len(playable_actions) == 1:
            return 0
        if obs.me not in self.players:
            self.players[obs.me] = LookaheadPlayer(obs.me, self.config, self.model)
        player = self.players[obs.me]
        if obs.cpu_left is not None and obs.cpu_left < .100:
            return player.planner.nano.decide(obs, playable_actions)
        selected = player.decide(obs.sample_world(self.rng), playable_actions, cpu_left=obs.cpu_left, capture=False)
        if self.trace_path:
            row = dict(player.last_decision, pid=os.getpid(), request=self.requests,
                       turn=obs.turn, seat=obs.me.value)
            with (self.trace_path / f"{os.getpid()}.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        return playable_actions.index(selected)
