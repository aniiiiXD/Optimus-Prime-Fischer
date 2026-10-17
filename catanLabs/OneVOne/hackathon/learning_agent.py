"""Official adapter with one shared planner and optional learned action residuals."""

import json
import os
from pathlib import Path

from agent import Agent as BaseAgent
from catanLabs.OneVOne.agent import DEFAULT_CONFIG, TRAINED_CONFIG
from catanLabs.OneVOne.learning import LearningPlayer


class Agent(BaseAgent):
    def __init__(self, model=None, trace_path=None, config_path=None):
        super().__init__()
        self.model = model
        self.requests = 0
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path is not None:
            self.trace_path.mkdir(parents=True, exist_ok=True)
        config = Path(config_path) if config_path is not None else (
            TRAINED_CONFIG if TRAINED_CONFIG.is_file() else DEFAULT_CONFIG
        )
        self.config = json.loads(config.read_text())

    def decide(self, obs, playable_actions):
        self.requests += 1
        if obs.me not in self.players:
            self.players[obs.me] = LearningPlayer(obs.me, self.config, self.model)
        if len(playable_actions) == 1:
            return 0
        world = obs.sample_world(self.rng)
        player = self.players[obs.me]
        selected = player.decide(world, playable_actions)
        if self.trace_path is not None:
            row = dict(player.last_decision, turn=obs.turn, seat=obs.me.value,
                       pid=os.getpid(), request=self.requests)
            with (self.trace_path / f"{os.getpid()}.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        return playable_actions.index(selected)
