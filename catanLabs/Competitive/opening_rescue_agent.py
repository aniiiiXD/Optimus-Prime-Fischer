"""Research intervention: switch frozen Transformer and fair-value phases.

This isolates the effect of selected decision phases on a trained policy's games.
The fair-value component has its own hidden-state invariance checks; this
hybrid is an experiment, not a promoted visible-only submission agent.
"""

from __future__ import annotations

import json
from pathlib import Path

from catanLabs.Competitive.fair_value_agent import FairValueAgent
from catanLabs.Transformer.agent import TransformerAgent
from catanLabs.Transformer.legacy_v2_agent import LegacyV2TransformerAgent
from catanLabs.Transformer.visible_observation import snapshot


OPENING_PHASES = {"BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD"}
KNOWN_PHASES = OPENING_PHASES | {"PLAY_TURN", "DISCARD", "MOVE_ROBBER",
                                 "DECIDE_TRADE", "DECIDE_ACCEPTEES"}


class OpeningRescueAgent:
    def __init__(self, color, config_path: Path):
        config_path = Path(config_path).resolve(strict=True)
        config = json.loads(config_path.read_text())
        model_path = (config_path.parent / config["transformer_checkpoint"]).resolve(strict=True)
        fair_path = (config_path.parent / config["fair_config"]).resolve(strict=True)
        schema = config.get("transformer_schema", "v3")
        model_agent = {"v2": LegacyV2TransformerAgent, "v3": TransformerAgent}.get(schema)
        if model_agent is None:
            raise ValueError(f"Unknown Transformer schema: {schema}")
        self.transformer = model_agent(color, model_path)
        self.fair = FairValueAgent(color, json.loads(fair_path.read_text()))
        if "model_phases" in config and "fair_phases" in config:
            raise ValueError("Specify either model_phases or fair_phases")
        self.model_phases = set(config["model_phases"]) if "model_phases" in config else None
        self.fair_phases = set(config.get("fair_phases", OPENING_PHASES))
        selected = self.model_phases if self.model_phases is not None else self.fair_phases
        if not selected or not selected <= KNOWN_PHASES:
            raise ValueError(f"Invalid intervention phases: {selected}")

    def decide(self, game, playable_actions):
        legal = tuple(playable_actions)
        if not legal:
            raise ValueError("No legal Catan action")
        phase = game.state.current_prompt.name
        fair_controls = (phase not in self.model_phases if self.model_phases is not None
                         else phase in self.fair_phases)
        if fair_controls:
            choice = self.fair.decide(game, legal)
            if choice not in legal:
                raise ValueError("Fair opening component returned an illegal action")
            return choice
        index = self.transformer.decide_visible(snapshot(game, legal))
        if type(index) is not int or not 0 <= index < len(legal):
            raise ValueError("Transformer returned an illegal action index")
        return legal[index]


def create_agent(color, checkpoint, seed):
    del seed
    if checkpoint is None:
        raise ValueError("Opening rescue requires a JSON checkpoint config")
    return OpeningRescueAgent(color, Path(checkpoint))
