"""Research-only phase gate between the frozen guided search and FairValue."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from catanLabs.Competitive.fair_value_agent import FairValueAgent
from catanLabs.Competitive.guided_search_agent import GuidedSearchAgent


OPENING_PHASES = frozenset({"BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD"})
MODES = frozenset({"opening", "turn", "all"})


class PhaseGuidedPolicy:
    def __init__(self, guided, fair, mode: str):
        if mode not in MODES:
            raise ValueError(f"Invalid guided phase mode: {mode}")
        self.guided = guided
        self.fair = fair
        self.mode = mode
        self.routing_counts = Counter()

    def decide(self, game, playable_actions):
        legal = tuple(playable_actions)
        if not legal:
            raise ValueError("No legal Catan action")
        phase = game.state.current_prompt.name
        if len(legal) == 1:
            self.routing_counts[(phase, "forced")] += 1
            return legal[0]
        use_guided = (self.mode == "all"
                      or self.mode == "opening" and phase in OPENING_PHASES
                      or self.mode == "turn" and phase == "PLAY_TURN")
        name, policy = ("guided", self.guided) if use_guided else ("fair", self.fair)
        choice = policy.decide(game, legal)
        if choice not in legal:
            raise ValueError(f"{name} component returned an illegal action")
        self.routing_counts[(phase, name)] += 1
        return choice


class PhaseGuidedAgent(PhaseGuidedPolicy):
    def __init__(self, color, config_path: Path):
        config_path = Path(config_path).resolve(strict=True)
        config = json.loads(config_path.read_text())
        if set(config) != {"mode", "fair_config", "guided_config"}:
            raise ValueError("Phase guided config needs mode, fair_config and guided_config")
        fair_path = (config_path.parent / config["fair_config"]).resolve(strict=True)
        guided_path = (config_path.parent / config["guided_config"]).resolve(strict=True)
        fair = FairValueAgent(color, json.loads(fair_path.read_text()))
        guided = GuidedSearchAgent(color, json.loads(guided_path.read_text()))
        super().__init__(guided, fair, config["mode"])


def create_agent(color, checkpoint, seed):
    del seed
    if checkpoint is None:
        raise ValueError("Phase guided agent needs a JSON config")
    return PhaseGuidedAgent(color, Path(checkpoint))
