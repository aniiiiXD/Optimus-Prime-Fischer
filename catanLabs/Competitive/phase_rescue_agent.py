"""Frozen Transformer with FairValue choices in one specified turn phase."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from catanLabs.Competitive.fair_value_agent import FairValueAgent
from catanLabs.Transformer.agent import TransformerAgent
from catanLabs.Transformer.visible_observation import snapshot


INTERVENTIONS = ("play_turn", "discard")
PHASES = {"play_turn": "PLAY_TURN", "discard": "DISCARD"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(config_path: Path) -> dict:
    config_path = Path(config_path).resolve(strict=True)
    config = json.loads(config_path.read_text())
    if config.get("intervention") not in INTERVENTIONS:
        raise ValueError(f"Unknown phase rescue intervention: {config.get('intervention')!r}")
    for stem in ("model_checkpoint", "fair_config"):
        path = (config_path.parent / config[stem]).resolve(strict=True)
        expected = config[stem + "_sha256"]
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen {stem} hash changed: {actual} != {expected}")
        config[stem + "_path"] = path
    return config


class PhaseFairRescuePolicy:
    """Evaluate the frozen model everywhere; substitute FairValue in one phase."""

    def __init__(self, model, fair, intervention: str, snapshot_fn=snapshot):
        if intervention not in INTERVENTIONS:
            raise ValueError(f"Unknown phase intervention: {intervention}")
        self.model = model
        self.fair = fair
        self.intervention = intervention
        self.snapshot_fn = snapshot_fn
        self.counters = Counter()
        self.phase_counters: dict[str, Counter] = {}

    def decide(self, game, playable_actions):
        legal = tuple(playable_actions)
        if not legal:
            raise ValueError("No legal Catan action")
        phase = game.state.current_prompt.name
        self.counters["decisions"] += 1
        phase_counts = self.phase_counters.setdefault(phase, Counter())
        phase_counts["decisions"] += 1
        if len(legal) == 1:
            self.counters["forced"] += 1
            phase_counts["forced"] += 1
            return legal[0]

        index = self.model.decide_visible(self.snapshot_fn(game, legal))
        if type(index) is not int or not 0 <= index < len(legal):
            raise ValueError("Transformer returned an illegal action index")
        model_action = legal[index]
        if phase != PHASES[self.intervention]:
            return model_action

        self.counters["consulted"] += 1
        phase_counts["consulted"] += 1
        fair_action = self.fair.decide(game, legal)
        if fair_action not in legal:
            raise ValueError("FairValue returned an illegal action")
        if fair_action == model_action:
            return model_action
        self.counters["disagreed"] += 1
        self.counters["overrides"] += 1
        phase_counts["disagreed"] += 1
        phase_counts["overrides"] += 1
        return fair_action

    def summary(self) -> dict:
        return {
            "intervention": self.intervention,
            **{key: int(self.counters[key]) for key in
               ("decisions", "forced", "consulted", "disagreed", "overrides")},
            "by_phase": {phase: dict(sorted(counts.items()))
                         for phase, counts in sorted(self.phase_counters.items())},
        }


class PhaseFairRescueAgent(PhaseFairRescuePolicy):
    def __init__(self, color, config_path: Path):
        config = load_config(config_path)
        model = TransformerAgent(color, config["model_checkpoint_path"])
        fair = FairValueAgent(color, json.loads(config["fair_config_path"].read_text()))
        super().__init__(model, fair, config["intervention"])


def create_agent(color, checkpoint, seed):
    del seed
    if checkpoint is None:
        raise ValueError("Phase rescue requires a JSON intervention config")
    return PhaseFairRescueAgent(color, Path(checkpoint))
