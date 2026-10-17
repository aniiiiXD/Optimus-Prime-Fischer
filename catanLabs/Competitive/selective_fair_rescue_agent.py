"""Frozen-model diagnostic with one narrowly selected FairValue intervention."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from catanLabs.Competitive.fair_value_agent import FairValueAgent
from catanLabs.Transformer.agent import TransformerAgent
from catanLabs.Transformer.visible_observation import snapshot


INTERVENTIONS = ("opening", "robber", "trade_end")
OPENING_PHASES = {"BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD"}
TRADE_END_TYPES = {"MARITIME_TRADE", "END_TURN"}


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
        raise ValueError(f"Unknown selective rescue intervention: {config.get('intervention')!r}")
    for stem in ("model_checkpoint", "fair_config"):
        path = (config_path.parent / config[stem]).resolve(strict=True)
        expected = config[stem + "_sha256"]
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen {stem} hash changed: {actual} != {expected}")
        config[stem + "_path"] = path
    return config


class SelectiveFairRescuePolicy:
    """Uses a visible-only model; consults FairValue only for selected choices."""

    def __init__(self, model, fair, intervention: str, snapshot_fn=snapshot):
        if intervention not in INTERVENTIONS:
            raise ValueError(f"Unknown intervention: {intervention}")
        self.model = model
        self.fair = fair
        self.intervention = intervention
        self.snapshot_fn = snapshot_fn
        self.counters = Counter()
        self.phase_counters: dict[str, Counter] = {}
        self.trade_directions = Counter()

    def _eligible(self, phase: str) -> bool:
        if self.intervention == "opening":
            return phase in OPENING_PHASES
        if self.intervention == "robber":
            return phase == "MOVE_ROBBER"
        return phase == "PLAY_TURN"

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

        visible = self.snapshot_fn(game, legal)
        index = self.model.decide_visible(visible)
        if type(index) is not int or not 0 <= index < len(legal):
            raise ValueError("Transformer returned an illegal action index")
        model_action = legal[index]
        if not self._eligible(phase):
            return model_action

        self.counters["consulted"] += 1
        phase_counts["consulted"] += 1
        fair_action = self.fair.decide(game, legal)
        if fair_action not in legal:
            raise ValueError("FairValue returned an illegal action")
        if fair_action == model_action:
            return model_action
        self.counters["disagreed"] += 1
        phase_counts["disagreed"] += 1

        if self.intervention == "trade_end":
            model_type = model_action.action_type.name
            fair_type = fair_action.action_type.name
            if {model_type, fair_type} != TRADE_END_TYPES:
                return model_action
            self.trade_directions[f"model_{model_type}__fair_{fair_type}"] += 1

        self.counters["overrides"] += 1
        phase_counts["overrides"] += 1
        return fair_action

    def summary(self) -> dict:
        return {
            "intervention": self.intervention,
            **{key: int(self.counters[key]) for key in
               ("decisions", "forced", "consulted", "disagreed", "overrides")},
            "by_phase": {phase: dict(sorted(counts.items()))
                         for phase, counts in sorted(self.phase_counters.items())},
            "trade_directions": dict(sorted(self.trade_directions.items())),
        }


class SelectiveFairRescueAgent(SelectiveFairRescuePolicy):
    def __init__(self, color, config_path: Path):
        config = load_config(config_path)
        model = TransformerAgent(color, config["model_checkpoint_path"])
        fair = FairValueAgent(color, json.loads(config["fair_config_path"].read_text()))
        super().__init__(model, fair, config["intervention"])


def create_agent(color, checkpoint, seed):
    del seed
    if checkpoint is None:
        raise ValueError("Selective rescue requires a JSON intervention config")
    return SelectiveFairRescueAgent(color, Path(checkpoint))
