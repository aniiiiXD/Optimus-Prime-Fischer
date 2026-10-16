"""Outcome critic and conservative OneVOne heuristic/value hybrid."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn

from catanLabs.OneVOne.agent import OneVOneAgent
from catanLabs.OneVOne.policy_features import ACTION_TYPES
from catanLabs.FromScratch.StrategyOne.train_legal_ranker import encode_state, stable_edge_index
from catanLabs.FromScratch.StrategyOne.generate_dataset import observation


HISTORY_LEN = 16
EVENT_DIM = 32
RERANK_TYPES = {"BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "MARITIME_TRADE"}
RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


def event_vector(event: dict[str, Any]) -> np.ndarray:
    out = np.zeros(EVENT_DIM, np.float32)
    kind = event.get("type")
    if kind in ACTION_TYPES:
        out[ACTION_TYPES.index(kind)] = 1
    out[18] = float(event.get("actor") == "self")
    out[19] = float(event.get("actor") == "opponent")
    value = event.get("value")

    def add_resources(item: Any) -> None:
        if isinstance(item, str) and item in RESOURCES:
            out[20 + RESOURCES.index(item)] += 0.25
        elif isinstance(item, (list, tuple)):
            for part in item:
                add_resources(part)

    add_resources(value)
    if isinstance(value, int):
        out[25] = value / 54.0
    elif isinstance(value, (list, tuple)):
        for i, part in enumerate(value[:3]):
            if isinstance(part, int):
                out[25 + i] = part / 54.0
    result = event.get("result")
    if isinstance(result, (list, tuple)) and len(result) == 2:
        if all(isinstance(x, int) for x in result):
            out[28] = sum(result) / 12.0
    out[29] = float(kind == "ROLL")
    out[30] = float(kind in RERANK_TYPES)
    out[31] = 1.0
    return out


def encode_history(obs: dict[str, Any]) -> np.ndarray:
    result = np.zeros((HISTORY_LEN, EVENT_DIM), np.float32)
    events = obs.get("public_events", [])[-HISTORY_LEN:]
    if events:
        result[-len(events):] = np.stack([event_vector(event) for event in events])
    return result


def encode_observation(obs: dict[str, Any], edge_index=None) -> tuple[np.ndarray, np.ndarray]:
    if edge_index is None:
        edge_index = stable_edge_index(obs)
    return np.asarray(encode_state(obs, edge_index), np.float32), encode_history(obs)


class OutcomeTransformer(nn.Module):
    def __init__(self, state_dim: int, width: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.width = width
        self.state = nn.Sequential(nn.Linear(state_dim, width), nn.LayerNorm(width), nn.GELU())
        self.event = nn.Linear(EVENT_DIM, width)
        self.position = nn.Parameter(torch.zeros(HISTORY_LEN, width))
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, 0.0,
                                           batch_first=True, norm_first=True)
        self.memory = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.value = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(),
                                   nn.Linear(width, 1))

    def forward(self, states: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        context = self.memory(self.event(events) + self.position[None]).mean(1)
        return self.value(torch.cat((self.state(states), context), dim=-1)).squeeze(-1)


def load_model(path: str | Path) -> OutcomeTransformer:
    data = torch.load(path, map_location="cpu", weights_only=True)
    model = OutcomeTransformer(data["state_dim"], data["width"])
    model.load_state_dict(data["model"])
    return model.cpu().eval()


class OneVOneOutcomeAgent(OneVOneAgent):
    """Keep OneVOne's action family and tactical choices; rerank like actions."""

    def __init__(self, color: Any, model: OutcomeTransformer, beta: float, config: dict):
        super().__init__(color, config)
        self.model = model.cpu().eval()
        self.beta = float(beta)
        self.changed = 0
        self.considered = 0
        self._edge_index = None

    @torch.inference_mode()
    def decide(self, game: Any, playable_actions: list[Any]) -> Any:
        baseline = super().decide(game, playable_actions)
        if self.beta == 0 or len(playable_actions) < 2:
            return baseline
        if getattr(self, "last_scan", {}).get("override", False):
            return baseline
        kind = baseline.action_type.name
        if kind not in RERANK_TYPES:
            return baseline
        choices = [action for action in playable_actions if action.action_type.name == kind]
        if len(choices) < 2:
            return baseline
        self.considered += 1
        states, histories, heuristic = [], [], []
        valid = []
        for action in choices:
            try:
                after = game.copy()
                after.execute(action, validate_action=True)
                if after.winning_color() == self.color:
                    return action
                obs = observation(after, self.color)
                if self._edge_index is None:
                    self._edge_index = stable_edge_index(obs)
                state, history = encode_observation(obs, self._edge_index)
                states.append(state)
                histories.append(history)
                heuristic.append(float(self.value_fn(after, self.color)))
                valid.append(action)
            except Exception:
                # Keep OneVOne's known-legal recommendation if a hypothetical
                # engine copy cannot resolve a candidate safely.
                return baseline
        if baseline not in valid:
            return baseline
        values = torch.sigmoid(self.model(torch.from_numpy(np.stack(states)),
                                          torch.from_numpy(np.stack(histories)))).numpy()
        raw = np.asarray(heuristic, np.float64)
        spread = max(1e-9, float(raw.max() - raw.min()))
        normalized = (raw - raw.max()) / spread
        score = normalized + self.beta * (values - values[valid.index(baseline)])
        selected = valid[int(score.argmax())]
        self.changed += selected != baseline
        return selected
