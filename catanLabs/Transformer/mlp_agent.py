"""Catanatron benchmark adapter for the v2 action-conditioned MLP."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from catanLabs.Transformer.mlp import CHECKPOINT_SCHEMA, CatanActionMLP
from catanLabs.Transformer.model import DATASET_SCHEMA, FEATURE_SCHEMA, encode_visible_observation
from catanLabs.Transformer.visible_observation import VisibleAgentAdapter, VisibleObservation


def _valid_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


class MLPAgent:
    def __init__(self, color, checkpoint_path):
        self.color = color
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (checkpoint.get("schema") != CHECKPOINT_SCHEMA
                or checkpoint.get("feature_schema") != FEATURE_SCHEMA
                or checkpoint.get("dataset_schema") != DATASET_SCHEMA):
            raise ValueError("Incompatible Catan MLP checkpoint or v2 feature schema")
        if not _valid_sha256(checkpoint.get("dataset_sha256")):
            raise ValueError("Catan MLP checkpoint is missing its dataset SHA-256")
        if checkpoint.get("split_file") is not None and not _valid_sha256(checkpoint.get("split_file_sha256")):
            raise ValueError("Catan MLP checkpoint is missing its split-file SHA-256")
        summary_path = checkpoint_path.with_name("training.json")
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if (summary.get("dataset_sha256") != checkpoint["dataset_sha256"]
                    or summary.get("split_file_sha256") != checkpoint.get("split_file_sha256")
                    or summary.get("feature_schema") != FEATURE_SCHEMA):
                raise ValueError("Catan MLP checkpoint hashes disagree with training.json")
        self.model = CatanActionMLP(**checkpoint["model_config"]).cpu()
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def score_visible(self, observation: VisibleObservation) -> list[float]:
        if not observation.legal_actions:
            raise ValueError("No legal actions in visible observation")
        arrays = encode_visible_observation(observation)
        with torch.inference_mode():
            scores = self.model(
                *(torch.from_numpy(array).unsqueeze(0) for array in arrays),
                torch.ones((1, len(observation.legal_actions)), dtype=torch.bool),
            )
        return scores.squeeze(0).tolist()

    def decide_visible(self, observation: VisibleObservation) -> int:
        if len(observation.legal_actions) == 1:
            return 0
        scores = self.score_visible(observation)
        return max(range(len(scores)), key=scores.__getitem__)


def create_agent(color, checkpoint, seed):
    del seed
    if checkpoint is None:
        raise ValueError("The trained MLP requires --checkpoint")
    return VisibleAgentAdapter(MLPAgent(color, checkpoint))
