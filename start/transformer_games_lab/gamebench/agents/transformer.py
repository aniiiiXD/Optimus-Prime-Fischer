"""Legal-action wrapper around a TinyGameTransformer checkpoint."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import torch

from gamebench.agents.base import Agent
from gamebench.games.base import GameState
from gamebench.models.tiny_transformer import TinyGameTransformer, TinyTransformerConfig


class TransformerAgent(Agent):
    def __init__(
        self,
        model: TinyGameTransformer,
        name: str = "transformer",
        checkpoint: str | None = None,
    ) -> None:
        self.model = model.eval().cpu()
        self.name = name
        self.checkpoint = checkpoint

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        expected_game: str | None = None,
    ) -> "TransformerAgent":
        checkpoint_path = Path(path).expanduser().resolve()
        payload: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        game = payload.get("game")
        if expected_game is not None and game != expected_game:
            raise ValueError(
                f"checkpoint was trained for {game!r}, not requested game {expected_game!r}"
            )
        try:
            config = TinyTransformerConfig(**payload["model_config"])
            model = TinyGameTransformer(config)
            model.load_state_dict(payload["state_dict"])
        except KeyError as error:
            raise ValueError(f"invalid GameBench checkpoint: missing {error.args[0]!r}") from error
        return cls(model, name=f"transformer:{checkpoint_path.name}", checkpoint=str(checkpoint_path))

    def select_action(self, state: GameState, rng: random.Random) -> int:
        del rng
        tokens = torch.tensor([state.history_tokens()], dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(tokens)[0]
        legal = state.legal_actions()
        masked = torch.full_like(logits, -torch.inf)
        masked[list(legal)] = logits[list(legal)]
        return int(masked.argmax())

    def metadata(self) -> dict[str, int | str | None]:
        return {
            **super().metadata(),
            "parameters": self.model.parameter_count,
            "checkpoint": self.checkpoint,
        }
