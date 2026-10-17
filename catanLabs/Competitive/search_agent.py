"""Search baseline adapters for controlled CPU comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import DEFAULT_WEIGHTS


def create_agent(color, checkpoint, seed):
    del seed
    config = json.loads(Path(checkpoint).read_text()) if checkpoint is not None else {}
    params = AlphaBetaPlayer.Params(
        depth=int(config.get("depth", 2)),
        prunning=bool(config.get("prunning", True)),
        value_fn="contender",
        weights={**DEFAULT_WEIGHTS, **config.get("weights", {})},
    )
    return AlphaBetaPlayer(color, params)
