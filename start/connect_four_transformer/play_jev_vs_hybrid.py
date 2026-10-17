"""Play one complete Connect Four game: official Jev vs our hybrid agent.

Jev is a remote, typed-choice model.  It sees a structured board and returns a
probability for each legal column.  Our local side uses the current
Transformer-guided alpha-beta agent.  The API key is deliberately read only
from ``JEV_API_KEY`` or ``TYPESAFE_API_KEY``; it is never written to a file or
printed.

Example:
    export JEV_API_KEY='apikey_...'
    ./.venv/bin/python -m connect_four_transformer.play_jev_vs_hybrid
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from connect_four_engine import PLAYER_ONE, ConnectFour
from connect_four_engine.play import render

from .hybrid_search import TransformerGuidedSearchAgent


API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "connect_four_distillation"
    / "artifacts"
    / "checkpoints"
    / "candidate_pons_phases_v3_5min.pt"
)


def _board_rows(game: ConnectFour) -> list[str]:
    symbols = {0: ".", PLAYER_ONE: "X", -PLAYER_ONE: "O"}
    return ["".join(symbols[cell] for cell in game.board[row * 7 : (row + 1) * 7]) for row in range(6)]


def choose_jev(
    game: ConnectFour, api_key: str, *, timeout: float
) -> tuple[int, dict[str, Any], dict[str, int], float]:
    """Ask Jev to select exactly one currently legal column."""
    legal = list(game.legal_actions())
    labels = {f"col_{action + 1}": f"Drop a piece in legal column {action + 1}." for action in legal}
    player = "X" if game.current_player == PLAYER_ONE else "O"
    state = {
        "game": "Connect Four",
        "board_rows_top_to_bottom": _board_rows(game),
        "columns": "1 2 3 4 5 6 7",
        "side_to_move": player,
        "legal_columns": [action + 1 for action in legal],
        "rule": "Pieces fall to the lowest empty cell. Four contiguous pieces horizontally, vertically, or diagonally wins.",
    }
    payload = {
        "model": "jev-latest",
        "state": state,
        "questions": {
            "move": {
                "type": "choice",
                "instructions": (
                    "You are the side to move in Connect Four. Choose the strongest legal move "
                    "to maximize your chance of winning. Check immediate wins, forced blocks, and forks."
                ),
                "criteria": labels,
            }
        },
    }
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"Jev HTTP {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach Jev: {error.reason}") from error
    elapsed_ms = (time.perf_counter() - started) * 1000

    answer = data["answers"]["move"]
    label = answer["choice"]
    if label not in labels:
        raise RuntimeError(f"Jev returned a non-legal choice label: {label!r}")
    action = int(label.removeprefix("col_")) - 1
    usage = data.get("usage", {})
    return action, answer, usage, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jev", choices=("X", "O"), default="X")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=5_000)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.depth <= 0 or args.nodes <= 0 or args.timeout <= 0:
        parser.error("depth, nodes, and timeout must be positive")

    api_key = os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        parser.error("set JEV_API_KEY (or TYPESAFE_API_KEY) in your shell; never commit it")
    jev_player = PLAYER_ONE if args.jev == "X" else -PLAYER_ONE
    hybrid = TransformerGuidedSearchAgent.from_checkpoint(
        args.checkpoint,
        max_depth=args.depth,
        node_limit=args.nodes,
        max_model_evaluations=1,
        model_value_weight=0.0,
    )
    game = ConnectFour()
    api_tokens = 0
    print(f"Jev is {args.jev}; hybrid Transformer + alpha-beta is {'O' if args.jev == 'X' else 'X'}.")
    while not game.is_terminal:
        render(game)
        if game.current_player == jev_player:
            action, answer, usage, elapsed_ms = choose_jev(game, api_key, timeout=args.timeout)
            probabilities = answer["probabilities"]
            ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            top = ", ".join(f"{name.replace('col_', 'c')}={prob:.1%}" for name, prob in ranked[:3])
            api_tokens += int(usage.get("input_tokens", 0))
            print(
                f"Jev chooses col {action + 1}; confidence={answer['confidence']:.1%}; "
                f"top: {top}; request={elapsed_ms:.1f} ms; input={usage.get('input_tokens', '?')} tok"
            )
        else:
            decision = hybrid.evaluate(game)
            stats = decision.stats
            action = decision.best_action
            print(
                f"hybrid chooses col {action + 1} ({decision.reason}); depth={stats.completed_depth}, "
                f"nodes={stats.nodes}, model calls={stats.model_evaluations}, "
                f"time={stats.elapsed_seconds * 1000:.1f} ms"
            )
        game.step(action)

    render(game)
    if game.is_draw:
        print("Draw.")
    else:
        print(f"{'X' if game.winner == PLAYER_ONE else 'O'} wins.")
    print(f"Jev total billed input tokens: {api_tokens}")


if __name__ == "__main__":
    main()
