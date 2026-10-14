"""Play Connect Four against the experimental Transformer-guided search agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from connect_four_engine import PLAYER_ONE, ConnectFour, IllegalMoveError
from connect_four_engine.play import render

from .hybrid_search import TransformerGuidedSearchAgent


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "connect_four_distillation"
    / "artifacts"
    / "checkpoints"
    / "candidate_pons_phases_v3_5min.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--human", choices=("X", "O"), default="X")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=5_000)
    parser.add_argument("--model-evals", type=int, default=1)
    parser.add_argument("--model-value-weight", type=float, default=0.0)
    args = parser.parse_args()
    if min(args.depth, args.nodes, args.model_evals) <= 0:
        parser.error("depth, nodes, and model-evals must be positive")
    if not 0.0 <= args.model_value_weight <= 1.0:
        parser.error("model-value-weight must be in [0,1]")

    agent = TransformerGuidedSearchAgent.from_checkpoint(
        args.checkpoint,
        max_depth=args.depth,
        node_limit=args.nodes,
        max_model_evaluations=args.model_evals,
        model_value_weight=args.model_value_weight,
    )
    human = PLAYER_ONE if args.human == "X" else -PLAYER_ONE
    game = ConnectFour()
    while not game.is_terminal:
        render(game)
        if game.current_player == human:
            try:
                game.step(int(input("Your column (1–7): ").strip()) - 1)
            except (ValueError, IllegalMoveError) as error:
                print(f"Invalid move: {error}")
            continue

        decision = agent.evaluate(game)
        stats = decision.stats
        print(
            f"hybrid chooses col {decision.best_action + 1} ({decision.reason}); "
            f"depth={stats.completed_depth}, nodes={stats.nodes}, "
            f"model calls={stats.model_evaluations}, "
            f"time={stats.elapsed_seconds * 1000:.1f} ms"
        )
        game.step(decision.best_action)

    render(game)
    if game.is_draw:
        print("Draw.")
    else:
        print(f"{'X' if game.winner == PLAYER_ONE else 'O'} wins.")


if __name__ == "__main__":
    main()
