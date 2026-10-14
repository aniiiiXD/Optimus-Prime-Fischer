"""Play against a trained Connect Four Transformer and see its top choices."""

from __future__ import annotations

import argparse
from pathlib import Path

from connect_four_engine import PLAYER_ONE, ConnectFour, IllegalMoveError
from connect_four_engine.play import render

from .agent import TransformerAgent


DEFAULT_CHECKPOINT = Path(__file__).parent / "artifacts" / "checkpoints" / "candidate_v0.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--human", choices=("X", "O"), default="X")
    parser.add_argument(
        "--no-lookahead",
        action="store_true",
        help="play the Transformer's raw highest-probability move without tactical simulation",
    )
    args = parser.parse_args()
    agent = TransformerAgent(args.checkpoint, tactical_guard=not args.no_lookahead)
    human_player = PLAYER_ONE if args.human == "X" else -PLAYER_ONE
    game = ConnectFour()
    while not game.is_terminal:
        render(game)
        if game.current_player == human_player:
            try:
                game.step(int(input("Your column (1–7): ").strip()) - 1)
            except (ValueError, IllegalMoveError) as error:
                print(f"Invalid move: {error}")
        else:
            decision = agent.inspect(game)
            top = decision["choices"][:3]
            print(
                f"model V(s)={decision['value']:+.3f}, forward {decision['latency_ms']:.2f} ms, "
                f"lookahead {decision['tactical_scan_ms']:.2f} ms, "
                f"decision total {decision['decision_latency_ms']:.2f} ms | "
                + " | ".join(
                    f"col {choice['column'] + 1}: {choice['probability']:.1%}, Q={choice['q']:+.2f}"
                    + (
                        ", safe" if choice.get("safe_next_turn")
                        else (
                            f", allows win in {choice['opponent_winning_replies']}"
                            if choice.get("safe_next_turn") is False
                            else ", unchecked"
                        )
                    )
                    + (" ← chosen" if choice["selected"] else "")
                    for choice in top
                )
            )
            chosen = int(decision["selected_action"])
            if chosen != int(decision["model_top_action"]):
                print(
                    f"one-move simulation overrides col {int(decision['model_top_action']) + 1}; "
                    f"chooses col {chosen + 1} ({decision['decision_reason']})."
                )
            game.step(chosen)
    render(game)
    if game.is_draw:
        print("Draw.")
    else:
        winner = "X" if game.winner == PLAYER_ONE else "O"
        print(f"{winner} wins.")


if __name__ == "__main__":
    main()
