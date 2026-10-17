"""Replay one game and expose the MLP's action scores at a chosen turn."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from run import (Catanatron, Color, Game, RandomPlayer, canonical_action_key,
                 install_v5_protocol, load_candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20262001)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--turn", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "move_probe.json")
    args = parser.parse_args()
    install_v5_protocol()
    create = load_candidate("strategyone")
    import evaluate_win_rate as evaluation

    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
                seed=args.seed, vps_to_win=10, discard_limit=7,
                friendly_robber=False, number_placement="official_spiral")
    game.playable_actions.sort(key=canonical_action_key)
    our_color = game.state.colors[args.seat]
    their_color = next(color for color in game.state.colors if color != our_color)
    ours = create(our_color, args.seed)
    players = {our_color: ours, their_color: Catanatron(their_color, Catanatron.Params())}

    for _ in range(10000):
        if game.winning_color() is not None:
            raise RuntimeError("Game ended before the requested turn")
        actor = game.state.current_color()
        legal = game.playable_actions
        action = players[actor].decide(game, legal)
        if actor == our_color and game.state.num_turns == args.turn and action.action_type.name == "END_TURN":
            visible = evaluation.observation(game, our_color)
            state = torch.tensor([evaluation.encode_state(visible, ours.edge_index)], dtype=torch.float32)
            actions = torch.tensor([[
                evaluation.encode_action(evaluation.action_json(candidate, our_color), visible, ours.edge_index)
                for candidate in legal
            ]], dtype=torch.float32)
            with torch.no_grad():
                scores = ours.adapter(ours.model(state, actions), state, actions)[0].tolist()
            candidates = []
            for candidate, score in zip(legal, scores, strict=True):
                row = {"type": candidate.action_type.name, "value": repr(candidate.value), "score": score}
                if candidate.action_type.name == "MARITIME_TRADE":
                    hypothetical = game.copy()
                    hypothetical.execute(candidate, validate_action=True)
                    row["next_build_options"] = sorted({
                        next_action.action_type.name for next_action in hypothetical.playable_actions
                        if next_action.action_type.name.startswith("BUILD_")
                        or next_action.action_type.name == "BUY_DEVELOPMENT_CARD"
                    })
                candidates.append(row)
            report = {"seed": args.seed, "seat": args.seat, "turn": args.turn,
                      "chosen": action.action_type.name, "hand": visible["own_hand"],
                      "candidates": sorted(candidates, key=lambda row: row["score"], reverse=True)}
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            return
        game.execute(action, validate_action=True)
        game.playable_actions.sort(key=canonical_action_key)
    raise RuntimeError("Requested decision not found")


if __name__ == "__main__":
    main()
