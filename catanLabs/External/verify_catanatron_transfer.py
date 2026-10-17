"""Smoke-test graph/action mapping, legality, and hidden-state invariance."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

import numpy as np  # noqa: E402
from catanatron import Color, Game, RandomPlayer  # noqa: E402
from catanatron.players.value import ValueFunctionPlayer  # noqa: E402
from catanLabs.Competitive.check_visibility import perturbed_hidden  # noqa: E402
from catanLabs.External.catanatron_transfer_agent import (  # noqa: E402
    TransferredPPOAgent, action_id, encode_visible, topology_map,
)


def main():
    checkpoint = WORKSPACE / "catanLabs/External/models/cpu10_finetuned.pt"
    tiles, vertices, edges = topology_map()
    reverse = ({value: i for i, value in enumerate(tiles)},
               {value: i for i, value in enumerate(vertices)},
               {value: i for i, value in enumerate(edges)})
    checked = 0
    action_kinds = set()
    for seed in (20265101, 20265102, 20265103):
        game = Game([RandomPlayer(color) for color in Color], seed=seed,
                    vps_to_win=10, number_placement="official_spiral")
        actor = game.state.colors[0]
        agent = TransferredPPOAgent(actor, checkpoint)
        opponents = {color: ValueFunctionPlayer(color) for color in Color if color != actor}
        for _ in range(1800):
            if game.winning_color() is not None:
                break
            current = game.state.current_color()
            actions = game.playable_actions
            if current == actor:
                for action in actions:
                    action_id(action, game, *reverse)
                    action_kinds.add(action.action_type.name)
                hidden = perturbed_hidden(game, actor)
                before = game.state.random.getstate()
                view = encode_visible(game, tiles, vertices, edges)
                changed_view = encode_visible(hidden, tiles, vertices, edges)
                if not np.array_equal(view, changed_view):
                    raise AssertionError("Private-state perturbation changed observation")
                chosen = agent.decide(game, actions)
                if chosen != agent.decide(hidden, actions):
                    raise AssertionError("Private-state perturbation changed decision")
                if chosen not in actions or game.state.random.getstate() != before:
                    raise AssertionError("Illegal action or RNG mutation")
                checked += 1
            else:
                chosen = opponents[current].decide(game, actions)
            game.execute(chosen, validate_action=True)
    print({"checked_decisions": checked,
           "action_types": sorted(action_kinds),
           "hidden_invariant": True, "legal_actions": True})


if __name__ == "__main__":
    main()
