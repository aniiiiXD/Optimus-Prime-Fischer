"""Check that the selected action ignores hidden hands, deck order, and RNG."""

from __future__ import annotations

import json
import argparse
import random
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanatron import Color, Game, RandomPlayer  # noqa: E402
from catanatron.models.actions import generate_playable_actions  # noqa: E402
from catanatron.players.value import ValueFunctionPlayer  # noqa: E402
from catanLabs.Competitive.fair_value_agent import FairValueAgent  # noqa: E402
from catanLabs.Competitive.guided_search_agent import GuidedSearchAgent  # noqa: E402


def perturbed_hidden(game, actor):
    clone = game.copy()
    state = clone.state
    for color in state.colors:
        if color == actor:
            continue
        prefix = f"P{state.color_to_index[color]}_"
        resources = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
        counts = [state.player_state[prefix + resource + "_IN_HAND"] for resource in resources]
        for resource, amount in zip(resources, counts[1:] + counts[:1]):
            state.player_state[prefix + resource + "_IN_HAND"] = amount
        cards = ("KNIGHT", "MONOPOLY", "YEAR_OF_PLENTY", "ROAD_BUILDING")
        count = [state.player_state[prefix + card + "_IN_HAND"] for card in cards]
        for card, amount in zip(cards, count[1:] + count[:1]):
            state.player_state[prefix + card + "_IN_HAND"] = amount
    state.development_listdeck.reverse()
    state.random = random.Random(981273)
    return clone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("fair", "guided"), default="fair")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    default_config = ("fair_army_finish.json" if args.agent == "fair"
                      else "guided_search_3.json")
    config_path = args.config or Path(__file__).parent / "configs" / default_config
    agent_class = FairValueAgent if args.agent == "fair" else GuidedSearchAgent
    checked = 0
    categories = set()
    for seed in (20262001, 20262002, 20262003):
        colors = list(Color)
        game = Game([RandomPlayer(color) for color in colors], seed=seed,
                    vps_to_win=10, discard_limit=7, friendly_robber=False,
                    number_placement="official_spiral")
        actor = game.state.colors[0]
        candidate = agent_class(
            actor,
            json.loads(config_path.read_text()),
        )
        opponents = {color: ValueFunctionPlayer(color) for color in colors if color != actor}
        checked_synthetic_cards = False
        for _ in range(1300):
            if game.winning_color() is not None:
                break
            color = game.state.current_color()
            actions = game.playable_actions
            if (not checked_synthetic_cards and color == actor
                    and not game.state.is_initial_build_phase
                    and game.state.current_prompt.name == "PLAY_TURN"):
                synthetic = game.copy()
                prefix = f"P{synthetic.state.color_to_index[actor]}_"
                for card in ("KNIGHT", "MONOPOLY"):
                    synthetic.state.player_state[prefix + card + "_IN_HAND"] += 1
                    synthetic.state.player_state[prefix + card + "_OWNED_AT_START"] = True
                synthetic.playable_actions = generate_playable_actions(synthetic.state)
                kinds = {a.action_type.name for a in synthetic.playable_actions}
                assert {"ROLL", "PLAY_KNIGHT_CARD", "PLAY_MONOPOLY"} <= kinds
                rng_before = synthetic.state.random.getstate()
                original = candidate.decide(synthetic, synthetic.playable_actions)
                assert synthetic.state.random.getstate() == rng_before, "decision advanced game RNG"
                alternate = candidate.decide(
                    perturbed_hidden(synthetic, actor), synthetic.playable_actions
                )
                assert original == alternate, (seed, original, alternate)
                checked += 1
                categories.update(kinds)
                checked_synthetic_cards = True
            if color == actor and len(actions) > 1:
                rng_before = game.state.random.getstate()
                original = candidate.decide(game, actions)
                assert game.state.random.getstate() == rng_before, "decision advanced game RNG"
                alternate = candidate.decide(perturbed_hidden(game, actor), actions)
                assert original == alternate, (seed, game.state.num_turns, original, alternate)
                checked += 1
                categories.update(action.action_type.name for action in actions)
            action = candidate.decide(game, actions) if color == actor else opponents[color].decide(game, actions)
            game.execute(action, validate_action=True)
    print({"checked_decisions": checked, "legal_action_types": sorted(categories),
           "hidden_perturbation_invariant": True})


if __name__ == "__main__":
    main()
