"""Win-oriented Catanatron agent adapted for a provisional two-player target."""

from __future__ import annotations

import json
from pathlib import Path

from catanatron.models.player import Player
from catanatron.state_functions import player_num_resource_cards

from competitive import CompetitiveAgent
from tactical_vp import TacticalVPAgent


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"


class OneVOneRobber(CompetitiveAgent):
    """Score robber moves as a two-player production-denial decision."""

    def __init__(self, color, weights=None, duel_config=None):
        super().__init__(color, weights)
        self.duel_config = {
            "own_production_cost": 1.2,
            "opponent_lead_sensitivity": 0.12,
            "min_opponent_pressure": 0.8,
            "max_opponent_pressure": 1.45,
            "theft_base": 0.2,
            "theft_per_public_card": 0.08,
            "max_theft_cards": 8,
            **(duel_config or {}),
        }

    def _robber_score(self, game, action):
        state = game.state
        enemies = [color for color in state.colors if color != self.color]
        if len(enemies) != 1:
            return super()._robber_score(game, action)

        opponent = enemies[0]
        coordinate, victim = action.value
        tile = state.board.map.land_tiles[coordinate]
        if tile.resource is None:
            return -2.0

        pips = (6 - abs(7 - tile.number)) / 36.0
        own_prefix = f"P{state.color_to_index[self.color]}_"
        opponent_prefix = f"P{state.color_to_index[opponent]}_"
        own_vp = state.player_state[own_prefix + "VICTORY_POINTS"]
        opponent_vp = state.player_state[opponent_prefix + "VICTORY_POINTS"]

        own_values = self._resource_values(self._production(game, self.color), own_vp)
        opponent_values = self._resource_values(
            self._production(game, opponent), opponent_vp
        )
        lead = opponent_vp - own_vp
        pressure = 1.0 + self.duel_config["opponent_lead_sensitivity"] * lead
        pressure = max(
            self.duel_config["min_opponent_pressure"],
            min(self.duel_config["max_opponent_pressure"], pressure),
        )

        own_blocked = 0.0
        opponent_blocked = 0.0
        for node in tile.nodes.values():
            building = state.board.buildings.get(node)
            if building is None:
                continue
            owner, building_type = building
            yield_count = 2 if building_type == "CITY" else 1
            if owner == self.color:
                own_blocked += pips * own_values[tile.resource] * yield_count
            elif owner == opponent:
                opponent_blocked += (
                    pips * opponent_values[tile.resource] * yield_count
                )

        score = self.weights["robber"] * (
            pressure * opponent_blocked
            - self.duel_config["own_production_cost"] * own_blocked
        )
        if victim == opponent:
            # Card count is public in Catan. Never use the hidden card identities.
            public_hand_size = player_num_resource_cards(state, opponent)
            if public_hand_size:
                score += self.duel_config["theft_base"]
                score += self.duel_config["theft_per_public_card"] * min(
                    public_hand_size, self.duel_config["max_theft_cards"]
                )
        return score


class OneVOneAgent(TacticalVPAgent, Player):
    """Local tactical VP agent with a robber policy specialized for one rival."""

    def __init__(self, color, config=None):
        config = dict(config or {})
        # Keep the normal Catanatron observer hooks so this can run both in
        # the tournament harness and directly in Game.play() against a human.
        Player.__init__(self, color)
        super().__init__(color, config)
        self.robber = OneVOneRobber(
            color,
            weights=config.get("robber_weights", {}),
            duel_config=config.get("onevone_robber", {}),
        )


def create_agent(color, checkpoint=None, seed=None):
    """Factory used by the local tournament runner and compatible adapters."""
    del seed  # Policy tie-breaking is deterministic; the game owns randomness.
    config_path = Path(checkpoint) if checkpoint is not None else DEFAULT_CONFIG
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Agent config must be a JSON object: {config_path}")
    return OneVOneAgent(color, config)
