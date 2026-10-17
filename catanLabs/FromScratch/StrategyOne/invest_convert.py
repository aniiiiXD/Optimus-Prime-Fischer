"""Visible-information invest/convert planner layered on the frozen MLP.

The planner expands deterministic actions within the current turn. It never
simulates a dice roll, development-card draw, or robber theft. Coefficients are
small and explicit so game-outcome tuning can inspect their effect.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "start/catan_lab/vendor/catanatron/catanatron"))
sys.path.insert(0, str(ROOT))

from catanatron.state_functions import get_actual_victory_points

from evaluate_win_rate import MLPPlayer
from same_turn_search import find_immediate_win
from settlement_target_head import SettlementTargetHead
from train_calibrated_ranker import ActionCalibrationAdapter
from train_legal_ranker import HERE, LegalRanker, stable_edge_index


RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
CONVERT_ACTIONS = {"BUILD_CITY", "BUILD_SETTLEMENT", "BUILD_ROAD", "MARITIME_TRADE"}
BUILD_ACTIONS = {"BUILD_CITY", "BUILD_SETTLEMENT", "BUILD_ROAD"}
ORDER = {"BUILD_CITY": 0, "BUILD_SETTLEMENT": 1, "MARITIME_TRADE": 2, "BUILD_ROAD": 3}


@dataclass(frozen=True)
class Coefficients:
    vp: float = 8.0
    production: float = 4.0
    access: float = 1.5
    hand: float = 0.5
    excess_cards: float = 0.9
    horizon_base: float = 13.0
    horizon_opp_vp: float = 1.05
    horizon_own_vp: float = 0.35
    override_margin: float = 0.20

    def json(self):
        return asdict(self)


class InvestConvertPlayer(MLPPlayer):
    def __init__(self, *args, coefficients: Coefficients | None = None,
                 max_roots: int = 48, max_followups: int = 160, **kwargs):
        # The old same-turn search only looks for a forced win. We run it once
        # here, before evaluating nonwinning investment and conversion lines.
        kwargs["same_turn_search"] = False
        super().__init__(*args, **kwargs)
        self.coefficients = coefficients or Coefficients()
        self.max_roots = max_roots
        self.max_followups = max_followups
        self.planner_consulted = 0
        self.planner_overrides = 0
        self.planner_trade_to_build = 0
        self.planner_invest = 0
        self.planner_convert = 0

    def _public_vp(self, game, color):
        state = game.state
        prefix = f"P{state.color_to_index[color]}_"
        return int(state.player_state[prefix + "VICTORY_POINTS"])

    def _horizon(self, game):
        other = next(color for color in game.state.colors if color != self.color)
        own = self._public_vp(game, self.color)
        opponent = self._public_vp(game, other)
        c = self.coefficients
        return max(1.0, min(12.0, c.horizon_base - c.horizon_opp_vp * opponent
                            - c.horizon_own_vp * own))

    def _production(self, game):
        state = game.state
        board = state.board
        blocked = board.map.land_tiles[board.robber_coordinate]
        rates = dict.fromkeys(RESOURCES, 0.0)
        for node, (owner, kind) in board.buildings.items():
            if owner != self.color:
                continue
            for tile in board.map.adjacent_tiles[node]:
                if tile.resource in rates and tile is not blocked:
                    rates[tile.resource] += (6 - abs(7 - tile.number)) / 36.0 * (
                        2 if kind == "CITY" else 1)
        return rates

    def _site_quality(self, game, node, weights):
        return sum(
            (6 - abs(7 - tile.number)) / 36.0 * weights[tile.resource]
            for tile in game.state.board.map.adjacent_tiles[node]
            if tile.resource in weights
        )

    def _score(self, game):
        if game.winning_color() == self.color:
            return 1e6
        state = game.state
        prefix = f"P{state.color_to_index[self.color]}_"
        vp = get_actual_victory_points(state, self.color)
        horizon = self._horizon(game)
        c = self.coefficients
        rates = self._production(game)
        # Resources that we produce poorly have greater marginal value.
        weights = {resource: min(1.6, 1.0 + max(0.0, .12 - rate) * 3.0)
                   for resource, rate in rates.items()}
        production = sum(weights[resource] * rate for resource, rate in rates.items())
        sites = state.board.buildable_node_ids(self.color)
        access = max((self._site_quality(game, node, weights) for node in sites),
                     default=0.0)
        hand = {resource: int(state.player_state[prefix + resource + "_IN_HAND"])
                for resource in RESOURCES}
        hand_value = sum(weights[r] * (min(n, 3) + .15 * max(0, n - 3))
                         for r, n in hand.items())
        excess = max(0, sum(hand.values()) - 7)
        return (c.vp * vp + c.production * horizon * production
                + c.access * horizon * access + c.hand * hand_value
                - c.excess_cards * excess)

    def _after(self, game, action):
        child = game.copy()
        child.execute(action, validate_action=True)
        return child

    def _planned(self, game, playable_actions, baseline):
        if game.state.current_prompt.name != "PLAY_TURN":
            return None
        prefix = f"P{game.state.color_to_index[self.color]}_"
        if not game.state.player_state[prefix + "HAS_ROLLED"]:
            return None

        roots = sorted(
            (action for action in playable_actions
             if action.action_type.name in CONVERT_ACTIONS),
            key=lambda action: (ORDER[action.action_type.name], repr(action.value)),
        )[:self.max_roots]
        if not roots:
            return None
        self.planner_consulted += 1
        starting_score = self._score(game)
        baseline_score = starting_score
        if baseline.action_type.name in CONVERT_ACTIONS:
            baseline_score = self._score(self._after(game, baseline))

        best_action = None
        best_score = float("-inf")
        best_line = None
        followups = 0
        for root in roots:
            child = self._after(game, root)
            score = self._score(child)
            if score > best_score:
                best_action, best_score, best_line = root, score, (root,)
            if child.winning_color() is not None or child.state.current_color() != self.color:
                continue
            next_actions = sorted(
                (action for action in child.playable_actions
                 if action.action_type.name in BUILD_ACTIONS),
                key=lambda action: (ORDER[action.action_type.name], repr(action.value)),
            )
            for second in next_actions:
                if followups >= self.max_followups:
                    break
                followups += 1
                later = self._after(child, second)
                # Slight preference for a one-action gain over a two-action gain.
                score = self._score(later) - .02
                if score > best_score:
                    best_action, best_score, best_line = root, score, (root, second)
            if followups >= self.max_followups:
                break

        if best_action is None or best_score <= baseline_score + self.coefficients.override_margin:
            return None
        if best_line and best_line[0].action_type.name == "MARITIME_TRADE" and len(best_line) > 1:
            self.planner_trade_to_build += 1
        # Investment is production/access growth without a public point this turn.
        if best_line:
            simulated = game
            for action in best_line:
                simulated = self._after(simulated, action)
            if self._public_vp(simulated, self.color) > self._public_vp(game, self.color):
                self.planner_convert += 1
            else:
                self.planner_invest += 1
        self.planner_overrides += best_action != baseline
        return best_action

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        immediate = find_immediate_win(game, self.color, max_depth=4)
        if immediate is not None:
            return immediate
        baseline = super().decide(game, playable_actions)
        planned = self._planned(game, playable_actions, baseline)
        return planned if planned is not None else baseline


@lru_cache(maxsize=1)
def _load_frozen_assets():
    device = torch.device("cpu")
    checkpoint = torch.load(HERE / "runs/calibrated_ranker_v1/model.pt",
                            map_location=device, weights_only=True)
    model = LegalRanker(checkpoint["state_size"], checkpoint["action_size"])
    model.load_state_dict(checkpoint["base_model"])
    model.eval()
    adapter = ActionCalibrationAdapter(checkpoint["state_size"])
    adapter.load_state_dict(checkpoint["adapter"])
    adapter.eval()
    target = torch.load(HERE / "runs/settlement_target_v1/epoch_4.pt",
                        map_location=device, weights_only=True)
    settlement = SettlementTargetHead(target["input_size"])
    settlement.load_state_dict(target["head"])
    settlement.eval()
    with (HERE / "data/catanatron_diverse.jsonl").open() as stream:
        first = json.loads(next(stream))
    return model, stable_edge_index(first["observation"]), device, adapter, settlement


def create_agent(color, checkpoint=None, seed=None):
    """CatanLabs-compatible factory. ``checkpoint`` is a coefficient JSON."""
    del seed
    coefficients = (Coefficients(**json.loads(Path(checkpoint).read_text()))
                    if checkpoint is not None else Coefficients())
    return InvestConvertPlayer(color, *_load_frozen_assets(),
                               coefficients=coefficients)
