"""Research policy: evaluate Road Building after its free roads are placed.

Only deterministic, legal own actions are executed in copied states. Roll,
theft, opponent hands, and the development deck are never sampled in search.
"""

from __future__ import annotations

import json
from pathlib import Path

from catanatron.models.enums import ActionType

from catanLabs.Competitive.fair_value_agent import FairValueAgent


class RoadBuildingAgent(FairValueAgent):
    def __init__(self, color, config):
        super().__init__(color, config)
        self.max_first_roads = int(config.get("road_building_max_first_roads", 64))
        self.max_second_roads = int(config.get("road_building_max_second_roads", 256))
        self.score_margin = float(config.get("road_building_score_margin", 0.0))
        if self.max_first_roads < 1 or self.max_second_roads < 1 or self.score_margin < 0:
            raise ValueError("Invalid Road Building search budget")
        self.last_scan = {}
        self._candidate_roads = ()
        self._pending_roads = ()

    def _vp(self, game):
        prefix = f"P{game.state.color_to_index[self.color]}_"
        return int(game.state.player_state[prefix + "ACTUAL_VICTORY_POINTS"])

    def _fair_score(self, game, action):
        """Return FairValue score, a chance-safe VP upper bound, and certain win."""
        kind = action.action_type
        if kind == ActionType.ROLL:
            return self._roll_value(game), self._vp(game), False
        if kind == ActionType.BUY_DEVELOPMENT_CARD:
            # The hidden top card may be a VP card. Do not inspect the deck;
            # conservatively protect that possible point from replacement.
            return self._buy_value(game), self._vp(game) + 1, False
        if kind == ActionType.PLAY_MONOPOLY:
            return self._monopoly_value(game, action.value), self._vp(game), False
        after = game.copy()
        after.execute(action, validate_action=True)
        return (self.value_fn(after, self.color), self._vp(after),
                after.winning_color() == self.color)

    def find_override(self, game, playable_actions, baseline):
        """Return Road Building iff its best legal free-road leaf beats FairValue."""
        card = next((action for action in playable_actions
                     if action.action_type == ActionType.PLAY_ROAD_BUILDING), None)
        info = {"eligible": card is not None, "first_roads": 0,
                "second_roads": 0, "truncated": False, "override": False}
        self._candidate_roads = ()
        if card is None or baseline == card:
            self.last_scan = info
            return None

        baseline_score, baseline_vp_upper, baseline_wins = self._fair_score(game, baseline)
        info["baseline_score"] = baseline_score
        info["baseline_vp_upper_bound"] = baseline_vp_upper
        if baseline_wins:
            info["baseline_wins"] = True
            self.last_scan = info
            return None

        after_card = game.copy()
        after_card.execute(card, validate_action=True)
        first_roads = [action for action in after_card.playable_actions
                       if action.action_type == ActionType.BUILD_ROAD]
        if len(first_roads) > self.max_first_roads:
            info["truncated"] = True
        best_key = (False, -1, float("-inf"))
        best_line = None
        best_roads = ()

        def consider(state, first, second=None):
            nonlocal best_key, best_line, best_roads
            wins = state.winning_color() == self.color
            score = self.value_fn(state, self.color)
            vp = self._vp(state)
            if (wins, vp, score) > best_key:
                best_key = (wins, vp, score)
                best_roads = (first,) if second is None else (first, second)
                best_line = {
                    "first_road": repr(first.value),
                    "second_road": repr(second.value) if second is not None else None,
                    "result_vp": vp,
                }

        for first in first_roads[:self.max_first_roads]:
            after_first = after_card.copy()
            after_first.execute(first, validate_action=True)
            info["first_roads"] += 1
            if after_first.winning_color() == self.color:
                consider(after_first, first)
                break
            second_roads = ([action for action in after_first.playable_actions
                             if action.action_type == ActionType.BUILD_ROAD]
                            if after_first.state.is_road_building else [])
            if not second_roads:
                consider(after_first, first)
                continue
            for second in second_roads:
                if info["second_roads"] >= self.max_second_roads:
                    info["truncated"] = True
                    break
                after_second = after_first.copy()
                after_second.execute(second, validate_action=True)
                info["second_roads"] += 1
                consider(after_second, first, second)
                if best_key[0]:
                    break
            if info["second_roads"] >= self.max_second_roads or best_key[0]:
                break

        info["best_line"] = best_line
        info["best_score"] = best_key[2]
        info["best_line_vp"] = best_key[1]
        # For an unknown purchase, a VP-card draw can score one immediately.
        # Require a strictly greater certain gain; otherwise keep FairValue.
        vp_safe = (best_key[1] > baseline_vp_upper if baseline.action_type == ActionType.BUY_DEVELOPMENT_CARD
                   else best_key[1] >= baseline_vp_upper)
        info["override"] = (best_line is not None and vp_safe
                            and (best_key[0] or best_key[2] > baseline_score + self.score_margin))
        if info["override"]:
            self._candidate_roads = best_roads
        self.last_scan = info
        return card if info["override"] else None

    def decide(self, game, playable_actions):
        if game.state.is_road_building and self._pending_roads:
            expected = self._pending_roads[0]
            planned = next((action for action in playable_actions if action == expected), None)
            if planned is not None:
                self._pending_roads = self._pending_roads[1:]
                self.last_scan = {"planned_road": True, "remaining": len(self._pending_roads)}
                return planned
        if not game.state.is_road_building or self._pending_roads:
            self._pending_roads = ()
        baseline = super().decide(game, playable_actions)
        override = self.find_override(game, playable_actions, baseline)
        if override is not None:
            self._pending_roads = self._candidate_roads
            return override
        return baseline


def create_agent(color, checkpoint, seed):
    del seed
    config = json.loads(Path(checkpoint).read_text()) if checkpoint is not None else {}
    return RoadBuildingAgent(color, config)
