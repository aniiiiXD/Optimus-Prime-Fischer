"""Research agent: exact, deterministic two-action VP tactics after rolling.

The ordinary fair-value policy remains the fallback. This scanner only follows
our own public/known actions; it never resolves a roll, theft, or unknown card.
"""

from __future__ import annotations

import json
from pathlib import Path

from catanatron.models.enums import ActionType

from fair_value import FairValueAgent


ROOT_TYPES = frozenset({
    ActionType.MARITIME_TRADE,
    ActionType.BUILD_ROAD,
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_CITY,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_ROAD_BUILDING,
})
POINT_TYPES = frozenset({
    ActionType.BUILD_ROAD,  # A road can claim Longest Road.
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_CITY,
})
# Knight play itself has no random outcome: it updates public Largest Army and
# then asks for a robber move. The later theft is deliberately not simulated.
SAFE_BASELINE_TYPES = ROOT_TYPES | {ActionType.PLAY_KNIGHT_CARD}


class TacticalVPAgent(FairValueAgent):
    def __init__(self, color, config):
        super().__init__(color, config)
        self.min_vp = int(config.get("tactical_min_vp", 7))
        self.max_roots = int(config.get("tactical_max_roots", 64))
        self.max_followups = int(config.get("tactical_max_followups", 256))
        if not 0 <= self.min_vp <= 10 or self.max_roots < 1 or self.max_followups < 1:
            raise ValueError("Invalid tactical search budget")
        self.last_scan = {}

    def _vp(self, game):
        prefix = f"P{game.state.color_to_index[self.color]}_"
        return int(game.state.player_state[prefix + "ACTUAL_VICTORY_POINTS"])

    def _eligible(self, game):
        if game.state.current_color() != self.color or game.state.current_prompt.name != "PLAY_TURN":
            return False
        prefix = f"P{game.state.color_to_index[self.color]}_"
        return (self._vp(game) >= self.min_vp
                and bool(game.state.player_state[prefix + "HAS_ROLLED"]))

    def find_override(self, game, playable_actions, baseline):
        """Return a root move only if it beats FairValue's immediate VP gain."""
        info = {"eligible": self._eligible(game), "roots": 0, "followups": 0,
                "found_line": False, "override": False, "truncated": False}
        if not info["eligible"] or len(playable_actions) < 2:
            self.last_scan = info
            return None

        baseline_vp = current_vp = self._vp(game)
        if baseline.action_type in SAFE_BASELINE_TYPES:
            baseline_after = game.copy()
            baseline_after.execute(baseline, validate_action=True)
            baseline_vp = self._vp(baseline_after)
        elif baseline.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            # The top card is hidden and might be a victory-point card. Use
            # its maximum immediate VP gain instead of peeking at the deck.
            baseline_vp += 1
        if baseline_vp >= game.vps_to_win:
            info["baseline_vp"] = baseline_vp
            info["best_line_vp"] = baseline_vp
            self.last_scan = info
            return None

        best_action = None
        best_key = (False, baseline_vp, float("-inf"))
        def consider(state, root, second=None):
            nonlocal best_action, best_key
            vp = self._vp(state)
            if vp <= current_vp:
                return
            info["found_line"] = True
            key = (state.winning_color() == self.color, vp,
                   self.value_fn(state, self.color))
            if key > best_key:
                best_key, best_action = key, root
                info["line"] = {
                    "first": (root.action_type.name, repr(root.value)),
                    "second": ((second.action_type.name, repr(second.value))
                               if second is not None else None),
                    "result_vp": vp,
                }

        roots = [action for action in playable_actions if action.action_type in ROOT_TYPES]
        if len(roots) > self.max_roots:
            info["truncated"] = True
        for action in roots[:self.max_roots]:
            after = game.copy()
            after.execute(action, validate_action=True)
            info["roots"] += 1
            consider(after, action)
            if best_key[0]:
                break
            if after.state.current_color() != self.color or after.winning_color() is not None:
                continue
            second_actions = sorted(
                (second for second in after.playable_actions if second.action_type in POINT_TYPES),
                key=lambda second: (second.action_type.name, repr(second.value)),
            )
            for second in second_actions:
                if info["followups"] >= self.max_followups:
                    info["truncated"] = True
                    break
                later = after.copy()
                later.execute(second, validate_action=True)
                info["followups"] += 1
                consider(later, action, second)
                if best_key[0]:
                    break
            if info["followups"] >= self.max_followups:
                break
            if best_key[0]:
                break

        info["baseline_vp"] = baseline_vp
        info["best_line_vp"] = best_key[1]
        info["override"] = best_action is not None and best_key[1] > baseline_vp
        self.last_scan = info
        return best_action if info["override"] else None

    def decide(self, game, playable_actions):
        baseline = super().decide(game, playable_actions)
        return self.find_override(game, playable_actions, baseline) or baseline


def create_agent(color, checkpoint, seed):
    del seed
    config = json.loads(Path(checkpoint).read_text()) if checkpoint is not None else {}
    return TacticalVPAgent(color, config)
