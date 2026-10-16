"""Bounded, chance-free same-turn search for an immediate winning sequence.

Only deterministic own-turn actions are expanded. This never reads the
opponent's resource hand or development cards, and never predicts dice.
"""

from __future__ import annotations

from collections import deque

from catanatron.state_functions import get_actual_victory_points

EXPAND = {"BUILD_CITY", "BUILD_SETTLEMENT", "BUILD_ROAD", "MARITIME_TRADE"}
PRIORITY = {"BUILD_CITY": 0, "BUILD_SETTLEMENT": 1, "BUILD_ROAD": 2, "MARITIME_TRADE": 3}


def find_immediate_win(game, color, max_depth: int = 4, max_nodes: int = 80):
    """Return the first move of a found win before ending this turn, or None."""
    if get_actual_victory_points(game.state, color) < 6:
        return None
    queue = deque([(game, None, 0)])
    visited = 0
    while queue and visited < max_nodes:
        state, first, depth = queue.popleft()
        visited += 1
        if depth >= max_depth or state.state.current_color() != color:
            continue
        options = sorted(
            (action for action in state.playable_actions if action.action_type.name in EXPAND),
            key=lambda action: (PRIORITY[action.action_type.name], repr(action.value)),
        )
        for action in options:
            if visited + len(queue) >= max_nodes:
                break
            child = state.copy()
            child.execute(action, validate_action=True)
            root_action = action if first is None else first
            if child.winning_color() == color:
                return root_action
            queue.append((child, root_action, depth + 1))
    return None
