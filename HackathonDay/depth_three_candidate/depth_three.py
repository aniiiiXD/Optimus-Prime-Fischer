"""Clock-bounded search for points after up to three known actions this turn.

Only actions with deterministic outcomes are expanded. In particular, this
planner never reads a hidden development draw or resolves dice/robber theft.
"""

from __future__ import annotations

from time import process_time

from catanatron.models.enums import ActionType


# Prep actions can turn a poor-looking immediate move into a point build.
PREP_TYPES = frozenset({
    ActionType.MARITIME_TRADE,
    ActionType.BUILD_ROAD,
    ActionType.PLAY_ROAD_BUILDING,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_MONOPOLY,
})
POINT_TYPES = frozenset({
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_CITY,
    ActionType.BUILD_ROAD,  # May claim Longest Road.
})
SEARCH_TYPES = PREP_TYPES | POINT_TYPES
ORDER = {
    # A free road card can hide a two-road Longest Road win. Explore it before
    # broad trade branches consume the search clock.
    ActionType.PLAY_ROAD_BUILDING: 0,
    ActionType.PLAY_YEAR_OF_PLENTY: 1,
    ActionType.MARITIME_TRADE: 2,
    ActionType.BUILD_ROAD: 3,
    ActionType.PLAY_MONOPOLY: 4,
    ActionType.BUILD_SETTLEMENT: 5,
    ActionType.BUILD_CITY: 6,
}


def _points(game, color):
    key = f"P{game.state.color_to_index[color]}_ACTUAL_VICTORY_POINTS"
    return int(game.state.player_state[key])


def _ordered(actions, allowed):
    return sorted(
        (action for action in actions if action.action_type in allowed),
        key=lambda action: (ORDER[action.action_type], repr(action.value)),
    )


def depth_three_override(game, color, playable_actions, baseline, cpu_left):
    """Return a first move with more guaranteed VP than baseline, if found.

    The existing one/two-action policy runs first. Search is limited to 24 ms
    and 240 simulated actions per decision, with a CPU reserve for the rest of
    the game. The caller keeps its original choice if no better line is found.
    """
    if (game.state.current_color() != color
            or game.state.current_prompt.name != "PLAY_TURN"
            or len(playable_actions) < 2):
        return None
    prefix = f"P{game.state.color_to_index[color]}_"
    if not game.state.player_state[prefix + "HAS_ROLLED"]:
        return None
    if cpu_left is not None and cpu_left < 0.25:
        return None

    budget = 0.024 if cpu_left is None else min(0.024, max(0.0, cpu_left - 0.22) / 4)
    if budget < 0.004:
        return None
    deadline = process_time() + budget
    best_vp = _points(game, color)
    # An unknown development draw is not a guaranteed point. All searched
    # actions have deterministic outcomes and can be scored exactly.
    if baseline.action_type in SEARCH_TYPES:
        after_baseline = game.copy()
        after_baseline.execute(baseline, validate_action=True)
        best_vp = _points(after_baseline, color)
    if best_vp >= game.vps_to_win:
        return None

    best_root = None
    nodes = 0
    roots = _ordered(playable_actions, SEARCH_TYPES)
    if baseline.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        # At nine points, the old policy can prefer an uncertain hidden draw
        # over a guaranteed trade/build win. Check trades before free roads.
        roots.sort(key=lambda action: (action.action_type != ActionType.MARITIME_TRADE,
                                       ORDER[action.action_type], repr(action.value)))
    for root in roots[:40]:
        if nodes >= 240 or process_time() >= deadline:
            break
        after = game.copy()
        after.execute(root, validate_action=True)
        nodes += 1
        vp = _points(after, color)
        if vp > best_vp:
            best_vp, best_root = vp, root
        if after.winning_color() == color:
            return root
        if after.state.current_color() != color:
            continue

        root_nodes = 0
        root_limit = 160 if root.action_type == ActionType.PLAY_ROAD_BUILDING else 24
        for second in _ordered(after.playable_actions, SEARCH_TYPES):
            if root_nodes >= root_limit or nodes >= 240 or process_time() >= deadline:
                break
            later = after.copy()
            later.execute(second, validate_action=True)
            nodes += 1
            root_nodes += 1
            vp = _points(later, color)
            if vp > best_vp:
                best_vp, best_root = vp, root
            if later.winning_color() == color:
                return root
            if later.state.current_color() != color:
                continue

            for third in _ordered(later.playable_actions, POINT_TYPES):
                if root_nodes >= root_limit or nodes >= 240 or process_time() >= deadline:
                    break
                final = later.copy()
                final.execute(third, validate_action=True)
                nodes += 1
                root_nodes += 1
                vp = _points(final, color)
                if vp > best_vp:
                    best_vp, best_root = vp, root
                if final.winning_color() == color:
                    return root

    return best_root
