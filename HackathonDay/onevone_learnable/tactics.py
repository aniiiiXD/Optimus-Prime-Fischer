"""Small deterministic winning-line guard. Strategic preferences stay learnable."""
from __future__ import annotations

from time import process_time
from features import prefix

SEARCH_NAMES = frozenset({"BUILD_CITY", "BUILD_SETTLEMENT", "BUILD_ROAD",
                          "PLAY_KNIGHT_CARD", "PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY",
                          "MARITIME_TRADE", "PLAY_ROAD_BUILDING"})
ORDER = {name: i for i, name in enumerate(("BUILD_CITY", "BUILD_SETTLEMENT", "PLAY_KNIGHT_CARD",
                                         "BUILD_ROAD", "PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY",
                                         "MARITIME_TRADE", "PLAY_ROAD_BUILDING"))}


def guaranteed_win(game, color):
    # Only our known VP is inspected; game.winning_color may reveal a sampled
    # opponent's private points and is therefore deliberately not consulted.
    return game.state.player_state[prefix(game, color)+"ACTUAL_VICTORY_POINTS"] >= game.vps_to_win


def winning_root(game, candidates, color, deadline, max_nodes=96, depth=3):
    """Return a legal root only when its known action sequence reaches target VP.

    No draws, dice, theft, or opponent replies are searched. The search is
    intentionally incomplete and cannot claim that an unsearched win is absent.
    """
    for c in candidates:
        if c.successor is not None and guaranteed_win(c.successor, color):
            return c.index, 0
    if game.state.player_state[prefix(game, color)+"ACTUAL_VICTORY_POINTS"] < 7:
        return None, 0
    nodes = 0
    seen = set()

    def visit(world, remaining):
        nonlocal nodes
        if guaranteed_win(world, color):
            return True
        if remaining == 0 or nodes >= max_nodes or process_time() >= deadline:
            return False
        if world.state.current_color() != color or world.state.current_prompt.name != "PLAY_TURN":
            return False
        own = prefix(world, color)
        key = (remaining, tuple(sorted(world.state.board.buildings.items())),
               tuple(sorted(world.state.board.roads.items())),
               tuple((k, v) for k, v in sorted(world.state.player_state.items()) if k.startswith(own)),
               world.state.is_road_building)
        if key in seen:
            return False
        seen.add(key)
        actions = sorted((a for a in world.playable_actions if a.action_type.name in SEARCH_NAMES),
                         key=lambda a: ORDER[a.action_type.name])
        for action in actions:
            if nodes >= max_nodes or process_time() >= deadline:
                break
            after = world.copy()
            after.execute(action, validate_action=False)
            nodes += 1
            if visit(after, remaining-1):
                return True
        return False

    for c in sorted(candidates, key=lambda c: ORDER.get(c.action.action_type.name, 100)):
        if process_time() >= deadline or nodes >= max_nodes:
            break
        if c.successor is not None and c.action.action_type.name in SEARCH_NAMES:
            if visit(c.successor, depth-1):
                return c.index, nodes
    return None, nodes
