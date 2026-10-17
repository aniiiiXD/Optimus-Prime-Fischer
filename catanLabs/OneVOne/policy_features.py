"""Visible-information state and legal-action features for the 1v1 MLP policy.

The encoder deliberately reads only the acting player's private hand, public
board/player state, the public current trade, and legal actions. It does not
inspect opponent resource identities, development deck contents, or future RNG.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


FEATURES_VERSION = "onevone-visible-v1"
RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEV_CARDS = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")
PROMPTS = (
    "BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD", "PLAY_TURN",
    "DISCARD", "MOVE_ROBBER", "DECIDE_TRADE", "DECIDE_ACCEPTEES",
)
ACTION_TYPES = (
    "ROLL", "MOVE_ROBBER", "DISCARD_RESOURCE", "BUILD_ROAD",
    "BUILD_SETTLEMENT", "BUILD_CITY", "BUY_DEVELOPMENT_CARD",
    "PLAY_KNIGHT_CARD", "PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY",
    "PLAY_ROAD_BUILDING", "MARITIME_TRADE", "OFFER_TRADE",
    "ACCEPT_TRADE", "REJECT_TRADE", "CONFIRM_TRADE", "CANCEL_TRADE",
    "END_TURN",
)
FAMILY_NAMES = ("build", "robber", "trade", "development", "turn")


def _name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "name", value)).upper()


def edge_key(edge: Any) -> tuple[int, int]:
    return tuple(sorted((int(edge[0]), int(edge[1]))))


def board_layout(board_map: Any) -> dict[str, Any]:
    """Return stable indices for the fixed BASE board geometry."""
    nodes = tuple(sorted(int(node) for node in board_map.land_nodes))
    tiles = tuple(sorted(board_map.land_tiles.keys(), key=repr))
    edges = tuple(sorted({
        edge_key(edge)
        for tile in board_map.land_tiles.values()
        for edge in tile.edges.values()
    }))
    if (len(nodes), len(tiles), len(edges)) != (54, 19, 72):
        raise ValueError(
            "The v1 MLP encoder expects the 54-node, 19-tile, 72-edge BASE map; "
            f"got {(len(nodes), len(tiles), len(edges))}."
        )
    return {
        "nodes": nodes,
        "node_index": {node: i for i, node in enumerate(nodes)},
        "tiles": tiles,
        "tile_index": {tile: i for i, tile in enumerate(tiles)},
        "edges": edges,
        "edge_index": {edge: i for i, edge in enumerate(edges)},
    }


def _one_hot(size: int, index: int | None) -> list[float]:
    values = [0.0] * size
    if index is not None and 0 <= index < size:
        values[index] = 1.0
    return values


def _state_prefix(state: Any, color: Any) -> str:
    return f"P{state.color_to_index[color]}_"


def _player_num_cards(state: Any, color: Any) -> int:
    # Import lazily; callers install the pinned Catanatron source first.
    from catanatron.state_functions import player_num_resource_cards
    return int(player_num_resource_cards(state, color))


def encode_state(game: Any, color: Any) -> np.ndarray:
    """Encode one player's legal observation as a fixed-length float vector."""
    state = game.state
    board = state.board
    layout = board_layout(board.map)
    enemy = next(player for player in state.colors if player != color)
    own_prefix = _state_prefix(state, color)
    enemy_prefix = _state_prefix(state, enemy)

    values: list[float] = []
    prompt = getattr(state.current_prompt, "name", str(state.current_prompt))
    values.extend(_one_hot(len(PROMPTS), PROMPTS.index(prompt) if prompt in PROMPTS else None))

    # Each tile: resource one-hot (including desert), number one-hot, robber flag.
    resource_names = (*RESOURCES, "DESERT")
    robber_coordinate = board.robber_coordinate
    for coordinate in layout["tiles"]:
        tile = board.map.land_tiles[coordinate]
        resource = _name(tile.resource) or "DESERT"
        values.extend(_one_hot(len(resource_names), resource_names.index(resource)
                               if resource in resource_names else None))
        number = getattr(tile, "number", None)
        values.extend(_one_hot(11, int(number) - 2 if number is not None else None))
        values.append(float(coordinate == robber_coordinate))

    # Each intersection: empty/own settlement/own city/enemy settlement/enemy city,
    # buildability, then port type (3:1 or one of the five specific ports).
    node_owner_kind = [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in layout["nodes"]]
    for node, (owner, kind) in board.buildings.items():
        idx = layout["node_index"][int(node)]
        label = _name(kind) or str(kind).upper()
        if owner == color:
            category = 2 if "CITY" in label else 1
        else:
            category = 4 if "CITY" in label else 3
        node_owner_kind[idx] = _one_hot(5, category)
    buildable = set(board.buildable_node_ids(color, True))
    ports = board.map.port_nodes
    port_names = ("3_TO_1", *RESOURCES)
    port_by_node: dict[int, int] = {}
    for resource, nodes in ports.items():
        port_name = "3_TO_1" if resource is None else _name(resource)
        if port_name not in port_names:
            continue
        port_slot = port_names.index(port_name)
        for node in nodes:
            port_by_node[int(node)] = port_slot
    for idx, node in enumerate(layout["nodes"]):
        values.extend(node_owner_kind[idx])
        values.append(float(node in buildable))
        values.extend(_one_hot(len(port_names), port_by_node.get(node)))

    # Each road edge: empty/ours/opponent.
    roads = {edge_key(edge): owner for edge, owner in board.roads.items()}
    for edge in layout["edges"]:
        owner = roads.get(edge)
        category = 0 if owner is None else (1 if owner == color else 2)
        values.extend(_one_hot(3, category))

    # Resource production is an expected number of cards per roll, with the robber
    # blocking the tile. Keep resource identities so the value head can learn recipes.
    production: dict[Any, dict[str, float]] = {
        player: dict.fromkeys(RESOURCES, 0.0) for player in state.colors
    }
    robber_tile = board.map.land_tiles[robber_coordinate]
    for node, (owner, kind) in board.buildings.items():
        yield_count = 2 if "CITY" in (_name(kind) or str(kind).upper()) else 1
        for tile in board.map.adjacent_tiles[int(node)]:
            if tile == robber_tile:
                continue
            resource = _name(tile.resource)
            number = getattr(tile, "number", None)
            if resource in RESOURCES and number is not None:
                probability = (6 - abs(7 - int(number))) / 36.0
                production[owner][resource] += probability * yield_count
    for player in (color, enemy):
        values.extend(min(2.0, production[player][r]) / 2.0 for r in RESOURCES)

    # Private information is limited to our hand/dev cards. Opponent resource
    # identities and hidden victory points are intentionally omitted.
    values.extend(min(15.0, _player_num_cards(state, color)) / 15.0 for _ in [0])
    for resource in RESOURCES:
        amount = state.player_state[own_prefix + resource + "_IN_HAND"]
        values.append(min(15.0, float(amount)) / 15.0)
    for card in DEV_CARDS:
        amount = state.player_state[own_prefix + card + "_IN_HAND"]
        values.append(min(10.0, float(amount)) / 10.0)
    values.append(min(15.0, _player_num_cards(state, enemy)) / 15.0)

    def scalar(prefix: str, key: str, maximum: float) -> float:
        return min(maximum, float(state.player_state.get(prefix + key, 0))) / maximum

    own_public = scalar(own_prefix, "VICTORY_POINTS", 10)
    own_actual = scalar(own_prefix, "ACTUAL_VICTORY_POINTS", 15)
    enemy_public = scalar(enemy_prefix, "VICTORY_POINTS", 10)
    values.extend((own_public, own_actual, enemy_public,
                   max(0.0, own_actual - own_public)))
    values.extend((
        scalar(own_prefix, "LONGEST_ROAD_LENGTH", 15),
        scalar(enemy_prefix, "LONGEST_ROAD_LENGTH", 15),
        scalar(own_prefix, "PLAYED_KNIGHT", 10),
        scalar(enemy_prefix, "PLAYED_KNIGHT", 10),
        scalar(own_prefix, "ROADS_AVAILABLE", 15),
        scalar(own_prefix, "SETTLEMENTS_AVAILABLE", 5),
        scalar(own_prefix, "CITIES_AVAILABLE", 4),
        scalar(enemy_prefix, "ROADS_AVAILABLE", 15),
        scalar(enemy_prefix, "SETTLEMENTS_AVAILABLE", 5),
        scalar(enemy_prefix, "CITIES_AVAILABLE", 4),
        scalar(own_prefix, "HAS_ROAD", 1),
        scalar(own_prefix, "HAS_ARMY", 1),
        scalar(own_prefix, "HAS_ROLLED", 1),
        scalar(own_prefix, "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN", 1),
        min(float(state.num_turns), 200.0) / 200.0,
    ))

    # Current domestic offer is public when answering or confirming a trade.
    current_trade = getattr(state, "current_trade", None)
    if current_trade is not None and len(current_trade) >= 10:
        values.extend(min(8.0, float(x)) / 8.0 for x in current_trade[:10])
    else:
        values.extend([0.0] * 10)

    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("State encoder produced a non-finite feature")
    return result


def action_family(action_type_name: str) -> int:
    if action_type_name in {"BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY"}:
        return 0
    if action_type_name in {"MOVE_ROBBER", "DISCARD_RESOURCE"}:
        return 1
    if action_type_name in {
        "MARITIME_TRADE", "OFFER_TRADE", "ACCEPT_TRADE", "REJECT_TRADE",
        "CONFIRM_TRADE", "CANCEL_TRADE",
    }:
        return 2
    if action_type_name in {
        "BUY_DEVELOPMENT_CARD", "PLAY_KNIGHT_CARD", "PLAY_YEAR_OF_PLENTY",
        "PLAY_MONOPOLY", "PLAY_ROAD_BUILDING",
    }:
        return 3
    return 4


ACTION_FEATURE_DIM = (
    len(ACTION_TYPES) + len(FAMILY_NAMES) + 54 + 72 + 19
    + 5 + 5 + 5 + 5 + 2
)


def encode_action(action: Any, board_map: Any) -> tuple[np.ndarray, int]:
    """Encode a legal action's type, target, resource choices, and trade terms."""
    layout = board_layout(board_map)
    kind = action.action_type.name
    family = action_family(kind)
    values: list[float] = []
    values.extend(_one_hot(len(ACTION_TYPES), ACTION_TYPES.index(kind)
                           if kind in ACTION_TYPES else None))
    values.extend(_one_hot(len(FAMILY_NAMES), family))
    node_bits = [0.0] * 54
    edge_bits = [0.0] * 72
    tile_bits = [0.0] * 19
    resource_a = [0.0] * 5
    resource_b = [0.0] * 5
    trade_give = [0.0] * 5
    trade_get = [0.0] * 5
    steals = 0.0
    has_player_target = 0.0

    value = action.value
    if kind in {"BUILD_SETTLEMENT", "BUILD_CITY"} and value is not None:
        node_bits[layout["node_index"][int(value)]] = 1.0
    elif kind == "BUILD_ROAD" and value is not None:
        edge_bits[layout["edge_index"][edge_key(value)]] = 1.0
    elif kind == "MOVE_ROBBER" and value is not None:
        coordinate, victim = value
        tile_bits[layout["tile_index"][coordinate]] = 1.0
        steals = float(victim is not None)
    elif kind in {"DISCARD_RESOURCE", "PLAY_MONOPOLY"}:
        resource = _name(value)
        if resource in RESOURCES:
            resource_a[RESOURCES.index(resource)] = 1.0
    elif kind == "PLAY_YEAR_OF_PLENTY":
        for i, resource_value in enumerate(value or ()):
            resource = _name(resource_value)
            if resource in RESOURCES:
                (resource_a if i == 0 else resource_b)[RESOURCES.index(resource)] = 1.0
    elif kind == "MARITIME_TRADE" and value is not None:
        # The first four slots are the resources given; the fifth is requested.
        for resource_value in value[:4]:
            resource = _name(resource_value)
            if resource in RESOURCES:
                trade_give[RESOURCES.index(resource)] += 0.25
        if len(value) >= 5:
            resource = _name(value[4])
            if resource in RESOURCES:
                trade_get[RESOURCES.index(resource)] = 1.0
    elif kind in {"OFFER_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "CONFIRM_TRADE"}:
        # Domestic offers are two five-resource frequency decks.
        if isinstance(value, (tuple, list)) and len(value) >= 10:
            for i in range(5):
                trade_give[i] = min(8.0, float(value[i])) / 8.0
                trade_get[i] = min(8.0, float(value[i + 5])) / 8.0
            has_player_target = float(len(value) > 10 and value[10] is not None)

    values.extend(node_bits)
    values.extend(edge_bits)
    values.extend(tile_bits)
    values.extend(resource_a)
    values.extend(resource_b)
    values.extend(trade_give)
    values.extend(trade_get)
    values.extend((steals, has_player_target))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (ACTION_FEATURE_DIM,):
        raise AssertionError(f"Action feature shape {result.shape}, expected {ACTION_FEATURE_DIM}")
    return result, family


def encode_legal_actions(playable_actions: list[Any], board_map: Any) -> tuple[np.ndarray, np.ndarray]:
    encoded = [encode_action(action, board_map) for action in playable_actions]
    return (
        np.stack([item[0] for item in encoded]),
        np.asarray([item[1] for item in encoded], dtype=np.int64),
    )
