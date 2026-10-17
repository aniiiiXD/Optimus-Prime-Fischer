"""Versioned, visible-information features shared by experts and learned scorers.

Opponent resource identities are public in this official two-player ruleset.
Opponent unplayed dev identities, private VP, deck order and RNG are never read.
"""
from __future__ import annotations

import numpy as np
from catanatron.features import build_production_features, reachability_features
from catanatron.models.board import STATIC_GRAPH
from catanatron.players.value import value_production

SCHEMA = "onevone-visible-v2"
RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEVS = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")
ACTIONS = (
    "ROLL", "END_TURN", "BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY",
    "MARITIME_TRADE", "BUY_DEVELOPMENT_CARD", "PLAY_YEAR_OF_PLENTY",
    "PLAY_ROAD_BUILDING", "PLAY_KNIGHT_CARD", "PLAY_MONOPOLY",
    "MOVE_ROBBER", "DISCARD_RESOURCE",
)
COMPONENT_NAMES = (
    "public_vp", "production", "enemy_production", "reachable_production_0",
    "reachable_production_1", "hand_synergy", "buildable_nodes", "num_tiles",
    "hand_resources", "discard_risk", "longest_road", "hand_devs", "army_size",
)
STATE_NAMES = (
    *COMPONENT_NAMES, "actual_vp", "opponent_public_vp", "opponent_dev_count",
    *("hand_" + r.lower() for r in RESOURCES),
    *("opponent_hand_" + r.lower() for r in RESOURCES),
    *("production_" + r.lower() for r in RESOURCES),
    *("opponent_production_" + r.lower() for r in RESOURCES),
    *("dev_" + d.lower() for d in DEVS),
    "opponent_army", "opponent_road", "roads_available", "settlements_available",
    "cities_available", "elapsed_turns", "has_rolled", "initial_placement",
)
STATE_SCALES = np.asarray(
    [10, 2, 2, 2, 2, 1, 10, 19, 10, 1, 15, 5, 6, 10, 10, 10]
    + [10] * 10 + [1] * 10 + [5] * 5
    + [6, 15, 15, 5, 4, 100, 1, 1], dtype=np.float32,
)
EXTRA_NAMES = ("expert_rank", "tactical_gain_div3", "robber_utility_div5",
               "expected_unknown_card", "is_stochastic", "target_node_div53")
CANDIDATE_NAMES = (
    *("after_" + n for n in STATE_NAMES),
    *("delta_" + n for n in STATE_NAMES),
    *("action_" + n.lower() for n in ACTIONS), *EXTRA_NAMES,
    *(f"node_{i}" for i in range(54)),
)
STATE_DIM = len(STATE_NAMES)
CANDIDATE_DIM = len(CANDIDATE_NAMES)
BOARD_DIM, TOKENS = 32, 55
PRODUCTION_FN = build_production_features(True)
NEIGHBORS = tuple(tuple(sorted(n for n in STATIC_GRAPH.neighbors(i) if n < 54)) for i in range(54))
ADJACENCY = np.eye(TOKENS, dtype=np.float32)
for i, neighbors in enumerate(NEIGHBORS):
    ADJACENCY[i, list(neighbors)] = 1


def prefix(game, color):
    return f"P{game.state.color_to_index[color]}_"


def state_features(game, color):
    state, board, ps = game.state, game.state.board, game.state.player_state
    opponent = next(c for c in state.colors if c != color)
    own, other = prefix(game, color), prefix(game, opponent)
    production = PRODUCTION_FN(game, color)
    reach = reachability_features(game, color, 2)
    hand = [ps[own + r + "_IN_HAND"] for r in RESOURCES]
    synergy = 1 - (sum(max(1 - hand[i], 0) for i in (0, 1, 2, 3)) / 4
                   + (max(2 - hand[3], 0) + max(3 - hand[4], 0)) / 5) / 2
    nodes = state.buildings_by_color[color]["SETTLEMENT"] + state.buildings_by_color[color]["CITY"]
    tiles = set(t for n in nodes for t in board.map.adjacent_tiles[n])
    components = [
        ps[own + "VICTORY_POINTS"], value_production(production, "P0"),
        value_production(production, "P1", False),
        sum(reach[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES),
        sum(reach[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES),
        synergy, len(board.buildable_node_ids(color)), len(tiles), sum(hand),
        float(sum(hand) > 7), ps[own + "LONGEST_ROAD_LENGTH"],
        sum(ps[own + d + "_IN_HAND"] for d in DEVS), ps[own + "PLAYED_KNIGHT"],
    ]
    values = components + [ps[own + "ACTUAL_VICTORY_POINTS"], ps[other + "VICTORY_POINTS"],
                           sum(ps[other + d + "_IN_HAND"] for d in DEVS)]
    values += hand + [ps[other + r + "_IN_HAND"] for r in RESOURCES]
    values += [production[f"EFFECTIVE_P0_{r}_PRODUCTION"] for r in RESOURCES]
    values += [production[f"EFFECTIVE_P1_{r}_PRODUCTION"] for r in RESOURCES]
    values += [ps[own + d + "_IN_HAND"] for d in DEVS]
    values += [ps[other + "PLAYED_KNIGHT"], ps[other + "LONGEST_ROAD_LENGTH"],
               ps[own + "ROADS_AVAILABLE"], ps[own + "SETTLEMENTS_AVAILABLE"],
               ps[own + "CITIES_AVAILABLE"], state.num_turns,
               bool(ps[own + "HAS_ROLLED"]), bool(state.is_initial_build_phase)]
    # Keep expert arithmetic in float64: the legacy VP coefficient is 3e14.
    # Casting its sum to float32 would erase production/hand distinctions.
    return np.asarray(values, dtype=np.float64)


def candidate_features(before, after, action, expert_rank, tactical_gain=0,
                       robber_utility=0, unknown_card=False, stochastic=False):
    nodes = np.zeros(54, dtype=np.float32)
    name = action.action_type.name
    marked = ()
    if name in ("BUILD_SETTLEMENT", "BUILD_CITY"):
        marked = (int(action.value),)
    elif name == "BUILD_ROAD":
        marked = tuple(action.value)
    for node in marked:
        nodes[node] = 1 / len(marked)
    one_hot = np.asarray([name == kind for kind in ACTIONS], dtype=np.float32)
    extra = [expert_rank, tactical_gain / 3, robber_utility / 5,
             float(unknown_card), float(stochastic), sum(marked) / max(1, len(marked)) / 53]
    return np.concatenate((after / STATE_SCALES, (after - before) / STATE_SCALES,
                           one_hot, np.asarray(extra, dtype=np.float32), nodes)).astype(np.float32)


def encode_board(game, color):
    """54 graph-node tokens plus one global token, relative to our seat.

    Layout follows the independently developed GoalEconomy visible board encoder;
    local road ownership uses get_edge_color to handle both edge orientations.
    """
    state, board, ps = game.state, game.state.board, game.state.player_state
    opponent = next(c for c in state.colors if c != color)
    own, other = prefix(game, color), prefix(game, opponent)
    x = np.zeros((TOKENS, BOARD_DIM), dtype=np.float32)
    own_sites, other_sites = set(board.buildable_node_ids(color)), set(board.buildable_node_ids(opponent))
    blocked = board.map.land_tiles[board.robber_coordinate]
    blocked_nodes = set(blocked.nodes.values())
    probability = (6 - abs(7 - blocked.number)) / 36 if blocked.resource else 0
    own_network, other_network = set(), set()
    for edge, owner in board.roads.items():
        (own_network if owner == color else other_network).update(edge)
    for node in range(54):
        for j, resource in enumerate(RESOURCES):
            x[node, j] = 3 * board.map.node_production[node].get(resource, 0)
            x[node, 5+j] = 3 * probability if node in blocked_nodes and resource == blocked.resource else 0
        building = board.buildings.get(node)
        if building:
            owner, kind = building
            x[node, 10 + (0 if owner == color else 2) + (kind == "CITY")] = 1
        x[node, 14:16] = [node in own_sites, node in other_sites]
        for j, resource in enumerate((None,) + RESOURCES):
            x[node, 16+j] = node in board.map.port_nodes.get(resource, ())
        for j, neighbor in enumerate(NEIGHBORS[node]):
            owner = board.get_edge_color((node, neighbor))
            x[node, 22+j], x[node, 25+j] = owner == color, owner == opponent
        x[node, 28:32] = [len(NEIGHBORS[node])/3, node/53, node in own_network, node in other_network]
    g = x[-1]
    g[0:2] = [ps[own + "ACTUAL_VICTORY_POINTS"]/10, ps[other + "VICTORY_POINTS"]/10]
    g[2:7] = [ps[own + r + "_IN_HAND"]/10 for r in RESOURCES]
    g[7:12] = [ps[other + r + "_IN_HAND"]/10 for r in RESOURCES]
    g[12:18] = [(15-ps[own + "ROADS_AVAILABLE"])/15, (15-ps[other + "ROADS_AVAILABLE"])/15,
                 ps[own + "LONGEST_ROAD_LENGTH"]/15, ps[other + "LONGEST_ROAD_LENGTH"]/15,
                 ps[own + "PLAYED_KNIGHT"]/6, ps[other + "PLAYED_KNIGHT"]/6]
    g[18:22] = [ps[own + "HAS_ARMY"], ps[other + "HAS_ARMY"], ps[own + "HAS_ROAD"], ps[other + "HAS_ROAD"]]
    g[22:27] = [ps[own + d + "_IN_HAND"]/5 for d in DEVS]
    g[27:32] = [state.num_turns/100, state.current_color() == color,
                  ps[own + "CITIES_AVAILABLE"]/4, ps[own + "SETTLEMENTS_AVAILABLE"]/5,
                  sum(ps[other + d + "_IN_HAND"] for d in DEVS)/10]
    return x
