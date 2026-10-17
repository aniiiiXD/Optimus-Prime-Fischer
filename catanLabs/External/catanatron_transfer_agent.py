"""Experimental, observation-only port of the upstream PPO policy to Catanatron.

The two engines label the same 19/54/72 board graph differently. We compute
an explicit graph isomorphism, translate public features and legal actions,
and mask every action that Catanatron does not offer. This is a *policy* port;
the upstream hidden-state rollout search cannot be used fairly here.
"""

from __future__ import annotations

from functools import lru_cache

import networkx as nx
import numpy as np
import torch
from torch.nn import functional as F

from catanLabs.Competitive.agent import CompetitiveAgent

RUST_RESOURCES = ("WHEAT", "SHEEP", "WOOD", "BRICK", "ORE")
RUST_DEV = ("KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY")
PIPS = (0, 0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1)
OBS_DIM = 1350
ACTION_DIM = 299


@lru_cache(maxsize=1)
def topology_map():
    """Return Rust tile/node/edge IDs mapped to Catanatron IDs.

    Include vertex adjacency in the isomorphism, so a matched tile alone
    cannot silently scramble its six corners. Verify all 72 mapped edges.
    """
    from catanatron.models.board import base_map
    from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS

    rust, catan = nx.Graph(), nx.Graph()
    corners = ((0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1))
    seen = {}
    rust_edges = set()
    tile_id = 0
    for row, count in enumerate((3, 4, 5, 4, 3)):
        cy = 3 * (row - 2)
        x0 = -(count - 1)
        for col in range(count):
            verts = []
            for dx, dy in corners:
                key = (x0 + 2 * col + dx, cy + dy)
                if key not in seen:
                    seen[key] = len(seen)
                verts.append(seen[key])
            rust.add_node(("t", tile_id), kind="tile")
            for i, vert in enumerate(verts):
                nxt = verts[(i + 1) % 6]
                rust.add_node(("v", vert), kind="vertex")
                rust.add_edge(("t", tile_id), ("v", vert))
                rust.add_edge(("v", vert), ("v", nxt))
                rust_edges.add(tuple(sorted((vert, nxt))))
            tile_id += 1

    catan_edges = set()
    for tile in base_map.land_tiles.values():
        catan.add_node(("t", tile.id), kind="tile")
        for vert in tile.nodes.values():
            catan.add_node(("v", vert), kind="vertex")
            catan.add_edge(("t", tile.id), ("v", vert))
        for a, b in tile.edges.values():
            catan.add_edge(("v", a), ("v", b))
            catan_edges.add(tuple(sorted((a, b))))

    match = nx.algorithms.isomorphism.GraphMatcher(
        rust, catan, node_match=lambda a, b: a["kind"] == b["kind"]
    )
    catan_port_pairs = set()
    for port in base_map.ports_by_id.values():
        a, b = PORT_DIRECTION_TO_NODEREFS[port.direction]
        catan_port_pairs.add(tuple(sorted((port.nodes[a], port.nodes[b]))))
    rust_port_pairs = ((0, 1), (7, 10), (12, 22), (35, 36), (45, 46),
                       (50, 53), (47, 48), (39, 40), (27, 28))
    # There are 12 rotations/reflections. Pick the one that preserves the
    # most fixed port sites, reducing a source-engine distribution shift.
    candidates = list(match.isomorphisms_iter())
    if not candidates:
        raise ValueError("Rust and Catanatron board graphs differ")
    mapping = max(candidates, key=lambda candidate: sum(
        tuple(sorted((candidate[("v", a)][1], candidate[("v", b)][1])))
        in catan_port_pairs for a, b in rust_port_pairs
    ))
    tiles = tuple(mapping[("t", i)][1] for i in range(19))
    vertices = tuple(mapping[("v", i)][1] for i in range(54))
    edges = tuple(tuple(sorted((vertices[a], vertices[b]))) for a, b in sorted(rust_edges))
    if len(set(tiles)) != 19 or len(set(vertices)) != 54 or set(edges) != catan_edges:
        raise ValueError("Incomplete Rust/Catanatron board mapping")
    return tiles, vertices, edges


def _pair_index(give, recv):
    return give * 4 + (recv if recv < give else recv - 1)


def _yop_index(first, second):
    first, second = sorted((first, second))
    return first * (11 - first) // 2 + second - first


def action_id(action, game, tile_inverse, vertex_inverse, edge_inverse):
    """Return an upstream action ID for a Catanatron legal move."""
    kind, value = action.action_type.name, action.value
    if kind == "BUILD_SETTLEMENT":
        return vertex_inverse[value]
    if kind == "BUILD_CITY":
        return 54 + vertex_inverse[value]
    if kind == "BUILD_ROAD":
        return 108 + edge_inverse[tuple(sorted(value))]
    if kind == "MOVE_ROBBER":
        tile = game.state.board.map.land_tiles[value[0]]
        return 180 + tile_inverse[tile.id]
    if kind == "DISCARD_RESOURCE":
        return 203 + RUST_RESOURCES.index(value)
    if kind == "PLAY_MONOPOLY":
        return 208 + RUST_RESOURCES.index(value)
    if kind == "PLAY_YEAR_OF_PLENTY":
        resources = [RUST_RESOURCES.index(r) for r in value]
        if len(resources) == 1:  # Catanatron allows a partial draw from a nearly empty bank.
            resources *= 2
        return 213 + _yop_index(*resources)
    if kind == "MARITIME_TRADE":
        give = next(r for r in value[:-1] if r is not None)
        recv = value[-1]
        return 228 + _pair_index(RUST_RESOURCES.index(give), RUST_RESOURCES.index(recv))
    constants = {"ROLL": 294, "BUY_DEVELOPMENT_CARD": 295,
                 "PLAY_KNIGHT_CARD": 296, "PLAY_ROAD_BUILDING": 297, "END_TURN": 298}
    if kind in constants:
        return constants[kind]
    raise ValueError(f"Unmapped legal Catanatron action: {kind}")


def encode_visible(game, tiles, vertices, edges):
    """Upstream-shaped observation using only public data and own private data.

    Opponents' resource and development *counts* are public tabletop data.
    Catanatron lacks a count-only API, so summation is isolated below; the
    model never sees the private composition or hidden development deck.
    """
    state = game.state
    board = state.board
    map_obj = board.map
    colors = state.colors
    seat = state.color_to_index[state.current_color()]
    out = np.zeros(OBS_DIM, dtype=np.float32)
    tile_by_id = {tile.id: tile for tile in map_obj.land_tiles.values()}
    robber_id = map_obj.land_tiles[board.robber_coordinate].id
    terrain = {name: i for i, name in enumerate(RUST_RESOURCES)}

    for i, catan_id in enumerate(tiles):
        tile = tile_by_id[catan_id]
        base = i * 8
        out[base + terrain.get(tile.resource, 5)] = 1
        if tile.number is not None:
            out[base + 6] = PIPS[tile.number] / 5.0
        out[base + 7] = float(catan_id == robber_id)

    node_port = {}
    for resource, nodes in map_obj.port_nodes.items():
        for node in nodes:
            node_port[node] = 0 if resource is None else 1 + terrain[resource]
    for i, node in enumerate(vertices):
        base = 152 + i * 14
        building = board.buildings.get(node)
        if building:
            owner, building_type = building
            rel = (state.color_to_index[owner] - seat) % 4
            out[base + rel * 2 + int(building_type == "CITY")] = 1
        if node in node_port:
            out[base + 8 + node_port[node]] = 1

    for i, edge in enumerate(edges):
        owner = board.get_edge_color(edge)
        if owner is not None:
            rel = (state.color_to_index[owner] - seat) % 4
            out[908 + i * 4 + rel] = 1

    for rel in range(4):
        color = colors[(seat + rel) % 4]
        player = state.color_to_index[color]
        key = f"P{player}_"
        base = 1196 + rel * 17
        out[base] = sum(state.player_state[key + r + "_IN_HAND"] for r in RUST_RESOURCES) / 19
        out[base + 1] = sum(state.player_state[key + d + "_IN_HAND"] for d in RUST_DEV) / 25
        out[base + 2] = state.player_state[key + "PLAYED_KNIGHT"] / 14
        out[base + 3] = state.player_state[key + "VICTORY_POINTS"] / game.vps_to_win
        out[base + 4] = state.player_state[key + "SETTLEMENTS_AVAILABLE"] / 5
        out[base + 5] = state.player_state[key + "CITIES_AVAILABLE"] / 4
        out[base + 6] = state.player_state[key + "ROADS_AVAILABLE"] / 15
        out[base + 7] = state.player_state[key + "LONGEST_ROAD_LENGTH"] / 15
        out[base + 8] = float(state.player_state[key + "HAS_ROAD"])
        out[base + 9] = float(state.player_state[key + "HAS_ARMY"])
        owned_ports = board.get_player_port_resources(color)
        out[base + 10] = float(None in owned_ports)
        for resource in RUST_RESOURCES:
            out[base + 11 + terrain[resource]] = float(resource in owned_ports)
        out[base + 16] = float(player == state.current_turn_index)

    own_key = f"P{seat}_"
    for i, resource in enumerate(RUST_RESOURCES):
        out[1264 + i] = state.player_state[own_key + resource + "_IN_HAND"] / 19
        out[1309 + i] = state.resource_freqdeck[("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE").index(resource)] / 19
    for i, dev in enumerate(RUST_DEV):
        out[1269 + i] = state.player_state[own_key + dev + "_IN_HAND"] / 5

    prompt = state.current_prompt.name
    if state.is_initial_build_phase:
        total_settlements = sum(len(state.buildings_by_color[c]["SETTLEMENT"]) for c in colors)
        out[1314 + (0 if total_settlements < 4 else 1)] = 1
    else:
        out[1316] = 1
        if prompt == "DISCARD":
            phase = 2
        elif prompt == "MOVE_ROBBER":
            phase = 3
        elif state.is_road_building:
            phase = 6
        elif not state.player_state[f"P{state.current_turn_index}_HAS_ROLLED"]:
            phase = 0
        else:
            phase = 5
        out[1318 + phase] = 1
    out[1328] = float(state.player_state[f"P{state.current_turn_index}_HAS_ROLLED"])
    out[1329] = state.num_turns / 1000
    out[1330] = game.vps_to_win / 10
    out[1331] = state.free_roads_available / 2
    if prompt == "DISCARD":
        out[1333] = state.discard_counts[seat] / 10
    return out


class TransferredPPOAgent:
    def __init__(self, color, checkpoint):
        self.color = color
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["obs_dim"] != OBS_DIM or payload["num_actions"] != ACTION_DIM:
            raise ValueError("Incompatible upstream observation/action schema")
        self.weights = payload["model_state"]
        self.robber = CompetitiveAgent(color)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        tiles, vertices, edges = topology_map()
        tile_inverse = {value: key for key, value in enumerate(tiles)}
        vertex_inverse = {value: key for key, value in enumerate(vertices)}
        edge_inverse = {value: key for key, value in enumerate(edges)}
        obs = encode_visible(game, tiles, vertices, edges)
        state = self.weights
        with torch.inference_mode():
            h = torch.from_numpy(obs)
            h = F.relu(F.linear(h, state["trunk.0.weight"], state["trunk.0.bias"]))
            h = F.relu(F.linear(h, state["trunk.2.weight"], state["trunk.2.bias"]))
            logits = F.linear(h, state["policy.weight"], state["policy.bias"]).numpy()
        mapped = []
        for action in playable_actions:
            action_index = action_id(action, game, tile_inverse, vertex_inverse, edge_inverse)
            bonus = 0.0
            if action.action_type.name == "MOVE_ROBBER":
                # Rust asks for the victim in a separate action; Catanatron
                # combines that choice with tile placement.
                bonus = 0.01 * self.robber._robber_score(game, action)
            mapped.append((float(logits[action_index]) + bonus, action))
        return max(mapped, key=lambda pair: pair[0])[1]


def create_agent(color, checkpoint, seed):
    del seed
    if checkpoint is None:
        raise ValueError("--checkpoint must name an upstream .pt file")
    return TransferredPPOAgent(color, checkpoint)
