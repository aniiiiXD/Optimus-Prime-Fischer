"""CPU-only, visible-information transfer of Benjamin's policy to Catanatron.

The board graph is mapped by isomorphism. All candidate actions come from
Catanatron's legal list; the Benjamin network only ranks those candidates.
This is an engine transfer test, not Benjamin's native R1-rule evaluation.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external/benjamin_catan_rl"
sys.path.insert(0, str(ROOT.parent / "Hyde/start/catan_lab/vendor/catanatron/catanatron"))
sys.path.insert(0, str(UPSTREAM / "src"))

from catanatron.models.board import base_map  # noqa: E402
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS  # noqa: E402
from catanatron.state_functions import (  # noqa: E402
    get_actual_victory_points,
    get_dev_cards_in_hand,
    get_played_dev_cards,
    get_player_freqdeck,
    get_visible_victory_points,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)
from catan_rl.checkpoint import load_checkpoint  # noqa: E402
from catan_rl.policy.board_geometry import build_geometry  # noqa: E402
from catan_rl.engine.board import catanBoard  # noqa: E402
from catan_rl.policy.network import CatanPolicy  # noqa: E402
from catan_rl.policy.obs_schema import ActionType, HEAD_DIMS, MASK_KEYS, RESOURCES_CW  # noqa: E402

RESOURCE_ONEHOT = ("BRICK", "ORE", "SHEEP", "WHEAT", "WOOD", "DESERT")
TOKENS = (None, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
PIPS = {None: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
DEV = ("KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY")


@lru_cache(maxsize=1)
def topology():
    """Benjamin tile/vertex/edge index -> Catanatron tile/node/edge."""
    geometry = build_geometry()
    ben, cat = nx.Graph(), nx.Graph()
    for hi, verts in enumerate(geometry.hex_to_vertex.tolist()):
        ben.add_node(("h", hi), kind="h")
        for vi in verts:
            ben.add_node(("v", vi), kind="v")
            ben.add_edge(("h", hi), ("v", vi))
    for a, b in geometry.edge_to_vertex.tolist():
        ben.add_edge(("v", a), ("v", b))
    for tile in base_map.land_tiles.values():
        cat.add_node(("h", tile.id), kind="h")
        for node in tile.nodes.values():
            cat.add_node(("v", node), kind="v")
            cat.add_edge(("h", tile.id), ("v", node))
        for a, b in tile.edges.values():
            cat.add_edge(("v", a), ("v", b))
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        ben, cat, node_match=lambda a, b: a["kind"] == b["kind"]
    )
    ben_board = catanBoard()
    ben_ports = {i for i, pixel in ben_board.vertex_index_to_pixel_dict.items()
                 if ben_board.boardGraph[pixel].port}
    cat_ports = set()
    for port in base_map.ports_by_id.values():
        for ref in PORT_DIRECTION_TO_NODEREFS[port.direction]:
            cat_ports.add(port.nodes[ref])
    mapping = max(matcher.isomorphisms_iter(),
                  key=lambda candidate: sum(candidate[("v", i)][1] in cat_ports for i in ben_ports),
                  default=None)
    if mapping is None:
        raise RuntimeError("Benjamin/Catanatron board graphs differ")
    tiles = tuple(mapping[("h", i)][1] for i in range(19))
    nodes = tuple(mapping[("v", i)][1] for i in range(54))
    edges = tuple(tuple(sorted((nodes[a], nodes[b]))) for a, b in geometry.edge_to_vertex.tolist())
    assert len(set(tiles)) == 19 and len(set(nodes)) == 54 and len(set(edges)) == 72
    return tiles, nodes, edges


def _player_features(game, color, viewer):
    state, board = game.state, game.state.board
    own = color == viewer
    prefix = player_key(state, color)
    feats = np.zeros(67 if own else 69, dtype=np.float32)
    if own:
        counts = dict(zip(("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"), get_player_freqdeck(state, color)))
        for j, res in enumerate(RESOURCES_CW):
            feats[j] = min(1, counts[res] / 8)
        feats[5] = get_actual_victory_points(state, color) / 15
        feats[54] = min(1, sum(counts.values()) / 9)
        feats[55] = min(1, max(0, sum(counts.values()) - 7) / 2)
    else:
        feats[5] = get_visible_victory_points(state, color) / 15
        feats[54 + min(5, player_num_dev_cards(state, color))] = 1
        feats[60] = min(1, player_num_resource_cards(state, color) / 20)
    income = np.zeros(5, dtype=np.float32)
    node_to_tile = {}
    for tile in board.map.land_tiles.values():
        for n in tile.nodes.values():
            node_to_tile.setdefault(n, []).append(tile)
    for node, (owner, kind) in board.buildings.items():
        if owner != color:
            continue
        for tile in node_to_tile.get(node, ()):
            if tile.resource in RESOURCES_CW and tile.number:
                income[RESOURCES_CW.index(tile.resource)] += PIPS[tile.number] * (2 if kind == "CITY" else 1) / 36
    feats[6:11] = income
    if not own:
        # Catanatron exposes the rival's total hand size, but not its private
        # composition. Estimate composition from public production; never read
        # P{opponent}_*_IN_HAND. This is necessarily less exact than Benjamin's
        # broadcast hand tracker in its native engine.
        weights = income + 0.03
        estimate = player_num_resource_cards(state, color) * weights / weights.sum()
        feats[:5] = np.minimum(1, estimate / 8)
    ports = board.get_player_port_resources(color)
    for j, res in enumerate(RESOURCES_CW):
        feats[11 + j] = .5 if res in ports else (1 / 3 if None in ports else .25)
    feats[16] = float(None in ports)
    for j, res in enumerate(RESOURCES_CW):
        feats[17 + j] = float(res in ports)
    feats[22] = float(state.player_state[f"{prefix}_HAS_ROAD"])
    feats[23] = float(state.player_state[f"{prefix}_HAS_ARMY"])
    feats[24] = min(1, state.player_state[f"{prefix}_LONGEST_ROAD_LENGTH"] / 15)
    feats[25] = min(1, state.player_state[f"{prefix}_PLAYED_KNIGHT"] / 8)
    feats[26] = state.player_state[f"{prefix}_SETTLEMENTS_AVAILABLE"] / 5
    feats[27] = state.player_state[f"{prefix}_CITIES_AVAILABLE"] / 4
    feats[28] = state.player_state[f"{prefix}_ROADS_AVAILABLE"] / 15
    if own:
        for j, card in enumerate(DEV):
            feats[29 + j] = min(1, get_dev_cards_in_hand(state, color, card) / 8)
    prompt = state.current_prompt.name
    feats[34] = float(state.is_initial_build_phase)
    feats[35] = float(prompt == "PLAY_TURN" and not state.player_state[f"P{state.current_turn_index}_HAS_ROLLED"])
    feats[36] = float(not state.is_initial_build_phase and prompt == "PLAY_TURN" and not feats[35])
    feats[37] = float(prompt == "MOVE_ROBBER")
    feats[38] = float(prompt == "DISCARD")
    purchases = sum(record.action.action_type.name == "BUY_DEVELOPMENT_CARD"
                    for record in state.action_records)
    feats[39] = max(0, 25 - purchases) / 25
    for record in reversed(state.action_records):
        if record.action.action_type.name == "ROLL":
            value = record.result
            if isinstance(value, (list, tuple)):
                value = sum(value)
            if isinstance(value, int) and 2 <= value <= 12:
                feats[40 + value - 2] = 1
            break
    return feats


def encode_visible(game, viewer):
    """Encode public board, own hand, and public opponent totals only."""
    state, board = game.state, game.state.board
    tiles, nodes, edges = topology()
    tile_by_id = {tile.id: tile for tile in board.map.land_tiles.values()}
    buildings = board.buildings
    ports = {node: res for res, values in board.map.port_nodes.items() for node in values}
    intrinsic = np.zeros((19, 19), dtype=np.float32)
    vertex = np.zeros((54, 16), dtype=np.float32)
    edge = np.zeros((72, 16), dtype=np.float32)
    tile_rep = np.zeros((19, 79), dtype=np.float32)
    for i, tile_id in enumerate(tiles):
        tile = tile_by_id[tile_id]
        intrinsic[i, RESOURCE_ONEHOT.index(tile.resource or "DESERT")] = 1
        intrinsic[i, 6 + TOKENS.index(tile.number)] = 1
        intrinsic[i, 17] = float(tile.id == board.map.land_tiles[board.robber_coordinate].id)
        intrinsic[i, 18] = PIPS[tile.number] / 5
        tile_rep[i, :19] = intrinsic[i]
        # Align the tile's six corner slots to Benjamin geometry.
        ben_corners = build_geometry().hex_to_vertex[i].tolist()
        cat_corners = [nodes[v] for v in ben_corners]
        for j, node in enumerate(cat_corners):
            owner, kind = buildings.get(node, (None, None))
            rel = 0 if owner is None else (1 if owner == viewer else 2)
            tile_rep[i, 19 + 6 * j + rel] = 1
            tile_rep[i, 19 + 6 * j + 3 + (0 if kind is None else (1 if kind == "SETTLEMENT" else 2))] = 1
            a, b = node, cat_corners[(j + 1) % 6]
            road_owner = board.get_edge_color((a, b))
            road_rel = 0 if road_owner is None else (1 if road_owner == viewer else 2)
            tile_rep[i, 55 + 4 * j + road_rel] = 1
            tile_rep[i, 55 + 4 * j + 3] = float(road_owner is not None)
    for i, node in enumerate(nodes):
        owner, kind = buildings.get(node, (None, None))
        vertex[i, 0 if owner is None else (1 if owner == viewer else 2)] = 1
        vertex[i, 3 + (0 if kind is None else (1 if kind == "SETTLEMENT" else 2))] = 1
        port = ports.get(node, "NONE")
        port_idx = 0 if port == "NONE" else (1 if port is None else 2 + RESOURCES_CW.index(port))
        vertex[i, 6 + port_idx] = 1
        vertex[i, 13] = float(kind is not None)
    for i, pair in enumerate(edges):
        owner = board.get_edge_color(pair)
        edge[i, 0 if owner is None else (1 if owner == viewer else 2)] = 1
        edge[i, 3] = float(owner is not None)
    opponent = next(c for c in state.colors if c != viewer)
    curr = _player_features(game, viewer, viewer)
    opp = _player_features(game, opponent, viewer)
    own_dev = np.array([get_dev_cards_in_hand(state, viewer, card) for card in DEV], dtype=np.float32)
    played_opp = np.array([get_played_dev_cards(state, opponent, card) if card != "VICTORY_POINT" else 0 for card in DEV], dtype=np.float32)
    global_features = np.zeros(14, dtype=np.float32)
    global_features[:5] = np.array([state.resource_freqdeck[("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE").index(r)] / 19 for r in RESOURCES_CW])
    played_own = np.array([get_played_dev_cards(state, viewer, card) if card != "VICTORY_POINT" else 0 for card in DEV], dtype=np.float32)
    global_features[5:10] = np.maximum(0, np.array([14, 5, 2, 2, 2]) - own_dev - played_own - played_opp) / np.array([14, 5, 2, 2, 2])
    return {
        "tile_representations": tile_rep,
        "current_player_main": curr,
        "next_player_main": opp,
        "current_dev_counts": own_dev,
        "next_played_dev_counts": played_opp,
        "hex_features": intrinsic,
        "vertex_features": vertex,
        "edge_features": edge,
        "global_features": global_features,
        "is_setup": np.array([float(state.is_initial_build_phase)], dtype=np.float32),
        "opponent_kind": np.int64(0),
        "opponent_policy_id": np.int64(100),
    }


def action_tuple(action, game):
    """Map a Catanatron legal action to Benjamin's six head indices."""
    kind, value = action.action_type.name, action.value
    tiles, nodes, edges = topology()
    t_inv = {v: i for i, v in enumerate(tiles)}
    n_inv = {v: i for i, v in enumerate(nodes)}
    e_inv = {v: i for i, v in enumerate(edges)}
    type_name = {
        "BUILD_SETTLEMENT": "BUILD_SETTLEMENT", "BUILD_CITY": "BUILD_CITY",
        "BUILD_ROAD": "BUILD_ROAD", "END_TURN": "END_TURN",
        "MOVE_ROBBER": "MOVE_ROBBER", "BUY_DEVELOPMENT_CARD": "BUY_DEV_CARD",
        "PLAY_KNIGHT_CARD": "PLAY_KNIGHT", "PLAY_YEAR_OF_PLENTY": "PLAY_YOP",
        "PLAY_MONOPOLY": "PLAY_MONOPOLY", "PLAY_ROAD_BUILDING": "PLAY_ROAD_BUILDER",
        "MARITIME_TRADE": "BANK_TRADE", "DISCARD_RESOURCE": "DISCARD", "ROLL": "ROLL_DICE",
    }[kind]
    result = [getattr(ActionType, type_name), 0, 0, 0, 0, 0]
    if kind in ("BUILD_SETTLEMENT", "BUILD_CITY"):
        result[1] = n_inv[value]
    elif kind == "BUILD_ROAD":
        result[2] = e_inv[tuple(sorted(value))]
    elif kind == "MOVE_ROBBER":
        result[3] = t_inv[game.state.board.map.land_tiles[value[0]].id]
    elif kind == "PLAY_KNIGHT_CARD":
        result[3] = t_inv[game.state.board.map.land_tiles[game.state.board.robber_coordinate].id]
    elif kind in ("PLAY_MONOPOLY", "DISCARD_RESOURCE"):
        result[4] = RESOURCES_CW.index(value)
    elif kind == "PLAY_YEAR_OF_PLENTY":
        vals = list(value)
        if len(vals) == 1:
            vals *= 2
        result[4:6] = [RESOURCES_CW.index(v) for v in vals[:2]]
    elif kind == "MARITIME_TRADE":
        result[4] = RESOURCES_CW.index(next(v for v in value[:-1] if v is not None))
        result[5] = RESOURCES_CW.index(value[-1])
    return result


def legal_masks(mapped):
    masks = {name: torch.zeros(1, size, dtype=torch.bool) for name, size in HEAD_DIMS}
    for name in MASK_KEYS:
        if name not in masks:
            dim = 54 if name.startswith("corner") else (72 if name == "edge" else (19 if name == "tile" else 5))
            masks[name] = torch.zeros(1, dim, dtype=torch.bool)
    for typ, corner, road, tile, r1, r2 in mapped:
        masks["type"][0, typ] = True
        if typ == ActionType.BUILD_SETTLEMENT: masks["corner_settlement"][0, corner] = True
        if typ == ActionType.BUILD_CITY: masks["corner_city"][0, corner] = True
        if typ == ActionType.BUILD_ROAD: masks["edge"][0, road] = True
        if typ in (ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT): masks["tile"][0, tile] = True
        if typ == ActionType.BANK_TRADE:
            masks["resource1_trade"][0, r1] = True
            masks["resource2_trade"][0, r2] = True
        if typ == ActionType.DISCARD: masks["resource1_discard"][0, r1] = True
        if typ == ActionType.PLAY_MONOPOLY: masks["resource1_default"][0, r1] = True
        if typ == ActionType.PLAY_YOP:
            masks["resource1_yop"][0, r1] = True
            masks["resource2_yop"][0, r2] = True
            if r1 == r2: masks["resource2_yop_same"][0, r2] = True
    for mask in masks.values():
        if not mask.any(): mask[0, 0] = True
    return masks


class BenjaminCatanatronAgent:
    def __init__(self, color, checkpoint, seed=None):
        del seed
        self.color = color
        self.policy = CatanPolicy()
        self.policy.set_board_geometry(build_geometry().as_dict_of_tensors())
        load_checkpoint(checkpoint, map_location="cpu").apply_to_policy(self.policy, strict=True)
        self.policy.cpu().eval()
        self.decisions = 0

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        mapped = [action_tuple(a, game) for a in playable_actions]
        masks = legal_masks(mapped)
        obs_np = encode_visible(game, self.color)
        obs = {k: (torch.as_tensor(v).unsqueeze(0).long() if k.startswith("opponent_") else torch.as_tensor(v).unsqueeze(0).float()) for k, v in obs_np.items()}
        with torch.inference_mode():
            out = self.policy.forward(obs)
            trunk = out["trunk"].expand(len(mapped), -1)
            nodes = {k: out[f"_node_{k}"].expand(len(mapped), -1, -1) for k in ("v", "e", "h")}
            batch_masks = {k: v.expand(len(mapped), -1) for k, v in masks.items()}
            acts = torch.tensor(mapped, dtype=torch.long)
            setup = out["_is_setup"].expand(len(mapped), -1)
            scores = self.policy.action_heads.evaluate_actions(trunk, acts, batch_masks, nodes, setup)["log_prob"]
            index = int(scores.argmax())
        self.decisions += 1
        return playable_actions[index]


def create_agent(color, checkpoint, seed):
    return BenjaminCatanatronAgent(color, checkpoint, seed)
