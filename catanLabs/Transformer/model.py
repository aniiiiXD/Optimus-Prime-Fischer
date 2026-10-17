"""Visible-state encoding and a small set-Transformer Catan policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from catanLabs.Transformer.visible_observation import (
    SCHEMA_VERSION as OBSERVATION_SCHEMA,
    VisibleAction,
    VisibleObservation,
    snapshot,
)


RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
# Encoded arrays have no raw board snapshot, so a feature change requires a
# new dataset and checkpoint rather than reinterpreting older tensors.
FEATURE_SCHEMA = "visible-state-tokens-v3"
DATASET_SCHEMA = "catanlabs-transformer-dataset-v3"
CHECKPOINT_SCHEMA = "catanlabs-action-transformer-v3"
TERRAIN = (*RESOURCES, "DESERT")
PROMPTS = (
    "BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD", "PLAY_TURN", "DISCARD",
    "MOVE_ROBBER", "DECIDE_TRADE", "DECIDE_ACCEPTEES",
)
ACTION_TYPES = (
    "ROLL", "MOVE_ROBBER", "DISCARD_RESOURCE", "BUILD_ROAD",
    "BUILD_SETTLEMENT", "BUILD_CITY", "BUY_DEVELOPMENT_CARD",
    "PLAY_KNIGHT_CARD", "PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY",
    "PLAY_ROAD_BUILDING", "MARITIME_TRADE", "OFFER_TRADE", "ACCEPT_TRADE",
    "REJECT_TRADE", "CONFIRM_TRADE", "CANCEL_TRADE", "END_TURN",
)

FEATURE_DIM = 40
ACTION_FEATURE_DIM = 20
STATE_TOKEN_COUNT = 1 + 4 + 19 + 54 + 72
KIND_GLOBAL, KIND_PLAYER, KIND_TILE, KIND_NODE, KIND_EDGE, KIND_ACTION = range(6)
GROUP_NONE, GROUP_TILE, GROUP_NODE, GROUP_EDGE = range(4)


def _unit(index: int, size: int) -> np.ndarray:
    row = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        row[index] = 1.0
    return row


def _edge_key(edge: Any) -> tuple[int, int]:
    return tuple(sorted((int(edge[0]), int(edge[1]))))


def _action_descriptor(
    action: VisibleAction,
    tile_id_by_coord: dict[tuple[int, ...], int],
    edge_to_id: dict[tuple[int, int], int],
) -> tuple[np.ndarray, int, int, int]:
    """Encode an action using only the frozen visible action vocabulary."""
    action_name = action.kind
    try:
        action_type = ACTION_TYPES.index(action_name)
    except ValueError as exc:
        raise ValueError(f"Unknown Catanatron action type: {action_name}") from exc

    features = np.zeros(ACTION_FEATURE_DIM, dtype=np.float32)
    group, entity_id = GROUP_NONE, 0
    value = action.value

    if action_name in ("BUILD_SETTLEMENT", "BUILD_CITY"):
        group, entity_id = GROUP_NODE, int(value)
    elif action_name == "BUILD_ROAD":
        group, entity_id = GROUP_EDGE, edge_to_id[_edge_key(value)]
    elif action_name == "MOVE_ROBBER":
        coordinate, victim = value
        group, entity_id = GROUP_TILE, tile_id_by_coord[coordinate]
        if victim is not None:
            features[15 + victim] = 1.0
    elif action_name == "DISCARD_RESOURCE" or action_name == "PLAY_MONOPOLY":
        if value in RESOURCES:
            features[RESOURCES.index(value)] = 1.0
    elif action_name == "PLAY_YEAR_OF_PLENTY":
        for resource in value:
            if resource in RESOURCES:
                features[RESOURCES.index(resource)] += 1.0
    elif action_name == "MARITIME_TRADE":
        # Catanatron stores four offered-card slots followed by the requested
        # resource. Resource frequencies make equivalent offer orderings match.
        for resource in value[:4]:
            if resource in RESOURCES:
                features[5 + RESOURCES.index(resource)] += 1.0
        requested = value[4]
        if requested in RESOURCES:
            features[10 + RESOURCES.index(requested)] = 1.0
    elif action_name in ("OFFER_TRADE", "CONFIRM_TRADE"):
        for i, amount in enumerate(value[:10]):
            features[(5 if i < 5 else 10) + (i % 5)] = float(amount)
        if len(value) > 10:
            features[15] = 1.0
    elif isinstance(value, str) and value in RESOURCES:
        features[RESOURCES.index(value)] = 1.0

    return features, action_type, group, entity_id


def encode_visible_observation(observation: VisibleObservation):
    """Encode only a frozen player-visible snapshot and its legal actions."""
    if observation.schema_version != OBSERVATION_SCHEMA:
        raise ValueError(f"Unsupported visible observation schema: {observation.schema_version!r}")
    if observation.acting_relative_seat != 0 or len(observation.players) != 4:
        raise ValueError("Expected a four-player snapshot relative to the actor")
    if (len(observation.tiles), len(observation.nodes), len(observation.edges)) != (19, 54, 72):
        raise ValueError("Unexpected board topology in visible observation")
    tile_by_id = sorted(observation.tiles, key=lambda tile: tile.id)
    node_by_id = {node.id: node for node in observation.nodes}
    edge_list = sorted(observation.edges, key=lambda edge: edge.nodes)
    tile_id_by_coord = {tile.coordinate: tile.id for tile in tile_by_id}
    edge_to_id = {edge.nodes: index for index, edge in enumerate(edge_list)}
    tokens = np.zeros((STATE_TOKEN_COUNT, FEATURE_DIM), dtype=np.float32)
    kinds = np.empty(STATE_TOKEN_COUNT, dtype=np.int64)
    ids = np.empty(STATE_TOKEN_COUNT, dtype=np.int64)
    cursor = 0

    row = tokens[cursor]
    row[:7] = _unit(PROMPTS.index(observation.phase) if observation.phase in PROMPTS else -1, 7)
    row[7] = min(observation.turn_count / 500.0, 2.0)
    row[8] = observation.turn_relative_seat / 4.0
    row[9:14] = (
        float(observation.is_initial_build_phase), float(observation.is_discarding),
        float(observation.is_moving_knight), float(observation.is_road_building),
        float(observation.is_resolving_trade),
    )
    row[14] = observation.free_roads_available / 2.0
    own = observation.players[0]
    if own.own_resources is None or own.own_development_cards is None:
        raise ValueError("Acting player's private cards are missing")
    row[15] = min(sum(own.own_resources) / 20.0, 2.0)
    row[16] = float(observation.own_has_rolled)
    row[17] = float(observation.own_played_development_card_this_turn)
    row[18:23] = np.asarray(observation.bank_resources, dtype=np.float32) / 19.0
    row[23:27] = np.asarray(observation.discard_counts, dtype=np.float32) / 7.0
    row[27] = float(observation.friendly_robber)
    row[28] = observation.discard_limit / 10.0
    row[29] = observation.target_victory_points / 10.0
    kinds[cursor], ids[cursor] = KIND_GLOBAL, 0
    cursor += 1

    for player in observation.players:
        row = tokens[cursor]
        rel = player.relative_seat
        row[:9] = (
            float(rel == 0), rel / 3.0, player.public_victory_points / 10.0,
            float(player.has_largest_army), float(player.has_longest_road),
            player.longest_road_length / 15.0, player.roads_available / 15.0,
            player.settlements_available / 5.0, player.cities_available / 4.0,
        )
        if rel == 0:
            row[9:14] = np.minimum(np.asarray(own.own_resources, dtype=np.float32) / 10.0, 2.0)
            row[14:19] = np.minimum(np.asarray(own.own_development_cards, dtype=np.float32) / 5.0, 2.0)
        # Preserve the existing token order: knight, monopoly, year of
        # plenty, road building, victory point. Snapshot stores the two
        # middle card types in the opposite order.
        played = player.played_development_cards
        row[19:24] = (played[0] / 14.0, played[2] / 2.0, played[1] / 2.0,
                      played[3] / 2.0, played[4] / 5.0)
        row[24:27] = (len(player.settlement_nodes) / 5.0,
                      len(player.city_nodes) / 4.0, len(player.road_edges) / 15.0)
        production = np.zeros(5, dtype=np.float32)
        for node_id in player.settlement_nodes:
            production += np.asarray(node_by_id[node_id].production, dtype=np.float32)
        for node_id in player.city_nodes:
            production += 2.0 * np.asarray(node_by_id[node_id].production, dtype=np.float32)
        row[27:32] = production
        # Public hand sizes are visible to every player even though the
        # opponent's resource and development-card identities are hidden.
        row[32] = min(player.resource_count / 19.0, 2.0)
        row[33] = min(player.development_card_count / 25.0, 2.0)
        kinds[cursor], ids[cursor] = KIND_PLAYER, rel
        cursor += 1

    for tile in tile_by_id:
        row = tokens[cursor]
        terrain = TERRAIN.index(tile.resource) if tile.resource in TERRAIN else TERRAIN.index("DESERT")
        row[:6] = _unit(terrain, 6)
        row[6:17] = _unit(tile.number - 2 if tile.number is not None else -1, 11)
        row[17] = float(tile.has_robber)
        for axis, value in enumerate(tile.coordinate[:3]):
            row[18 + axis] = value / 5.0
        kinds[cursor], ids[cursor] = KIND_TILE, tile.id
        cursor += 1

    for node_id in range(54):
        node = node_by_id[node_id]
        row = tokens[cursor]
        owner = node.owner_relative_seat
        row[:5] = _unit(owner + 1 if owner is not None else 0, 5)
        row[5:7] = (float(node.building_type == "SETTLEMENT"), float(node.building_type == "CITY"))
        row[7:12] = node.production
        port = 0 if node.port is None else (1 if node.port == "ANY" else 2 + RESOURCES.index(node.port))
        row[12:19] = _unit(port, 7)
        row[19] = float(node.actor_can_build)
        row[20] = len(node.adjacent_tile_ids) / 3.0
        kinds[cursor], ids[cursor] = KIND_NODE, node_id
        cursor += 1

    for edge_id, edge in enumerate(edge_list):
        row = tokens[cursor]
        owner = edge.owner_relative_seat
        row[:5] = _unit(owner + 1 if owner is not None else 0, 5)
        row[5] = float(edge.actor_can_build)
        for endpoint, node_id in enumerate(edge.nodes):
            building = node_by_id[node_id]
            if building.owner_relative_seat is not None:
                row[6 + endpoint * 5 + building.owner_relative_seat] = 1.0
                row[16 + endpoint] = float(building.building_type == "CITY")
        kinds[cursor], ids[cursor] = KIND_EDGE, edge_id
        cursor += 1

    encoded = [_action_descriptor(action, tile_id_by_coord, edge_to_id)
               for action in observation.legal_actions]
    action_features = (np.stack([item[0] for item in encoded]) if encoded
                       else np.zeros((0, ACTION_FEATURE_DIM), dtype=np.float32))
    action_types = np.asarray([item[1] for item in encoded], dtype=np.int64)
    groups = np.asarray([item[2] for item in encoded], dtype=np.int64)
    target_ids = np.asarray([item[3] for item in encoded], dtype=np.int64)
    return tokens, kinds, ids, action_features, action_types, groups, target_ids


def encode_observation(game: Any, playable_actions: list[Any]):
    """Compatibility wrapper: freeze the Game before encoding anything."""
    return encode_visible_observation(snapshot(game, tuple(playable_actions)))


class CatanActionTransformer(nn.Module):
    """Small permutation-equivariant Transformer over board and legal actions."""

    def __init__(self, width: int = 64, layers: int = 2, heads: int = 4, dropout: float = 0.05):
        super().__init__()
        self.width = width
        self.state_projection = nn.Linear(FEATURE_DIM, width)
        self.action_projection = nn.Linear(ACTION_FEATURE_DIM, width)
        self.kind_embedding = nn.Embedding(6, width)
        self.entity_embedding = nn.Embedding(73, width)
        self.action_type_embedding = nn.Embedding(len(ACTION_TYPES), width)
        self.target_group_embedding = nn.Embedding(4, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.action_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(
        self,
        state_features: torch.Tensor,
        state_kinds: torch.Tensor,
        state_ids: torch.Tensor,
        action_features: torch.Tensor,
        action_types: torch.Tensor,
        target_groups: torch.Tensor,
        target_ids: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        state_tokens = self.state_projection(state_features)
        state_tokens = state_tokens + self.kind_embedding(state_kinds) + self.entity_embedding(state_ids)
        action_tokens = self.action_projection(action_features)
        action_tokens = (
            action_tokens + self.kind_embedding.weight[KIND_ACTION]
            + self.action_type_embedding(action_types)
            + self.target_group_embedding(target_groups)
            + self.entity_embedding(target_ids)
        )
        tokens = torch.cat((state_tokens, action_tokens), dim=1)
        state_mask = torch.zeros(
            state_features.shape[:2], dtype=torch.bool, device=state_features.device
        )
        padding_mask = torch.cat((state_mask, ~action_mask), dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.action_head(encoded[:, state_features.shape[1]:]).squeeze(-1)


@dataclass
class EncodedRecord:
    state_features: np.ndarray
    state_kinds: np.ndarray
    state_ids: np.ndarray
    action_features: np.ndarray
    action_types: np.ndarray
    target_groups: np.ndarray
    target_ids: np.ndarray
    label: int
    game_id: int
    phase: str
    behavior: str
    actor_color: str
    outcome: int = -1
    final_vp: int = -1

