"""A player-visible, immutable boundary for Catanatron decision agents.

Only :func:`snapshot` receives a Catanatron ``Game``. Agents behind
``VisibleAgentAdapter`` receive this value object and return a legal-action
index; they cannot inspect the simulator's hidden hands, deck, or RNG through
their decision argument. The original legal objects remain in the adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Protocol, TypeAlias


RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEVELOPMENT_CARDS = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")
SCHEMA_VERSION = "catanlabs-visible-observation-v1"

Primitive: TypeAlias = None | bool | int | float | str
VisibleValue: TypeAlias = Primitive | tuple["VisibleValue", ...]


@dataclass(frozen=True, slots=True)
class VisibleAction:
    kind: str
    value: VisibleValue
    actor_relative_seat: int


@dataclass(frozen=True, slots=True)
class VisibleTile:
    id: int
    coordinate: tuple[int, ...]
    resource: str
    number: int | None
    has_robber: bool
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class VisibleNode:
    id: int
    owner_relative_seat: int | None
    building_type: str | None
    production: tuple[float, ...]
    port: str | None  # None=no port, "ANY"=3:1 port, resource name=2:1 port
    adjacent_tile_ids: tuple[int, ...]
    actor_can_build: bool


@dataclass(frozen=True, slots=True)
class VisibleEdge:
    nodes: tuple[int, int]
    owner_relative_seat: int | None
    actor_can_build: bool


@dataclass(frozen=True, slots=True)
class VisiblePlayer:
    relative_seat: int
    public_victory_points: int
    has_largest_army: bool
    has_longest_road: bool
    longest_road_length: int
    roads_available: int
    settlements_available: int
    cities_available: int
    resource_count: int
    development_card_count: int
    played_development_cards: tuple[int, ...]
    settlement_nodes: tuple[int, ...]
    city_nodes: tuple[int, ...]
    road_edges: tuple[tuple[int, int], ...]
    own_resources: tuple[int, ...] | None
    own_development_cards: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class VisibleObservation:
    schema_version: str
    acting_relative_seat: int
    turn_relative_seat: int
    phase: str
    turn_count: int
    target_victory_points: int
    discard_limit: int
    friendly_robber: bool
    is_initial_build_phase: bool
    is_discarding: bool
    is_moving_knight: bool
    is_road_building: bool
    is_resolving_trade: bool
    free_roads_available: int
    own_has_rolled: bool
    own_played_development_card_this_turn: bool
    bank_resources: tuple[int, ...]
    discard_counts: tuple[int, ...]
    players: tuple[VisiblePlayer, ...]
    tiles: tuple[VisibleTile, ...]
    nodes: tuple[VisibleNode, ...]
    edges: tuple[VisibleEdge, ...]
    legal_actions: tuple[VisibleAction, ...]


def observation_to_dict(observation: VisibleObservation) -> dict:
    """Return JSON-compatible data without retaining simulator references."""
    if observation.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported visible schema: {observation.schema_version}")
    return asdict(observation)


def _require_plain_tree(value: object) -> None:
    """Reject mutable or simulator-owned objects at the serialization edge."""
    if hasattr(value, "__dataclass_fields__"):
        for field in fields(value):
            _require_plain_tree(getattr(value, field.name))
    elif isinstance(value, tuple):
        for part in value:
            _require_plain_tree(part)
    elif value is not None and type(value) not in (bool, int, float, str):
        raise ValueError(f"Non-primitive visible value: {type(value).__name__}")


def observation_from_dict(data: dict) -> VisibleObservation:
    """Reconstruct a typed, immutable snapshot from a JSON object."""
    def checked(cls, value):
        if type(value) is not dict or set(value) != {field.name for field in fields(cls)}:
            raise ValueError(f"Invalid {cls.__name__} fields")
        return value

    source = checked(VisibleObservation, data)
    if source["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported visible schema: {source['schema_version']}")
    players = []
    for value in source["players"]:
        item = checked(VisiblePlayer, value).copy()
        for key in ("played_development_cards", "settlement_nodes", "city_nodes", "own_resources", "own_development_cards"):
            if item[key] is not None:
                item[key] = tuple(item[key])
        item["road_edges"] = tuple(tuple(edge) for edge in item["road_edges"])
        players.append(VisiblePlayer(**item))
    tiles = []
    for value in source["tiles"]:
        item = checked(VisibleTile, value).copy()
        for key in ("coordinate", "nodes"):
            item[key] = tuple(item[key])
        item["edges"] = tuple(tuple(edge) for edge in item["edges"])
        tiles.append(VisibleTile(**item))
    nodes = []
    for value in source["nodes"]:
        item = checked(VisibleNode, value).copy()
        item["production"] = tuple(item["production"])
        item["adjacent_tile_ids"] = tuple(item["adjacent_tile_ids"])
        nodes.append(VisibleNode(**item))
    edges = []
    for value in source["edges"]:
        item = checked(VisibleEdge, value).copy()
        item["nodes"] = tuple(item["nodes"])
        edges.append(VisibleEdge(**item))
    def restore_value(value):
        if isinstance(value, list):
            return tuple(restore_value(item) for item in value)
        if value is None or type(value) in (bool, int, float, str):
            return value
        raise ValueError(f"Invalid visible action value: {type(value).__name__}")
    actions = []
    for value in source["legal_actions"]:
        item = checked(VisibleAction, value).copy()
        item["value"] = restore_value(item["value"])
        actions.append(VisibleAction(**item))
    item = source.copy()
    item["bank_resources"] = tuple(item["bank_resources"])
    item["discard_counts"] = tuple(item["discard_counts"])
    item["players"] = tuple(players)
    item["tiles"] = tuple(tiles)
    item["nodes"] = tuple(nodes)
    item["edges"] = tuple(edges)
    item["legal_actions"] = tuple(actions)
    observation = VisibleObservation(**item)
    _require_plain_tree(observation)
    return observation


def _name(value: object) -> str:
    return str(getattr(value, "value", value))


def _freeze_value(value: object, relative_index: dict[object, int]) -> VisibleValue:
    """Convert action arguments to primitives; colors become relative seats."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        if value in relative_index:
            return relative_index[value]
        return _name(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(part, relative_index) for part in value)
    raise TypeError(f"Unsupported legal-action value type: {type(value).__name__}")


def snapshot(game: object, playable_actions: object) -> VisibleObservation:
    """Copy exactly the acting player's visible state and legal action set.

    Catanatron exposes private opponent arrays on ``Game``. This function
    reads only their sums to reproduce public card counts. It never exposes
    their card identities, the development deck, or the simulator RNG.
    """
    state = game.state
    board = state.board
    map_obj = board.map
    actor = state.current_color()
    absolute_colors = tuple(state.colors)
    actor_seat = state.color_to_index[actor]
    colors = tuple(absolute_colors[(actor_seat + i) % len(absolute_colors)] for i in range(len(absolute_colors)))
    relative_index = {color: i for i, color in enumerate(colors)}

    players = []
    for relative_seat, color in enumerate(colors):
        prefix = f"P{state.color_to_index[color]}_"
        resources = tuple(int(state.player_state[prefix + resource + "_IN_HAND"]) for resource in RESOURCES)
        development_cards = tuple(int(state.player_state[prefix + card + "_IN_HAND"]) for card in DEVELOPMENT_CARDS)
        buildings = state.buildings_by_color[color]
        players.append(VisiblePlayer(
            relative_seat=relative_seat,
            public_victory_points=int(state.player_state[prefix + "VICTORY_POINTS"]),
            has_largest_army=bool(state.player_state[prefix + "HAS_ARMY"]),
            has_longest_road=bool(state.player_state[prefix + "HAS_ROAD"]),
            longest_road_length=int(state.player_state[prefix + "LONGEST_ROAD_LENGTH"]),
            roads_available=int(state.player_state[prefix + "ROADS_AVAILABLE"]),
            settlements_available=int(state.player_state[prefix + "SETTLEMENTS_AVAILABLE"]),
            cities_available=int(state.player_state[prefix + "CITIES_AVAILABLE"]),
            resource_count=sum(resources),
            development_card_count=sum(development_cards),
            played_development_cards=tuple(int(state.player_state[prefix + "PLAYED_" + card]) for card in DEVELOPMENT_CARDS),
            settlement_nodes=tuple(sorted(int(node) for node in buildings["SETTLEMENT"])),
            city_nodes=tuple(sorted(int(node) for node in buildings["CITY"])),
            road_edges=tuple(sorted(tuple(sorted((int(a), int(b)))) for a, b in buildings["ROAD"])),
            own_resources=resources if relative_seat == 0 else None,
            own_development_cards=development_cards if relative_seat == 0 else None,
        ))

    coordinates = {int(tile.id): tuple(int(axis) for axis in coordinate)
                   for coordinate, tile in map_obj.land_tiles.items()}
    tiles = tuple(
        VisibleTile(
            id=int(tile.id), coordinate=coordinates[int(tile.id)],
            resource=_name(tile.resource), number=int(tile.number) if tile.number is not None else None,
            has_robber=board.robber_coordinate == coordinates[int(tile.id)],
            nodes=tuple(sorted(int(node) for node in tile.nodes.values())),
            edges=tuple(sorted(tuple(sorted((int(a), int(b)))) for a, b in tile.edges.values())),
        )
        for tile in sorted(map_obj.land_tiles.values(), key=lambda tile: int(tile.id))
    )

    port_by_node: dict[int, str] = {}
    for resource, port_nodes in map_obj.port_nodes.items():
        for node in port_nodes:
            port_by_node[int(node)] = "ANY" if resource is None else _name(resource)
    buildable_nodes = set(board.buildable_node_ids(actor))
    nodes = tuple(
        VisibleNode(
            id=node,
            owner_relative_seat=relative_index[board.buildings[node][0]] if node in board.buildings else None,
            building_type=_name(board.buildings[node][1]) if node in board.buildings else None,
            production=tuple(float(map_obj.node_production[node].get(resource, 0.0)) for resource in RESOURCES),
            port=port_by_node.get(node),
            adjacent_tile_ids=tuple(sorted(int(tile.id) for tile in map_obj.adjacent_tiles.get(node, ()))),
            actor_can_build=node in buildable_nodes,
        )
        for node in range(54)
    )
    all_edges = sorted({edge for tile in tiles for edge in tile.edges})
    buildable_edges = {tuple(sorted((int(a), int(b)))) for a, b in board.buildable_edges(actor)}
    edges = tuple(
        VisibleEdge(
            nodes=edge,
            owner_relative_seat=relative_index[board.get_edge_color(edge)]
            if board.get_edge_color(edge) is not None else None,
            actor_can_build=edge in buildable_edges,
        )
        for edge in all_edges
    )
    legal_actions = tuple(
        VisibleAction(
            kind=action.action_type.name,
            value=_freeze_value(action.value, relative_index),
            actor_relative_seat=relative_index[action.color],
        )
        for action in playable_actions
    )
    own_prefix = f"P{actor_seat}_"
    return VisibleObservation(
        schema_version=SCHEMA_VERSION,
        acting_relative_seat=0,
        turn_relative_seat=(state.current_turn_index - actor_seat) % len(colors),
        phase=state.current_prompt.name,
        turn_count=int(state.num_turns),
        target_victory_points=int(game.vps_to_win),
        discard_limit=int(state.discard_limit),
        friendly_robber=bool(state.friendly_robber),
        is_initial_build_phase=bool(state.is_initial_build_phase),
        is_discarding=bool(state.is_discarding),
        is_moving_knight=bool(state.is_moving_knight),
        is_road_building=bool(state.is_road_building),
        is_resolving_trade=bool(state.is_resolving_trade),
        free_roads_available=int(state.free_roads_available),
        own_has_rolled=bool(state.player_state[own_prefix + "HAS_ROLLED"]),
        own_played_development_card_this_turn=bool(state.player_state[own_prefix + "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]),
        bank_resources=tuple(int(value) for value in state.resource_freqdeck),
        discard_counts=tuple(int(state.discard_counts[state.color_to_index[color]]) for color in colors),
        players=tuple(players), tiles=tiles, nodes=nodes, edges=edges,
        legal_actions=legal_actions,
    )


class VisibleDecisionAgent(Protocol):
    def decide_visible(self, observation: VisibleObservation) -> int:
        """Return the index of a legal action in ``observation.legal_actions``."""


class VisibleAgentAdapter:
    """Run an observation-only policy in Catanatron's current agent contract."""

    def __init__(self, agent: VisibleDecisionAgent):
        self.agent = agent

    def decide(self, game: object, playable_actions: object):
        legal = tuple(playable_actions)
        if not legal:
            raise ValueError("No legal actions")
        choice = self.agent.decide_visible(snapshot(game, legal))
        if type(choice) is not int or not 0 <= choice < len(legal):
            raise ValueError(f"Visible agent returned invalid legal-action index: {choice!r}")
        return legal[choice]
