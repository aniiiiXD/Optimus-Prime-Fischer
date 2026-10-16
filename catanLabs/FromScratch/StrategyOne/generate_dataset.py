#!/usr/bin/env python3
"""Collect 1v1 trajectories labelled only by Catanatron's existing bots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
CATANATRON_ROOT = WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"
SOURCE_MANIFEST = WORKSPACE / "start/catan_lab/sources.json"
RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEVELOPMENT_CARDS = (
    "KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"
)

sys.path.insert(0, str(CATANATRON_ROOT))

from catanatron import Color, Game, RandomPlayer  # noqa: E402
from catanatron.players.search import VictoryPointPlayer  # noqa: E402
from catanatron.players.value import ValueFunctionPlayer  # noqa: E402
from catanatron.players.weighted_random import WeightedRandomPlayer  # noqa: E402
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


def relative(value: Any, viewer: Color) -> Any:
    if isinstance(value, Color):
        return "self" if value == viewer else "opponent"
    if isinstance(value, tuple):
        return [relative(item, viewer) for item in value]
    if isinstance(value, list):
        return [relative(item, viewer) for item in value]
    if isinstance(value, dict):
        return {str(key): relative(item, viewer) for key, item in value.items()}
    if hasattr(value, "name"):
        return value.name
    return value


def action_json(action: Any, viewer: Color) -> dict[str, Any]:
    return {
        "actor": "self" if action.color == viewer else "opponent",
        "type": action.action_type.name,
        "value": relative(action.value, viewer),
    }


def resources(cards: list[int]) -> dict[str, int]:
    return dict(zip(RESOURCES, cards, strict=True))


def board_json(game: Game, viewer: Color) -> dict[str, Any]:
    board = game.state.board
    catan_map = board.map
    tiles = [
        {
            "id": tile.id,
            "coordinate": list(coordinate),
            "resource": tile.resource,
            "dice_number": tile.number,
            "nodes": sorted(tile.nodes.values()),
            "robber": coordinate == board.robber_coordinate,
        }
        for coordinate, tile in sorted(catan_map.land_tiles.items(), key=lambda item: item[1].id)
    ]
    buildings = [
        {
            "node": node,
            "owner": "self" if color == viewer else "opponent",
            "kind": kind,
        }
        for node, (color, kind) in sorted(board.buildings.items())
    ]
    seen_edges = set()
    roads = []
    for edge, color in board.roads.items():
        canonical = tuple(sorted(edge))
        if canonical not in seen_edges:
            seen_edges.add(canonical)
            roads.append({"edge": list(canonical), "owner": "self" if color == viewer else "opponent"})
    ports = [
        {"resource": resource, "nodes": sorted(nodes)}
        for resource, nodes in sorted(catan_map.port_nodes.items(), key=lambda item: str(item[0]))
    ]
    return {"tiles": tiles, "buildings": buildings, "roads": roads, "ports": ports}


def public_events(game: Game, viewer: Color) -> list[dict[str, Any]]:
    """Results of card draws and steals are deliberately excluded."""
    events = []
    for record in game.state.action_records[-64:]:
        event = action_json(record.action, viewer)
        if record.action.action_type.name == "ROLL":
            event["result"] = relative(record.result, viewer)
        events.append(event)
    return events


def observation(game: Game, viewer: Color) -> dict[str, Any]:
    state = game.state
    own_key = player_key(state, viewer)
    enemy = next(color for color in state.colors if color != viewer)
    enemy_key = player_key(state, enemy)
    return {
        "board": board_json(game, viewer),
        "own_hand": resources(get_player_freqdeck(state, viewer)),
        "own_development_cards": {
            card: get_dev_cards_in_hand(state, viewer, card) for card in DEVELOPMENT_CARDS
        },
        "own_actual_vp": get_actual_victory_points(state, viewer),
        "opponent_public": {
            "visible_vp": get_visible_victory_points(state, enemy),
            "resource_count": player_num_resource_cards(state, enemy),
            "development_card_count": player_num_dev_cards(state, enemy),
            "played_development_cards": {
                card: get_played_dev_cards(state, enemy, card)
                for card in DEVELOPMENT_CARDS
                if card != "VICTORY_POINT"
            },
            "roads_available": state.player_state[f"{enemy_key}_ROADS_AVAILABLE"],
            "settlements_available": state.player_state[f"{enemy_key}_SETTLEMENTS_AVAILABLE"],
            "cities_available": state.player_state[f"{enemy_key}_CITIES_AVAILABLE"],
            "longest_road_length": state.player_state[f"{enemy_key}_LONGEST_ROAD_LENGTH"],
            "has_longest_road": state.player_state[f"{enemy_key}_HAS_ROAD"],
            "has_largest_army": state.player_state[f"{enemy_key}_HAS_ARMY"],
        },
        "bank_resources_assumed_observable": resources(state.resource_freqdeck),
        "turn": {
            "has_rolled": state.player_state[f"{own_key}_HAS_ROLLED"],
            "initial_build_phase": state.is_initial_build_phase,
            "discarding": state.is_discarding,
            "moving_robber": state.is_moving_knight,
            "road_building": state.is_road_building,
            "free_roads_available": state.free_roads_available,
            "resolving_trade": state.is_resolving_trade,
        },
        "public_events": public_events(game, viewer),
    }


TEACHERS = ("F_BASE", "F_CONTENDER", "VP")
OPPONENTS = ("F_BASE", "F_CONTENDER", "VP", "W", "R")


def make_bot(name: str, color: Color) -> Any:
    if name == "F_BASE":
        return ValueFunctionPlayer(
            color, ValueFunctionPlayer.Params(value_fn="base", epsilon=None)
        )
    if name == "F_CONTENDER":
        return ValueFunctionPlayer(
            color, ValueFunctionPlayer.Params(value_fn="contender", epsilon=None)
        )
    if name == "VP":
        return VictoryPointPlayer(color)
    if name == "W":
        return WeightedRandomPlayer(color)
    if name == "R":
        return RandomPlayer(color)
    raise ValueError(f"Unsupported Catanatron bot code: {name}")


def matchup_for_game(game_id: int) -> tuple[str, str]:
    """Cycle all 15 external teacher/opponent combinations evenly."""
    teacher = TEACHERS[game_id % len(TEACHERS)]
    opponent = OPPONENTS[(game_id // len(TEACHERS)) % len(OPPONENTS)]
    return teacher, opponent


def teacher_label(code: str) -> str:
    return {
        "F_BASE": "Catanatron ValueFunctionPlayer(base, epsilon=None)",
        "F_CONTENDER": "Catanatron ValueFunctionPlayer(contender, epsilon=None)",
        "VP": "Catanatron VictoryPointPlayer",
    }[code]


def collect_game(game_id: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Save only external teacher moves; no local policy produces labels."""
    colors = (Color.BLUE, Color.RED)
    game = Game([RandomPlayer(color) for color in colors], seed=seed, vps_to_win=10)
    teacher_color = game.state.colors[game_id % 2]
    teacher_code, opponent_code = matchup_for_game(game_id)
    opponent_color = next(color for color in colors if color != teacher_color)
    policies = {
        teacher_color: make_bot(teacher_code, teacher_color),
        opponent_color: make_bot(opponent_code, opponent_color),
    }
    rows = []
    termination = "turn_cap"
    for ply in range(10_000):
        actor = game.state.current_color()
        legal = list(game.playable_actions)
        bot = policies[actor]
        chosen = bot.decide(game, legal)
        if chosen not in legal:
            raise RuntimeError(f"{type(bot).__name__} chose an illegal action")
        if actor == teacher_color:
            rows.append(
                {
                    "schema_version": "catanatron-diverse-jsonl-v1",
                    "game_id": game_id,
                    "seed": seed,
                    "ply": ply,
                    "teacher": teacher_label(teacher_code),
                    "opponent": opponent_code,
                    "observation": observation(game, actor),
                    "legal_actions": [action_json(action, actor) for action in legal],
                    "chosen_action": action_json(chosen, actor),
                }
            )
        game.execute(chosen, validate_action=True)
        if game.winning_color() is not None:
            termination = "win"
            break
    winner = game.winning_color()
    for row in rows:
        row["outcome"] = {"won": winner == teacher_color, "termination_reason": termination}
    return rows, {
        "game_id": game_id,
        "seed": seed,
        "teacher_seat": game_id % 2,
        "teacher": teacher_label(teacher_code),
        "opponent": opponent_code,
        "teacher_won": winner == teacher_color,
        "termination_reason": termination,
        "teacher_decisions": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--output", type=Path, default=HERE / "data/catanatron_diverse.jsonl")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], []
    for game_id in range(args.games):
        game_rows, summary = collect_game(game_id, args.seed + game_id)
        rows.extend(game_rows)
        summaries.append(summary)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    metadata = {
        "schema_version": "catanatron-diverse-metadata-v1",
        "teachers": [teacher_label(code) for code in TEACHERS],
        "opponents": list(OPPONENTS),
        "engine": json.loads(SOURCE_MANIFEST.read_text())["catanatron"],
        "rules": {"players": 2, "vps_to_win": 10, "number_placement": "official_spiral"},
        "rows": len(rows),
        "games": summaries,
        "note": "Every saved label is a move from the pinned external Catanatron implementation. The game contract remains provisional until the organizer publishes the official interface.",
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(rows)} diverse Catanatron-labelled rows from {args.games} games")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
