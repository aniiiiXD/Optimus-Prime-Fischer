#!/usr/bin/env python3
"""Measure where the trained MLP loses to Catanatron Value Base; no training or policy changes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from train_legal_ranker import HERE, LegalRanker, encode_action, encode_state, stable_edge_index
from train_calibrated_ranker import ActionCalibrationAdapter

CATANATRON_ROOT = HERE.parents[2] / "start/catan_lab/vendor/catanatron/catanatron"
sys.path.insert(0, str(CATANATRON_ROOT))
sys.path.insert(0, str(HERE.parents[2]))

from catanatron import Color, Game, RandomPlayer  # noqa: E402
from catanatron.models.enums import CITY, SETTLEMENT  # noqa: E402
from catanatron.state_functions import get_actual_victory_points, get_player_freqdeck  # noqa: E402
from catanLabs.Catanatron.tournament_protocol import install_v5_protocol  # noqa: E402
from generate_dataset import action_json, make_bot, observation  # noqa: E402


def action_category(action_type: str) -> str:
    if action_type in {"BUILD_ROAD", "BUILD_SETTLEMENT"}:
        return "territory"
    if action_type == "BUILD_CITY":
        return "vp_building"
    if action_type in {"MOVE_ROBBER", "PLAY_KNIGHT_CARD"}:
        return "robber"
    if action_type in {"MARITIME_TRADE", "BUY_DEVELOPMENT_CARD"}:
        return "economy"
    if action_type.startswith("PLAY_"):
        return "development"
    return "turn_flow"


def expected_pips(game: Game, color: Color) -> int:
    """Offline diagnostic: expected dice pips from owned buildings, excluding robber-blocked tiles."""
    total = 0
    board = game.state.board
    for node, (owner, kind) in board.buildings.items():
        if owner != color:
            continue
        multiplier = 2 if kind == CITY else 1
        for coordinate, tile in board.map.land_tiles.items():
            if node in tile.nodes.values() and tile.number is not None and coordinate != board.robber_coordinate:
                total += multiplier * (6 - abs(7 - tile.number))
    return total


def metrics(game: Game, color: Color) -> dict[str, int]:
    board = game.state.board
    buildings = [kind for _, (owner, kind) in board.buildings.items() if owner == color]
    roads = len({tuple(sorted(edge)) for edge, owner in board.roads.items() if owner == color})
    return {
        "actual_vp": get_actual_victory_points(game.state, color),
        "resource_cards": sum(get_player_freqdeck(game.state, color)),
        "production_pips": expected_pips(game, color),
        "roads": roads,
        "settlements": buildings.count(SETTLEMENT),
        "cities": buildings.count(CITY),
    }


class Policy:
    def __init__(self, color: Color, model: LegalRanker, edge_index: dict[tuple[int, int], int], device: torch.device, adapter: ActionCalibrationAdapter | None = None):
        self.color, self.model, self.edge_index, self.device, self.adapter = color, model, edge_index, device, adapter

    @torch.no_grad()
    def choose(self, game: Game, legal: list[Any]) -> Any:
        visible = observation(game, self.color)
        state = torch.tensor([encode_state(visible, self.edge_index)], dtype=torch.float32, device=self.device)
        actions = torch.tensor([[encode_action(action_json(action, self.color), visible, self.edge_index) for action in legal]], dtype=torch.float32, device=self.device)
        scores = self.model(state, actions)
        if self.adapter is not None:
            scores = self.adapter(scores, state, actions)
        return legal[int(scores.argmax(dim=1).item())]


def phase(own_decisions: int) -> str:
    if own_decisions < 20:
        return "opening"
    if own_decisions < 60:
        return "midgame"
    return "late_game"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20264001)
    parser.add_argument("--seed-stride", type=int, default=1, help="Spacing between seeded games; use 5 for a cyclic five-opponent screen.")
    parser.add_argument("--checkpoint", type=Path, default=HERE / "runs/legal_ranker_v1/model.pt")
    parser.add_argument("--output", type=Path, default=HERE / "reports/mlp_loss_diagnosis_v1.json")
    args = parser.parse_args()
    # ValueFunctionPlayer evaluates hypothetical copies.  Upstream copies share the
    # live RNG, so isolate them before asking for diagnostic recommendations.
    install_v5_protocol()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = LegalRanker(checkpoint["state_size"], checkpoint["action_size"])
    adapter = None
    if "base_model" in checkpoint:
        model.load_state_dict(checkpoint["base_model"])
        adapter = ActionCalibrationAdapter(checkpoint["state_size"])
        adapter.load_state_dict(checkpoint["adapter"])
        adapter.eval()
    else:
        model.load_state_dict(checkpoint.get("model", checkpoint.get("policy")))
    model.eval()
    with (HERE / "data/catanatron_diverse.jsonl").open(encoding="utf-8") as handle:
        first = json.loads(next(handle))
    edge_index = stable_edge_index(first["observation"])

    events, games = [], []
    for game_index in range(args.games):
        colors = (Color.BLUE, Color.RED)
        game_seed = args.seed + game_index * args.seed_stride
        game = Game([RandomPlayer(color) for color in colors], seed=game_seed, vps_to_win=10)
        mlp_color = game.state.colors[(game_index * args.seed_stride) % 2]
        teacher_color = next(color for color in colors if color != mlp_color)
        mlp = Policy(mlp_color, model, edge_index, torch.device("cpu"), adapter)
        teacher = make_bot("F_BASE", teacher_color)
        own_decisions = 0
        for ply in range(10_000):
            actor = game.state.current_color()
            legal = list(game.playable_actions)
            if actor == mlp_color:
                before_mlp, before_teacher = metrics(game, mlp_color), metrics(game, teacher_color)
                action = mlp.choose(game, legal)
                # This is only an offline comparison; it never changes the live action.
                teacher_recommendation = teacher.decide(game, legal)
                events.append(
                    {
                        "game_id": game_index,
                        "phase": phase(own_decisions),
                        "mlp_action": action.action_type.name,
                        "teacher_action": teacher_recommendation.action_type.name,
                        "exact_teacher_match": action == teacher_recommendation,
                        "mlp_before": before_mlp,
                        "teacher_before": before_teacher,
                    }
                )
                own_decisions += 1
            else:
                action = teacher.decide(game, legal)
            game.execute(action, validate_action=True)
            if game.winning_color() is not None:
                break
        winner = game.winning_color()
        games.append(
            {
                "game_id": game_index,
                "seed": game_seed,
                "mlp_won": winner == mlp_color,
                "completed": winner is not None,
                "plies": len(game.state.action_records),
                "mlp_final": metrics(game, mlp_color),
                "teacher_final": metrics(game, teacher_color),
            }
        )

    by_phase: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    action_counts, teacher_counts, agreement = Counter(), Counter(), Counter()
    action_pairs, phase_action_pairs, action_type_pairs = Counter(), Counter(), Counter()
    exact_by_category, total_by_category = Counter(), Counter()
    for event in events:
        bucket = by_phase[event["phase"]]
        for key, value in event["mlp_before"].items():
            bucket[f"mlp_{key}"].append(value)
            bucket[f"teacher_{key}"].append(event["teacher_before"][key])
        mlp_category = action_category(event["mlp_action"])
        teacher_category = action_category(event["teacher_action"])
        action_counts[mlp_category] += 1
        teacher_counts[teacher_category] += 1
        total_by_category[mlp_category] += 1
        exact_by_category[mlp_category] += event["exact_teacher_match"]
        action_pairs[f"{mlp_category} -> {teacher_category}"] += 1
        phase_action_pairs[f"{event['phase']}: {mlp_category} -> {teacher_category}"] += 1
        action_type_pairs[f"{event['mlp_action']} -> {event['teacher_action']}"] += 1
        agreement[event["phase"]] += event["exact_teacher_match"]
        agreement[f"{event['phase']}_total"] += 1

    phase_summary = {}
    for name, values in by_phase.items():
        phase_summary[name] = {
            key: round(sum(series) / len(series), 3) for key, series in values.items()
        }
        phase_summary[name]["teacher_action_match_rate"] = round(
            agreement[name] / agreement[f"{name}_total"], 3
        )
    losses = [game for game in games if not game["mlp_won"]]
    loss_ids = {game["game_id"] for game in losses}
    loss_action_type_pairs = Counter()
    loss_action_category_pairs = Counter()
    for event in events:
        if event["game_id"] not in loss_ids:
            continue
        loss_action_type_pairs[f"{event['mlp_action']} -> {event['teacher_action']}"] += 1
        loss_action_category_pairs[
            f"{action_category(event['mlp_action'])} -> {action_category(event['teacher_action'])}"
        ] += 1
    final_gap = {
        key: round(
            sum(game["mlp_final"][key] - game["teacher_final"][key] for game in losses) / len(losses), 3
        )
        for key in games[0]["mlp_final"]
    }
    report = {
        "games": args.games,
        "wins": sum(game["mlp_won"] for game in games),
        "win_rate": sum(game["mlp_won"] for game in games) / len(games),
        "events": len(events),
        "mlp_action_categories": action_counts,
        "teacher_action_categories": teacher_counts,
        "action_category_pairs": action_pairs,
        "phase_action_category_pairs": phase_action_pairs,
        "action_type_pairs": action_type_pairs,
        "exact_teacher_match_by_mlp_category": {
            category: round(exact_by_category[category] / total, 3)
            for category, total in total_by_category.items()
        },
        "phase_summary": phase_summary,
        "average_final_gap_in_losses_mlp_minus_teacher": final_gap,
        "loss_action_type_pairs": loss_action_type_pairs,
        "loss_action_category_pairs": loss_action_category_pairs,
        "games_detail": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "games_detail"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
