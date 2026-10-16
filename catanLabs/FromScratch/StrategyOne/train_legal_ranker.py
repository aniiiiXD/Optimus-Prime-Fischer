#!/usr/bin/env python3
"""Train a compact MLP to rank legal Catan actions from external teacher data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
DATASET = HERE / "data/catanatron_diverse.jsonl"
CATANATRON_ROOT = HERE.parents[2] / "start/catan_lab/vendor/catanatron/catanatron"
RESOURCE_INDEX = {name: index for index, name in enumerate(("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"))}
TERRAIN_INDEX = {name: index for index, name in enumerate(("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE", None))}
ACTION_TYPES = (
    "ROLL", "MOVE_ROBBER", "DISCARD_RESOURCE", "BUILD_ROAD", "BUILD_SETTLEMENT",
    "BUILD_CITY", "BUY_DEVELOPMENT_CARD", "PLAY_KNIGHT_CARD", "PLAY_YEAR_OF_PLENTY",
    "PLAY_MONOPOLY", "PLAY_ROAD_BUILDING", "MARITIME_TRADE", "OFFER_TRADE",
    "ACCEPT_TRADE", "REJECT_TRADE", "CONFIRM_TRADE", "CANCEL_TRADE", "END_TURN",
)
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_TYPES)}
DEV_CARDS = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")

sys.path.insert(0, str(CATANATRON_ROOT))


def stable_edge_index(observation: dict[str, Any]) -> dict[tuple[int, int], int]:
    """Catan topology is static; derive all 72 edges from the pinned engine."""
    # All map variants used here share Catanatron's base topology.
    from catanatron.models.board import get_edges

    return {tuple(sorted(edge)): index for index, edge in enumerate(sorted(get_edges()))}


def one_hot(size: int, index: int | None) -> list[float]:
    result = [0.0] * size
    if index is not None and 0 <= index < size:
        result[index] = 1.0
    return result


def resource_values(values: dict[str, int], scale: float = 20.0) -> list[float]:
    return [float(values.get(resource, 0)) / scale for resource in RESOURCE_INDEX]


def encode_state(observation: dict[str, Any], edge_index: dict[tuple[int, int], int]) -> list[float]:
    board = observation["board"]
    features: list[float] = []

    # Tile terrain, exact dice number, and robber location: 19 * 8 values.
    tiles = {tile["id"]: tile for tile in board["tiles"]}
    for tile_id in range(19):
        tile = tiles[tile_id]
        features.extend(one_hot(6, TERRAIN_INDEX[tile["resource"]]))
        features.append(float(tile["dice_number"] or 0) / 12.0)
        features.append(float(tile["robber"]))

    # Building ownership/type: self settlement, self city, enemy settlement, enemy city.
    buildings = {building["node"]: building for building in board["buildings"]}
    for node in range(54):
        building = buildings.get(node)
        features.extend(
            [
                float(building is not None and building["owner"] == "self" and building["kind"] == "SETTLEMENT"),
                float(building is not None and building["owner"] == "self" and building["kind"] == "CITY"),
                float(building is not None and building["owner"] == "opponent" and building["kind"] == "SETTLEMENT"),
                float(building is not None and building["owner"] == "opponent" and building["kind"] == "CITY"),
            ]
        )

    # Road ownership: 72 * 2 values.
    roads = {tuple(road["edge"]): road["owner"] for road in board["roads"]}
    for edge, _ in sorted(edge_index.items(), key=lambda item: item[1]):
        owner = roads.get(edge)
        features.extend([float(owner == "self"), float(owner == "opponent")])

    # Ports: 54 nodes * five 2:1 resources plus generic 3:1.
    port_lookup: dict[int, str | None] = {}
    for port in board["ports"]:
        for node in port["nodes"]:
            port_lookup[node] = port["resource"]
    for node in range(54):
        port = port_lookup.get(node, "missing")
        features.extend(one_hot(6, 5 if port is None else RESOURCE_INDEX.get(port)))

    features.extend(resource_values(observation["own_hand"]))
    features.extend([float(observation["own_development_cards"].get(card, 0)) / 5.0 for card in DEV_CARDS])
    features.append(float(observation["own_actual_vp"]) / 10.0)

    enemy = observation["opponent_public"]
    features.extend(
        [
            float(enemy["visible_vp"]) / 10.0,
            float(enemy["resource_count"]) / 20.0,
            float(enemy["development_card_count"]) / 5.0,
            float(enemy["roads_available"]) / 15.0,
            float(enemy["settlements_available"]) / 5.0,
            float(enemy["cities_available"]) / 4.0,
            float(enemy["longest_road_length"]) / 15.0,
            float(enemy["has_longest_road"]),
            float(enemy["has_largest_army"]),
        ]
    )
    features.extend(resource_values(observation["bank_resources_assumed_observable"], scale=19.0))
    turn = observation["turn"]
    features.extend(
        [
            float(turn["has_rolled"]), float(turn["initial_build_phase"]),
            float(turn["discarding"]), float(turn["moving_robber"]),
            float(turn["road_building"]), float(turn["resolving_trade"]),
            float(turn["free_roads_available"]) / 2.0,
        ]
    )
    return features


def collect_resources(value: Any, output: list[float]) -> None:
    if isinstance(value, str) and value in RESOURCE_INDEX:
        output[RESOURCE_INDEX[value]] += 1.0
    elif isinstance(value, list):
        for item in value:
            collect_resources(item, output)


def encode_action(action: dict[str, Any], observation: dict[str, Any], edge_index: dict[tuple[int, int], int]) -> list[float]:
    """Encode type, target node/edge/tile, selected resources, and victim bit."""
    action_type = action["type"]
    value = action["value"]
    features = one_hot(len(ACTION_TYPES), ACTION_INDEX[action_type])
    node = None
    edge = None
    tile = None
    victim = 0.0
    if action_type in {"BUILD_SETTLEMENT", "BUILD_CITY"}:
        node = value
    elif action_type == "BUILD_ROAD":
        edge = tuple(sorted(value))
    elif action_type == "MOVE_ROBBER":
        coordinate, victim_name = value
        victim = float(victim_name == "opponent")
        tile = next(
            (entry["id"] for entry in observation["board"]["tiles"] if entry["coordinate"] == coordinate),
            None,
        )
    features.extend(one_hot(54, node))
    features.extend(one_hot(72, edge_index.get(edge) if edge is not None else None))
    features.extend(one_hot(19, tile))
    resource_counts = [0.0] * 5
    collect_resources(value, resource_counts)
    features.extend([count / 4.0 for count in resource_counts])
    features.append(victim)
    return features


def action_key(action: dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def candidate_set(row: dict[str, Any], negatives: int) -> list[dict[str, Any]]:
    """Chosen move first, then deterministic alternatives for a contrastive label."""
    chosen = row["chosen_action"]
    chosen_key = action_key(chosen)
    alternatives = [action for action in row["legal_actions"] if action_key(action) != chosen_key]
    if not alternatives:
        return [chosen]
    digest = hashlib.sha256(f"{row['game_id']}:{row['ply']}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "little")).shuffle(alternatives)
    return [chosen, *alternatives[:negatives]]


def iter_rows(dataset: Path, split: str, negatives: int, test_modulus: int) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    with dataset.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            is_test = row["game_id"] % test_modulus == 0
            if (split == "test") != is_test:
                continue
            candidates = candidate_set(row, negatives)
            if len(candidates) > 1:
                yield row, candidates


class LegalRanker(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size + action_size, 384), nn.ReLU(), nn.LayerNorm(384),
            nn.Linear(384, 192), nn.ReLU(), nn.Linear(192, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # states: [batch, state], actions: [batch, candidates, action]
        expanded = states.unsqueeze(1).expand(-1, actions.shape[1], -1)
        return self.network(torch.cat((expanded, actions), dim=-1)).squeeze(-1)


def batch_iter(dataset: Path, split: str, batch_size: int, negatives: int, edge_index: dict[tuple[int, int], int], test_modulus: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    states, actions = [], []
    for row, candidates in iter_rows(dataset, split, negatives, test_modulus):
        state = encode_state(row["observation"], edge_index)
        candidate_features = [encode_action(action, row["observation"], edge_index) for action in candidates]
        while len(candidate_features) < negatives + 1:
            candidate_features.append(candidate_features[-1])
        states.append(state)
        actions.append(candidate_features)
        if len(states) == batch_size:
            yield torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)
            states, actions = [], []
    if states:
        yield torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: LegalRanker, dataset: Path, device: torch.device, batch_size: int, negatives: int, edge_index: dict[tuple[int, int], int], test_modulus: int) -> dict[str, float]:
    model.eval()
    total = correct = 0
    total_loss = 0.0
    for states, actions in batch_iter(dataset, "test", batch_size, negatives, edge_index, test_modulus):
        logits = model(states.to(device), actions.to(device))
        labels = torch.zeros(len(states), dtype=torch.long, device=device)
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(dim=1) == 0).sum())
        total += len(states)
    return {"examples": total, "exact_choice_accuracy": correct / total, "loss": total_loss / total}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--test-modulus", type=int, default=5, help="Every Nth game is held out; 5 means 20%.")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output", type=Path, default=HERE / "runs/legal_ranker_v1")
    args = parser.parse_args()
    if args.negatives < 1:
        parser.error("--negatives must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu")

    with args.dataset.open(encoding="utf-8") as handle:
        first = json.loads(next(handle))
    edge_index = stable_edge_index(first["observation"])
    state_size = len(encode_state(first["observation"], edge_index))
    action_size = len(encode_action(first["chosen_action"], first["observation"], edge_index))
    model = LegalRanker(state_size, action_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        train_total = train_correct = 0
        train_loss = 0.0
        for states, actions in batch_iter(args.dataset, "train", args.batch_size, args.negatives, edge_index, args.test_modulus):
            states, actions = states.to(device), actions.to(device)
            logits = model(states, actions)
            labels = torch.zeros(len(states), dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_total += len(states)
            train_loss += float(loss) * len(states)
            train_correct += int((logits.argmax(dim=1) == 0).sum())
        test = evaluate(model, args.dataset, device, args.batch_size, args.negatives, edge_index, args.test_modulus)
        metrics = {
            "epoch": epoch,
            "seconds": round(time.perf_counter() - start, 2),
            "train_examples": train_total,
            "train_accuracy": train_correct / train_total,
            "train_loss": train_loss / train_total,
            "test": test,
        }
        history.append(metrics)
        print(json.dumps(metrics))
    torch.save({"state_size": state_size, "action_size": action_size, "model": model.state_dict()}, args.output / "model.pt")
    (args.output / "metrics.json").write_text(json.dumps({"device": str(device), "test_modulus": args.test_modulus, "history": history}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
