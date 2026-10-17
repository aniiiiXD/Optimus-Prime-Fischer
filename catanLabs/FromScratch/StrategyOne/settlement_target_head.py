#!/usr/bin/env python3
"""Train a learned local-board classifier for settlement destinations.

The frozen calibrated policy still decides whether to build a settlement.
This head only reranks legal settlement nodes after that decision. Labels come
from external Catanatron bots; no handcrafted target scores are used.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from train_legal_ranker import HERE, LegalRanker, encode_action, encode_state, stable_edge_index
from train_calibrated_ranker import ActionCalibrationAdapter


RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
NUMBERS = (None, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
RESOURCE_INDEX = {resource: i for i, resource in enumerate(RESOURCES)}
NUMBER_INDEX = {number: i for i, number in enumerate(NUMBERS)}
PORT_INDEX = {resource: i for i, resource in enumerate(RESOURCES)}
LOCAL_SIZE = 5 * len(NUMBERS) + 7 + 7 + 4 + 5 * len(NUMBERS) + 5 + 4


def _one_hot(size: int, index: int | None) -> list[float]:
    values = [0.0] * size
    if index is not None:
        values[index] = 1.0
    return values


def settlement_features(observation: dict[str, Any], node: int) -> list[float]:
    """Represent the board around a candidate node without a node-ID weight."""
    board = observation["board"]
    touching_tiles = [tile for tile in board["tiles"] if node in tile["nodes"]]
    terrain_number = [0.0] * (5 * len(NUMBERS))
    blocked = own_neighbors = opponent_neighbors = 0.0
    for tile in touching_tiles:
        resource = tile["resource"]
        if resource in RESOURCE_INDEX:
            terrain_number[RESOURCE_INDEX[resource] * len(NUMBERS) + NUMBER_INDEX.get(tile["dice_number"], 0)] += 1.0
        blocked += float(tile["robber"])
        tile_nodes = set(tile["nodes"])
        for building in board["buildings"]:
            if building["node"] in tile_nodes:
                if building["owner"] == "self":
                    own_neighbors += 2.0 if building["kind"] == "CITY" else 1.0
                else:
                    opponent_neighbors += 2.0 if building["kind"] == "CITY" else 1.0

    port = next((entry["resource"] for entry in board["ports"] if node in entry["nodes"]), "missing")
    port_features = _one_hot(7, 6 if port == "missing" else 5 if port is None else PORT_INDEX[port])
    own_roads = opponent_roads = 0.0
    for road in board["roads"]:
        if node in road["edge"]:
            if road["owner"] == "self":
                own_roads += 1.0
            else:
                opponent_roads += 1.0

    current_production = [0.0] * (5 * len(NUMBERS))
    tiles = board["tiles"]
    for building in board["buildings"]:
        if building["owner"] != "self":
            continue
        multiplier = 2.0 if building["kind"] == "CITY" else 1.0
        for tile in tiles:
            resource = tile["resource"]
            if building["node"] in tile["nodes"] and resource in RESOURCE_INDEX:
                current_production[RESOURCE_INDEX[resource] * len(NUMBERS) + NUMBER_INDEX.get(tile["dice_number"], 0)] += multiplier

    owner_counts = {
        (owner, kind): sum(
            building["owner"] == owner and building["kind"] == kind
            for building in board["buildings"]
        )
        for owner in ("self", "opponent")
        for kind in ("SETTLEMENT", "CITY")
    }

    result = [
        *terrain_number,
        float(len(touching_tiles)) / 3.0,
        blocked / 3.0,
        own_neighbors / 18.0,
        opponent_neighbors / 18.0,
        own_roads / 3.0,
        opponent_roads / 3.0,
        float(observation["turn"]["initial_build_phase"]),
        *port_features,
        *[owner_counts[(owner, kind)] / 5.0 for owner in ("self", "opponent") for kind in ("SETTLEMENT", "CITY")],
        *[value / 10.0 for value in current_production],
        *[float(observation["own_hand"].get(resource, 0)) / 20.0 for resource in RESOURCES],
        float(observation["own_actual_vp"]) / 10.0,
        float(observation["opponent_public"]["visible_vp"]) / 10.0,
        float(len(board["buildings"])) / 15.0,
        float(len(board["roads"])) / 30.0,
    ]
    if len(result) != LOCAL_SIZE:
        raise RuntimeError(f"Local feature size {len(result)} != {LOCAL_SIZE}")
    return result


class SettlementTargetHead(nn.Module):
    def __init__(self, input_size: int = LOCAL_SIZE) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass
class Example:
    features: torch.Tensor
    base_scores: torch.Tensor
    label: int
    opening: bool
    teacher: str


def load_frozen_policy(checkpoint_path: Path, device: torch.device) -> tuple[LegalRanker, ActionCalibrationAdapter]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "base_model" not in checkpoint:
        raise ValueError("Settlement target head requires a calibrated-ranker checkpoint")
    base = LegalRanker(checkpoint["state_size"], checkpoint["action_size"]).to(device)
    base.load_state_dict(checkpoint["base_model"])
    adapter = ActionCalibrationAdapter(checkpoint["state_size"]).to(device)
    adapter.load_state_dict(checkpoint["adapter"])
    base.eval()
    adapter.eval()
    return base, adapter


@torch.no_grad()
def load_examples(dataset: Path, checkpoint: Path, test_modulus: int, device: torch.device) -> tuple[list[Example], list[Example]]:
    base, adapter = load_frozen_policy(checkpoint, device)
    with dataset.open(encoding="utf-8") as handle:
        first = json.loads(next(handle))
    edge_index = stable_edge_index(first["observation"])
    train: list[Example] = []
    test: list[Example] = []
    with dataset.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["chosen_action"]["type"] != "BUILD_SETTLEMENT":
                continue
            candidates = [action for action in row["legal_actions"] if action["type"] == "BUILD_SETTLEMENT"]
            if len(candidates) < 2:
                continue
            observation = row["observation"]
            state = torch.tensor([encode_state(observation, edge_index)], dtype=torch.float32, device=device)
            actions = torch.tensor(
                [[encode_action(action, observation, edge_index) for action in candidates]],
                dtype=torch.float32,
                device=device,
            )
            logits = adapter(base(state, actions), state, actions).squeeze(0).cpu()
            features = torch.tensor(
                [settlement_features(observation, int(action["value"])) for action in candidates],
                dtype=torch.float32,
            )
            label = next(i for i, action in enumerate(candidates) if action == row["chosen_action"])
            example = Example(
                features, logits, label, bool(observation["turn"]["initial_build_phase"]), row["teacher"]
            )
            (test if row["game_id"] % test_modulus == 0 else train).append(example)
    return train, test


@torch.no_grad()
def evaluate(examples: list[Example], head: SettlementTargetHead, device: torch.device) -> dict[str, Any]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        key = "opening" if example.opening else "later"
        targets = ("all", key, example.teacher)
        base_choice = int(example.base_scores.argmax().item())
        scored = example.base_scores.to(device) + head(example.features.to(device))
        target_choice = int(scored.argmax().item())
        for target in targets:
            buckets[target]["count"] += 1
            buckets[target]["base_correct"] += base_choice == example.label
            buckets[target]["target_correct"] += target_choice == example.label
    return {
        name: {
            "examples": counts["count"],
            "base_accuracy": round(counts["base_correct"] / counts["count"], 4),
            "target_accuracy": round(counts["target_correct"] / counts["count"], 4),
        }
        for name, counts in buckets.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=HERE / "data/catanatron_diverse.jsonl")
    parser.add_argument("--base-checkpoint", type=Path, default=HERE / "runs/calibrated_ranker_v1/model.pt")
    parser.add_argument("--output", type=Path, default=HERE / "runs/settlement_target_v1")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--test-modulus", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20266001)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.test_modulus < 2:
        parser.error("--epochs must be positive and --test-modulus at least 2")
    torch.manual_seed(args.seed)
    device = torch.device("mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu")
    train, test = load_examples(args.dataset, args.base_checkpoint, args.test_modulus, device)
    head = SettlementTargetHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history = [{"epoch": 0, "holdout": evaluate(test, head, device)}]
    print(json.dumps(history[-1]), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(len(train)).tolist()
        losses = []
        for index in order:
            example = train[index]
            logits = example.base_scores.to(device) + head(example.features.to(device))
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([example.label], device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        head.eval()
        row = {"epoch": epoch, "train_loss": round(sum(losses) / len(losses), 5), "holdout": evaluate(test, head, device)}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save({"head": head.state_dict(), "input_size": LOCAL_SIZE, "base_checkpoint": str(args.base_checkpoint), "epoch": epoch}, args.output / f"epoch_{epoch}.pt")
    (args.output / "metrics.json").write_text(json.dumps({"train_examples": len(train), "test_examples": len(test), "history": history}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
