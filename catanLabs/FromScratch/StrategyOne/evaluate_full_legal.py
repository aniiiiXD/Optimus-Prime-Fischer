#!/usr/bin/env python3
"""Evaluate a trained legal-action ranker against every legal move."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from train_legal_ranker import (
    DATASET,
    HERE,
    LegalRanker,
    action_key,
    encode_action,
    encode_state,
    stable_edge_index,
)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "runs/legal_ranker_v1/model.pt")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-modulus", type=int, default=5, help="Every Nth game is held out; 5 means 20%.")
    args = parser.parse_args()
    device = torch.device("mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = LegalRanker(checkpoint["state_size"], checkpoint["action_size"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with args.dataset.open(encoding="utf-8") as handle:
        first = json.loads(next(handle))
    edge_index = stable_edge_index(first["observation"])

    buffers: dict[int, list[tuple[list[float], list[list[float]], int]]] = defaultdict(list)
    total = correct = non_forced = non_forced_correct = 0

    def flush(count: int) -> None:
        nonlocal total, correct, non_forced, non_forced_correct
        batch = buffers.pop(count, [])
        if not batch:
            return
        states = torch.tensor([item[0] for item in batch], dtype=torch.float32, device=device)
        actions = torch.tensor([item[1] for item in batch], dtype=torch.float32, device=device)
        predicted = model(states, actions).argmax(dim=1).cpu().tolist()
        for prediction, (_, _, label) in zip(predicted, batch, strict=True):
            total += 1
            correct += prediction == label
            if count > 1:
                non_forced += 1
                non_forced_correct += prediction == label

    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["game_id"] % args.test_modulus != 0:
                continue
            candidates = row["legal_actions"]
            chosen_key = action_key(row["chosen_action"])
            label = next(index for index, action in enumerate(candidates) if action_key(action) == chosen_key)
            observation = row["observation"]
            item = (
                encode_state(observation, edge_index),
                [encode_action(action, observation, edge_index) for action in candidates],
                label,
            )
            buffers[len(candidates)].append(item)
            if len(buffers[len(candidates)]) >= args.batch_size:
                flush(len(candidates))
    for count in list(buffers):
        flush(count)
    report = {
        "all_held_out_states": total,
        "all_held_out_exact_accuracy": correct / total,
        "non_forced_states": non_forced,
        "non_forced_exact_accuracy": non_forced_correct / non_forced,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
