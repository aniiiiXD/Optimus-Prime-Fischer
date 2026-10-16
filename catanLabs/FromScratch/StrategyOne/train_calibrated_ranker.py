#!/usr/bin/env python3
"""Train a learned action-calibration adapter on top of the frozen legal-action MLP.

The adapter has two new learned heads: state-conditioned residuals for each
action type and a state-conditioned END_TURN adjustment.  BUY_DEVELOPMENT_CARD
versus END_TURN examples receive an explicit balanced pairwise loss.  No
hand-written game-play rule or reward is introduced.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from train_legal_ranker import (
    ACTION_INDEX,
    ACTION_TYPES,
    HERE,
    LegalRanker,
    candidate_set,
    encode_action,
    encode_state,
    stable_edge_index,
)


class ActionCalibrationAdapter(nn.Module):
    """Learned state-conditioned residuals over legal action types."""

    def __init__(self, state_size: int) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(state_size, 192), nn.ReLU(), nn.LayerNorm(192),
            nn.Linear(192, 96), nn.ReLU(),
        )
        self.action_residual = nn.Linear(96, len(ACTION_TYPES))
        self.end_turn_adjustment = nn.Linear(96, 1)
        # Start as the existing policy exactly. Training discovers every
        # residual from teacher examples rather than applying a fixed boost.
        nn.init.zeros_(self.action_residual.weight)
        nn.init.zeros_(self.action_residual.bias)
        nn.init.zeros_(self.end_turn_adjustment.weight)
        nn.init.zeros_(self.end_turn_adjustment.bias)

    def forward(self, base_scores: torch.Tensor, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        context = self.context(states)
        type_residuals = self.action_residual(context)
        action_types = actions[:, :, : len(ACTION_TYPES)]
        residual = torch.einsum("bca,ba->bc", action_types, type_residuals)
        end_mask = action_types[:, :, ACTION_INDEX["END_TURN"]]
        end_adjustment = self.end_turn_adjustment(context).squeeze(-1).unsqueeze(1)
        return base_scores + residual - end_mask * end_adjustment


def iter_rows(dataset: Path, split: str, test_modulus: int) -> Iterator[dict]:
    with dataset.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            is_test = row["game_id"] % test_modulus == 0
            if (split == "test") == is_test:
                yield row


def general_batches(
    dataset: Path,
    split: str,
    batch_size: int,
    negatives: int,
    edge_index: dict[tuple[int, int], int],
    test_modulus: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    states, actions = [], []
    for row in iter_rows(dataset, split, test_modulus):
        candidates = candidate_set(row, negatives)
        if len(candidates) == 1:
            continue
        features = [encode_action(action, row["observation"], edge_index) for action in candidates]
        while len(features) < negatives + 1:
            features.append(features[-1])
        states.append(encode_state(row["observation"], edge_index))
        actions.append(features)
        if len(states) == batch_size:
            yield torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)
            states, actions = [], []
    if states:
        yield torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)


def buy_end_batches(
    dataset: Path,
    split: str,
    batch_size: int,
    edge_index: dict[tuple[int, int], int],
    test_modulus: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield teacher Buy Dev first and legal End Turn second."""
    states, actions = [], []
    for row in iter_rows(dataset, split, test_modulus):
        if row["chosen_action"]["type"] != "BUY_DEVELOPMENT_CARD":
            continue
        end_turn = next((action for action in row["legal_actions"] if action["type"] == "END_TURN"), None)
        if end_turn is None:
            continue
        states.append(encode_state(row["observation"], edge_index))
        actions.append(
            [
                encode_action(row["chosen_action"], row["observation"], edge_index),
                encode_action(end_turn, row["observation"], edge_index),
            ]
        )
        if len(states) == batch_size:
            yield torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)
            states, actions = [], []
    if states:
        yield torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)


def scores(base: LegalRanker, adapter: ActionCalibrationAdapter, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        base_scores = base(states, actions)
    return adapter(base_scores, states, actions)


@torch.no_grad()
def full_legal_metrics(
    dataset: Path,
    base: LegalRanker,
    adapter: ActionCalibrationAdapter,
    edge_index: dict[tuple[int, int], int],
    device: torch.device,
    test_modulus: int,
) -> dict[str, float]:
    total = correct = nonforced = nonforced_correct = buy_total = buy_correct = 0
    predicted = Counter()
    for row in iter_rows(dataset, "test", test_modulus):
        legal = row["legal_actions"]
        chosen = row["chosen_action"]
        state = torch.tensor([encode_state(row["observation"], edge_index)], dtype=torch.float32, device=device)
        actions = torch.tensor(
            [[encode_action(action, row["observation"], edge_index) for action in legal]],
            dtype=torch.float32,
            device=device,
        )
        logits = scores(base, adapter, state, actions).squeeze(0)
        prediction = int(logits.argmax().item())
        predicted[legal[prediction]["type"]] += 1
        selected = legal[prediction] == chosen
        total += 1
        correct += selected
        if len(legal) > 1:
            nonforced += 1
            nonforced_correct += selected
        if chosen["type"] == "BUY_DEVELOPMENT_CARD":
            buy_total += 1
            buy_correct += selected
    return {
        "full_legal_accuracy": round(correct / total, 6),
        "nonforced_accuracy": round(nonforced_correct / nonforced, 6),
        "buy_development_accuracy": round(buy_correct / buy_total, 6),
        "buy_development_examples": buy_total,
        "predicted_action_types": dict(predicted),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=HERE / "data/catanatron_diverse.jsonl")
    parser.add_argument("--base-checkpoint", type=Path, default=HERE / "runs/legal_ranker_v1/model.pt")
    parser.add_argument("--output", type=Path, default=HERE / "runs/calibrated_ranker_v1")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--test-modulus", type=int, default=5, help="Every fifth game is sealed for validation.")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.negatives, args.test_modulus) < 1:
        parser.error("epoch, batch, negative, and split parameters must be positive")
    if args.pairwise_weight <= 0:
        parser.error("--pairwise-weight must be positive")
    device = torch.device("mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    base = LegalRanker(checkpoint["state_size"], checkpoint["action_size"]).to(device)
    base.load_state_dict(checkpoint.get("model", checkpoint.get("policy")))
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    with args.dataset.open(encoding="utf-8") as handle:
        first_row = json.loads(next(handle))
    edge_index = stable_edge_index(first_row["observation"])
    adapter = ActionCalibrationAdapter(checkpoint["state_size"]).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        pair_iterator = itertools.cycle(
            buy_end_batches(args.dataset, "train", args.batch_size, edge_index, args.test_modulus)
        )
        total = general_loss = pair_loss = 0.0
        for state_batch, action_batch in general_batches(
            args.dataset, "train", args.batch_size, args.negatives, edge_index, args.test_modulus
        ):
            states, actions = state_batch.to(device), action_batch.to(device)
            general_logits = scores(base, adapter, states, actions)
            labels = torch.zeros(len(states), dtype=torch.long, device=device)
            imitation_loss = F.cross_entropy(general_logits, labels)
            pair_states, pair_actions = next(pair_iterator)
            pair_logits = scores(base, adapter, pair_states.to(device), pair_actions.to(device))
            # Teacher Buy Dev is index zero, End Turn is index one.
            ranking_loss = F.softplus(-(pair_logits[:, 0] - pair_logits[:, 1])).mean()
            loss = imitation_loss + args.pairwise_weight * ranking_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += 1
            general_loss += float(imitation_loss.detach())
            pair_loss += float(ranking_loss.detach())
        adapter.eval()
        metrics = {
            "epoch": epoch,
            "steps": int(total),
            "imitation_loss": round(general_loss / total, 6),
            "buy_end_pairwise_loss": round(pair_loss / total, 6),
            **full_legal_metrics(args.dataset, base, adapter, edge_index, device, args.test_modulus),
        }
        history.append(metrics)
        print(json.dumps(metrics))
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save(
        {
            "schema": "catan-calibrated-ranker-v1",
            "state_size": checkpoint["state_size"],
            "action_size": checkpoint["action_size"],
            "base_model": base.state_dict(),
            "adapter": adapter.state_dict(),
            "config": config,
        },
        args.output / "model.pt",
    )
    (args.output / "metrics.json").write_text(json.dumps({"config": config, "history": history}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
