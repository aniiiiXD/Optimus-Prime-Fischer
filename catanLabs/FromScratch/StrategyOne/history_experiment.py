#!/usr/bin/env python3
"""Compare a public-history Transformer with a state-only residual control."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_legal_ranker import ACTION_TYPES, HERE, LegalRanker, candidate_set, encode_action, encode_state, stable_edge_index
from train_calibrated_ranker import ActionCalibrationAdapter

HISTORY = 16
EVENT_EXTRA = 14  # self/opponent, roll 2..12, present


def encode_history(obs: dict, edge_index: dict, length: int = HISTORY) -> list[list[float]]:
    action_size = len(ACTION_TYPES) + 54 + 72 + 19 + 5 + 1
    width = action_size + EVENT_EXTRA
    result = [[0.0] * width for _ in range(length)]
    for position, event in enumerate(obs.get("public_events", [])[-length:], start=max(0, length - len(obs.get("public_events", [])))):
        vector = encode_action(event, obs, edge_index)
        extras = [float(event.get("actor") == "self"), float(event.get("actor") == "opponent")]
        roll = event.get("result") if event.get("type") == "ROLL" else None
        extras.extend(float(roll == number) for number in range(2, 13))
        extras.append(1.0)
        result[position] = vector + extras
    return result


class HistoryResidual(nn.Module):
    def __init__(self, state_size: int, action_size: int, use_history: bool):
        super().__init__()
        self.use_history = use_history
        self.state = nn.Sequential(nn.Linear(state_size, 64), nn.ReLU())
        self.action = nn.Sequential(nn.Linear(action_size, 64), nn.ReLU())
        if use_history:
            self.event = nn.Linear(action_size + EVENT_EXTRA, 64)
            layer = nn.TransformerEncoderLayer(64, 2, 128, 0.0, batch_first=True, norm_first=False)
            self.sequence = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            input_size = 192
        else:
            input_size = 128
        self.output = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, state: torch.Tensor, actions: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        n = actions.shape[1]
        pieces = [self.state(state).unsqueeze(1).expand(-1, n, -1), self.action(actions)]
        if self.use_history:
            # Mask padding by averaging only real tokens. A fixed positional
            # signal lets the encoder distinguish repeated actions over time.
            present = history[:, :, -1] > 0
            positions = torch.arange(history.shape[1], device=history.device).float()
            positional = torch.sin(positions[None, :, None] / (10000 ** (torch.arange(64, device=history.device).float()[None, None, :] / 64)))
            encoded = self.sequence(self.event(history) + positional)
            summary = (encoded * present.unsqueeze(-1)).sum(1) / present.sum(1).clamp(min=1).unsqueeze(-1)
            pieces.append(summary.unsqueeze(1).expand(-1, n, -1))
        return self.output(torch.cat(pieces, -1)).squeeze(-1)


def load_examples(path: Path, edge_index: dict, max_train: int, max_test: int, negatives: int):
    train, test = [], []
    seen_train = seen_test = 0
    rng = random.Random(3402)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            candidates = candidate_set(row, negatives)
            if len(candidates) < 2:
                continue
            is_test = row["game_id"] % 5 == 0
            seen_test += is_test
            seen_train += not is_test
            target, seen, limit = (test, seen_test, max_test) if is_test else (train, seen_train, max_train)
            # Bounded reservoirs preserve game and phase coverage across 1,028 games.
            replacement = rng.randrange(seen) if len(target) >= limit else None
            if replacement is not None and replacement >= limit:
                continue
            obs = row["observation"]
            actions = [encode_action(action, obs, edge_index) for action in candidates]
            while len(actions) < negatives + 1:
                actions.append(actions[-1])
            sample = (encode_state(obs, edge_index), actions, encode_history(obs, edge_index))
            if len(target) < limit:
                target.append(sample)
            else:
                target[replacement] = sample
    return train, test


def batches(examples: list, size: int, shuffle: bool):
    order = list(range(len(examples)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), size):
        chunk = [examples[index] for index in order[start:start + size]]
        yield tuple(torch.tensor([item[j] for item in chunk], dtype=torch.float32) for j in range(3))


def score(base, adapter, residual, state, actions, history):
    with torch.no_grad():
        original = adapter(base(state, actions), state, actions)
    return original + residual(state, actions, history)


@torch.no_grad()
def evaluate(base, adapter, residual, examples, batch_size):
    residual.eval()
    total = correct = 0
    loss = 0.0
    for state, actions, history in batches(examples, batch_size, False):
        logits = score(base, adapter, residual, state, actions, history)
        labels = torch.zeros(len(state), dtype=torch.long)
        correct += int((logits.argmax(1) == 0).sum())
        loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        total += len(state)
    return {"examples": total, "accuracy": correct / total, "loss": loss / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=HERE / "runs/calibrated_ranker_v1/model.pt")
    parser.add_argument("--output", type=Path, default=HERE / "runs/history_experiment")
    parser.add_argument("--train-examples", type=int, default=12000)
    parser.add_argument("--test-examples", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--negatives", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(3402)
    random.seed(3402)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = LegalRanker(checkpoint["state_size"], checkpoint["action_size"])
    base.load_state_dict(checkpoint["base_model"])
    base.eval()
    adapter = ActionCalibrationAdapter(checkpoint["state_size"])
    adapter.load_state_dict(checkpoint["adapter"])
    adapter.eval()
    with (HERE / "data/catanatron_diverse.jsonl").open(encoding="utf-8") as handle:
        first = json.loads(next(handle))
    edge_index = stable_edge_index(first["observation"])
    started = time.monotonic()
    train, test = load_examples(HERE / "data/catanatron_diverse.jsonl", edge_index, args.train_examples, args.test_examples, args.negatives)
    print(json.dumps({"loaded_train": len(train), "loaded_test": len(test), "seconds": round(time.monotonic()-started, 1)}), flush=True)
    results = {}
    for name, use_history in (("state_control", False), ("history_transformer", True)):
        torch.manual_seed(3402)
        residual = HistoryResidual(checkpoint["state_size"], checkpoint["action_size"], use_history)
        optimizer = torch.optim.AdamW(residual.parameters(), lr=2e-4, weight_decay=1e-4)
        trajectory = []
        for epoch in range(args.epochs + 1):
            validation = evaluate(base, adapter, residual, test, 64)
            trajectory.append({"epoch": epoch, **validation})
            print(json.dumps({"model": name, **trajectory[-1]}), flush=True)
            if epoch == args.epochs:
                break
            residual.train()
            for state, actions, history in batches(train, 64, True):
                logits = score(base, adapter, residual, state, actions, history)
                labels = torch.zeros(len(state), dtype=torch.long)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        torch.save({"residual": residual.state_dict(), "state_size": checkpoint["state_size"], "action_size": checkpoint["action_size"], "use_history": use_history}, args.output / f"{name}.pt")
        results[name] = trajectory
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
