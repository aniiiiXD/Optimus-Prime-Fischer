#!/usr/bin/env python3
"""Attribute the MLP's BUY_DEVELOPMENT_CARD-versus-END_TURN failure.

This is an inspection tool, not training.  It reports the action-score collapse
and feature-group ablations on a game-level held-out split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

from train_legal_ranker import ACTION_TYPES, HERE, LegalRanker, encode_action, encode_state, stable_edge_index


STATE_GROUPS = (
    ("tiles_and_robber", 19 * 8),
    ("building_layout", 54 * 4),
    ("road_layout", 72 * 2),
    ("ports", 54 * 6),
    ("own_resources", 5),
    ("own_development_cards", 5),
    ("own_vp", 1),
    ("opponent_public", 9),
    ("bank_resources", 5),
    ("turn_phase", 7),
)


def state_slices() -> dict[str, slice]:
    start, result = 0, {}
    for name, size in STATE_GROUPS:
        result[name] = slice(start, start + size)
        start += size
    return result


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=HERE / "data/catanatron_diverse.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=HERE / "runs/legal_ranker_v1/model.pt")
    parser.add_argument("--holdout-modulus", type=int, default=10, help="Use game IDs divisible by this number.")
    parser.add_argument("--output", type=Path, default=HERE / "reports/mlp_buy_development_attribution.json")
    args = parser.parse_args()
    if args.holdout_modulus < 2:
        parser.error("--holdout-modulus must be at least 2")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = LegalRanker(checkpoint["state_size"], checkpoint["action_size"])
    model.load_state_dict(checkpoint.get("model", checkpoint.get("policy")))
    model.eval()
    with args.dataset.open(encoding="utf-8") as handle:
        first_row = json.loads(next(handle))
    edge_index = stable_edge_index(first_row["observation"])
    groups = state_slices()
    if sum(size for _, size in STATE_GROUPS) != checkpoint["state_size"]:
        raise RuntimeError("State feature layout changed; update STATE_GROUPS before attribution.")

    examples = correct = 0
    predicted_types: Counter[str] = Counter()
    buy_ranks, buy_minus_end, top_minus_buy = [], [], []
    ablation_deltas: dict[str, list[float]] = defaultdict(list)
    gradient_times_input: dict[str, list[float]] = defaultdict(list)
    for line in args.dataset.open(encoding="utf-8"):
        row = json.loads(line)
        if row["game_id"] % args.holdout_modulus != 0:
            continue
        if row["chosen_action"]["type"] != "BUY_DEVELOPMENT_CARD":
            continue
        legal = row["legal_actions"]
        buy_index = next((i for i, action in enumerate(legal) if action["type"] == "BUY_DEVELOPMENT_CARD"), None)
        end_index = next((i for i, action in enumerate(legal) if action["type"] == "END_TURN"), None)
        if buy_index is None or end_index is None:
            continue
        state_features = encode_state(row["observation"], edge_index)
        action_features = [encode_action(action, row["observation"], edge_index) for action in legal]
        state = torch.tensor([state_features], dtype=torch.float32, requires_grad=True)
        actions = torch.tensor([action_features], dtype=torch.float32)
        scores = model(state, actions).squeeze(0)
        predicted_index = int(scores.argmax().item())
        predicted_types[legal[predicted_index]["type"]] += 1
        correct += predicted_index == buy_index
        examples += 1
        rank = int((scores > scores[buy_index]).sum().item()) + 1
        buy_ranks.append(float(rank))
        margin = scores[buy_index] - scores[end_index]
        buy_minus_end.append(float(margin.item()))
        top_minus_buy.append(float((scores.max() - scores[buy_index]).item()))

        # Local feature sensitivity for the BUY-minus-END score margin.
        model.zero_grad(set_to_none=True)
        margin.backward()
        for name, section in groups.items():
            gradient_times_input[name].append(float((state.grad[0, section] * state.detach()[0, section]).sum().item()))

        # Counterfactual ablation: remove one observed feature group while
        # retaining the exact same legal action pair. Positive means removing
        # the group made BUY score better relative to END.
        with torch.no_grad():
            for name, section in groups.items():
                masked = state.detach().clone()
                masked[:, section] = 0.0
                masked_scores = model(masked, actions).squeeze(0)
                masked_margin = masked_scores[buy_index] - masked_scores[end_index]
                ablation_deltas[name].append(float((masked_margin - margin.detach()).item()))

    if not examples:
        raise RuntimeError("No held-out BUY_DEVELOPMENT_CARD examples found.")
    action_offset = checkpoint["state_size"]
    first_layer = model.network[0].weight.detach()
    action_type_norms = {
        action: round(float(first_layer[:, action_offset + index].norm().item()), 6)
        for index, action in enumerate(ACTION_TYPES)
    }
    ablation_summary = {
        group: {
            "mean_margin_change_when_masked": mean(values),
            "mean_absolute_margin_change_when_masked": mean([abs(value) for value in values]),
            "mean_gradient_times_input": mean(gradient_times_input[group]),
        }
        for group, values in ablation_deltas.items()
    }
    report = {
        "purpose": "Held-out attribution for teacher BUY_DEVELOPMENT_CARD choices. Positive masked-margin change means the observed group suppresses BUY relative to END.",
        "checkpoint": str(args.checkpoint),
        "holdout_modulus": args.holdout_modulus,
        "examples": examples,
        "buy_development_exact_accuracy": round(correct / examples, 6),
        "predicted_action_types": dict(predicted_types),
        "mean_buy_rank_among_legal_actions": mean(buy_ranks),
        "mean_buy_minus_end_score": mean(buy_minus_end),
        "mean_top_minus_buy_score": mean(top_minus_buy),
        "state_group_ablation": ablation_summary,
        "first_layer_action_type_l2_norm": action_type_norms,
        "interpretation": "Action-type norms are capacity diagnostics only; use held-out score/rank and state-group ablations to identify behavioural failures.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
