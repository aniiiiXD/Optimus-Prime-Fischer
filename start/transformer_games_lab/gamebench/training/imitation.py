"""Minimal next-action imitation trainer; intentionally not reinforcement learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import time
from typing import Any

import torch
from torch.nn import functional as F

from gamebench.agents.base import Agent
from gamebench.games.base import GameAdapter
from gamebench.models.tiny_transformer import TinyGameTransformer, TinyTransformerConfig


Example = tuple[tuple[int, ...], int, tuple[int, ...]]


@dataclass(frozen=True)
class ImitationConfig:
    games: int = 100
    epochs: int = 4
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    validation_fraction: float = 0.2
    seed: int = 0


def generate_examples(game: GameAdapter, teacher: Agent, games: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    for game_index in range(games):
        state = game.reset(seed + game_index)
        while not state.is_terminal():
            action = teacher.select_action(state.clone(), rng)
            examples.append((state.history_tokens(), action, state.legal_actions()))
            state.apply_action(action)
    return examples


def _batch(examples: list[Example], indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, ...]]]:
    selected = [examples[index] for index in indices]
    max_length = max(len(tokens) for tokens, _, _ in selected)
    tokens = torch.zeros((len(selected), max_length), dtype=torch.long)
    lengths = torch.empty(len(selected), dtype=torch.long)
    targets = torch.empty(len(selected), dtype=torch.long)
    legal: list[tuple[int, ...]] = []
    for row, (sequence, target, legal_actions) in enumerate(selected):
        tokens[row, : len(sequence)] = torch.tensor(sequence)
        lengths[row] = len(sequence)
        targets[row] = target
        legal.append(legal_actions)
    return tokens, lengths, targets, legal


def _evaluate(model: TinyGameTransformer, examples: list[Example], batch_size: int) -> dict[str, float]:
    model.eval()
    correct = 0
    legal_predictions = 0
    losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            indices = list(range(start, min(start + batch_size, len(examples))))
            tokens, lengths, targets, legal = _batch(examples, indices)
            logits = model(tokens, lengths)
            losses.append(float(F.cross_entropy(logits, targets)))
            predicted = logits.argmax(dim=-1).tolist()
            correct += sum(prediction == int(target) for prediction, target in zip(predicted, targets))
            legal_predictions += sum(prediction in allowed for prediction, allowed in zip(predicted, legal))
    count = max(1, len(examples))
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "teacher_action_accuracy": correct / count,
        "raw_legal_action_rate": legal_predictions / count,
        "examples": len(examples),
    }


def train_imitation(
    game: GameAdapter,
    teacher: Agent,
    model_config: TinyTransformerConfig,
    train_config: ImitationConfig | None = None,
) -> tuple[TinyGameTransformer, dict[str, Any]]:
    config = train_config or ImitationConfig()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    examples = generate_examples(game, teacher, config.games, config.seed)
    rng = random.Random(config.seed)
    rng.shuffle(examples)
    split = max(1, min(len(examples) - 1, int(len(examples) * (1 - config.validation_fraction))))
    training, validation = examples[:split], examples[split:]
    model = TinyGameTransformer(model_config).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    epoch_metrics: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        order = list(range(len(training)))
        rng.shuffle(order)
        losses: list[float] = []
        for offset in range(0, len(order), config.batch_size):
            indices = order[offset : offset + config.batch_size]
            tokens, lengths, targets, _ = _batch(training, indices)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, lengths)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_metrics = _evaluate(model, validation, config.batch_size)
        epoch_metrics.append({
            "epoch": float(epoch + 1),
            "training_loss": sum(losses) / max(1, len(losses)),
            **validation_metrics,
        })
    report = {
        "game": game.metadata(),
        "teacher": teacher.metadata(),
        "model": {**model_config.as_dict(), "parameters": model.parameter_count},
        "training": asdict(config),
        "dataset": {"total": len(examples), "train": len(training), "validation": len(validation)},
        "epochs": epoch_metrics,
        "wall_seconds": time.perf_counter() - started,
    }
    return model, report
