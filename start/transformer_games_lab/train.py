#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gamebench.models.tiny_transformer import TinyTransformerConfig
from gamebench.registry import make_agent, make_game
from gamebench.training.imitation import ImitationConfig, train_imitation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny next-action Transformer by imitation.")
    parser.add_argument("--game", choices=("othello", "connect4"), default="othello")
    parser.add_argument("--model", choices=("transformer",), default="transformer")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--teacher", default="greedy")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/latest_transformer.pt"))
    args = parser.parse_args()

    game = make_game(args.game)
    teacher = make_agent(args.teacher)
    model_config = TinyTransformerConfig(
        vocab_size=game.spec.token_vocab_size,
        action_size=game.spec.action_space_size,
        context_length=game.spec.max_game_length + 1,
        d_model=args.width,
        n_heads=args.heads,
        n_layers=args.layers,
        d_ff=args.mlp_width,
    )
    training_config = ImitationConfig(
        games=args.games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    model, report = train_imitation(game, teacher, model_config, training_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 1,
        "game": args.game,
        "model_config": model_config.as_dict(),
        "state_dict": model.state_dict(),
        "training_report": report,
    }, args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    final = report["epochs"][-1]
    print(f"trained {report['model']['parameters']:,} parameters on {report['dataset']['train']} examples")
    print(f"validation teacher accuracy: {final['teacher_action_accuracy']:.1%}")
    print(f"validation raw legal-action rate: {final['raw_legal_action_rate']:.1%}")
    print(f"checkpoint: {args.output}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
