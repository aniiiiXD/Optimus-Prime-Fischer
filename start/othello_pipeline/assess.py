"""Compare checkpoints on the same group-held-out teacher positions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch

from .dataset import deduplicate_examples, load_examples, split_examples
from .model import ModelConfig, OthelloTransformer
from .train import run_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out Othello teacher agreement and regret.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    dataset_sha = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    examples, _ = deduplicate_examples(load_examples(args.dataset))
    results = []
    for path in args.checkpoints:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        report = saved["report"]
        if report["dataset_sha256"] != dataset_sha:
            raise ValueError(f"checkpoint does not belong to this dataset: {path}")
        config = report["training_config"]
        _, validation, _ = split_examples(examples, config["validation_fraction"], config["seed"])
        model = OthelloTransformer(ModelConfig(**saved["model_config"])).eval()
        model.load_state_dict(saved["state_dict"])
        metrics = run_epoch(
            model, validation, config["batch_size"], config["temperature"], config["score_scale"],
            None, random.Random(config["seed"]), config.get("legality_pos_weight", 1.0),
        )
        random_optimal = statistics.mean(
            sum(score == max(example.teacher_scores) for score in example.teacher_scores) / len(example.legal)
            for example in validation
        )
        random_regret = statistics.mean(
            max(example.teacher_scores) - statistics.mean(example.teacher_scores)
            for example in validation
        )
        results.append({"checkpoint": str(path), "heldout_examples": len(validation), "metrics": metrics,
                        "random_legal_optimal_rate": random_optimal, "random_legal_regret": random_regret})
        print(f"{path.name}: optimal {metrics['teacher_optimal_rate']:.1%} "
              f"(random legal {random_optimal:.1%}), regret {metrics['teacher_regret']:.2f} "
              f"(random legal {random_regret:.2f}), raw legal recall {metrics['raw_legal_recall']:.1%}")
    report = {"dataset_sha256": dataset_sha, "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
