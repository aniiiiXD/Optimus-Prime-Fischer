"""CPU teacher-distillation trainer. Requires a verified labeled JSONL file."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import random
import resource
import time

import torch
from torch.nn import functional as F

from .dataset import Example, deduplicate_examples, load_examples, split_examples
from .model import ModelConfig, OthelloTransformer, encode_board


def make_batch(examples: list[Example], temperature: float, score_scale: float) -> dict[str, torch.Tensor]:
    if temperature <= 0 or score_scale <= 0:
        raise ValueError("temperature and score scale must be positive")
    batch = len(examples)
    board = torch.stack([encode_board(list(example.board), example.player) for example in examples])
    side = torch.tensor([example.player for example in examples], dtype=torch.long)
    candidates = torch.arange(65, dtype=torch.long).expand(batch, -1)
    candidate_mask = torch.ones((batch, 65), dtype=torch.bool)
    legal = torch.zeros((batch, 65), dtype=torch.bool)
    teacher_probs = torch.zeros((batch, 65), dtype=torch.float32)
    q_targets = torch.zeros((batch, 65), dtype=torch.float32)
    raw_scores = torch.full((batch, 65), float("-inf"), dtype=torch.float32)
    results = torch.zeros(batch, dtype=torch.float32)
    result_mask = torch.zeros(batch, dtype=torch.bool)
    for index, example in enumerate(examples):
        actions = torch.tensor(example.legal, dtype=torch.long)
        scores = torch.tensor(example.teacher_scores, dtype=torch.float32)
        legal[index, actions] = True
        teacher_probs[index, actions] = torch.softmax(scores / temperature, dim=0)
        q_targets[index, actions] = torch.tanh(scores / score_scale)
        raw_scores[index, actions] = scores
        if example.result is not None:
            results[index] = example.result
            result_mask[index] = True
    return {
        "board": board,
        "side": side,
        "candidates": candidates,
        "candidate_mask": candidate_mask,
        "legal": legal,
        "teacher_probs": teacher_probs,
        "q_targets": q_targets,
        "raw_scores": raw_scores,
        "results": results,
        "result_mask": result_mask,
    }


def loss_and_metrics(
    model: OthelloTransformer, batch: dict[str, torch.Tensor], legality_pos_weight: float = 1.0
) -> tuple[torch.Tensor, dict[str, float]]:
    if legality_pos_weight <= 0:
        raise ValueError("legality positive weight must be positive")
    output = model(batch["board"], batch["side"], batch["candidates"], batch["candidate_mask"])
    masked_policy = output.policy_logits.masked_fill(~batch["legal"], torch.finfo(output.policy_logits.dtype).min)
    log_policy = F.log_softmax(masked_policy, dim=-1)
    policy_loss = -(batch["teacher_probs"] * log_policy).sum(dim=-1).mean()
    q_loss = F.smooth_l1_loss(output.action_values[batch["legal"]], batch["q_targets"][batch["legal"]])
    legality_loss = F.binary_cross_entropy_with_logits(
        output.legality_logits,
        batch["legal"].float(),
        pos_weight=torch.tensor(legality_pos_weight, dtype=output.legality_logits.dtype),
    )
    if batch["result_mask"].any():
        value_loss = F.smooth_l1_loss(output.state_value[batch["result_mask"]], batch["results"][batch["result_mask"]])
    else:
        value_loss = output.state_value.sum() * 0.0
    loss = policy_loss + 0.5 * q_loss + 0.25 * value_loss + 0.1 * legality_loss

    with torch.no_grad():
        chosen = masked_policy.argmax(dim=-1)
        best = batch["raw_scores"].argmax(dim=-1)
        batch_ids = torch.arange(chosen.shape[0])
        regret = batch["raw_scores"][batch_ids, best] - batch["raw_scores"][batch_ids, chosen]
        legal_pred = output.legality_logits > 0
        true_legal = batch["legal"]
        metrics = {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "q_loss": float(q_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "legality_loss": float(legality_loss.detach()),
            "teacher_top1": float((chosen == best).float().mean()),
            "teacher_optimal_rate": float((regret <= 0).float().mean()),
            "teacher_regret": float(regret.mean()),
            "raw_legal_recall": float((legal_pred & true_legal).sum() / true_legal.sum()),
            "raw_illegal_false_positive_rate": float((legal_pred & ~true_legal).sum() / (~true_legal).sum()),
        }
    return loss, metrics


def run_epoch(
    model: OthelloTransformer,
    examples: list[Example],
    batch_size: int,
    temperature: float,
    score_scale: float,
    optimizer: torch.optim.Optimizer | None,
    rng: random.Random,
    legality_pos_weight: float = 1.0,
) -> dict[str, float]:
    model.train(optimizer is not None)
    order = list(range(len(examples)))
    if optimizer is not None:
        rng.shuffle(order)
    totals: dict[str, float] = {}
    seen = 0
    for start in range(0, len(order), batch_size):
        selected = [examples[index] for index in order[start : start + batch_size]]
        batch = make_batch(selected, temperature, score_scale)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = loss_and_metrics(model, batch, legality_pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.inference_mode():
                _, metrics = loss_and_metrics(model, batch, legality_pos_weight)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * len(selected)
        seen += len(selected)
    return {key: value / seen for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Othello student from verified teacher scores on CPU.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--score-scale", type=float, default=16.0)
    parser.add_argument("--legality-pos-weight", type=float, default=8.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--mlp-width", type=int, default=128)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) <= 0 or args.learning_rate <= 0 or args.legality_pos_weight <= 0:
        parser.error("epochs, batch size, threads, and learning rate must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    examples, duplicates_removed = deduplicate_examples(load_examples(args.dataset))
    training, validation, overlap_removed = split_examples(examples, args.validation_fraction, args.seed)
    config = ModelConfig(args.width, args.heads, args.blocks, args.mlp_width)
    model = OthelloTransformer(config).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    started = time.perf_counter()
    epochs: list[dict[str, object]] = []
    best_regret = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, training, args.batch_size, args.temperature, args.score_scale, optimizer, rng, args.legality_pos_weight)
        validation_metrics = run_epoch(model, validation, args.batch_size, args.temperature, args.score_scale, None, rng, args.legality_pos_weight)
        epochs.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["teacher_regret"] < best_regret:
            best_regret = validation_metrics["teacher_regret"]
            best_weights = copy.deepcopy(model.state_dict())
        print(f"epoch {epoch}: train loss {train_metrics['loss']:.3f}, held-out regret {validation_metrics['teacher_regret']:.3f}, top1 {validation_metrics['teacher_top1']:.1%}")

    wall_seconds = time.perf_counter() - started
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = maxrss if platform.system() == "Darwin" else maxrss * 1024
    fingerprint = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "dataset_sha256": fingerprint,
        "teacher_id": examples[0].teacher_id,
        "examples_total": len(examples),
        "duplicates_removed_by_symmetry": duplicates_removed,
        "train_examples": len(training),
        "validation_examples": len(validation),
        "overlap_removed_from_validation": overlap_removed,
        "model_config": config.to_dict(),
        "parameters": model.parameter_count,
        "training_config": {key: value for key, value in vars(args).items() if key not in ("dataset", "output")},
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "epochs": epochs,
        "best_validation_regret": best_regret,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": config.to_dict(), "state_dict": best_weights, "report": report}, args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved best student: {args.output} ({model.parameter_count:,} params, {wall_seconds:.1f} s)")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
