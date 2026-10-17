"""Train either scorer on outcomes; split whole board seeds and branch siblings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
# Training can use a separate torch environment without installing the engine.
# features imports the engine, so use its installed sources if training elsewhere.
from features import SCHEMA
from networks import MLP, Transformer
from runtime import NeuralScorer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--architecture", choices=("mlp", "transformer"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()
    if min(args.epochs, args.threads) < 1:
        p.error("epochs and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with np.load(args.data, allow_pickle=False) as data:
        if str(data["schema"].item()) != SCHEMA:
            raise ValueError("Dataset schema mismatch")
        arrays = {k: data[k] for k in ("board", "candidate", "outcome", "weight", "seed", "group")}
    seeds = np.unique(arrays["seed"])
    if len(seeds) < 4:
        raise ValueError("Need at least four independent boards")
    shuffled = rng.permutation(seeds)
    valid_seeds = shuffled[:max(1, len(seeds)//5)]
    valid = np.flatnonzero(np.isin(arrays["seed"], valid_seeds))
    train = np.flatnonzero(~np.isin(arrays["seed"], valid_seeds))
    tensors = {k: torch.as_tensor(arrays[k], dtype=torch.float32)
               for k in ("board", "candidate", "outcome", "weight")}
    model = MLP() if args.architecture == "mlp" else Transformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    history, best, best_state, best_epoch = [], float("inf"), None, 0
    started = time.monotonic()

    def loss_for(indices):
        logits = model(tensors["board"][indices], tensors["candidate"][indices])
        labels, weights = tensors["outcome"][indices], tensors["weight"][indices]
        losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        return (losses*weights).sum()/weights.sum(), logits

    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0
        permutation = rng.permutation(train)
        for offset in range(0, len(train), 128):
            indices = permutation[offset:offset+128]
            optimizer.zero_grad()
            loss, _ = loss_for(indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            total += loss.item()*len(indices)
        model.eval()
        with torch.no_grad():
            numerator = denominator = 0
            for offset in range(0, len(valid), 128):
                indices = valid[offset:offset+128]
                loss, _ = loss_for(indices)
                weight = float(tensors["weight"][indices].sum())
                numerator += loss.item()*weight
                denominator += weight
            validation_loss = numerator/denominator
        row = {"epoch": epoch, "train_bce": total/len(train), "validation_bce": validation_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best:
            best, best_epoch = validation_loss, epoch
            best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = args.output/"model.npz"
    np.savez_compressed(checkpoint, schema=SCHEMA, architecture=args.architecture,
                        **{k:v.numpy() for k,v in best_state.items()})
    runtime = NeuralScorer(checkpoint)
    sample = valid[:min(24, len(valid))]
    with torch.no_grad():
        reference = model(tensors["board"][sample], tensors["candidate"][sample]).numpy()
    exported = np.asarray([runtime.predict(arrays["board"][i], arrays["candidate"][i:i+1])[0] for i in sample])
    difference = float(np.max(np.abs(reference-exported)))
    if difference > 1e-4:
        raise AssertionError(f"Torch/NumPy export mismatch: {difference}")
    base = float(np.clip(arrays["outcome"][train].mean(), 1e-6, 1-1e-6))
    y = arrays["outcome"][valid]
    report = {"architecture": args.architecture, "schema": SCHEMA,
              "parameters": sum(p.numel() for p in model.parameters()),
              "rows": len(y)+len(train), "train_rows": len(train), "validation_rows": len(valid),
              "train_seeds": sorted(set(int(s) for s in seeds)-set(int(s) for s in valid_seeds)),
              "validation_seeds": sorted(int(s) for s in valid_seeds),
              "selection": "minimum validation BCE; final match seeds unused",
              "selected_epoch": best_epoch, "validation_bce": best,
              "constant_validation_bce": float(-(y*np.log(base)+(1-y)*np.log(1-base)).mean()),
              "max_export_logit_difference": difference, "history": history,
              "wall_seconds": time.monotonic()-started,
              "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (args.output/"training.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("history", "train_seeds", "validation_seeds")}, indent=2))


if __name__ == "__main__":
    main()
