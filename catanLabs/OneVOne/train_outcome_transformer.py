"""Train a history-aware outcome critic on an 80/20 split of 1,028 games."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from catanLabs.OneVOne.outcome_transformer import OutcomeTransformer, encode_observation


MAIN = Path("/Users/anixd/Documents/Hyde")
DEFAULT_META = MAIN / "catanLabs/FromScratch/StrategyOne/data/catanatron_diverse.meta.json"
DEFAULT_DATA = MAIN / "catanLabs/FromScratch/StrategyOne/data/catanatron_diverse.jsonl"


def split_games(meta_path: Path, seed: int) -> tuple[set[int], set[int], list[dict]]:
    games = json.loads(meta_path.read_text())["games"]
    ids = [int(item["game_id"]) for item in games]
    if len(ids) != 1028 or len(set(ids)) != 1028:
        raise ValueError("Expected exactly 1,028 distinct dataset games")
    random.Random(seed).shuffle(ids)
    n_test = round(len(ids) * 0.2)
    return set(ids[n_test:]), set(ids[:n_test]), games


def sample_game(rows: list[dict], max_rows: int):
    positions = np.linspace(0, len(rows) - 1, min(max_rows, len(rows)), dtype=int)
    return [rows[int(index)] for index in sorted(set(positions))]


def encode_dataset(data_path: Path, fit_ids: set[int], valid_ids: set[int], test_ids: set[int],
                   max_rows: int, cache: Path):
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        data = np.load(cache)
        return {name: data[name] for name in data.files}
    groups = {split: {"state": [], "history": [], "label": [], "game_id": []}
              for split in ("fit", "valid", "test")}
    current_id, current_rows = None, []
    edge_index = None

    def finish():
        nonlocal edge_index
        if current_id is None:
            return
        split = "fit" if current_id in fit_ids else "valid" if current_id in valid_ids else "test"
        target = groups[split]
        for row in sample_game(current_rows, max_rows):
            if edge_index is None:
                from catanLabs.FromScratch.StrategyOne.train_legal_ranker import stable_edge_index
                edge_index = stable_edge_index(row["observation"])
            state, history = encode_observation(row["observation"], edge_index)
            target["state"].append(state)
            target["history"].append(history)
            target["label"].append(float(row["outcome"]["won"]))
            target["game_id"].append(current_id)

    with data_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            game_id = int(row["game_id"])
            if game_id not in fit_ids and game_id not in valid_ids and game_id not in test_ids:
                raise ValueError(f"Unexpected game ID {game_id}")
            if current_id is not None and game_id != current_id:
                finish()
                current_rows = []
            current_id = game_id
            current_rows.append(row)
        finish()
    arrays = {}
    for split, group in groups.items():
        for name, values in group.items():
            dtype = np.float32 if name != "game_id" else np.int32
            arrays[f"{split}_{name}"] = np.asarray(values, dtype=dtype)
    if len(set(arrays["fit_game_id"])) != len(fit_ids):
        raise AssertionError("Fit games missing from encoded dataset")
    if len(set(arrays["valid_game_id"])) != len(valid_ids):
        raise AssertionError("Validation games missing from encoded dataset")
    if len(set(arrays["test_game_id"])) != len(test_ids):
        raise AssertionError("Holdout games missing from encoded dataset")
    np.savez(cache, **arrays)
    return arrays


@torch.inference_mode()
def metrics(model, dataset, batch_size, drop_history=False):
    model.eval()
    prediction, target = [], []
    for states, history, labels in DataLoader(dataset, batch_size=batch_size):
        if drop_history:
            history = torch.zeros_like(history)
        p = torch.sigmoid(model(states, history))
        prediction.append(p.numpy())
        target.append(labels.numpy())
    p = np.concatenate(prediction).astype(np.float64)
    y = np.concatenate(target).astype(np.float64)
    eps = 1e-8
    nll = float(np.mean(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))))
    brier = float(np.mean((p - y) ** 2))
    # Pairwise-rank AUC via ranks, with ties credited half.
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), np.float64)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and sorted_p[j] == sorted_p[i]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    positive, negative = int(y.sum()), int(len(y) - y.sum())
    auc = float((ranks[y == 1].sum() - positive * (positive + 1) / 2)
                / (positive * negative)) if positive and negative else None
    return {"n": len(y), "win_fraction": float(y.mean()),
            "nll": nll, "brier": brier, "auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "runs/outcome_1028")
    parser.add_argument("--split-seed", type=int, default=20261017)
    parser.add_argument("--max-rows-per-game", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261017)
    np.random.seed(20261017)
    args.out.mkdir(parents=True, exist_ok=True)
    train_ids, test_ids, games = split_games(args.meta, args.split_seed)
    valid_ids = set(random.Random(args.split_seed + 1).sample(sorted(train_ids),
                                                          round(len(train_ids) * 0.1)))
    fit_ids = train_ids - valid_ids
    split = {"seed": args.split_seed, "train_game_ids": sorted(train_ids),
             "fit_game_ids": sorted(fit_ids), "validation_game_ids": sorted(valid_ids),
             "test_game_ids": sorted(test_ids),
             "train_seeds": sorted(g["seed"] for g in games if g["game_id"] in train_ids),
             "test_seeds": sorted(g["seed"] for g in games if g["game_id"] in test_ids)}
    (args.out / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    started = time.perf_counter()
    data = encode_dataset(args.data, fit_ids, valid_ids, test_ids, args.max_rows_per_game,
                          args.out / "encoded.npz")
    print("encoded", {k: v.shape for k, v in data.items()}, flush=True)
    def tensors(prefix):
        return TensorDataset(torch.from_numpy(data[prefix + "_state"]),
                             torch.from_numpy(data[prefix + "_history"]),
                             torch.from_numpy(data[prefix + "_label"]))
    train, valid, test = tensors("fit"), tensors("valid"), tensors("test")
    model = OutcomeTransformer(data["fit_state"].shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    best_nll = float("inf")
    report = {"source_games": len(games), "train_games": len(train_ids),
              "validation_games": len(valid_ids), "test_games": len(test_ids),
              "fit_rows": len(train), "validation_rows": len(valid),
              "test_rows": len(test), "history_len": 16,
              "parameters": sum(p.numel() for p in model.parameters()),
              "epochs": [], "source_data": str(args.data)}
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for states, history, labels in DataLoader(train, batch_size=args.batch_size,
                                                  shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            logits = model(states, history)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        val = metrics(model, valid, args.batch_size)
        row = {"epoch": epoch + 1, "train_nll": float(np.mean(losses)), **val}
        report["epochs"].append(row)
        print(row, flush=True)
        if val["nll"] < best_nll:
            best_nll = val["nll"]
            torch.save({"model": model.state_dict(), "state_dim": model.state_dim,
                        "width": model.width}, args.out / "best.pt")
        (args.out / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    from catanLabs.OneVOne.outcome_transformer import load_model
    best_model = load_model(args.out / "best.pt")
    report["best_holdout"] = metrics(best_model, test, args.batch_size)
    report["best_holdout_no_history"] = metrics(best_model, test, args.batch_size,
                                                 drop_history=True)
    report["wall_seconds"] = round(time.perf_counter() - started, 2)
    (args.out / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print("report", args.out / "training.json", flush=True)


if __name__ == "__main__":
    main()
