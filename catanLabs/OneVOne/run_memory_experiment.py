"""Train and test a tiny CPU Transformer using only OneVOne teacher games.

The seed split is by board, so seat-swapped games never cross train/holdout.
History is reconstructed from earlier observations of the same player.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catanLabs.Catanatron.engine import source_info
from catanLabs.Catanatron.tournament_protocol import install_v5_protocol
from catanLabs.OneVOne.memory_transformer import MemoryPolicy, load_agent, make_sequence

ENGINE = source_info()
sys.path.insert(0, ENGINE["python_root"])
install_v5_protocol()


def load_examples(path: Path, history_len: int, holdout_boards: int):
    games = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["teacher"] == "onevone":
                games[(row["seed"], row["seat"])].append(row)
    seeds = sorted({key[0] for key in games})
    if len(seeds) <= holdout_boards:
        raise ValueError("Need more seeds than holdout boards")
    holdout = set(seeds[-holdout_boards:])
    train, valid = [], []
    for (seed, _seat), rows in sorted(games.items()):
        prior = []
        for row in sorted(rows, key=lambda x: x["decision"]):
            current = np.asarray(row["state"], dtype=np.float32)
            if len(row["actions"]) > 1:
                example = (make_sequence(current, prior, history_len),
                           np.asarray(row["actions"], dtype=np.float32),
                           int(row["chosen"]))
                (valid if seed in holdout else train).append(example)
            prior.append(current)
    return train, valid, sorted(holdout), len(games)


def collate(rows, use_history):
    batch = len(rows)
    seq_len, state_dim = rows[0][0].shape
    action_dim = rows[0][1].shape[1]
    max_actions = max(len(row[1]) for row in rows)
    states = np.stack([row[0] for row in rows])
    if not use_history:
        states[:, :-1] = 0
    history_mask = np.any(states != 0, axis=2)
    actions = np.zeros((batch, max_actions, action_dim), np.float32)
    action_mask = np.zeros((batch, max_actions), bool)
    labels = np.zeros(batch, np.int64)
    for i, (_, choices, chosen) in enumerate(rows):
        actions[i, :len(choices)] = choices
        action_mask[i, :len(choices)] = True
        labels[i] = chosen
    return (torch.from_numpy(states), torch.from_numpy(history_mask),
            torch.from_numpy(actions), torch.from_numpy(action_mask),
            torch.from_numpy(labels))


def evaluate(model, rows, use_history, batch_size):
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: collate(x, use_history))
    model.eval()
    correct = count = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for states, history_mask, actions, action_mask, labels in loader:
            logits = model(states, history_mask, actions, action_mask)
            correct += int((logits.argmax(1) == labels).sum())
            loss_sum += float(nn.functional.cross_entropy(logits, labels,
                                                          reduction="sum"))
            count += len(labels)
    return {"choices": count, "top1": correct / count,
            "nll": loss_sum / count}


def train_variant(train, valid, width, history_len, use_history,
                  epochs, batch_size, seed, checkpoint):
    torch.manual_seed(seed)
    random.seed(seed)
    model = MemoryPolicy(train[0][0].shape[1], train[0][1].shape[1],
                         width=width, history_len=history_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4)
    started = time.perf_counter()
    records = []
    best = -1.0
    for epoch in range(epochs):
        loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda x: collate(x, use_history))
        model.train()
        for states, history_mask, actions, action_mask, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(states, history_mask, actions, action_mask)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        score = evaluate(model, valid, use_history, batch_size)
        records.append({"epoch": epoch + 1, **score})
        print(f"{checkpoint.stem} epoch {epoch+1}: {score}", flush=True)
        if score["top1"] > best:
            best = score["top1"]
            torch.save({"model": model.state_dict(), "state_dim": model.state_proj[0].in_features,
                        "action_dim": model.action_proj.in_features, "width": width,
                        "history_len": history_len, "use_history": use_history}, checkpoint)
    return {"best_holdout_top1": best, "epochs": records,
            "train_seconds": round(time.perf_counter() - started, 2),
            "parameter_count": sum(p.numel() for p in model.parameters())}


def game_tests(checkpoints, seeds, max_game_seconds):
    from catanLabs.FromScratch.StrategyOne.experiments.catanatron_80pct.run import run_game
    from catanLabs.OneVOne.agent import create_agent
    from catanatron import Color
    factories = {
        "onevone": lambda color, seed: create_agent(color, None, seed),
        **{name: (lambda color, seed, path=path: load_agent(color, path))
           for name, path in checkpoints.items()},
    }
    results = {}
    for name, factory in factories.items():
        games = []
        for seed in seeds:
            for seat in range(2):
                game = run_game(seed, seat, candidate_factory=factory,
                                max_seconds=max_game_seconds, max_actions=10000,
                                max_turns=500, trace=False)
                games.append(game)
                print(f"{name} seed={seed} seat={seat}: "
                      f"{game['status']} win={game['ours_won']}", flush=True)
        latencies = [game["decision_ms"]["ours"]["median"] for game in games
                     if game["decision_ms"]["ours"]["median"] is not None]
        results[name] = {"games": len(games),
                         "wins": sum(game["ours_won"] for game in games),
                         "completed": sum(game["status"] == "completed" for game in games),
                         "errors": [game.get("error") for game in games if game["status"] == "error"],
                         "capped": sum(game["status"] == "capped" for game in games),
                         "median_game_decision_ms": statistics.median(latencies) if latencies else None,
                         "per_game": games}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(
        "/Users/anixd/Documents/Hyde/catanLabs/OneVOne/data/bootstrap_20272000_64boards.jsonl.gz"))
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "runs/memory_pilot")
    parser.add_argument("--holdout-boards", type=int, default=12)
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--game-seeds", type=int, default=4)
    parser.add_argument("--max-game-seconds", type=float, default=60.0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train, valid, holdout, total_games = load_examples(args.data, args.history_len,
                                                        args.holdout_boards)
    print(f"loaded {len(train)} train / {len(valid)} validation choices "
          f"from {total_games} OneVOne games", flush=True)
    report = {"data": str(args.data), "engine": ENGINE, "train_choices": len(train),
              "holdout_choices": len(valid), "holdout_seeds": holdout,
              "teacher_games": total_games, "history_len": args.history_len,
              "width": args.width, "variants": {}}
    checkpoints = {}
    for name, use_history in (("memoryless", False), ("memory", True)):
        checkpoint = args.out / f"{name}.pt"
        report["variants"][name] = train_variant(
            train, valid, args.width, args.history_len, use_history,
            args.epochs, args.batch_size, 20261017, checkpoint)
        checkpoints[name] = checkpoint
        (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    if args.game_seeds:
        seeds = list(range(20269900, 20269900 + args.game_seeds))
        report["fresh_game_seeds"] = seeds
        report["fresh_games"] = game_tests(checkpoints, seeds, args.max_game_seconds)
    report["total_seconds"] = round(time.perf_counter() - started, 2)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print("REPORT", args.out / "report.json", flush=True)


if __name__ == "__main__":
    main()
