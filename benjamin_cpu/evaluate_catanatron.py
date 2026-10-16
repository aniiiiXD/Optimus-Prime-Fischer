"""Paired-seat direct Benjamin-policy transfer versus Catanatron web bot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent / "Hyde"
sys.path.insert(0, str(MAIN))
from catanLabs.Catanatron.tournament_protocol import install_v5_protocol  # noqa: E402
from benjamin_cpu.catanatron_adapter import create_agent  # noqa: E402


def run_game_function():
    path = MAIN / "catanLabs/FromScratch/StrategyOne/experiments/catanatron_80pct/run.py"
    spec = importlib.util.spec_from_file_location("catanatron_80pct_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_game


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boards", type=int, default=20)
    p.add_argument("--seeds", type=Path, default=ROOT / "benjamin_cpu/reports/heldout_206_seeds.json")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    install_v5_protocol()
    run_game = run_game_function()
    boards = json.loads(args.seeds.read_text())["boards"][:args.boards]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.out.exists():
        existing = [json.loads(line) for line in args.out.read_text().splitlines() if line]
    done = {(r["seed"], r["seat"]) for r in existing}
    for board in boards:
        for seat in (0, 1):
            if (board["seed"], seat) in done:
                continue
            row = run_game(
                board["seed"], seat,
                candidate_factory=lambda color, seed: create_agent(color, args.checkpoint, seed),
                max_seconds=120, max_actions=10000, max_turns=500, trace=False,
            )
            with args.out.open("a") as out:
                out.write(json.dumps(row) + "\n")
            existing.append(row)
            print(f"{args.checkpoint.name} {len(existing)}/{2*len(boards)} wins={sum(r['ours_won'] for r in existing)} last={row['status']} error={row.get('error')}", flush=True)
    print(json.dumps({"games": len(existing), "wins": sum(r["ours_won"] for r in existing),
                      "errors": [r.get("error") for r in existing if r["status"] == "error"],
                      "capped": sum(r["status"] == "capped" for r in existing)}, indent=2))


if __name__ == "__main__":
    main()
