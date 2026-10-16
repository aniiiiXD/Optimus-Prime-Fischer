"""Evaluate Benjamin checkpoints on identical CPU-only R1 game seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from catan_rl.checkpoint import load_checkpoint
from catan_rl.eval.harness import EvalHarness
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.network import CatanPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--games-per-seat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = {}
    for path in args.checkpoints:
        policy = CatanPolicy()
        policy.set_board_geometry(build_geometry().as_dict_of_tensors())
        load_checkpoint(path, map_location="cpu").apply_to_policy(policy, strict=True)
        policy.cpu().eval()
        harness = EvalHarness(opponent_types=("heuristic",),
                              n_games_per_seat=args.games_per_seat,
                              seed=args.seed, device="cpu", max_turns=400,
                              ruleset="R1", audit_rules=False)
        started = time.perf_counter()
        result = harness.run(policy).results[0]
        results[str(path)] = {
            "wins": result.wins, "games": result.n, "win_rate": result.wr,
            "seat0_win_rate": result.wr_seat0,
            "seat1_win_rate": result.wr_seat1,
            "truncated": result.n_truncated,
            "wall_seconds": round(time.perf_counter() - started, 2),
        }
        print(path, results[str(path)], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"device": "cpu", "ruleset": "R1",
                                    "opponent": "Benjamin fixed heuristic",
                                    "seed": args.seed,
                                    "games_per_seat": args.games_per_seat,
                                    "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
