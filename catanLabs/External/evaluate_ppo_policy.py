"""CPU-only seat-rotated evaluation of an upstream PPO checkpoint without search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "catan-rl"
sys.path.insert(0, str(UPSTREAM / "training"))

import catan_py  # noqa: E402
from ppo import PolicyValueNet  # noqa: E402
from evaluate_upstream import wilson  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=int, choices=(7, 10), default=10)
    parser.add_argument("--opponent", choices=("heuristic", "heuristic_v2"), default="heuristic")
    parser.add_argument("--visibility", choices=("perfect", "realistic"), default="perfect")
    parser.add_argument("--games", type=int, default=48, help="Episodes per seat")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (checkpoint["codec_version"] != catan_py.CODEC_VERSION or
            checkpoint["obs_version"] != catan_py.OBS_VERSION or
            checkpoint["num_actions"] != catan_py.NUM_ACTIONS or
            checkpoint["obs_dim"] != catan_py.OBS_DIM):
        raise ValueError("Checkpoint and environment schemas do not match")
    net = PolicyValueNet(catan_py.OBS_DIM, catan_py.NUM_ACTIONS,
                         checkpoint["config"]["hidden"]).cpu().eval()
    net.load_state_dict(checkpoint["model_state"])
    results = []
    started = time.perf_counter()
    with torch.inference_mode():
        for seat in range(4):
            seats = [args.opponent] * 4
            seats[seat] = "policy"
            env = catan_py.VecEnv(min(args.games, 48), victory_target=args.target,
                                  visibility=args.visibility, seed=args.seed,
                                  seats=seats)
            obs, masks, _ = env.observe()
            completed = 0
            wins = 0
            vp_sum = 0
            caps = 0
            while completed < args.games:
                logits, _ = net(torch.as_tensor(obs), torch.as_tensor(masks))
                actions = logits.argmax(dim=1).numpy().astype(np.uint32)
                obs, masks, _, _, _, _ = env.step(actions)
                for turns, winner, vps, capped in env.take_episode_stats():
                    if completed >= args.games:
                        break
                    completed += 1
                    wins += int(winner == seat)
                    vp_sum += int(vps[seat])
                    caps += int(capped)
            results.append({"seat": seat, "games": completed, "wins": wins,
                            "win_rate": round(wins / completed, 4),
                            "mean_vp": round(vp_sum / completed, 3), "caps": caps})
            print(f"seat {seat}: {wins}/{completed} ({wins/completed:.1%})", flush=True)
    games = sum(row["games"] for row in results)
    wins = sum(row["wins"] for row in results)
    report = {
        "protocol": "upstream-catan-rl-ppo-greedy-seat-rotation-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_global_step": checkpoint["global_step"],
        "target_vp": args.target,
        "opponent": args.opponent,
        "visibility": args.visibility,
        "games_per_seat": args.games,
        "seed": args.seed,
        "by_seat": results,
        "total": {"games": games, "wins": wins, "win_rate": round(wins / games, 4),
                  "wilson95": wilson(wins, games),
                  "mean_vp": round(sum(row["mean_vp"] * row["games"] for row in results) / games, 3),
                  "caps": sum(row["caps"] for row in results)},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["total"], indent=2))


if __name__ == "__main__":
    main()
