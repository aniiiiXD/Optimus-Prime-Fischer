"""Run isolated, bounded baseline workers; never trains or changes weights.

Example: .venv/bin/python -m catan_lab.benchmark --agents eli --games 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parent
AGENTS = ["benjamin", "catanatron", "catanatron_1v1", "settlers", "eli"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=AGENTS)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--classical-policy", choices=["weighted", "value", "alphabeta"], default="weighted")
    parser.add_argument("--classical-opponents", choices=["random", "weighted", "value"], default="random")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reports/baselines.json")
    args = parser.parse_args()
    if args.games < 1 or args.threads < 1 or args.seconds <= 0:
        parser.error("games, threads and seconds must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "machine": platform.platform(), "python": sys.version,
              "hardware": {"physical_cores": psutil.cpu_count(logical=False),
                           "logical_cores": psutil.cpu_count(),
                           "physical_ram_bytes": psutil.virtual_memory().total},
              "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ("benchmark.py", "worker.py")},
              "sources": json.loads((ROOT / "sources.json").read_text()),
              "protocol": vars(args) | {"output": str(args.output)}, "results": []}
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), RAYON_NUM_THREADS=str(args.threads),
               SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    for agent in args.agents:
        target = args.output.parent / f"{args.output.stem}_{agent}.json"
        # Never mistake an older successful worker file for this invocation.
        target = target.with_name(f"{target.stem}_{time.time_ns()}.json")
        command = [sys.executable, "-m", "catan_lab.worker", "--agent", agent,
                   "--games", str(args.games), "--threads", str(args.threads),
                   "--seconds", str(args.seconds), "--seed", str(args.seed),
                   "--classical-policy", args.classical_policy,
                   "--classical-opponents", args.classical_opponents,
                   "--output", str(target.resolve())]
        print(f"{agent}: starting, limit {args.seconds:g}s", flush=True)
        started = time.perf_counter()
        try:
            process = subprocess.run(command, cwd=ROOT.parent, env=env, capture_output=True,
                                     text=True, timeout=args.seconds + 45)
            if process.returncode or not target.exists():
                result = {"agent": agent, "status": "error", "returncode": process.returncode,
                          "details": (process.stderr + process.stdout)[-7000:]}
            else:
                result = json.loads(target.read_text())
        except subprocess.TimeoutExpired:
            result = {"agent": agent, "status": "process_timeout", "limit_s": args.seconds + 45}
        result["process_wall_s"] = time.perf_counter() - started
        report["results"].append(result)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"{agent}: {result['status']}; {result.get('completed_games', 0)} games; "
              f"{result['process_wall_s']:.2f}s", flush=True)
    report["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
