"""Small parallel paired-seat screen using the official match harness."""

import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import platform
import sys
import time

import critic_experiment as experiment
import action_learning  # Install the same optional clock wrapper in spawned workers.


def main():
    experiment.ensure_hash_seed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, default=experiment.HERE.parents[2] / "my_team/nano.py")
    parser.add_argument("--clocked", action="store_true", help="Enforce the official clock; local iteration is uncapped by default")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1040000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ["ONEVONE_UNCLOCKED"] = "0" if args.clocked else "1"
    if args.games < 2 or args.games % 2 or args.workers < 1:
        parser.error("games must be positive and even; workers must be positive")
    output = args.output or experiment.HERE.parent / "reports/quick_eval" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    experiment.WORKERS = args.workers
    source_paths = set(experiment.HERE.glob("*.py")) | set((experiment.HERE.parent.parent / "Competitive").glob("*.py"))
    source_paths.update((experiment.HERE.parent / "agent.py", experiment.HERE.parent / "configs/default.json", args.a.resolve()))
    source_paths.update((experiment.HERE / "vendor/kit/catan_kit").glob("*.py"))
    source_paths.add(experiment.HERE / "vendor/kit/examples/f_agent.py")
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_paths)}
    metadata = {"runtime": {"platform": platform.platform(), "python": sys.version, "executable": sys.executable},
                "seed0": args.seed, "games": args.games, "workers": args.workers,
                "agent": str(args.a.resolve()), "opponent": "F",
                "clock": "official" if args.clocked else "unclocked", "source_sha256": source_hashes}
    experiment.write_json(output / "manifest.json", metadata)
    start = time.perf_counter()
    result, _ = experiment.run(output.resolve(), "eval", args.a.resolve(strict=True), args.games, args.seed)
    result["wall_s"] = round(time.perf_counter() - start, 3)
    result.update(metadata)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in source_hashes.items()):
        raise RuntimeError("Source changed during evaluation; discard this run")
    experiment.write_json(output / "results.json", result)
    print(f"{args.games} games in {result['wall_s']:.1f}s; records: {output}", flush=True)


if __name__ == "__main__":
    main()
