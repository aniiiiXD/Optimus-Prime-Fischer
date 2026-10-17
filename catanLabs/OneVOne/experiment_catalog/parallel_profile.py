"""Finish a slow benchmark profile in independent, resumable shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmark_1028 import DEFAULT_OUTPUT, board_rows, load_benchmark, load_factory


def read_rows(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def run_shard(output: Path, profile: str, index: int, count: int):
    main_done = {(row["seed"], row["seat"])
                 for row in read_rows(output / f"{profile}.jsonl")}
    path = output / f"{profile}.shard{index}.jsonl"
    rows = read_rows(path)
    done = {(row["seed"], row["seat"]) for row in rows}
    if len(done) != len(rows) or done & main_done:
        raise RuntimeError(f"Duplicate game in shard {index}")
    benchmark = load_benchmark()
    benchmark.install_v5_protocol()
    factory = load_factory(profile)
    with path.open("a", buffering=1) as stream:
        for board_index, (game_id, seed) in enumerate(board_rows()):
            if board_index % count != index:
                continue
            for seat in (0, 1):
                key = (seed, seat)
                if key in main_done or key in done:
                    continue
                row = benchmark.run_game(
                    seed, seat, candidate_factory=factory, max_seconds=60.0,
                    max_actions=10000, max_turns=500, trace=False,
                )
                row["game_id"] = game_id
                row["profile"] = profile
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                done.add(key)
                if len(done) % 100 == 0:
                    print(f"shard {index}: {len(done)} games", flush=True)


def combine(output: Path, profile: str, count: int):
    main = output / f"{profile}.jsonl"
    paths = [main] + [output / f"{profile}.shard{i}.jsonl" for i in range(count)]
    rows = [row for path in paths for row in read_rows(path)]
    keys = [(row["seed"], row["seat"]) for row in rows]
    expected = {(seed, seat) for _, seed in board_rows() for seat in (0, 1)}
    if len(keys) != len(expected) or set(keys) != expected:
        raise RuntimeError(f"Incomplete/duplicated {profile} schedule: {len(keys)}")
    temp = main.with_suffix(".merged")
    with temp.open("w") as stream:
        for row in sorted(rows, key=lambda row: (row["game_id"], row["seat"])):
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    temp.replace(main)
    print(f"merged {len(rows)} {profile} matches", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", default="alphabeta2_unpruned")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.shards < 1 or args.shard is not None and not 0 <= args.shard < args.shards:
        parser.error("Invalid shard configuration")
    if args.shard is not None:
        run_shard(output, args.profile, args.shard, args.shards)
        return
    processes = []
    for index in range(args.shards):
        log = output / f"{args.profile}.shard{index}.log"
        stream = log.open("a")
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                   "--profile", args.profile, "--shards", str(args.shards),
                   "--shard", str(index)]
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONHASHSEED": "0",
                                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        processes.append((process, stream))
    failures = []
    for index, (process, stream) in enumerate(processes):
        code = process.wait()
        stream.close()
        print(f"shard {index}: exit {code}", flush=True)
        if code:
            failures.append(index)
    if failures:
        raise RuntimeError(f"{args.profile} shards failed: {failures}")
    combine(output, args.profile, args.shards)


if __name__ == "__main__":
    main()
