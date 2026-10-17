"""Record games under the unmodified official harness, including every fault."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
KIT = ROOT.parent/"official-kit"
sys.path.insert(0, str(KIT))
from catan_kit import match
from catan_kit._patch import ensure_hash_seed
from catan_kit.evaluate import summarise


def hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix in (".py", ".json", ".npz")
            and not any(x in p.parts for x in ("__pycache__", "reports", "data", "tests"))}


def main():
    ensure_hash_seed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=ROOT/"agent.py")
    parser.add_argument("--opponent", type=Path, default=KIT/"examples/f_agent.py")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed0", type=int, default=2026850000)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.games < 2 or args.games%2 or not args.label.replace("_", "").replace("-", "").isalnum():
        parser.error("Use complete board pairs and an alphanumeric label")
    folder = ROOT/"reports"
    folder.mkdir(exist_ok=True)
    output = folder/(args.label+".json")
    if output.exists():
        raise FileExistsError(output)
    kit_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=KIT, text=True).strip()
    if kit_commit != "04fffac90e00610f60e0b06d919f262f4b3ed9db":
        raise RuntimeError("Unexpected official kit version")
    engine = json.loads(metadata.distribution("catanatron").read_text("direct_url.json"))["vcs_info"]["commit_id"]
    if engine != "d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816":
        raise RuntimeError("Unexpected engine version")
    record = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "agent": str(args.agent.resolve()), "opponent": str(args.opponent.resolve()),
              "kit_commit": kit_commit, "engine_commit": engine, "clock": "official defaults, enforced",
              "seed0": args.seed0, "games": args.games, "boards": args.games//2,
              "agent_hashes": hashes(args.agent.resolve().parent),
              "core_hashes": hashes(ROOT),
              "opponent_hashes": hashes(args.opponent.resolve().parent)}
    output.write_text(json.dumps(record, indent=2)+"\n")
    start = time.monotonic()
    original = match.play_game
    completed = 0

    def progress(*a, **kw):
        nonlocal completed
        result = original(*a, **kw)
        completed += 1
        if completed%20 == 0 or completed == args.games:
            print(f"{args.label}: {completed}/{args.games}", flush=True)
        return result

    match.play_game = progress
    try:
        records, load = match.play_chunk(str(args.agent.resolve()), str(args.opponent.resolve()),
                                         [args.seed0+i//2 for i in range(args.games)], list(range(args.games)))
        record.update(status="completed", a=summarise(records, "a"), b=summarise(records, "b"),
                      records=records, load=load, wall_seconds=time.monotonic()-start)
    except BaseException as error:
        record.update(status="failed", error=repr(error), completed_games=completed)
        raise
    finally:
        match.play_game = original
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["source_unchanged"] = record["agent_hashes"] == hashes(args.agent.resolve().parent)
        record["core_unchanged"] = record["core_hashes"] == hashes(ROOT)
        output.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k:record[k] for k in ("status", "a", "b", "wall_seconds", "source_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
