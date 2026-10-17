"""Run an official-kit match and save complete, reproducible results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official-kit"))

from catan_kit import match
from catan_kit._patch import ensure_hash_seed
from catan_kit.evaluate import summarise


def main():
    ensure_hash_seed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed0", type=int, default=2026110000)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control", type=Path, help="Same-seed records to compare by game")
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error("--games must be a positive even number")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")

    records, load = match.play_chunk(str(args.a.resolve()), str(args.b.resolve()),
                                     [args.seed0 + i // 2 for i in range(args.games)],
                                     list(range(args.games)))
    result = {
        "kit_commit": "04fffac90e00610f60e0b06d919f262f4b3ed9db",
        "seed0": args.seed0,
        "games": args.games,
        "a": str(args.a),
        "b": str(args.b),
        "a_agent_sha256": hashlib.sha256(args.a.read_bytes()).hexdigest(),
        "b_agent_sha256": hashlib.sha256(args.b.read_bytes()).hexdigest(),
        "summary_a": summarise(records),
        "summary_b": summarise(records, "b"),
        "load": load,
        "records": records,
    }
    if args.control:
        previous = json.loads(args.control.read_text())
        if isinstance(previous, list):
            if len(previous) != args.games or any(
                row["board_seed"] != args.seed0 + i // 2
                for i, row in enumerate(previous)
            ):
                parser.error("Control board seeds do not match")
            old = [row["harness"] for row in previous]
        else:
            if previous.get("seed0") != args.seed0 or previous.get("games") != args.games:
                parser.error("Control seed schedule does not match")
            old = previous["records"]
        gain = sum(p["winner"] != "a" and r["winner"] == "a"
                   for p, r in zip(old, records))
        loss = sum(p["winner"] == "a" and r["winner"] != "a"
                   for p, r in zip(old, records))
        result["paired_vs_control"] = {"gained_games": gain, "lost_games": loss,
                                        "net_games": gain - loss}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "load")}, indent=2))


if __name__ == "__main__":
    main()
