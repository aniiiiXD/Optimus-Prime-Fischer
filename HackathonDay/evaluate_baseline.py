"""Record an unmodified official self-check or evaluator CLI invocation.

Run with HackathonDay/.venv/bin/python. This is a development driver, not
part of the submitted agent. It does not change the official harness.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
KIT = ROOT / "official-kit"
AGENT = ROOT / "onevone_baseline"
KIT_COMMIT = "04fffac90e00610f60e0b06d919f262f4b3ed9db"
ENGINE_COMMIT = "d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816"


def parse_summary(output: str) -> dict:
    patterns = {
        "wins_games": r"A win rate\s+([\d.]+)\s+\+-([\d.]+)\s+\((\d+)/(\d+)\)",
        "mean_cpu_ms": r"A mean CPU / game\s+([\d.]+) ms",
        "worst_cpu_ms": r"A worst CPU / game\s+([\d.]+) ms",
        "games_over_budget": r"A games over budget\s+(\d+)",
        "random_moves_after_budget": r"random moves played after the limit:\s*(\d+)",
        "errors_decisions": r"A errors\s+(\d+)\s+illegal index or exception, of (\d+) decisions",
        "timeouts_crashes": r"A timeouts/crashes\s+(\d+)\s*/\s*(\d+)",
        "load_seconds": r"A load time\s+([\d.]+) s",
        "wall_seconds": r"wall clock\s+([\d.]+) s",
    }
    found = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if not match:
            raise ValueError(f"Official evaluator output missing {key}")
        found[key] = match.groups()
    rate, interval, wins, games = found.pop("wins_games")
    errors, decisions = found.pop("errors_decisions")
    timeouts, crashes = found.pop("timeouts_crashes")
    result = {
        "wins": int(wins), "games": int(games),
        "win_rate": float(rate), "reported_ci95_half_width": float(interval),
        "errors": int(errors), "decisions": int(decisions),
        "timeouts": int(timeouts), "crashes": int(crashes),
    }
    for key, value in found.items():
        result[key] = int(value[0]) if key in (
            "games_over_budget", "random_moves_after_budget"
        ) else float(value[0])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed0", type=int, default=2026101800)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("Use letters, digits, underscores and dashes for the label")
    if args.games <= 0 or args.games % 2:
        parser.error("Use a positive even game count for complete seat pairs")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=KIT, text=True).strip()
    if commit != KIT_COMMIT:
        raise RuntimeError(f"Official kit commit changed: {commit}")
    engine = json.loads(metadata.distribution("catanatron").read_text("direct_url.json"))
    if engine.get("vcs_info", {}).get("commit_id") != ENGINE_COMMIT:
        raise RuntimeError("Installed engine is not the official pinned commit")

    folder = ROOT / "reports" / "onevone_baseline"
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / f"{args.label}.txt"
    manifest = folder / f"{args.label}.json"
    if log.exists() or manifest.exists():
        raise FileExistsError(f"Choose a new label; {args.label} already exists")
    command = [sys.executable, "-u", "-m", "catan_kit.selfcheck"] if args.selfcheck else [
        sys.executable, "-u", "-m", "catan_kit.evaluate",
        "--a", str(AGENT / "agent.py"),
        "--b", str(KIT / "examples" / "f_agent.py"),
        "--games", str(args.games), "--seed0", str(args.seed0),
    ]
    record = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "working_directory": str(KIT),
        "kit_commit": commit, "engine_commit": ENGINE_COMMIT,
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: metadata.version(name) for name in ("catanatron", "numpy", "networkx", "psutil")},
        "unused_ml_packages": "torch, scipy and scikit-learn omitted; this heuristic does not import them",
        "agent_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(AGENT.iterdir()) if p.suffix in (".py", ".json")},
    }
    if not args.selfcheck:
        record.update(games=args.games, boards=args.games // 2,
                      seed_start=args.seed0, seed_end=args.seed0 + args.games // 2 - 1,
                      opponent="official examples/f_agent.py", clock="official defaults, enforced")
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    lines = []
    print("Running", " ".join(command), flush=True)
    with log.open("w") as out:
        process = subprocess.Popen(command, cwd=KIT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            out.write(line)
            out.flush()
            lines.append(line)
            print(line, end="", flush=True)
        code = process.wait()
    record.update(exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat(),
                  status="passed" if code == 0 else "failed")
    if code == 0 and not args.selfcheck:
        try:
            record["result"] = parse_summary("".join(lines))
        except ValueError as exc:
            record.update(status="parse_failed", parse_error=str(exc))
            code = 1
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
