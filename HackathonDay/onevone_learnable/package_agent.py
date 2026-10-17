"""Build a small, explicit runtime-only ZIP for the official upload script."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
RUNTIME = ("agent.py", "policy.py", "expert.py", "features.py", "runtime.py", "tactics.py", "config.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest_path = output.with_suffix(".json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(output)
    config = json.loads((ROOT/"config.json").read_text())
    files = {name: (ROOT/name).read_bytes() for name in RUNTIME}
    if config["scorer"] != "heuristic":
        checkpoint = Path(config["checkpoint"])
        if checkpoint.is_absolute() or ".." in checkpoint.parts:
            raise ValueError("Submission checkpoints must be inside the runtime folder")
        files[checkpoint.as_posix()] = (ROOT/checkpoint).read_bytes()
    files["LICENSE"] = (ROOT.parent/"official-kit/LICENSE").read_bytes()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files["NOTICE.md"] = (
        "# Roofline Maxxers OneVOne\n\n"
        "Official entry point: agent.py. Runtime dependencies: Python 3.11, NumPy, "
        "and the official pinned catanatron engine. No PyTorch is imported during play.\n\n"
        "Source: https://github.com/aniiiiXD/Optimus-Prime-Fischer\n"
        f"Source commit before packaging: {commit}\n\n"
        "Uses catanatron by Bryan Collazo (GPL-3.0-or-later), installed by the official kit: "
        "https://github.com/bcollazo/catanatron\n"
        "Official interface and engine: https://github.com/lyric-hyde/catan-hackathon-agent\n\n"
        "The heuristic is derived from this team's pre-event OneVOne experiments. "
        "This package does not certify compliance with the event's built-during-event rule; "
        "organizers determine eligibility. The neural redesign and outcome checkpoints "
        "were created during the current development session.\n"
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or "agent.py" not in archive.namelist():
            raise RuntimeError("Archive verification failed")
        if set(archive.namelist()) != set(files):
            raise RuntimeError("Unexpected archive contents")
    if output.stat().st_size > 100_000_000:
        raise ValueError("Archive exceeds the official 100 MB limit")
    record = {
        "status": "packaged_not_uploaded", "team": "Roofline Maxxers",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit, "scorer": config["scorer"],
        "zip": output.name, "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "files_sha256": {name: hashlib.sha256(payload).hexdigest() for name,payload in sorted(files.items())},
        "eligibility": "Derivative of pre-event team code; organizer confirmation required",
    }
    manifest_path.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k != "files_sha256"}, indent=2))


if __name__ == "__main__":
    main()
