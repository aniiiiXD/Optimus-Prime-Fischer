"""Extract only selected pretrained weights from the pinned Git trees."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
FILES = [
    ("eli", "models/catan-512-best.pt", "eli_best.pt"),
    ("eli", "models/catan-512.ctnn", "eli.ctnn"),
    ("settlers", "RL/results/default_after_update_3825.pt", "settlers_3825.pt"),
]


def main() -> None:
    sources = json.loads((ROOT / "sources.json").read_text())
    output = ROOT / "artifacts" / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, git_path, filename in FILES:
        source = sources[name]
        repo = ROOT / "vendor" / name
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if actual != source["commit"]:
            raise RuntimeError(f"Unexpected source revision: {name}")
        spec = f"{source['commit']}:{git_path}"
        blob_id = subprocess.check_output(["git", "rev-parse", spec], cwd=repo, text=True).strip()
        target = output / filename
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".part")
            with temporary.open("wb") as stream:
                subprocess.run(["git", "show", spec], cwd=repo, stdout=stream, check=True)
            if temporary.stat().st_size > 64 * 1024**2:
                raise RuntimeError(f"Unexpectedly large weight: {temporary}")
            if temporary.read_bytes()[:80].startswith(b"version https://git-lfs"):
                raise RuntimeError(f"LFS pointer, not checkpoint: {git_path}")
            temporary.replace(target)
        actual_blob = subprocess.check_output(["git", "hash-object", str(target)], text=True).strip()
        if actual_blob != blob_id:
            raise RuntimeError(f"Checkpoint differs from pinned Git blob: {target}")
        record = {"source": name, "source_commit": actual, "git_path": git_path,
                  "git_blob": blob_id, "file": filename, "bytes": target.stat().st_size,
                  "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        records.append(record)
        print(f"{filename}: {record['bytes']:,} bytes, SHA256 {record['sha256']}", flush=True)
    (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
