"""Fetch pinned source checkouts, excluding model archives and run outputs.

Run with the workspace Python; this creates no new environment and installs no
packages. Git LFS smudging is disabled. Selected checkpoint downloads are separate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
EXCLUDES = [
    "/*", "!*.pt", "!*.pth", "!*.ckpt", "!*.ctnn", "!*.npy", "!*.npz",
    "!*.pkl", "!*.pickle", "!*.zip", "!*.tar.gz", "!*.mp4", "!*.gif",
    "!/runs/", "!/data/", "!/RL/results/", "!/models/",
]


def git(args: list[str], cwd: Path, *, capture: bool = False) -> str:
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    cmd = ["git", "-c", "filter.lfs.smudge=", "-c", "filter.lfs.required=false", *args]
    for attempt in range(3):
        result = subprocess.run(cmd, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE if capture else None)
        if result.returncode == 0:
            return result.stdout.strip() if capture else ""
        if attempt < 2:
            time.sleep(1)
    raise RuntimeError(f"Git command failed: {args}")


def main() -> None:
    manifest = json.loads((ROOT / "sources.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", metavar="REPOSITORY")
    args = parser.parse_args()
    if any(name not in manifest for name in args.names):
        parser.error(f"Unknown repository; choose from {', '.join(manifest)}")
    for name in args.names or manifest:
        source = manifest[name]
        destination = ROOT / "vendor" / name
        destination.mkdir(parents=True, exist_ok=True)
        if not (destination / ".git").exists():
            if any(destination.iterdir()):
                raise RuntimeError(f"Refusing nonempty non-git directory: {destination}")
            git(["init", "--quiet"], destination)
            git(["remote", "add", "origin", source["url"]], destination)
            git(["config", "remote.origin.promisor", "true"], destination)
            git(["config", "remote.origin.partialclonefilter", "blob:none"], destination)
        elif git(["remote", "get-url", "origin"], destination, capture=True) != source["url"]:
            raise RuntimeError(f"Unexpected existing repository: {destination}")
        status = git(["status", "--porcelain"], destination, capture=True)
        if status:
            raise RuntimeError(f"Preserving modified checkout: {destination}")
        current = subprocess.run(["git", "rev-parse", "--verify", "HEAD"],
                                 cwd=destination, text=True, capture_output=True)
        if current.returncode == 0 and current.stdout.strip() != source["commit"]:
            raise RuntimeError(f"Existing revision differs; preserving {destination}")
        if current.returncode != 0:
            git(["fetch", "--depth", "1", "--filter=blob:none", "origin", source["commit"]], destination)
            git(["sparse-checkout", "init", "--no-cone"], destination)
            git(["sparse-checkout", "set", "--no-cone", *EXCLUDES], destination)
            git(["checkout", "--detach", source["commit"]], destination)
        print(f"{name}: {source['commit']} ready", flush=True)


if __name__ == "__main__":
    main()
