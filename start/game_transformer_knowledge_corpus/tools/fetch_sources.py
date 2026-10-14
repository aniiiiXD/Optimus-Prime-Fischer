#!/usr/bin/env python3
"""Fetch the corpus sources reproducibly.

GitHub source trees are fetched file-by-file so a single TLS reset does not lose a
large archive. Existing non-empty repositories are left untouched. Files larger
than MAX_GITHUB_BLOB_BYTES are recorded instead of downloaded; these are normally
datasets or model checkpoints, not source code.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
MAX_GITHUB_BLOB_BYTES = 8 * 1024 * 1024
TIMEOUT = 90
SOURCE_ONLY_REPOS = {"Emergent-World-Representations-Othello"}
SOURCE_SUFFIXES = {
    ".py", ".ipynb", ".md", ".rst", ".toml", ".cfg", ".ini", ".yml", ".yaml",
    ".json", ".sh", ".html", ".css", ".js", ".png", ".jpg", ".jpeg", ".svg",
}
SOURCE_BASENAMES = {"license", "requirements.txt", "setup.py", "manifest.in", ".gitignore"}

REPOSITORIES = [
    ("NandhaKishorM/laya", "main", "laya"),
    ("likenneth/othello_world", "master", "othello_world"),
    ("alxndrTL/othello_mamba", "main", "othello_mamba"),
    ("tchauffi/ChessTransformer", "main", "ChessTransformer"),
    ("sgrvinod/chess-transformers", "main", "chess-transformers"),
    ("edarsem/chessformer", "master", "chessformer"),
    ("Atenrev/chessformers", "master", "chessformers"),
    ("AHMerrill/connect-4", "main", "connect-4"),
    ("DeanHazineh/Emergent-World-Representations-Othello", "main", "Emergent-World-Representations-Othello"),
]

PAPERS = [
    ("2412.12119", "01_mastering_board_games_external_internal_planning.pdf"),
    ("2210.13382", "02_emergent_world_representations_othello_gpt.pdf"),
    ("2008.04057", "03_chess_transformer.pdf"),
    ("2007.03500", "04_go_transformer.pdf"),
    ("2605.19091", "05_chessformer_unified_architecture.pdf"),
    ("2409.12272", "06_mastering_chess_transformer.pdf"),
    ("2207.08766", "07_word_play_othello.pdf"),
]


def request(session: requests.Session, url: str, *, stream: bool = False) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = session.get(url, timeout=TIMEOUT, stream=stream)
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(str(last_error))


def safe_path(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"unsafe repository path: {relative}")
    return target


def fetch_repo(session: requests.Session, slug: str, branch: str, local_name: str) -> dict[str, Any]:
    destination = ROOT / "repos" / local_name
    existing = sum(1 for p in destination.rglob("*") if p.is_file()) if destination.exists() else 0
    commit = request(session, f"https://api.github.com/repos/{slug}/commits/{branch}").json()["sha"]
    if existing and local_name not in SOURCE_ONLY_REPOS:
        return {"repo": slug, "branch": branch, "commit": commit, "status": "existing", "files": existing, "skipped": []}

    destination.mkdir(parents=True, exist_ok=True)
    tree = request(session, f"https://api.github.com/repos/{slug}/git/trees/{commit}?recursive=1").json()
    skipped: list[dict[str, Any]] = []
    downloaded = 0
    for entry in tree.get("tree", []):
        if entry.get("type") != "blob":
            continue
        size = int(entry.get("size", 0))
        path = str(entry["path"])
        if local_name in SOURCE_ONLY_REPOS:
            suffix = Path(path).suffix.lower()
            basename = Path(path).name.lower()
            if suffix not in SOURCE_SUFFIXES and basename not in SOURCE_BASENAMES:
                skipped.append({"path": path, "bytes": size, "reason": "dataset/checkpoint/output excluded from source corpus"})
                continue
        if size > MAX_GITHUB_BLOB_BYTES:
            skipped.append({"path": path, "bytes": size, "reason": "larger than source-corpus limit"})
            continue
        target = safe_path(destination, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == size:
            downloaded += 1
            continue
        raw = f"https://raw.githubusercontent.com/{slug}/{commit}/{path}"
        target.write_bytes(request(session, raw).content)
        downloaded += 1
    return {"repo": slug, "branch": branch, "commit": commit, "status": "downloaded", "files": downloaded, "skipped": skipped}


def fetch_paper(session: requests.Session, arxiv_id: str, filename: str) -> dict[str, Any]:
    destination = ROOT / "papers" / filename
    if destination.exists() and destination.stat().st_size > 10_000:
        return {"arxiv": arxiv_id, "status": "existing", "bytes": destination.stat().st_size}
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = request(session, f"https://arxiv.org/pdf/{arxiv_id}", stream=True)
    digest = hashlib.sha256()
    total = 0
    with destination.open("wb") as handle:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                handle.write(chunk)
                digest.update(chunk)
                total += len(chunk)
    if destination.read_bytes()[:4] != b"%PDF":
        raise RuntimeError(f"downloaded file is not a PDF: {destination}")
    return {"arxiv": arxiv_id, "status": "downloaded", "bytes": total, "sha256": digest.hexdigest()}


def main() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "game-transformer-knowledge-corpus/1.0"})
    report: dict[str, Any] = {"repositories": [], "papers": []}
    for repo in REPOSITORIES:
        print(f"repository: {repo[0]}", flush=True)
        report["repositories"].append(fetch_repo(session, *repo))
    for paper in PAPERS:
        print(f"paper: arXiv {paper[0]}", flush=True)
        report["papers"].append(fetch_paper(session, *paper))
    output = ROOT / "download_manifest.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
