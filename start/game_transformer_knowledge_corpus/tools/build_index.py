#!/usr/bin/env python3
"""Build a small local JSONL retrieval index from notes and repository READMEs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rag"
MAX_CHARS = 1800
OVERLAP = 180


def selected_files() -> list[Path]:
    files = list((ROOT / "notes").glob("*.md"))
    files.extend((ROOT / "docs").glob("*.md"))
    files.append(ROOT / "README.md")
    files.append(ROOT / "REQUESTED_SET.md")
    for repo in sorted((ROOT / "repos").iterdir()) if (ROOT / "repos").is_dir() else []:
        if not repo.is_dir():
            continue
        readmes = sorted(p for p in repo.iterdir() if p.is_file() and p.name.lower().startswith("readme"))
        if readmes:
            files.append(readmes[0])
    return [p for p in files if p.exists() and p.is_file()]


def sections(text: str) -> list[tuple[str, str]]:
    heading = "Introduction"
    body: list[str] = []
    output: list[tuple[str, str]] = []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            if body:
                output.append((heading, "\n".join(body).strip()))
            heading, body = match.group(1), []
        else:
            body.append(line)
    if body:
        output.append((heading, "\n".join(body).strip()))
    return output


def chunks(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text] if text else []
    result: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        if end < len(text):
            split = text.rfind("\n", start + MAX_CHARS // 2, end)
            if split > start:
                end = split
        result.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - OVERLAP, start + 1)
    return [part for part in result if part]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    manifest = []
    for path in selected_files():
        relative = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        manifest.append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        index = 0
        for heading, section in sections(text):
            for part in chunks(section):
                records.append({
                    "id": hashlib.sha1(f"{relative}:{heading}:{index}".encode()).hexdigest()[:16],
                    "source": relative,
                    "heading": heading,
                    "chunk_index": index,
                    "text": part,
                })
                index += 1
    (OUT / "chunks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"documents": manifest, "chunks": len(records)}, indent=2) + "\n", encoding="utf-8")
    print(f"indexed {len(manifest)} documents into {len(records)} chunks")


if __name__ == "__main__":
    main()
