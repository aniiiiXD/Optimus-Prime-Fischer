"""Persistent cache for expensive, deterministic search-teacher labels.

This cache stores *position labels*, not model activations or gradients.  Its
key includes every setting that changes a label so stale supervision is never
silently reused after the teacher/search recipe changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


TEACHER_LABEL_VERSION = "negamax-ab-heuristic-v1-label-v1"


class TeacherLabelCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS teacher_labels (
                cache_key TEXT PRIMARY KEY,
                canonical_board TEXT NOT NULL,
                current_player INTEGER NOT NULL,
                depth INTEGER NOT NULL,
                node_limit INTEGER,
                temperature REAL NOT NULL,
                teacher_version TEXT NOT NULL,
                payload TEXT NOT NULL
            )"""
        )
        self.connection.commit()
        self.pending_writes = 0

    @staticmethod
    def make_key(
        board: tuple[int, ...],
        player: int,
        *,
        depth: int,
        node_limit: int | None,
        temperature: float,
    ) -> str:
        identity = {
            "board": list(board),
            "current_player": player,
            "depth": depth,
            "node_limit": node_limit,
            "temperature": temperature,
            "teacher_version": TEACHER_LABEL_VERSION,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(
        self,
        board: tuple[int, ...],
        player: int,
        *,
        depth: int,
        node_limit: int | None,
        temperature: float,
    ) -> dict[str, Any] | None:
        key = self.make_key(
            board,
            player,
            depth=depth,
            node_limit=node_limit,
            temperature=temperature,
        )
        result = self.connection.execute(
            "SELECT payload FROM teacher_labels WHERE cache_key = ?", (key,)
        ).fetchone()
        return json.loads(result[0]) if result else None

    def put(
        self,
        row: dict[str, Any],
        board: tuple[int, ...],
        player: int,
        *,
        depth: int,
        node_limit: int | None,
        temperature: float,
    ) -> None:
        key = self.make_key(
            board,
            player,
            depth=depth,
            node_limit=node_limit,
            temperature=temperature,
        )
        payload = json.dumps(row, separators=(",", ":"), allow_nan=False)
        self.connection.execute(
            """INSERT OR IGNORE INTO teacher_labels
               (cache_key, canonical_board, current_player, depth, node_limit,
                temperature, teacher_version, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                key,
                json.dumps(board, separators=(",", ":")),
                player,
                depth,
                node_limit,
                temperature,
                TEACHER_LABEL_VERSION,
                payload,
            ),
        )
        self.pending_writes += 1
        if self.pending_writes >= 100:
            self.connection.commit()
            self.pending_writes = 0

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def __enter__(self) -> "TeacherLabelCache":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = ["TEACHER_LABEL_VERSION", "TeacherLabelCache"]
