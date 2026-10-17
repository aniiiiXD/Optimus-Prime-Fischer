"""Lossless, safe raw Catan observations aligned with the tensor dataset.

Each gzip JSONL line is one unique decision. ``observation.legal_actions`` is
stored in the *same canonical order* as the corresponding NPZ candidate axis.
No pickle or simulator objects are written.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

from catanLabs.Transformer.visible_observation import (
    VisibleObservation, observation_from_dict, observation_to_dict,
)


RAW_SCHEMA = "catanlabs-visible-raw-v2"


@dataclass(frozen=True, slots=True)
class RawRecord:
    observation: VisibleObservation
    canonical_state_key: str
    game_id: int
    board_seed: int
    actor_color: str
    actor_seat: int
    phase: str
    behavior: str
    teacher: str
    teacher_config_sha256: str
    teacher_source_sha256: str
    target_action_index: int
    outcome: int
    final_vp: int
    benchmark_protocol: str
    rules_manifest_sha256: str
    engine_commit: str
    feature_schema: str


def record_to_dict(record: RawRecord) -> dict:
    if not 0 <= record.target_action_index < len(record.observation.legal_actions):
        raise ValueError("Teacher target is outside the canonical legal-action set")
    return {
        "raw_schema": RAW_SCHEMA,
        "observation": observation_to_dict(record.observation),
        "canonical_state_key": record.canonical_state_key,
        "game_id": record.game_id,
        "board_seed": record.board_seed,
        "actor_color": record.actor_color,
        "actor_seat": record.actor_seat,
        "phase": record.phase,
        "behavior": record.behavior,
        "teacher": record.teacher,
        "teacher_config_sha256": record.teacher_config_sha256,
        "teacher_source_sha256": record.teacher_source_sha256,
        "target_action_index": record.target_action_index,
        "outcome": record.outcome,
        "final_vp": record.final_vp,
        "benchmark_protocol": record.benchmark_protocol,
        "rules_manifest_sha256": record.rules_manifest_sha256,
        "engine_commit": record.engine_commit,
        "feature_schema": record.feature_schema,
    }


def record_from_dict(data: dict) -> RawRecord:
    if type(data) is not dict or data.get("raw_schema") != RAW_SCHEMA:
        raise ValueError("Unsupported raw-record schema")
    expected = set(RawRecord.__dataclass_fields__) | {"raw_schema"}
    if set(data) != expected:
        raise ValueError("Invalid raw-record fields")
    fields = {key: value for key, value in data.items() if key != "raw_schema"}
    fields["observation"] = observation_from_dict(fields["observation"])
    record = RawRecord(**fields)
    record_to_dict(record)  # Validate target index and snapshot schema.
    return record


def write_raw_records(path: Path, records: Iterable[RawRecord]) -> tuple[int, str]:
    """Write deterministic gzip JSONL; return row count and compressed SHA256."""
    writer = RawRecordWriter(path)
    try:
        for record in records:
            writer.write(record)
    finally:
        digest = writer.close()
    return writer.count, digest


class RawRecordWriter:
    """Stream records as games finish, after their outcomes are known."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("wb")
        self.compressed = gzip.GzipFile(filename="", fileobj=self.handle, mode="wb", mtime=0)
        self.count = 0
        self._digest: str | None = None

    def write(self, record: RawRecord) -> None:
        if self._digest is not None:
            raise ValueError("Raw writer is closed")
        line = json.dumps(record_to_dict(record), sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
        self.compressed.write(line.encode("utf-8") + b"\n")
        self.count += 1

    def close(self) -> str:
        if self._digest is None:
            self.compressed.close()
            self.handle.close()
            self._digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return self._digest


def read_raw_records(path: Path) -> Iterator[RawRecord]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                yield record_from_dict(json.loads(line))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid raw record at line {number}: {exc}") from exc
