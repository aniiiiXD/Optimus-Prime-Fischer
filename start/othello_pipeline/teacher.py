"""Batched labels from the pinned Neural Reversi `solve --all-moves` CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile

from gamebench.games.othello import OthelloState, PASS_ACTION
from .encoding import action_from_gtp, board_to_obf


POSITION_RE = re.compile(r"^Position #(\d+)\s+Depth:")
SCORE_ROW_RE = re.compile(r"^\|\s*([+-]\d+)\s*\|\s*(.*?)\s*\|")


@dataclass(frozen=True)
class TeacherConfig:
    binary: Path
    eval_file: Path
    eval_sm_file: Path
    level: int = 8
    threads: int = 1
    hash_mb: int = 64
    timeout_seconds: int = 120
    exact: bool = False

    @property
    def teacher_id(self) -> str:
        mode = "exact" if self.exact else f"L{self.level}"
        return f"neural-reversi-v6.3.0-e6bbc4f6-{mode}-T{self.threads}-H{self.hash_mb}"

    def check_files(self) -> None:
        for path in (self.binary, self.eval_file, self.eval_sm_file):
            if not path.is_file():
                raise FileNotFoundError(path)
        if min(self.level, self.threads, self.hash_mb, self.timeout_seconds) <= 0:
            raise ValueError("teacher level, threads, hash, and timeout must be positive")


def parse_solve_output(output: str, states: list[OthelloState]) -> list[dict[int, float]]:
    """Reject partial or unexpected engine output instead of training on it."""
    tables: dict[int, list[tuple[str, float]]] = {}
    current: int | None = None
    for line in output.splitlines():
        position = POSITION_RE.match(line)
        if position:
            current = int(position.group(1)) - 1
            if current in tables or not 0 <= current < len(states):
                raise ValueError(f"unexpected teacher position index: {current + 1}")
            tables[current] = []
            continue
        row = SCORE_ROW_RE.match(line)
        if row and current is not None:
            token = row.group(2).split()[0] if row.group(2).split() else ""
            tables[current].append((token, float(int(row.group(1)))))

    if set(tables) != set(range(len(states))):
        raise ValueError("teacher output is missing one or more positions")
    labels: list[dict[int, float]] = []
    for index, state in enumerate(states):
        legal = set(state.legal_actions())
        rows = tables[index]
        if not rows:
            raise ValueError(f"teacher returned no scores for position {index + 1}")
        if legal == {PASS_ACTION}:
            if rows[0][0].lower() != "ps":
                raise ValueError(f"teacher pass marker missing for position {index + 1}")
            scores = {PASS_ACTION: rows[0][1]}
        else:
            scores: dict[int, float] = {}
            for token, score in rows:
                try:
                    action = action_from_gtp(token)
                except ValueError as error:
                    raise ValueError(f"bad teacher move {token!r} at position {index + 1}") from error
                if action in scores:
                    raise ValueError(f"duplicate teacher move {token!r} at position {index + 1}")
                scores[action] = score
        if set(scores) != legal:
            raise ValueError(f"teacher/rules legal-move mismatch at position {index + 1}: {set(scores)} != {legal}")
        labels.append(scores)
    return labels


def label_positions(states: list[OthelloState], config: TeacherConfig) -> list[dict[int, float]]:
    if not states:
        return []
    config.check_files()
    for state in states:
        if state.is_terminal():
            raise ValueError("teacher labels require nonterminal positions")
    with tempfile.TemporaryDirectory(prefix="othello_teacher_") as directory:
        positions = Path(directory) / "positions.obf"
        positions.write_text("\n".join(board_to_obf(state) for state in states) + "\n", encoding="ascii")
        command = [
            str(config.binary.resolve()), "solve", str(positions), "--all-moves",
            "--level", str(config.level), "--threads", str(config.threads),
            "--hash-size", str(config.hash_mb),
            "--eval-file", str(config.eval_file.resolve()),
            "--eval-sm-file", str(config.eval_sm_file.resolve()),
        ]
        if config.exact:
            command.append("--exact")
        completed = subprocess.run(command, text=True, capture_output=True, timeout=config.timeout_seconds, check=False)
        if completed.returncode:
            raise RuntimeError(f"teacher exited {completed.returncode}: {completed.stderr[-1200:]}")
        return parse_solve_output(completed.stdout, states)
