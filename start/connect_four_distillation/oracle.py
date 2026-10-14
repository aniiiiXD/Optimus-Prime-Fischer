"""Query the pinned Pascal Pons CPU solver without changing its source code.

The solver speaks one-based move-history strings (for example, ``"4453"``),
whereas our ConnectFour engine uses zero-based integer columns internally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import subprocess
from typing import Any

from connect_four_engine import ConnectFour


SOLVER = Path(__file__).parent / "vendor" / "connect4" / "c4solver"
INVALID_MOVE = -1000


def validate_history(moves: str) -> ConnectFour:
    if not moves or any(column not in "1234567" for column in moves):
        raise ValueError("moves must be a nonempty string of columns 1–7")
    game = ConnectFour.from_moves([int(column) - 1 for column in moves])
    if game.is_terminal:
        raise ValueError("the position is already terminal; there is no next move to score")
    return game


class PonsOracle:
    """Keep one CPU solver process alive so its transposition cache can be reused."""

    def __init__(
        self,
        *,
        executable: Path = SOLVER,
        weak: bool = False,
        timeout_seconds: float = 5.0,
        opening_book: Path | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.executable = Path(executable)
        self.weak = weak
        self.timeout_seconds = timeout_seconds
        self.opening_book = opening_book
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "PonsOracle":
        if not self.executable.is_file():
            raise FileNotFoundError(f"solver not built: {self.executable}; run make c4solver")
        command = [str(self.executable.resolve()), "-a"]
        if self.weak:
            command.append("-w")
        if self.opening_book is not None:
            if not self.opening_book.is_file():
                raise FileNotFoundError(f"opening book not found: {self.opening_book}")
            command.extend(["-b", str(self.opening_book.resolve())])
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # Missing optional 7x6.book warning.
            text=True,
            bufsize=1,
            cwd=self.executable.parent,
        )
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.process is None:
            return
        process = self.process
        self.process = None
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()

    def analyze(self, moves: str) -> dict[str, Any]:
        """Return exact move scores, or weak W/D/L scores if ``weak=True``."""
        game = validate_history(moves)
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("use PonsOracle inside a with block")
        process.stdin.write(moves + "\n")
        process.stdin.flush()
        ready, _, _ = select.select([process.stdout], [], [], self.timeout_seconds)
        if not ready:
            process.kill()
            process.wait()
            raise TimeoutError(
                f"solver exceeded {self.timeout_seconds:g}s at ply {game.move_count}; "
                "try later positions or an opening book"
            )
        response = process.stdout.readline().split()
        if len(response) != 8 or response[0] != moves:
            raise RuntimeError(f"unexpected solver response: {response!r}")
        scores = [int(value) for value in response[1:]]
        legal = list(game.legal_actions())
        if any((scores[column] == INVALID_MOVE) != (column not in legal) for column in range(7)):
            raise RuntimeError("solver legality disagrees with our Connect Four engine")
        best_score = max(scores[column] for column in legal)
        return {
            "moves": moves,
            "ply": game.move_count,
            "board": game.board,
            "current_player": game.current_player,
            "legal_columns": [column + 1 for column in legal],
            "scores": scores,
            "best_columns": [column + 1 for column in legal if scores[column] == best_score],
            "best_score": best_score,
            "score_mode": "win_draw_loss" if self.weak else "exact",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("moves", help="one-based move history, e.g. 135567...")
    parser.add_argument("--weak", action="store_true", help="return W/D/L, not exact win distance")
    parser.add_argument("--timeout", type=float, default=5.0, help="seconds per position")
    parser.add_argument("--book", type=Path, help="optional compatible 7x6 opening book")
    args = parser.parse_args()
    with PonsOracle(weak=args.weak, timeout_seconds=args.timeout, opening_book=args.book) as oracle:
        print(json.dumps(oracle.analyze(args.moves), indent=2))


if __name__ == "__main__":
    main()
