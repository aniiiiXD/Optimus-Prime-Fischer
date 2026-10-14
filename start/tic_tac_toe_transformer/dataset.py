"""Generate Minimax-labelled Tic-Tac-Toe positions for supervised learning."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tic_tac_toe_lab.tic_tac_toe import available_moves, winner

from .input import tokenize_position


def other(player: str) -> str:
    return "O" if player == "X" else "X"


@lru_cache(maxsize=None)
def minimax_value(board_key: str, current_player: str) -> int:
    """Return the perfect-play outcome for the player to move: win=1, draw=0, loss=-1."""
    board = list(board_key)
    if winner(board) is not None:
        # The preceding player just won, so the player now nominally to move lost.
        return -1

    legal_moves = available_moves(board)
    if not legal_moves:
        return 0

    best_value = -2
    for square in legal_moves:
        board[square - 1] = current_player
        value = -minimax_value("".join(board), other(current_player))
        board[square - 1] = " "
        best_value = max(best_value, value)
    return best_value


def reachable_positions() -> list[tuple[str, str]]:
    """Enumerate unique legal, nonterminal boards and the player to move."""
    positions: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def visit(board: list[str], current_player: str) -> None:
        if winner(board) is not None or not available_moves(board):
            return

        board_key = "".join(board)
        state = (board_key, current_player)
        if state in seen:
            return
        seen.add(state)
        positions.append(state)

        for square in available_moves(board):
            board[square - 1] = current_player
            visit(board, other(current_player))
            board[square - 1] = " "

    visit([" "] * 9, "X")
    return positions


def make_example(board_key: str, current_player: str) -> dict[str, Any]:
    """Build one model input and its exact legal, policy, and value targets."""
    board = list(board_key)
    legal_moves = available_moves(board)
    if winner(board) is not None or not legal_moves:
        raise ValueError("dataset examples must be nonterminal positions")

    move_values: dict[int, int] = {}
    for square in legal_moves:
        board[square - 1] = current_player
        move_values[square] = -minimax_value("".join(board), other(current_player))
        board[square - 1] = " "

    best_value = max(move_values.values())
    optimal_moves = [square for square, value in move_values.items() if value == best_value]
    target_policy = [0.0] * 9
    for square in optimal_moves:
        target_policy[square - 1] = 1.0 / len(optimal_moves)

    legal_mask = [square in legal_moves for square in range(1, 10)]
    return {
        "board": board_key,
        "current_player": current_player,
        "tokens": tokenize_position(board, current_player),
        "legal_mask": legal_mask,
        "target_policy": target_policy,
        "optimal_moves": optimal_moves,
        "minimax_value": best_value,
    }


def make_dataset() -> list[dict[str, Any]]:
    """Create one Minimax-labelled example for every unique reachable position."""
    return [make_example(board, player) for board, player in reachable_positions()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "minimax_positions.jsonl",
        help="JSONL destination (default: tic_tac_toe_transformer/artifacts/minimax_positions.jsonl)",
    )
    args = parser.parse_args()

    examples = make_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(json.dumps(example, separators=(",", ":")) + "\n")

    values = {value: sum(item["minimax_value"] == value for item in examples) for value in (-1, 0, 1)}
    print(f"Saved {len(examples):,} positions to {args.output}")
    print(f"Minimax outcomes from side-to-move perspective: {values}")
    print("First example:")
    print(json.dumps(examples[0], indent=2))


if __name__ == "__main__":
    main()
