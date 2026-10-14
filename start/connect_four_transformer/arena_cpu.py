"""CPU-only, Game Arena-inspired Connect Four diagnostic runner.

The current checkpoint consumes structured board tokens. The text observation
and ``Final Answer: N`` response make its decisions inspectable alongside the
public Four in a Row harness; they do not turn it into a text language model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import platform
import re
import resource
import statistics
import sys
import time
from typing import Any

import torch

from connect_four_engine import ConnectFour, find_winner

from .agent import TransformerAgent
from .data import grouped_split, load_rows
from .search import AlphaBetaTeacher


HERE = Path(__file__).parent
DEFAULT_DATASET = HERE / "artifacts" / "data" / "positions_v0_2000.jsonl"
DEFAULT_CHECKPOINT = HERE / "artifacts" / "checkpoints" / "candidate_v0.pt"
DEFAULT_REPORT = HERE / "artifacts" / "reports" / "arena_cpu_v0.json"
ANSWER = re.compile(r"^Final Answer: ([0-6])$", re.MULTILINE)

# Zero-based columns, alternating X/O. The final O decision misses X's
# immediate diagonal win in column 2 (one-based).
HUMAN_MATCH = (0, 3, 4, 2, 4, 3, 4, 3, 3, 4, 1, 2, 2, 2, 1)
HUMAN_FIXTURES = {
    "vertical_block_missed": HUMAN_MATCH[:7],
    "diagonal_block_missed": HUMAN_MATCH[:13],
}


def from_dataset_row(row: dict[str, Any]) -> ConnectFour:
    """Restore a nonterminal, gravity-valid state without needing its history."""
    board = list(row["board"])
    if len(board) != 42 or any(cell not in (-1, 0, 1) for cell in board):
        raise ValueError("invalid board encoding")
    game = ConnectFour()
    game.board = board
    game.current_player = int(row["current_player"])
    game.move_count = sum(cell != 0 for cell in board)
    game.column_heights = []
    for column in range(7):
        cells = [board[r * 7 + column] for r in range(6)]
        height = sum(cell != 0 for cell in cells)
        if cells != [0] * (6 - height) + cells[6 - height :]:
            raise ValueError("board violates gravity")
        game.column_heights.append(height)
    if game.current_player not in (-1, 1) or find_winner(board) is not None:
        raise ValueError("expected a nonterminal position")
    if game.move_count != int(row["ply"]):
        raise ValueError("move count does not match board")
    return game


def text_board(game: ConnectFour) -> str:
    marks = {0: ".", 1: "X", -1: "O"}
    lines = [" ".join(marks[game.board[row * 7 + col]] for col in range(7)) for row in range(6)]
    return "\n".join(lines + ["0 1 2 3 4 5 6"])


def observation(game: ConnectFour) -> str:
    player = "X" if game.current_player == 1 else "O"
    return f"Player to move: {player}\n{text_board(game)}\nFinal Answer: <column_index>"


def parse_observation(text: str) -> ConnectFour:
    """Decode the local text-board contract before structured CPU inference."""
    lines = text.splitlines()
    if len(lines) != 9 or lines[0] not in ("Player to move: X", "Player to move: O"):
        raise ValueError("expected player, six board rows, column labels and answer format")
    if lines[7] != "0 1 2 3 4 5 6" or lines[8] != "Final Answer: <column_index>":
        raise ValueError("unexpected column labels or answer format")
    marks = {".": 0, "X": 1, "O": -1}
    board = []
    for line in lines[1:7]:
        cells = line.split()
        if len(cells) != 7 or any(cell not in marks for cell in cells):
            raise ValueError("expected seven board cells per row")
        board.extend(marks[cell] for cell in cells)
    return from_dataset_row({
        "board": board,
        "current_player": 1 if lines[0].endswith("X") else -1,
        "ply": sum(cell != 0 for cell in board),
    })


def parse_answer(response: str, legal: tuple[int, ...]) -> int:
    """Require an unambiguous, legal zero-based column on the final line."""
    lines = response.strip().splitlines()
    if not lines:
        raise ValueError("missing answer")
    match = ANSWER.fullmatch(lines[-1].strip())
    if match is None:
        raise ValueError("expected final line 'Final Answer: N' with N in 0..6")
    action = int(match.group(1))
    if action not in legal:
        raise ValueError(f"column {action} is illegal")
    return action


def immediate_wins(game: ConnectFour, player: int) -> list[int]:
    result = []
    for action in game.legal_actions():
        branch = game.copy()
        branch.current_player = player
        if branch.step(action).winner == player:
            result.append(action)
    return result


def category_and_critical(game: ConnectFour) -> tuple[str, list[int]]:
    own = immediate_wins(game, game.current_player)
    if own:
        return "win_now", own
    opponent = immediate_wins(game, -game.current_player)
    if len(opponent) == 1:
        return "forced_block", opponent
    if len(opponent) > 1:
        return "multiple_threats", opponent
    return "ordinary", []


def threat_orientation(game: ConnectFour, kind: str, critical: list[int]) -> str | None:
    if kind not in ("win_now", "forced_block"):
        return None
    actor = game.current_player if kind == "win_now" else -game.current_player
    directions = set()
    for action in critical:
        branch = game.copy()
        branch.current_player = actor
        line = branch.step(action).winning_line
        if len(line) < 2:
            raise RuntimeError("critical action does not create a four-in-a-row")
        dr = abs(line[1][0] - line[0][0])
        dc = abs(line[1][1] - line[0][1])
        directions.add("vertical" if dc == 0 else "horizontal" if dr == 0 else "diagonal")
    return next(iter(directions)) if len(directions) == 1 else "mixed"


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    tactical = [row for row in records if row["category"] in ("win_now", "forced_block")]
    return {
        "positions": len(records),
        "teacher_best_move_rate": sum(row["teacher_best_move"] for row in records) / len(records),
        "teacher_regret": statistics.mean(row["teacher_regret"] for row in records),
        "critical_move_rate": (
            sum(row["critical_move"] for row in tactical) / len(tactical) if tactical else None
        ),
        "guarded_critical_move_rate": (
            sum(row["guarded_critical_move"] for row in tactical) / len(tactical)
            if tactical else None
        ),
        "critical_positions": len(tactical),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    torch.set_num_threads(args.threads)
    rows = load_rows(args.dataset)
    test_ids = grouped_split(rows, seed=args.seed)["test"]
    agent = TransformerAgent(args.checkpoint)
    for _ in range(10):
        agent.inspect(ConnectFour())

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    model_latencies: list[float] = []
    for index in test_ids:
        row = rows[index]
        game = from_dataset_row(row)
        kind, critical = category_and_critical(game)
        started = time.perf_counter_ns()
        prompt = observation(game)
        parsed = parse_observation(prompt)
        decision = agent.inspect(parsed)
        response = f"Final Answer: {decision['choices'][0]['column']}"
        action = parse_answer(response, parsed.legal_actions())
        guarded_action = int(decision["selected_action"])
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        model_latencies.append(decision["latency_ms"])
        teacher_values = row["action_values"]
        best_value = max(teacher_values[a] for a in game.legal_actions())
        chosen_value = teacher_values[action]
        records.append({
            "source": "held_out_test",
            "dataset_index": index,
            "category": kind,
            "player": "X" if game.current_player == 1 else "O",
            "orientation": threat_orientation(game, kind, critical),
            "critical_columns": critical,
            "chosen_column": action,
            "guarded_column": guarded_action,
            "teacher_best_move": abs(chosen_value - best_value) < 1e-6,
            "teacher_regret": best_value - chosen_value,
            "critical_move": action in critical if critical else None,
            "guarded_critical_move": guarded_action in critical if critical else None,
            "top3": decision["choices"][:3],
            "prompt": prompt if kind in ("win_now", "forced_block") and action not in critical else None,
        })

    teacher = AlphaBetaTeacher()
    fixture_records = []
    for name, moves in HUMAN_FIXTURES.items():
        game = ConnectFour.from_moves(moves)
        kind, critical = category_and_critical(game)
        prompt = observation(game)
        decision = agent.inspect(parse_observation(prompt))
        choice = decision["choices"][0]["column"]
        guarded_choice = int(decision["selected_action"])
        label = teacher.evaluate(game, max_depth=4, node_limit=50_000)
        fixture_records.append({
            "name": name,
            "moves_zero_based": list(moves),
            "category": kind,
            "orientation": threat_orientation(game, kind, critical),
            "critical_columns": critical,
            "model_column": choice,
            "guarded_column": guarded_choice,
            "model_top3": decision["choices"][:3],
            "teacher_best_column": label.best_action,
            "teacher_values": label.action_values,
            "passed": choice in critical,
            "guarded_passed": guarded_choice in critical,
            "guard_reason": decision["decision_reason"],
            "text_observation": prompt,
        })

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    side_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    orientation_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["category"]].append(record)
        side_groups[f"{record['category']}:{record['player']}"].append(record)
        if record["orientation"]:
            orientation_groups[f"{record['category']}:{record['orientation']}"].append(record)
    report = {
        "scope": "structured-token Transformer; Game Arena-style text is an observation/response adapter",
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "split": "held-out test opening groups only",
        "seed": args.seed,
        "overall": summarize(records),
        "categories": {name: summarize(group) for name, group in sorted(groups.items())},
        "by_category_and_side": {name: summarize(group) for name, group in sorted(side_groups.items())},
        "by_category_and_orientation": {name: summarize(group) for name, group in sorted(orientation_groups.items())},
        "human_match_fixtures": fixture_records,
        "cpu": {
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "threads": args.threads,
            "wall_seconds": time.perf_counter() - wall_start,
            "process_cpu_seconds": time.process_time() - cpu_start,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
                1024 * 1024 if sys.platform == "darwin" else 1024
            ),
            "model_forward_median_ms": statistics.median(model_latencies),
            "model_forward_p95_ms": percentile(model_latencies, 0.95),
            "full_move_median_ms": statistics.median(latencies),
            "full_move_p95_ms": percentile(latencies, 0.95),
        },
        "failures": [record for record in records if record["critical_move"] is False],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"held-out positions: {len(records)}")
    for name, group in sorted(groups.items()):
        stats = summarize(group)
        metric = stats["critical_move_rate"]
        print(f"{name:<18} {len(group):>3} | critical accuracy: {metric:.1%}" if metric is not None else f"{name:<18} {len(group):>3}")
    print(
        "human-game fixtures:", ", ".join(
            f"{r['name']}=raw:{'PASS' if r['passed'] else 'FAIL'}, "
            f"guarded:{'PASS' if r['guarded_passed'] else 'FAIL'}"
            for r in fixture_records
        )
    )
    print(f"CPU full move median/p95: {report['cpu']['full_move_median_ms']:.3f}/{report['cpu']['full_move_p95_ms']:.3f} ms")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
