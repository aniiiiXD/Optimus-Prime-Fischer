"""Live rules and score-sign checks against the installed teacher engine."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import random

from gamebench.games.othello import OthelloAdapter, OthelloState, PASS_ACTION
from .teacher import TeacherConfig, label_positions


@lru_cache(maxsize=None)
def exact_score(board: tuple[int, ...], player: int, passes: int) -> int:
    """Tiny reference solver, used only for late positions with <=4 empties."""
    state = OthelloState(list(board), player, [], passes)
    if state.is_terminal():
        own = state.board.count(player)
        opponent = state.board.count(1 - player)
        empty = state.board.count(-1)
        diff = own - opponent
        return diff + empty if diff > 0 else diff - empty if diff < 0 else 0
    values = []
    for action in state.legal_actions():
        child = state.clone()
        child.apply_action(action)
        values.append(-exact_score(tuple(child.board), child.current_player, child.consecutive_passes))
    return max(values)


def main() -> None:
    root = Path(__file__).resolve().parent
    config = TeacherConfig(
        root / "vendor/neural-reversi/target/release/cli",
        root / "artifacts/teacher_weights/eval-e6bbc4f6.zst",
        root / "artifacts/teacher_weights/eval_sm-e6bbc4f6.zst",
        level=4,
        exact=True,
    )
    opening = OthelloAdapter().reset()
    opening_scores = label_positions([opening], TeacherConfig(config.binary, config.eval_file, config.eval_sm_file, level=4))
    assert set(opening_scores[0]) == set(opening.legal_actions())
    print(f"opening: {len(opening_scores[0])} legal moves agree")

    rng = random.Random(2026)
    late: OthelloState | None = None
    forced_pass: OthelloState | None = None
    for _ in range(100):
        state = OthelloAdapter().reset()
        while not state.is_terminal():
            if state.legal_actions() == (PASS_ACTION,) and forced_pass is None:
                forced_pass = state.clone()
            if state.board.count(-1) <= 4 and len(state.legal_actions()) >= 2 and late is None:
                late = state.clone()
            state.apply_action(rng.choice(state.legal_actions()))
        if late is not None and forced_pass is not None:
            break
    if late is None or forced_pass is None:
        raise RuntimeError("seeded random rollouts did not cover both late and pass positions")

    scores = label_positions([late], config)[0]
    expected = {}
    for action in late.legal_actions():
        child = late.clone()
        child.apply_action(action)
        expected[action] = -exact_score(tuple(child.board), child.current_player, child.consecutive_passes)
    if scores != expected:
        raise AssertionError(f"teacher score-perspective mismatch: teacher={scores}, reference={expected}")
    print(f"late exact scores agree for {len(scores)} moves: {scores}")

    pass_scores = label_positions([forced_pass], TeacherConfig(config.binary, config.eval_file, config.eval_sm_file, level=4))[0]
    assert set(pass_scores) == {PASS_ACTION}
    print(f"forced pass agrees: {pass_scores}")


if __name__ == "__main__":
    main()
