"""Paired-color Othello games for a trained student checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch

from gamebench.agents.base import Agent
from gamebench.games.othello import OthelloAdapter, OthelloState
from gamebench.evaluation.tournament import TournamentConfig, run_tournament
from gamebench.registry import make_agent
from .model import ModelConfig, OthelloTransformer, encode_board


class OthelloStudentAgent(Agent):
    name = "othello_student"

    def __init__(self, model: OthelloTransformer) -> None:
        self.model = model.cpu().eval()

    @classmethod
    def from_checkpoint(cls, path: Path) -> "OthelloStudentAgent":
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = OthelloTransformer(ModelConfig(**saved["model_config"]))
        model.load_state_dict(saved["state_dict"])
        return cls(model)

    def select_action(self, state: OthelloState, rng: random.Random) -> int:
        del rng
        legal = state.legal_actions()
        if not legal:
            raise ValueError("student received a terminal position")
        board = encode_board(state.board, state.current_player).unsqueeze(0)
        side = torch.tensor([state.current_player], dtype=torch.long)
        candidates = torch.tensor([legal], dtype=torch.long)
        mask = torch.ones_like(candidates, dtype=torch.bool)
        with torch.inference_mode():
            logits = self.model(board, side, candidates, mask).policy_logits[0]
        return legal[int(logits.argmax())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Student-versus-baseline Othello tournament.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponents", default="random,greedy")
    parser.add_argument("--games", type=int, default=20, help="games per opponent, with colors swapped")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.games <= 0 or args.threads <= 0:
        parser.error("games and threads must be positive")
    torch.set_num_threads(args.threads)
    student = OthelloStudentAgent.from_checkpoint(args.checkpoint)
    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    reports = {}
    for name in opponents:
        opponent = make_agent(name)
        report = run_tournament(
            OthelloAdapter(), [student, opponent],
            TournamentConfig(games_per_pair=args.games, seed=args.seed, record_games=True),
        )
        reports[name] = report
        student_stats = report["standings"][student.name]
        print(f"vs {name}: W/D/L={student_stats['wins']}/{student_stats['draws']}/{student_stats['losses']} "
              f"score={student_stats['score_rate']:.1%}, median move={student_stats['decision_latency_ms']['median']:.2f} ms")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"checkpoint": str(args.checkpoint), "opponents": reports}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
