"""Evaluate saved full-data checkpoints in real games against fixed baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tic_tac_toe_transformer.agents.minimax_agent import MinimaxAgent
from tic_tac_toe_transformer.agents.random_agent import RandomAgent
from tic_tac_toe_transformer.agents.transformer_agent import TransformerAgent
from tic_tac_toe_transformer.evaluation.tournament import run_matchup

from tic_tac_toe_shared_transformer.agent import SharedTransformerAgent


DEFAULT_CHECKPOINT_DIR = Path(__file__).parent / "artifacts" / "checkpoints"
DEFAULT_OUTPUT = Path(__file__).parent / "artifacts" / "reports" / "tournaments.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--random-games", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    agents = {
        "classic_policy": TransformerAgent(
            args.checkpoint_dir / f"classic_policy_seed{args.seed}.pt"
        ),
        "candidate_policy": SharedTransformerAgent(
            args.checkpoint_dir / f"candidate_policy_seed{args.seed}.pt"
        ),
        "shared_multitask": SharedTransformerAgent(
            args.checkpoint_dir / f"shared_multitask_seed{args.seed}.pt"
        ),
    }
    report = {"seed": args.seed, "random_games_per_side": args.random_games, "models": {}}
    for name, agent in agents.items():
        report["models"][name] = {
            "as_x_vs_random": run_matchup(agent, RandomAgent(101), args.random_games),
            "as_o_vs_random": run_matchup(RandomAgent(202), agent, args.random_games),
            "as_x_vs_minimax": run_matchup(agent, MinimaxAgent(), 1),
            "as_o_vs_minimax": run_matchup(MinimaxAgent(), agent, 1),
        }
        print(f"finished {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
