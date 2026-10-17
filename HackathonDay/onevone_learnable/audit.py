"""Verify that the trained scorer changes executed choices in complete games."""
import argparse
import json
from pathlib import Path
import random
from collections import Counter
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/"official-kit"))
from catan_kit import match
from catan_kit._patch import ensure_hash_seed
from catanatron.game import Game
from catanatron.models.player import RandomPlayer
from collect import Context, F_MODULE, play_to_end
from policy import LearnableOneVOne


class AuditPolicy(LearnableOneVOne):
    def __init__(self, color):
        super().__init__(color)
        self.examples = []

    def decide(self, game, actions, cpu_left=None):
        index = super().decide(game, actions, cpu_left)
        row = self.last_decision
        if row.get("logits") is not None and index != row["prior_choice"] and row["reason"] != "guaranteed_win":
            if len(self.examples) < 3:
                self.examples.append({"turn": game.state.num_turns,
                    "expert_action": repr(actions[row["prior_choice"]]),
                    "executed_action": repr(actions[index]),
                    "expert_logit": float(row["logits"][row["prior_choice"]]),
                    "chosen_logit": float(row["logits"][index])})
        return index


def main():
    ensure_hash_seed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed0", type=int, default=2026840000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    totals, rows = Counter(), []
    for gi in range(args.games):
        seed = args.seed0+gi//2
        color, enemy = match.COLORS[gi%2], match.COLORS[1-gi%2]
        policy = AuditPolicy(color)
        game = Game([RandomPlayer(c) for c in match.COLORS], seed=seed)
        random.seed(match.dice_seed(seed, gi//2))
        contexts = {color: Context(color, policy, seed*13+gi),
                    enemy: Context(enemy, F_MODULE.Agent(), seed*17+gi, official=True)}
        winner = play_to_end(game, contexts)
        totals.update(policy.stats)
        rows.append({"seed": seed, "first": gi%2 == 0, "stats": policy.stats,
                     "won": winner == color, "examples": policy.examples})
    report = {"totals": dict(totals), "games": rows,
              "purpose": "activity audit; unclocked and not a playing-strength estimate"}
    if totals["model_calls"] == 0 or totals["model_changes"] == 0 or totals["model_errors"]:
        raise AssertionError(f"Model activity gate failed: {totals}")
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
