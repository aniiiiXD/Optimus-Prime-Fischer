"""Check whether GuidedSearch would actually select the follow-up it scored."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanatron import Color, Game, RandomPlayer
from catanatron.models.enums import ActionType
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer

from catanLabs.Catanatron.tournament_protocol import canonical_action_key, install_v5_protocol
from catanLabs.Competitive.guided_search_agent import (
    FOLLOWUPS, SEARCH_ROOTS, GuidedSearchAgent,
)


class TracedGuidedSearchAgent(GuidedSearchAgent):
    def _continuation_bonus(self, game, action, immediate_score):
        self.expanded_roots.add(action)
        return super()._continuation_bonus(game, action, immediate_score)

    def decide(self, game, playable_actions):
        self.expanded_roots = set()
        return super().decide(game, playable_actions)


def searched_followup(agent, game, root):
    """Repeat the frozen search's leaf ranking for its selected deterministic root."""
    immediate = agent._score_action(game, root)
    after = game.copy()
    after.execute(root, validate_action=True)
    if after.winning_color() is not None or after.state.current_color() != agent.color:
        return None, None
    best_action, best_score = None, immediate
    for action in after.playable_actions:
        if action.action_type not in FOLLOWUPS:
            continue
        if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            score = agent._buy_value(after)
        else:
            later = after.copy()
            later.execute(action, validate_action=True)
            score = agent.value_fn(later, agent.color)
        if score > best_score:
            best_action, best_score = action, score
    return after, best_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20272800)
    parser.add_argument("--max-game-seconds", type=float, default=20)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.boards < 1 or args.max_game_seconds <= 0:
        parser.error("boards and max-game-seconds must be positive")
    install_v5_protocol()
    config = json.loads((Path(__file__).parent / "configs/guided_search_3.json").read_text())
    totals = Counter()
    examples, games = [], []
    colors = list(Color)
    for board_index in range(args.boards):
        seed = args.seed + board_index
        for seat in range(4):
            game = Game([RandomPlayer(color) for color in colors], seed=seed,
                        vps_to_win=10, discard_limit=7, friendly_robber=False,
                        number_placement="official_spiral")
            game.playable_actions.sort(key=canonical_action_key)
            actor_color = game.state.colors[seat]
            agent = TracedGuidedSearchAgent(actor_color, config)
            others = [color for color in game.state.colors if color != actor_color]
            names = ("value", "alphabeta1", "alphabeta1")
            rotation = (board_index + seat) % 3
            opponents = {}
            for index, color in enumerate(others):
                name = names[(index + rotation) % 3]
                opponents[color] = (ValueFunctionPlayer(color) if name == "value" else
                                    AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=1)))
            started = time.perf_counter()
            status, actions = "capped", 0
            pending = None
            try:
                while actions < 15000 and game.state.num_turns < 500:
                    if game.winning_color() is not None:
                        status = "completed"
                        break
                    if time.perf_counter() - started > args.max_game_seconds:
                        break
                    current = game.state.current_color()
                    legal = game.playable_actions
                    if current == actor_color:
                        rng_before = game.state.random.getstate()
                        choice = agent.decide(game, legal)
                        totals["own_decisions"] += 1
                        if pending is not None:
                            totals["actual_next_checked"] += 1
                            if choice != pending["shadow"]:
                                totals["shadow_actual_mismatch"] += 1
                            pending = None
                        if choice.action_type in SEARCH_ROOTS and choice in agent.expanded_roots:
                            totals["selected_expanded_roots"] += 1
                            after, expected = searched_followup(agent, game, choice)
                            if expected is not None:
                                totals["roots_with_improving_followup"] += 1
                                realized = agent.decide(after, after.playable_actions)
                                pending = {"shadow": realized}
                                match = realized == expected
                                totals["followup_matches" if match else "followup_diverges"] += 1
                                if len(examples) < 100:
                                    examples.append({"seed": seed, "seat": seat,
                                                     "turn": game.state.num_turns,
                                                     "phase": game.state.current_prompt.name,
                                                     "root": choice.action_type.name,
                                                     "expected": expected.action_type.name,
                                                     "realized": realized.action_type.name,
                                                     "match": match})
                        if game.state.random.getstate() != rng_before:
                            raise AssertionError("Probe advanced the live game RNG")
                        action = choice
                    else:
                        action = opponents[current].decide(game, legal)
                    if action not in legal:
                        raise ValueError(f"Illegal action: {action}")
                    game.execute(action, validate_action=True)
                    actions += 1
                if game.winning_color() is not None:
                    status = "completed"
            except Exception as exc:
                status = f"error: {type(exc).__name__}: {exc}"
            games.append({"seed": seed, "seat": seat, "status": status,
                          "actions": actions,
                          "wall_seconds": round(time.perf_counter() - started, 3)})
    report = {"probe": "guided root follow-up consistency",
              "seed": args.seed, "boards": args.boards,
              "opponents": ["value", "alphabeta1", "alphabeta1"],
              "totals": dict(totals), "examples": examples, "games": games}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"totals": dict(totals),
                      "completed": sum(game["status"] == "completed" for game in games),
                      "games": len(games)}, indent=2))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
