"""Shadow robber victim ties while FairValue plays every actual move."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
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
from catanLabs.Competitive.check_visibility import perturbed_hidden
from catanLabs.Competitive.fair_value_agent import FairValueAgent
from catanLabs.Competitive.robber_victim_agent import RobberVictimAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20272700)
    parser.add_argument("--max-game-seconds", type=float, default=20)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.boards < 1 or args.max_game_seconds <= 0:
        parser.error("boards and max-game-seconds must be positive")
    install_v5_protocol()
    config = json.loads((Path(__file__).parent / "configs/robber_victim_research.json").read_text())
    totals = {"own_decisions": 0, "robber_decisions": 0,
              "chosen_tile_multi_victim": 0, "overrides": 0,
              "visibility_checks": 0}
    latencies, examples, games = [], [], []
    colors = list(Color)
    for board_index in range(args.boards):
        seed = args.seed + board_index
        for seat in range(4):
            game = Game([RandomPlayer(color) for color in colors], seed=seed,
                        vps_to_win=10, discard_limit=7, friendly_robber=False,
                        number_placement="official_spiral")
            game.playable_actions.sort(key=canonical_action_key)
            actor_color = game.state.colors[seat]
            fair = FairValueAgent(actor_color, config)
            candidate = RobberVictimAgent(actor_color, config)
            others = [color for color in game.state.colors if color != actor_color]
            mix = ("value", "alphabeta1", "alphabeta1")
            rotation = (board_index + seat) % 3
            opponents = {}
            for index, color in enumerate(others):
                kind = mix[(index + rotation) % 3]
                opponents[color] = (ValueFunctionPlayer(color) if kind == "value" else
                                    AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=1)))
            started = time.perf_counter()
            status, actions = "capped", 0
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
                        baseline = fair.decide(game, legal)
                        totals["own_decisions"] += 1
                        if baseline.action_type == ActionType.MOVE_ROBBER:
                            totals["robber_decisions"] += 1
                            same_tile = [action for action in legal
                                         if action.value[0] == baseline.value[0]
                                         and action.value[1] is not None]
                            if len(same_tile) >= 2:
                                totals["chosen_tile_multi_victim"] += 1
                                before = time.perf_counter_ns()
                                alternative = candidate.find_override(game, legal, baseline)
                                latencies.append((time.perf_counter_ns() - before) / 1e6)
                                if alternative is not None:
                                    if alternative not in legal or alternative.value[0] != baseline.value[0]:
                                        raise AssertionError("Victim override changed tile or legality")
                                    totals["overrides"] += 1
                                    if len(examples) < 12:
                                        state = game.state
                                        def vp(action):
                                            victim = action.value[1]
                                            return state.player_state[
                                                f"P{state.color_to_index[victim]}_VICTORY_POINTS"]
                                        examples.append({"seed": seed, "seat": seat,
                                                         "turn": state.num_turns,
                                                         "baseline_victim_vp": vp(baseline),
                                                         "selected_victim_vp": vp(alternative)})
                                if totals["visibility_checks"] < 8:
                                    changed = perturbed_hidden(game, actor_color)
                                    hidden_baseline = fair.decide(changed, changed.playable_actions)
                                    hidden_alternative = candidate.find_override(
                                        changed, changed.playable_actions, hidden_baseline)
                                    if hidden_baseline != baseline or hidden_alternative != alternative:
                                        raise AssertionError("Hidden-state perturbation changed choice")
                                    totals["visibility_checks"] += 1
                        if game.state.random.getstate() != rng_before:
                            raise AssertionError("Shadow decision advanced live RNG")
                        action = baseline
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
    report = {"probe": "FairValue plays; robber victim tie-break shadowed",
              "seed": args.seed, "boards": args.boards, "seats_per_board": 4,
              "opponents": ["value", "alphabeta1", "alphabeta1"],
              "config": config, "totals": totals,
              "scanner_latency_ms": {
                  "median": round(statistics.median(latencies), 4) if latencies else None,
                  "p95": round(sorted(latencies)[int(.95 * (len(latencies) - 1))], 4)
                  if len(latencies) >= 20 else None,
                  "max": round(max(latencies), 4) if latencies else None},
              "examples": examples, "games": games}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"totals": totals, "latency_ms": report["scanner_latency_ms"],
                      "completed": sum(game["status"] == "completed" for game in games),
                      "games": len(games)}, indent=2))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
