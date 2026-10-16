#!/usr/bin/env python3
"""Play the trained MLP against external Catanatron bots and report win rates."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from train_legal_ranker import HERE, LegalRanker, encode_action, encode_state, stable_edge_index
from train_calibrated_ranker import ActionCalibrationAdapter
from settlement_target_head import SettlementTargetHead, settlement_features
from heuristic_experiments import own_hand_size, settlement_pips
from history_experiment import HistoryResidual, encode_history


CATANATRON_ROOT = HERE.parents[2] / "start/catan_lab/vendor/catanatron/catanatron"
sys.path.insert(0, str(CATANATRON_ROOT))
sys.path.insert(0, str(HERE.parents[2]))

from catanatron import Color, Game, RandomPlayer  # noqa: E402
from same_turn_search import find_immediate_win  # noqa: E402
from catanLabs.Catanatron.tournament_protocol import install_v5_protocol  # noqa: E402
from generate_dataset import action_json, make_bot, observation  # noqa: E402


class MLPPlayer:
    def __init__(self, color: Color, model: LegalRanker, edge_index: dict[tuple[int, int], int], device: torch.device, adapter: ActionCalibrationAdapter | None = None, settlement_head: SettlementTargetHead | None = None, settlement_heuristic: str = "none", buy_dev_end_hand_threshold: int | None = None, history_residual: HistoryResidual | None = None, same_turn_search: bool = False, same_turn_depth: int = 4, history_scale: float = 1.0):
        self.color = color
        self.model = model
        self.edge_index = edge_index
        self.device = device
        self.adapter = adapter
        self.settlement_head = settlement_head
        self.settlement_heuristic = settlement_heuristic
        self.buy_dev_end_hand_threshold = buy_dev_end_hand_threshold
        self.history_residual = history_residual
        self.history_scale = history_scale
        self.same_turn_search = same_turn_search
        self.same_turn_depth = same_turn_depth
        self.same_turn_search_changes = 0
        self.latencies_ms: list[float] = []
        self.settlement_reranks = 0
        self.settlement_changes = 0
        self.settlement_heuristic_changes = 0
        self.buy_dev_heuristic_changes = 0

    @torch.no_grad()
    def decide(self, game: Game, playable_actions: list[Any]) -> Any:
        started = time.perf_counter_ns()
        visible = observation(game, self.color)
        state = torch.tensor([encode_state(visible, self.edge_index)], dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            [[encode_action(action_json(action, self.color), visible, self.edge_index) for action in playable_actions]],
            dtype=torch.float32,
            device=self.device,
        )
        scores = self.model(state, actions)
        if self.adapter is not None:
            scores = self.adapter(scores, state, actions)
        if self.history_residual is not None:
            history = torch.tensor([encode_history(visible, self.edge_index)], dtype=torch.float32, device=self.device)
            scores = scores + self.history_scale * self.history_residual(state, actions, history)
        choice = int(scores.argmax(dim=1).item())
        if self.settlement_head is not None and playable_actions[choice].action_type.name == "BUILD_SETTLEMENT":
            candidate_indices = [
                index for index, action in enumerate(playable_actions)
                if action.action_type.name == "BUILD_SETTLEMENT"
            ]
            if len(candidate_indices) > 1:
                local = torch.tensor(
                    [settlement_features(visible, int(playable_actions[index].value)) for index in candidate_indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                local_scores = scores[0, candidate_indices] + self.settlement_head(local)
                refined_choice = candidate_indices[int(local_scores.argmax().item())]
                self.settlement_reranks += 1
                self.settlement_changes += refined_choice != choice
                choice = refined_choice
        if self.settlement_heuristic == "pips" and playable_actions[choice].action_type.name == "BUILD_SETTLEMENT":
            candidate_indices = [
                index for index, action in enumerate(playable_actions)
                if action.action_type.name == "BUILD_SETTLEMENT"
            ]
            if len(candidate_indices) > 1:
                refined_choice = max(
                    candidate_indices,
                    key=lambda index: (
                        settlement_pips(visible, int(playable_actions[index].value)),
                        float(scores[0, index]),
                    ),
                )
                self.settlement_heuristic_changes += refined_choice != choice
                choice = refined_choice
        if (
            self.buy_dev_end_hand_threshold is not None
            and playable_actions[choice].action_type.name == "END_TURN"
            and own_hand_size(visible) >= self.buy_dev_end_hand_threshold
        ):
            buy_index = next(
                (index for index, action in enumerate(playable_actions)
                 if action.action_type.name == "BUY_DEVELOPMENT_CARD"),
                None,
            )
            if buy_index is not None:
                self.buy_dev_heuristic_changes += 1
                choice = buy_index
        if self.same_turn_search:
            searched = find_immediate_win(game, self.color, max_depth=self.same_turn_depth)
            if searched is not None:
                refined_choice = playable_actions.index(searched)
                self.same_turn_search_changes += refined_choice != choice
                choice = refined_choice
        self.latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        return playable_actions[choice]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "runs/legal_ranker_v1/model.pt")
    parser.add_argument("--settlement-target-checkpoint", type=Path, help="Optional learned settlement-location classifier.")
    parser.add_argument("--settlement-heuristic", choices=("none", "pips"), default="none")
    parser.add_argument("--buy-dev-end-hand-threshold", type=int, help="If ending with at least this many cards, buy a legal development card.")
    parser.add_argument("--history-checkpoint", type=Path, help="Optional state-only or public-history residual experiment.")
    parser.add_argument("--history-scale", type=float, default=1.0)
    parser.add_argument("--same-turn-search", action="store_true", help="Use bounded deterministic search for a win this turn.")
    parser.add_argument("--same-turn-depth", type=int, default=4)
    games = parser.add_mutually_exclusive_group()
    games.add_argument("--games-per-opponent", type=int, default=10)
    games.add_argument("--games-total", type=int, help="Run this many sealed games, cycling opponents and seats.")
    parser.add_argument("--seed", type=int, default=20263001)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, default=HERE / "reports/mlp_win_rate_v1.json")
    args = parser.parse_args()
    if args.games_total is None and args.games_per_opponent < 2:
        parser.error("--games-per-opponent must be at least 2 so seats can rotate")
    if args.games_total is not None and args.games_total < 1:
        parser.error("--games-total must be positive")
    if args.buy_dev_end_hand_threshold is not None and args.buy_dev_end_hand_threshold < 1:
        parser.error("--buy-dev-end-hand-threshold must be positive")
    if args.same_turn_depth < 1:
        parser.error("--same-turn-depth must be positive")
    if args.history_scale < 0:
        parser.error("--history-scale must be nonnegative")
    install_v5_protocol()
    device = torch.device("mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = LegalRanker(checkpoint["state_size"], checkpoint["action_size"]).to(device)
    # Imitation checkpoints store ``model``; PPO actor-critic checkpoints
    # store the same legal-action network under ``policy``.
    adapter = None
    if "base_model" in checkpoint:
        model.load_state_dict(checkpoint["base_model"])
        adapter = ActionCalibrationAdapter(checkpoint["state_size"]).to(device)
        adapter.load_state_dict(checkpoint["adapter"])
        adapter.eval()
    else:
        model.load_state_dict(checkpoint.get("model", checkpoint.get("policy")))
    model.eval()
    settlement_head = None
    if args.settlement_target_checkpoint is not None:
        if adapter is None:
            parser.error("Settlement target head requires the calibrated base checkpoint")
        target_checkpoint = torch.load(args.settlement_target_checkpoint, map_location=device, weights_only=True)
        if Path(target_checkpoint["base_checkpoint"]).resolve() != args.checkpoint.resolve():
            parser.error("Settlement target head was trained for a different calibrated checkpoint")
        settlement_head = SettlementTargetHead(target_checkpoint["input_size"]).to(device)
        settlement_head.load_state_dict(target_checkpoint["head"])
        settlement_head.eval()
    history_residual = None
    if args.history_checkpoint is not None:
        history_checkpoint = torch.load(args.history_checkpoint, map_location=device, weights_only=True)
        if history_checkpoint["state_size"] != checkpoint["state_size"] or history_checkpoint["action_size"] != checkpoint["action_size"]:
            parser.error("History residual and base checkpoint sizes differ")
        history_residual = HistoryResidual(checkpoint["state_size"], checkpoint["action_size"], history_checkpoint["use_history"]).to(device)
        history_residual.load_state_dict(history_checkpoint["residual"])
        history_residual.eval()

    # Any saved row supplies the same Catan topology used by the model.
    with (HERE / "data/catanatron_diverse.jsonl").open(encoding="utf-8") as handle:
        first_row = json.loads(next(handle))
    edge_index = stable_edge_index(first_row["observation"])
    opponents = ("F_BASE", "F_CONTENDER", "VP", "W", "R")
    results = []
    all_latencies = []
    settlement_reranks = settlement_changes = settlement_heuristic_changes = buy_dev_heuristic_changes = same_turn_search_changes = 0
    if args.games_total is None:
        schedule = [
            opponent_code for opponent_code in opponents for _ in range(args.games_per_opponent)
        ]
    else:
        schedule = [opponents[index % len(opponents)] for index in range(args.games_total)]
    for game_index, opponent_code in enumerate(schedule):
        seed = args.seed + game_index
        colors = (Color.BLUE, Color.RED)
        game = Game([RandomPlayer(color) for color in colors], seed=seed, vps_to_win=10)
        mlp_color = game.state.colors[game_index % 2]
        opponent_color = next(color for color in colors if color != mlp_color)
        mlp = MLPPlayer(
            mlp_color, model, edge_index, device, adapter, settlement_head,
            args.settlement_heuristic, args.buy_dev_end_hand_threshold,
            history_residual,
            args.same_turn_search,
            args.same_turn_depth,
            args.history_scale,
        )
        policies = {mlp_color: mlp, opponent_color: make_bot(opponent_code, opponent_color)}
        error = None
        for _ in range(10_000):
            actor = game.state.current_color()
            action = policies[actor].decide(game, game.playable_actions)
            if action not in game.playable_actions:
                raise RuntimeError("Agent selected an illegal action")
            game.execute(action, validate_action=True)
            if game.winning_color() is not None:
                break
        winner = game.winning_color()
        all_latencies.extend(mlp.latencies_ms)
        settlement_reranks += mlp.settlement_reranks
        settlement_changes += mlp.settlement_changes
        settlement_heuristic_changes += mlp.settlement_heuristic_changes
        buy_dev_heuristic_changes += mlp.buy_dev_heuristic_changes
        same_turn_search_changes += mlp.same_turn_search_changes
        results.append(
            {
                "opponent": opponent_code,
                "seed": seed,
                "mlp_seat": game_index % 2,
                "mlp_won": winner == mlp_color,
                "completed": winner is not None,
                "plies": len(game.state.action_records),
                "error": error,
                "same_turn_search_changes": mlp.same_turn_search_changes,
            }
        )
    summary = {}
    for opponent_code in opponents:
        matches = [row for row in results if row["opponent"] == opponent_code]
        summary[opponent_code] = {
            "games": len(matches),
            "wins": sum(row["mlp_won"] for row in matches),
            "win_rate": sum(row["mlp_won"] for row in matches) / len(matches) if matches else None,
            "completed": sum(row["completed"] for row in matches),
        }
    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "games_per_opponent": args.games_per_opponent if args.games_total is None else None,
        "games_total": len(schedule),
        "settlement_target_checkpoint": str(args.settlement_target_checkpoint) if args.settlement_target_checkpoint else None,
        "settlement_target_reranks": settlement_reranks,
        "settlement_target_changes": settlement_changes,
        "settlement_heuristic": args.settlement_heuristic,
        "settlement_heuristic_changes": settlement_heuristic_changes,
        "buy_dev_end_hand_threshold": args.buy_dev_end_hand_threshold,
        "buy_dev_heuristic_changes": buy_dev_heuristic_changes,
        "history_checkpoint": str(args.history_checkpoint) if args.history_checkpoint else None,
        "history_scale": args.history_scale,
        "same_turn_search": args.same_turn_search,
        "same_turn_depth": args.same_turn_depth,
        "same_turn_search_changes": same_turn_search_changes,
        "summary": summary,
        "latency_ms": {
            "median": statistics.median(all_latencies),
            "p95": sorted(all_latencies)[int(0.95 * (len(all_latencies) - 1))],
        },
        "games": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "latency_ms": report["latency_ms"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
