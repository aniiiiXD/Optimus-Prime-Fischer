#!/usr/bin/env python3
"""Offline root-UCB counterfactual audit of late decisions in lost games.

This deliberately starts from the engine's complete game state, including
unknown opponent cards. It is an oracle diagnostic, never an online policy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

from train_legal_ranker import HERE, LegalRanker, encode_action, encode_state, stable_edge_index
from train_calibrated_ranker import ActionCalibrationAdapter
from settlement_target_head import SettlementTargetHead

WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))
sys.path.insert(0, str(WORKSPACE))
from catanatron import Color, Game, RandomPlayer  # noqa: E402
from catanatron.state_functions import get_actual_victory_points, get_visible_victory_points  # noqa: E402
from catanLabs.Catanatron.tournament_protocol import install_v5_protocol  # noqa: E402
from generate_dataset import action_json, make_bot, observation  # noqa: E402
from evaluate_win_rate import MLPPlayer  # noqa: E402
from diagnose_losses import expected_pips  # noqa: E402


def candidate_actions(game, player, legal, limit=3):
    obs = observation(game, player.color)
    state = torch.tensor([encode_state(obs, player.edge_index)], dtype=torch.float32)
    actions = torch.tensor([[encode_action(action_json(action, player.color), obs, player.edge_index) for action in legal]], dtype=torch.float32)
    with torch.no_grad():
        scores = player.adapter(player.model(state, actions), state, actions)[0].tolist()
    baseline = player.decide(game, legal)
    selected = [baseline]
    if baseline.action_type.name == "BUILD_SETTLEMENT":
        # Location counterfactuals are central to the expansion bottleneck.
        for index in sorted(range(len(legal)), key=lambda i: scores[i], reverse=True):
            if legal[index].action_type.name == "BUILD_SETTLEMENT" and legal[index] not in selected:
                selected.append(legal[index])
            if len(selected) == limit:
                return selected
    # Pick strong alternatives of different types so the audit studies choices
    # such as build versus wait, rather than nearly identical trade tuples.
    for index in sorted(range(len(legal)), key=lambda i: scores[i], reverse=True):
        option = legal[index]
        if option not in selected and all(option.action_type != prior.action_type for prior in selected):
            selected.append(option)
        if len(selected) == limit:
            break
    return selected


def rollout(root, action, color, player, opponent, seed, horizon):
    game = root.copy()
    rng = random.Random(seed)
    game.random = rng
    game.state.random = rng
    game.execute(action, validate_action=True)
    for _ in range(horizon):
        if game.winning_color() is not None:
            break
        actor = game.state.current_color()
        policy = player if actor == color else opponent
        chosen = policy.decide(game, list(game.playable_actions))
        game.execute(chosen, validate_action=True)
    winner = game.winning_color()
    if winner is not None:
        return (1.0 if winner == color else -1.0), True
    enemy = next(c for c in game.state.colors if c != color)
    vp_gap = get_actual_victory_points(game.state, color) - get_visible_victory_points(game.state, enemy)
    production_gap = expected_pips(game, color) - expected_pips(game, enemy)
    return math.tanh(0.30 * vp_gap + 0.025 * production_gap), False


def search(root, actions, color, player, opponent, seed, simulations, horizon):
    visits = [0] * len(actions)
    totals = [0.0] * len(actions)
    terminals = [0] * len(actions)
    for simulation in range(simulations):
        unvisited = next((i for i, count in enumerate(visits) if count == 0), None)
        if unvisited is not None:
            index = unvisited
        else:
            index = max(range(len(actions)), key=lambda i: totals[i] / visits[i] + 0.7 * math.sqrt(math.log(simulation + 1) / visits[i]))
        value, terminal = rollout(root, actions[index], color, player, opponent, seed + simulation * 1597, horizon)
        visits[index] += 1
        totals[index] += value
        terminals[index] += terminal
    return [
        {"action": action_json(action, color), "visits": visits[i], "mean_value": round(totals[i] / visits[i], 3), "terminal_rollouts": terminals[i]}
        for i, action in enumerate(actions)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20267400)
    parser.add_argument("--simulations", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--states-per-loss", type=int, default=3)
    parser.add_argument("--validation-rollouts", type=int, default=8)
    parser.add_argument("--output", type=Path, default=HERE / "reports/loss_counterfactual_search.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    install_v5_protocol()
    checkpoint = torch.load(HERE / "runs/calibrated_ranker_v1/model.pt", map_location="cpu", weights_only=True)
    model = LegalRanker(checkpoint["state_size"], checkpoint["action_size"])
    model.load_state_dict(checkpoint["base_model"])
    model.eval()
    adapter = ActionCalibrationAdapter(checkpoint["state_size"])
    adapter.load_state_dict(checkpoint["adapter"])
    adapter.eval()
    target_checkpoint = torch.load(HERE / "runs/settlement_target_v1/epoch_4.pt", map_location="cpu", weights_only=True)
    target_head = SettlementTargetHead(target_checkpoint["input_size"])
    target_head.load_state_dict(target_checkpoint["head"])
    target_head.eval()
    with (HERE / "data/catanatron_diverse.jsonl").open(encoding="utf-8") as handle:
        edge_index = stable_edge_index(json.loads(next(handle))["observation"])
    reports = []
    for game_id in range(args.games):
        game = Game([RandomPlayer(Color.BLUE), RandomPlayer(Color.RED)], seed=args.seed + game_id, vps_to_win=10)
        color = game.state.colors[game_id % 2]
        enemy = next(c for c in game.state.colors if c != color)
        player = MLPPlayer(color, model, edge_index, torch.device("cpu"), adapter, target_head)
        opponent = make_bot("F_BASE", enemy)
        snapshots = []
        for _ in range(10_000):
            actor = game.state.current_color()
            legal = list(game.playable_actions)
            if actor == color:
                action = player.decide(game, legal)
                enemy_vp = get_visible_victory_points(game.state, enemy)
                if len(legal) > 1 and (enemy_vp >= 4 or sum(a.action_type.name == "BUILD_SETTLEMENT" for a in legal) >= 2):
                    snapshots.append((game.copy(), list(legal), action, len(game.state.action_records), enemy_vp))
            else:
                action = opponent.decide(game, legal)
            game.execute(action, validate_action=True)
            if game.winning_color() is not None:
                break
        if game.winning_color() == color or not snapshots:
            continue
        selected = []
        build_item = next((item for item in reversed(snapshots) if sum(a.action_type.name == "BUILD_SETTLEMENT" for a in item[1]) >= 2 and item[3] > 5), None)
        if build_item is not None:
            selected.append(build_item)
        for threshold in (4, 6, 8):
            item = next((item for item in snapshots if item[4] >= threshold and item[3] not in {x[3] for x in selected}), None)
            if item is not None:
                selected.append(item)
            if len(selected) >= args.states_per_loss:
                break
        for root, legal, actual, ply, enemy_vp in selected:
            options = candidate_actions(root, player, legal)
            if len(options) < 2:
                continue
            estimates = search(root, options, color, player, opponent, args.seed + game_id * 100000 + ply * 1000, args.simulations, args.horizon)
            paired = [
                [rollout(root, option, color, player, opponent,
                         args.seed + game_id * 100000 + ply * 1000 + 10_000_000 + repeat * 7919,
                         args.horizon)[0] for repeat in range(args.validation_rollouts)]
                for option in options
            ]
            for index, branch in enumerate(estimates):
                branch["paired_validation_mean"] = round(sum(paired[index]) / len(paired[index]), 3)
                branch["paired_validation_wins"] = sum(value == 1.0 for value in paired[index])
            best = max(range(len(estimates)), key=lambda i: estimates[i]["paired_validation_mean"])
            result = {"game_id": game_id, "seed": args.seed + game_id, "ply": ply,
                      "visible_opponent_vp": enemy_vp,
                      "actual_move": action_json(actual, color), "branches": estimates,
                      "search_preferred_alternative": best != 0}
            reports.append(result)
            print(json.dumps({"game_id": game_id, "ply": ply, "branches": estimates}), flush=True)
    output = {"oracle_diagnostic": True, "games_attempted": args.games, "loss_states_analyzed": len(reports), "search_preferred_alternative": sum(row["search_preferred_alternative"] for row in reports), "states": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "states"}), flush=True)


if __name__ == "__main__":
    main()
