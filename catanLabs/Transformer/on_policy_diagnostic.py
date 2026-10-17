"""Trace a trained Transformer in a few native games against its fair teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanLabs.Catanatron.engine import source_info
from catanLabs.Catanatron.tournament_protocol import PROTOCOL, canonical_action_key, install_v5_protocol
from catanLabs.Competitive.fair_value_agent import FairValueAgent
from catanLabs.Transformer.agent import TransformerAgent
from catanLabs.Transformer.legacy_v2_agent import LegacyV2TransformerAgent
from catanLabs.Transformer.visible_observation import RESOURCES, snapshot


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_opponent(name, color, fair_config):
    from catanatron.players.value import ValueFunctionPlayer
    from catanatron.players.minimax import AlphaBetaPlayer
    if name == "value":
        return ValueFunctionPlayer(color)
    if name == "alphabeta1":
        return AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=1))
    if name == "fair":
        return FairValueAgent(color, fair_config)
    raise ValueError(name)


def action_dict(action):
    return {"kind": action.kind, "value": action.value}


def placement_proxy(observation, action):
    """Public production summary; this is not a strategic value estimate."""
    if action.kind != "BUILD_SETTLEMENT":
        return None
    node = observation.nodes[int(action.value)]
    production = {name: round(float(value), 6)
                  for name, value in zip(RESOURCES, node.production)}
    return {
        "node": node.id,
        "expected_resource_cards_per_roll": round(sum(node.production), 6),
        "resource_diversity": sum(value > 0 for value in node.production),
        "resource_production_per_roll": production,
        "port": node.port,
        "adjacent_tile_ids": node.adjacent_tile_ids,
        "meaning": "Production/diversity proxy only; ignores competition, roads, scarcity and future dice.",
    }


def score_groups(decisions, field):
    groups = {}
    for decision in decisions:
        key = decision[field]
        row = groups.setdefault(key, {"decisions": 0, "exact_agreement": 0,
                                      "action_type_agreement": 0, "nonforced": 0,
                                      "nonforced_exact_agreement": 0})
        row["decisions"] += 1
        row["exact_agreement"] += int(decision["matches_teacher"])
        row["action_type_agreement"] += int(decision["same_action_type"])
        if decision["legal_action_count"] > 1:
            row["nonforced"] += 1
            row["nonforced_exact_agreement"] += int(decision["matches_teacher"])
    for row in groups.values():
        row["exact_rate"] = row["exact_agreement"] / row["decisions"]
        row["nonforced_exact_rate"] = (
            row["nonforced_exact_agreement"] / row["nonforced"] if row["nonforced"] else None
        )
    return dict(sorted(groups.items()))


def candidate_vp(game, color):
    state = game.state
    prefix = f"P{state.color_to_index[color]}_"
    return {
        "public": int(state.player_state[prefix + "VICTORY_POINTS"]),
        "actual": int(state.player_state[prefix + "ACTUAL_VICTORY_POINTS"]),
    }


def opening_summary(decisions):
    # Read the actual two opening placement decisions. Settlements may later
    # become cities, so the final board is not an opening record.
    placements = [row["opening_model"] for row in decisions if row.get("opening_model")]
    nodes = [placement["node"] for placement in placements]
    production = [0.0] * len(RESOURCES)
    for placement in placements:
        for index, name in enumerate(RESOURCES):
            production[index] += float(placement["resource_production_per_roll"][name])
    return {
        "settlement_nodes": nodes,
        "expected_resource_cards_per_roll": round(sum(production), 6),
        "resource_diversity": sum(value > 0 for value in production),
        "resource_production_per_roll": {name: round(production[i], 6)
                                         for i, name in enumerate(RESOURCES)},
    }


def markdown(report):
    rows = [
        "# On-policy Transformer diagnostic",
        "",
        "The trained Transformer played these native Catanatron games. At each of its visited decisions, the fair-value teacher independently chose from the **same legal actions in the same state**. Disagreement is a diagnostic, not proof that either move is better.",
        "",
        f"Protocol: `{report['protocol']}`; opponents: {', '.join(report['opponents'])}; "
        f"games: **{len(report['games'])}**; complete: **{sum(game['status'] == 'completed' for game in report['games'])}**.",
        "",
        "| Phase | Candidate decisions | Exact teacher match | Match when choice exists |",
        "|---|---:|---:|---:|",
    ]
    for phase, values in report["by_phase"].items():
        choice = values["nonforced_exact_rate"]
        rows.append(f"| {phase} | {values['decisions']} | {100 * values['exact_rate']:.1f}% | "
                    f"{100 * choice:.1f}% |" if choice is not None else
                    f"| {phase} | {values['decisions']} | {100 * values['exact_rate']:.1f}% | — |")
    rows += ["", "## Opening and VP traces", ""]
    for game in report["games"]:
        rows.append(f"- Seed {game['seed']}, seat {game['candidate_seat']}: {game['status']}; "
                    f"candidate final VP {game['candidate_final_vp']['actual']}; "
                    f"opening production {game['opening']['expected_resource_cards_per_roll']:.3f} "
                    f"cards/roll across {game['opening']['resource_diversity']} resource types. "
                    f"VP trajectory: {[(point['turn'], point['candidate_vp']['actual']) for point in game['vp_trajectory'][::max(1, len(game['vp_trajectory']) // 10)]]}.")
    rows += [
        "",
        "The JSON report contains each candidate decision, its legal count, model and teacher choices, own resource hand, opening-node production proxies, and the turn-by-turn VP trajectory. The production proxy omits road access, contention and resource scarcity. A few games cannot establish playing strength or cause of loss.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"start/.venv/bin/python catanLabs/Transformer/on_policy_diagnostic.py --games {report['requested_games']} "
        f"--seed {report['first_seed']} --checkpoint {report['checkpoint']} "
        f"--checkpoint-schema {report.get('checkpoint_schema', 'v2')} "
        f"--max-total-seconds {report['limits']['max_total_seconds']} "
        f"--max-game-seconds {report['limits']['max_game_seconds']} "
        f"--max-turns {report['limits']['max_turns']} "
        f"--output {report['output']}",
        "```",
        "",
        f"Checkpoint SHA-256: `{report['checkpoint_sha256']}`  ",
        f"Rules-manifest SHA-256: `{report['rules_manifest_sha256']}`",
    ]
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=WORKSPACE / "catanLabs/Transformer/runs/gate3_matched/transformer/best.pt")
    parser.add_argument("--checkpoint-schema", choices=("v2", "v3"), default="v2")
    parser.add_argument("--fair-config", type=Path,
                        default=WORKSPACE / "catanLabs/Competitive/configs/fair_army_finish.json")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20269200)
    parser.add_argument("--max-total-seconds", type=float, default=55.0)
    parser.add_argument("--max-game-seconds", type=float, default=20.0)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--output", type=Path,
                        default=WORKSPACE / "catanLabs/Transformer/runs/gate4_onpolicy/diagnostic.json")
    args = parser.parse_args()
    if not 2 <= args.games <= 4 or min(args.max_total_seconds, args.max_game_seconds, args.max_turns) <= 0:
        parser.error("Use 2-4 games and positive wall/turn caps")
    from catanatron import Color, Game, RandomPlayer
    engine = source_info()
    install_v5_protocol()
    config = json.loads(args.fair_config.read_text())
    manifest_path = WORKSPACE / "catanLabs/rules_manifest.json"
    rules = json.loads(manifest_path.read_text())
    if rules["local_protocol"] != PROTOCOL or rules["engine"]["commit"] != engine["commit"]:
        raise RuntimeError("Local rules manifest disagrees with v5 protocol or pinned engine")
    checkpoint = args.checkpoint.resolve(strict=True)
    opponent_names = ("value", "alphabeta1", "fair")
    all_decisions = []
    games = []
    began = time.perf_counter()
    for game_index in range(args.games):
        if time.perf_counter() - began >= args.max_total_seconds:
            break
        seed = args.seed + game_index
        game = Game([RandomPlayer(color) for color in Color], seed=seed, vps_to_win=10,
                    discard_limit=7, friendly_robber=False,
                    number_placement="official_spiral")
        game.playable_actions.sort(key=canonical_action_key)
        candidate_seat = game_index % 4
        candidate_color = game.state.colors[candidate_seat]
        agent_class = LegacyV2TransformerAgent if args.checkpoint_schema == "v2" else TransformerAgent
        candidate = agent_class(candidate_color, checkpoint)
        teacher = FairValueAgent(candidate_color, config)
        other_colors = [color for color in game.state.colors if color != candidate_color]
        opponents = {
            color: make_opponent(opponent_names[(index + game_index) % 3], color, config)
            for index, color in enumerate(other_colors)
        }
        game_started = time.perf_counter()
        decisions = []
        vp_trajectory = [{"turn": int(game.state.num_turns), "candidate_vp": candidate_vp(game, candidate_color)}]
        actions = 0
        status = "capped"
        while actions < 15000 and game.state.num_turns < args.max_turns:
            if game.winning_color() is not None:
                status = "completed"
                break
            if (time.perf_counter() - game_started >= args.max_game_seconds
                    or time.perf_counter() - began >= args.max_total_seconds):
                break
            actor = game.state.current_color()
            legal = tuple(game.playable_actions)
            if actor == candidate_color:
                observation = snapshot(game, legal)
                before_rng = game.state.random.getstate()
                teacher_action = teacher.decide(game, legal)
                if game.state.random.getstate() != before_rng:
                    raise RuntimeError("Teacher changed live game RNG while scoring")
                if tuple(game.playable_actions) != legal:
                    raise RuntimeError("Teacher changed the live legal-action set")
                teacher_index = legal.index(teacher_action)
                decision_started = time.perf_counter_ns()
                choice = candidate.decide_visible(observation)
                decision_ms = (time.perf_counter_ns() - decision_started) / 1e6
                if type(choice) is not int or not 0 <= choice < len(legal):
                    raise ValueError(f"Model returned illegal action index {choice!r}")
                action = legal[choice]
                model_choice = observation.legal_actions[choice]
                teacher_choice = observation.legal_actions[teacher_index]
                record = {
                    "game_index": game_index, "seed": seed, "candidate_seat": candidate_seat,
                    "turn": int(game.state.num_turns), "phase": observation.phase,
                    "legal_action_count": len(legal),
                    "model_action": action_dict(model_choice),
                    "teacher_action": action_dict(teacher_choice),
                    "teacher_action_type": teacher_choice.kind,
                    "matches_teacher": choice == teacher_index,
                    "same_action_type": model_choice.kind == teacher_choice.kind,
                    "own_resources": dict(zip(RESOURCES, observation.players[0].own_resources)),
                    "own_vp": candidate_vp(game, candidate_color),
                    "model_decision_ms": round(decision_ms, 4),
                }
                if observation.phase == "BUILD_INITIAL_SETTLEMENT":
                    record["opening_model"] = placement_proxy(observation, model_choice)
                    record["opening_teacher"] = placement_proxy(observation, teacher_choice)
                decisions.append(record)
                all_decisions.append(record)
            else:
                action = opponents[actor].decide(game, legal)
            if action not in legal:
                raise ValueError(f"Illegal action in game {game_index}: {action}")
            previous_turn = int(game.state.num_turns)
            game.execute(action, validate_action=True)
            actions += 1
            if int(game.state.num_turns) != previous_turn:
                vp_trajectory.append({"turn": int(game.state.num_turns),
                                      "candidate_vp": candidate_vp(game, candidate_color)})
        if game.winning_color() is not None:
            status = "completed"
        games.append({
            "seed": seed, "candidate_seat": candidate_seat,
            "candidate_color": candidate_color.value,
            "opponents": {color.value: type(player).__name__ for color, player in opponents.items()},
            "status": status, "winner": game.winning_color().value if game.winning_color() else None,
            "actions": actions, "turns": int(game.state.num_turns),
            "candidate_decisions": len(decisions),
            "candidate_final_vp": candidate_vp(game, candidate_color),
            "opening": opening_summary(decisions),
            "vp_trajectory": vp_trajectory,
            "wall_seconds": round(time.perf_counter() - game_started, 3),
        })
    if not games:
        raise RuntimeError("No games could start within the total wall cap")
    result = {
        "description": "On-policy action-disagreement trace; not a controlled win-rate benchmark",
        "protocol": PROTOCOL, "engine_commit": engine["commit"],
        "rules_manifest_sha256": sha256(manifest_path),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_schema": args.checkpoint_schema,
        "fair_config": str(args.fair_config), "fair_config_sha256": sha256(args.fair_config),
        "requested_games": args.games, "first_seed": args.seed,
        "opponents": opponent_names,
        "limits": {"max_total_seconds": args.max_total_seconds,
                   "max_game_seconds": args.max_game_seconds, "max_turns": args.max_turns},
        "wall_seconds": round(time.perf_counter() - began, 3),
        "candidate_decisions": len(all_decisions),
        "illegal_actions": 0,
        "teacher_live_rng_changes": 0,
        "exact_teacher_matches": sum(row["matches_teacher"] for row in all_decisions),
        "by_phase": score_groups(all_decisions, "phase"),
        "by_teacher_action_type": score_groups(all_decisions, "teacher_action_type"),
        "games": games, "decisions": all_decisions,
        "output": str(args.output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(result))
    print(json.dumps({"output": str(args.output),
                      "games": len(games), "completed": sum(g["status"] == "completed" for g in games),
                      "candidate_decisions": len(all_decisions),
                      "exact_teacher_matches": result["exact_teacher_matches"],
                      "wall_seconds": result["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
