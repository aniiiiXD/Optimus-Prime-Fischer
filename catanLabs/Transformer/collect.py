"""Collect visible-state/teacher-action examples from mixed Catanatron games."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanLabs.Catanatron.engine import source_info
from catanLabs.Catanatron.tournament_protocol import (
    PROTOCOL, canonical_action_key, install_v5_protocol,
)
from catanLabs.Transformer.model import (
    DATASET_SCHEMA, FEATURE_SCHEMA, EncodedRecord, encode_visible_observation,
)
from catanLabs.Transformer.visible_observation import snapshot
from catanLabs.Transformer.raw_storage import RawRecord, RAW_SCHEMA, RawRecordWriter


def make_player(name, color, fair_config, tactical_config):
    from catanatron import RandomPlayer
    from catanatron.players.weighted_random import WeightedRandomPlayer
    from catanatron.players.value import ValueFunctionPlayer
    from catanatron.players.minimax import AlphaBetaPlayer
    from catanLabs.Competitive.fair_value_agent import FairValueAgent
    from catanLabs.Competitive.tactical_vp_agent import TacticalVPAgent

    if name == "random":
        return RandomPlayer(color)
    if name == "weighted":
        return WeightedRandomPlayer(color)
    if name == "value":
        return ValueFunctionPlayer(color)
    if name == "alphabeta1":
        return AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=1))
    if name == "fair":
        return FairValueAgent(color, fair_config)
    if name == "tactical":
        return TacticalVPAgent(color, tactical_config)
    raise ValueError(name)


def canonical_example(encoded, label_action, legal_actions):
    state_features, state_kinds, state_ids, action_features, action_types, groups, target_ids = encoded
    order = sorted(
        range(len(legal_actions)),
        key=lambda i: (
            int(action_types[i]), int(groups[i]), int(target_ids[i]),
            action_features[i].tobytes(),
        ),
    )
    label_original = next(i for i, action in enumerate(legal_actions) if action == label_action)
    label = order.index(label_original)
    action_features = action_features[order]
    action_types = action_types[order]
    groups = groups[order]
    target_ids = target_ids[order]
    digest = hashlib.blake2b(digest_size=20)
    for array in (state_features, state_kinds, state_ids, action_features, action_types, groups, target_ids):
        digest.update(np.ascontiguousarray(array).tobytes())
    return EncodedRecord(
        state_features=state_features,
        state_kinds=state_kinds,
        state_ids=state_ids,
        action_features=action_features,
        action_types=action_types,
        target_groups=groups,
        target_ids=target_ids,
        label=label,
        game_id=-1,
        phase="",
        behavior="",
        actor_color="",
    ), digest.hexdigest(), tuple(order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-examples", type=int, default=4000)
    parser.add_argument("--wall-seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--teacher", choices=("fair", "tactical", "alphabeta1"), default="fair")
    parser.add_argument("--fair-config", type=Path,
                        default=WORKSPACE / "catanLabs/Competitive/configs/fair_army_finish.json")
    parser.add_argument("--tactical-config", type=Path,
                        default=WORKSPACE / "catanLabs/Competitive/configs/tactical_vp_min0.json")
    parser.add_argument("--trajectory", choices=("mixed", "tactical"), default="mixed",
                        help="Mixed behavior policies or all-seat TacticalVP self-play trajectories")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs/pilot/dataset.npz")
    parser.add_argument("--raw-output", type=Path,
                        help="Compressed JSONL of canonical visible snapshots; defaults beside --output")
    args = parser.parse_args()
    if min(args.games, args.max_examples) < 1 or args.wall_seconds <= 0:
        parser.error("games/examples/wall-seconds must be positive")

    engine = source_info()
    rules_manifest_path = WORKSPACE / "catanLabs/rules_manifest.json"
    rules_manifest_bytes = rules_manifest_path.read_bytes()
    rules_manifest = json.loads(rules_manifest_bytes)
    if rules_manifest["local_protocol"] != PROTOCOL or rules_manifest["engine"]["commit"] != engine["commit"]:
        raise RuntimeError("Collector protocol/engine disagree with catanLabs/rules_manifest.json")
    rules_manifest_sha256 = hashlib.sha256(rules_manifest_bytes).hexdigest()
    sys.path.insert(0, engine["python_root"])
    from catanatron import Color, Game, RandomPlayer
    install_v5_protocol()

    fair_config_path = args.fair_config.resolve(strict=True)
    fair_config_bytes = fair_config_path.read_bytes()
    fair_config = json.loads(fair_config_bytes)
    tactical_config_path = args.tactical_config.resolve(strict=True)
    tactical_config_bytes = tactical_config_path.read_bytes()
    tactical_config = json.loads(tactical_config_bytes)
    if args.teacher == "fair":
        teacher_config_sha256 = hashlib.sha256(fair_config_bytes).hexdigest()
        teacher_source_path = WORKSPACE / "catanLabs/Competitive/fair_value_agent.py"
    elif args.teacher == "tactical":
        teacher_config_sha256 = hashlib.sha256(tactical_config_bytes).hexdigest()
        teacher_source_path = WORKSPACE / "catanLabs/Competitive/tactical_vp_agent.py"
    else:
        from catanatron.players.minimax import AlphaBetaPlayer
        teacher_config_sha256 = hashlib.sha256(repr(AlphaBetaPlayer.Params(depth=1)).encode()).hexdigest()
        teacher_source_path = WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron/catanatron/players/minimax.py"
    teacher_source_sha256 = hashlib.sha256(teacher_source_path.read_bytes()).hexdigest()
    policy_names = ("random", "weighted", "value", "fair", "tactical", "alphabeta1")
    teacher_name = args.teacher
    records: list[EncodedRecord] = []
    raw_path = args.raw_output or args.output.with_name(args.output.stem + ".raw.jsonl.gz")
    raw_writer = RawRecordWriter(raw_path)
    seen: set[str] = set()
    game_summaries = []
    phase_counts: dict[str, int] = {}
    policy_counts: dict[str, int] = {}
    timings_ns = {"observation_encoding": 0, "teacher_scoring": 0,
                  "behavior_decision": 0, "game_stepping": 0}
    started = time.perf_counter()
    stop = False

    for game_id in range(args.games):
        if time.perf_counter() - started >= args.wall_seconds or len(records) >= args.max_examples:
            break
        # Tactical trajectories expose the teacher to its own reachable states.
        # The mixed mode retains the broader native-state coverage used by earlier data.
        names = (['tactical'] * 4 if args.trajectory == 'tactical' else
                 [policy_names[(seat + game_id) % len(policy_names)] for seat in range(4)])
        colors = list(Color)
        game = Game([RandomPlayer(color) for color in colors], seed=args.seed + game_id,
                    vps_to_win=10, discard_limit=7,
                    friendly_robber=False, number_placement="official_spiral")
        game.playable_actions.sort(key=canonical_action_key)
        name_by_color = {color: names[i] for i, color in enumerate(colors)}
        behavior_agents = {color: make_player(names[i], color, fair_config, tactical_config)
                           for i, color in enumerate(colors)}
        record_indices: list[int] = []
        pending_raw: list[RawRecord] = []
        action_count = 0
        game_started = time.perf_counter()
        game_status = "capped"

        while action_count < 15000 and game.state.num_turns < 500:
            winner = game.winning_color()
            if winner is not None:
                game_status = "completed"
                break
            if time.perf_counter() - started >= args.wall_seconds:
                stop = True
                break
            actor = game.state.current_color()
            legal = list(game.playable_actions)
            behavior = name_by_color[actor]
            teacher = make_player(teacher_name, actor, fair_config, tactical_config)

            tick = time.perf_counter_ns()
            visible = snapshot(game, legal)
            encoded = encode_visible_observation(visible)
            timings_ns["observation_encoding"] += time.perf_counter_ns() - tick
            tick = time.perf_counter_ns()
            teacher_action = teacher.decide(game, legal)
            timings_ns["teacher_scoring"] += time.perf_counter_ns() - tick
            if teacher_action not in legal:
                raise ValueError(f"Teacher selected an illegal action: {teacher_action}")
            record, digest, order = canonical_example(encoded, teacher_action, legal)
            if digest not in seen and len(records) < args.max_examples:
                record.game_id = game_id
                record.phase = game.state.current_prompt.name
                record.behavior = behavior
                record.actor_color = actor.value
                record_indices.append(len(records))
                records.append(record)
                canonical_visible = replace(
                    visible,
                    legal_actions=tuple(visible.legal_actions[index] for index in order),
                )
                pending_raw.append(RawRecord(
                    observation=canonical_visible,
                    canonical_state_key=digest,
                    game_id=game_id,
                    board_seed=args.seed + game_id,
                    actor_color=actor.value,
                    actor_seat=int(game.state.color_to_index[actor]),
                    phase=record.phase,
                    behavior=behavior,
                    teacher=teacher_name,
                    teacher_config_sha256=teacher_config_sha256,
                    teacher_source_sha256=teacher_source_sha256,
                    target_action_index=record.label,
                    outcome=-1,
                    final_vp=-1,
                    benchmark_protocol=PROTOCOL,
                    rules_manifest_sha256=rules_manifest_sha256,
                    engine_commit=engine["commit"],
                    feature_schema=FEATURE_SCHEMA,
                ))
                seen.add(digest)
                phase_counts[record.phase] = phase_counts.get(record.phase, 0) + 1
                policy_counts[behavior] = policy_counts.get(behavior, 0) + 1

            actor_player = behavior_agents[actor]
            if behavior == teacher_name:
                chosen = teacher_action
            else:
                tick = time.perf_counter_ns()
                chosen = actor_player.decide(game, legal)
                timings_ns["behavior_decision"] += time.perf_counter_ns() - tick
            if chosen not in legal:
                raise ValueError(f"Behavior policy selected an illegal action: {chosen}")
            tick = time.perf_counter_ns()
            game.execute(chosen, validate_action=True)
            timings_ns["game_stepping"] += time.perf_counter_ns() - tick
            action_count += 1

        winner = game.winning_color()
        if winner is not None:
            game_status = "completed"
            for pending_index, record_index in enumerate(record_indices):
                record = records[record_index]
                actor_color = next(color for color in game.state.colors if color.value == record.actor_color)
                record.outcome = int(winner == actor_color)
                record.final_vp = int(game.state.player_state[
                    f"P{game.state.color_to_index[actor_color]}_ACTUAL_VICTORY_POINTS"
                ])
                pending_raw[pending_index] = replace(pending_raw[pending_index],
                                                     outcome=record.outcome,
                                                     final_vp=record.final_vp)
        for raw_record in pending_raw:
            raw_writer.write(raw_record)
        game_summaries.append({
            "game_id": game_id,
            "seed": args.seed + game_id,
            "behavior_by_input_color": names,
            "status": game_status,
            "winner": winner.value if winner is not None else None,
            "actions": action_count,
            "turns": int(game.state.num_turns),
            "wall_seconds": round(time.perf_counter() - game_started, 3),
        })
        print(f"collection game {game_id + 1}/{args.games}: {game_status}, {action_count} actions, {len(records)} unique examples", flush=True)
        if stop:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    max_candidates = max((len(record.action_features) for record in records), default=1)
    count = len(records)
    state_features = np.stack([record.state_features for record in records]) if count else np.zeros((0, 150, 40), dtype=np.float32)
    state_kinds = np.stack([record.state_kinds for record in records]) if count else np.zeros((0, 150), dtype=np.int64)
    state_ids = np.stack([record.state_ids for record in records]) if count else np.zeros((0, 150), dtype=np.int64)
    action_features = np.zeros((count, max_candidates, 20), dtype=np.float32)
    action_types = np.zeros((count, max_candidates), dtype=np.int64)
    target_groups = np.zeros((count, max_candidates), dtype=np.int64)
    target_ids = np.zeros((count, max_candidates), dtype=np.int64)
    counts = np.zeros(count, dtype=np.int64)
    for i, record in enumerate(records):
        n = len(record.action_features)
        action_features[i, :n] = record.action_features
        action_types[i, :n] = record.action_types
        target_groups[i, :n] = record.target_groups
        target_ids[i, :n] = record.target_ids
        counts[i] = n
    np.savez_compressed(
        args.output,
        dataset_schema=np.asarray(DATASET_SCHEMA),
        feature_schema=np.asarray(FEATURE_SCHEMA),
        state_features=state_features,
        state_kinds=state_kinds,
        state_ids=state_ids,
        action_features=action_features,
        action_types=action_types,
        target_groups=target_groups,
        target_ids=target_ids,
        candidate_counts=counts,
        labels=np.asarray([record.label for record in records], dtype=np.int64),
        game_ids=np.asarray([record.game_id for record in records], dtype=np.int32),
        phases=np.asarray([record.phase for record in records]),
        behaviors=np.asarray([record.behavior for record in records]),
        actor_colors=np.asarray([record.actor_color for record in records]),
        outcomes=np.asarray([record.outcome for record in records], dtype=np.int8),
        final_vps=np.asarray([record.final_vp for record in records], dtype=np.int8),
    )
    raw_sha256 = raw_writer.close()
    raw_count = raw_writer.count
    if raw_count != count:
        raise AssertionError("Raw and tensor record counts differ")
    manifest = {
        "schema": DATASET_SCHEMA,
        "dataset_schema": DATASET_SCHEMA,
        "raw_schema": RAW_SCHEMA,
        "raw_file": str(raw_path),
        "raw_records": raw_count,
        "raw_sha256": raw_sha256,
        "feature_schema": FEATURE_SCHEMA,
        "benchmark_protocol": PROTOCOL,
        "rules_manifest_sha256": rules_manifest_sha256,
        "seed": args.seed,
        "teacher": teacher_name,
        "teacher_depth": 1 if teacher_name == "alphabeta1" else None,
        "teacher_config": str(fair_config_path) if teacher_name == "fair" else (str(tactical_config_path) if teacher_name == "tactical" else None),
        "teacher_config_sha256": (hashlib.sha256(fair_config_bytes).hexdigest() if teacher_name == "fair" else (hashlib.sha256(tactical_config_bytes).hexdigest() if teacher_name == "tactical" else None)),
        "raw_teacher_config_sha256": teacher_config_sha256,
        "raw_teacher_source_sha256": teacher_source_sha256,
        "teacher_source_sha256": hashlib.sha256(teacher_source_path.read_bytes()).hexdigest() if teacher_name in {"fair", "tactical"} else None,
        "teacher_information_boundary": (
            "fair policy uses public game state and own private state; hidden perturbation smoke tested"
            if teacher_name == "fair" else (
                "tactical policy uses public game state and own private state; its exact scan does not sample hidden outcomes"
                if teacher_name == "tactical" else
            "Catanatron AlphaBeta1 receives full simulator Game; privileged/unverified labels"
            )
        ),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "visible_snapshot_sha256": hashlib.sha256(
            (WORKSPACE / "catanLabs/Transformer/visible_observation.py").read_bytes()
        ).hexdigest(),
        "encoder_sha256": hashlib.sha256(
            (WORKSPACE / "catanLabs/Transformer/model.py").read_bytes()
        ).hexdigest(),
        "teacher_action_contract": "Selected before the real action executes; never trained from future outcomes.",
        "outcome_contract": "Post-game analysis only; incomplete games use -1; excluded from action-policy inputs.",
        "chance_contract": {"isolated_copy_rng": True, "forced_search_outcomes": True,
                            "canonical_action_order": True, "python_hash_seed": os.environ.get("PYTHONHASHSEED")},
        "engine": engine,
        "unique_examples": count,
        "unique_key": "player-relative visible state tokens + sorted legal actions",
        "feature_boundary": "own private cards; public board/opponents/bank; no opponent hands or unplayed card identities",
        "max_legal_actions": max_candidates,
        "games": game_summaries,
        "phases": phase_counts,
        "trajectory_policies": policy_counts,
        "trajectory_mode": args.trajectory,
        "collection_wall_seconds": round(time.perf_counter() - started, 3),
        "profile_seconds": {key: round(value / 1e9, 6) for key, value in timings_ns.items()},
        "dataset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"dataset": str(args.output), "examples": count, "phases": phase_counts,
                      "policies": policy_counts, "collection_seconds": manifest["collection_wall_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
