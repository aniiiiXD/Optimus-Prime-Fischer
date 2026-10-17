"""Collect fair-teacher labels on complete Transformer self-play games.

The behavior policy receives only immutable visible observations. Teacher
labels are taken before the behavior action; outcomes are attached only after
a complete game. A wall-capped or otherwise incomplete game contributes no
training rows.
"""

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
import torch


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanLabs.Catanatron.engine import source_info
from catanLabs.Catanatron.tournament_protocol import (
    PROTOCOL, canonical_action_key, install_v5_protocol,
)
from catanLabs.Transformer.agent import TransformerAgent
from catanLabs.Transformer.collect import canonical_example
from catanLabs.Transformer.model import (
    ACTION_FEATURE_DIM, CHECKPOINT_SCHEMA, DATASET_SCHEMA, FEATURE_DIM,
    FEATURE_SCHEMA, STATE_TOKEN_COUNT, EncodedRecord, encode_visible_observation,
)
from catanLabs.Transformer.raw_storage import RAW_SCHEMA, RawRecord, RawRecordWriter
from catanLabs.Transformer.visible_observation import snapshot


BEHAVIOR_NAME = "transformer_selfplay"
COLLECTOR_SCHEMA = "catanlabs-on-policy-fair-labels-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pack(records: list[EncodedRecord], output: Path) -> int:
    count = len(records)
    max_candidates = max(len(record.action_features) for record in records)
    action_features = np.zeros((count, max_candidates, ACTION_FEATURE_DIM), dtype=np.float32)
    action_types = np.zeros((count, max_candidates), dtype=np.int64)
    target_groups = np.zeros((count, max_candidates), dtype=np.int64)
    target_ids = np.zeros((count, max_candidates), dtype=np.int64)
    counts = np.zeros(count, dtype=np.int64)
    for index, record in enumerate(records):
        width = len(record.action_features)
        action_features[index, :width] = record.action_features
        action_types[index, :width] = record.action_types
        target_groups[index, :width] = record.target_groups
        target_ids[index, :width] = record.target_ids
        counts[index] = width
    state_features = np.stack([record.state_features for record in records])
    state_kinds = np.stack([record.state_kinds for record in records])
    state_ids = np.stack([record.state_ids for record in records])
    if state_features.shape[1:] != (STATE_TOKEN_COUNT, FEATURE_DIM):
        raise ValueError("Unexpected state tensor shape")
    np.savez_compressed(
        output,
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
    return max_candidates


def collect(checkpoint_path: Path, output: Path, games: int, seed: int,
            wall_seconds: float, max_examples: int, fair_config_path: Path,
            *, behavior_agent=None) -> dict:
    if games < 1 or wall_seconds <= 0 or max_examples < 1:
        raise ValueError("games, wall_seconds and max_examples must be positive")
    if DATASET_SCHEMA != "catanlabs-transformer-dataset-v3" or FEATURE_SCHEMA != "visible-state-tokens-v3":
        raise RuntimeError("On-policy collector requires the v3 feature and dataset schema")
    output = output.resolve()
    raw_path = output.with_name(output.stem + ".raw.jsonl.gz")
    if any(path.exists() for path in (output, output.with_suffix(".json"), raw_path)):
        raise FileExistsError("On-policy output already exists; collector never overwrites a dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_path.resolve(strict=True)
    checkpoint_sha256 = _sha256(checkpoint_path)
    checkpoint_metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint_metadata.get("schema") != CHECKPOINT_SCHEMA
            or checkpoint_metadata.get("feature_schema") != FEATURE_SCHEMA
            or checkpoint_metadata.get("dataset_schema") != DATASET_SCHEMA):
        raise ValueError("Behavior checkpoint is not a v3 Transformer checkpoint")

    engine = source_info()
    rules_path = WORKSPACE / "catanLabs/rules_manifest.json"
    rules = json.loads(rules_path.read_text())
    if rules["local_protocol"] != PROTOCOL or rules["engine"]["commit"] != engine["commit"]:
        raise RuntimeError("Collector rules/engine disagree with provisional rules manifest")
    rules_sha256 = _sha256(rules_path)
    config = json.loads(fair_config_path.read_text())
    config_sha256 = _sha256(fair_config_path)
    teacher_source_path = WORKSPACE / "catanLabs/Competitive/fair_value_agent.py"
    teacher_source_sha256 = _sha256(teacher_source_path)

    sys.path.insert(0, engine["python_root"])
    from catanatron import Color, Game, RandomPlayer
    from catanLabs.Competitive.fair_value_agent import FairValueAgent
    install_v5_protocol()
    torch.set_num_threads(1)
    colors = list(Color)
    # The Transformer consumes player-relative observations, so one network
    # instance can safely decide for all four actor seats.
    smoke_stub = behavior_agent is not None
    behavior_name = "smoke_visible_stub" if smoke_stub else BEHAVIOR_NAME
    policy = behavior_agent if smoke_stub else TransformerAgent(colors[0], checkpoint_path)
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("Behavior checkpoint changed while it was loaded")
    teachers = {color: FairValueAgent(color, config) for color in colors}
    records: list[EncodedRecord] = []
    writer = RawRecordWriter(raw_path)
    seen: set[str] = set()
    summaries = []
    phase_counts: dict[str, int] = {}
    timings_ns = {"observation_encoding": 0, "teacher_scoring": 0,
                  "behavior_decision": 0, "game_stepping": 0}
    started = time.perf_counter()

    for game_id in range(games):
        if time.perf_counter() - started >= wall_seconds or len(records) >= max_examples:
            break
        game = Game([RandomPlayer(color) for color in colors], seed=seed + game_id,
                    vps_to_win=10, discard_limit=7, friendly_robber=False,
                    number_placement="official_spiral")
        game.playable_actions.sort(key=canonical_action_key)
        pending: list[tuple[EncodedRecord, RawRecord]] = []
        pending_keys: set[str] = set()
        game_started = time.perf_counter()
        actions = 0
        status = "capped"
        while actions < 15000 and game.state.num_turns < 500:
            winner = game.winning_color()
            if winner is not None:
                status = "completed"
                break
            if time.perf_counter() - started >= wall_seconds:
                status = "wall_capped"
                break
            actor = game.state.current_color()
            legal = list(game.playable_actions)
            tick = time.perf_counter_ns()
            visible = snapshot(game, legal)
            encoded = encode_visible_observation(visible)
            timings_ns["observation_encoding"] += time.perf_counter_ns() - tick

            record_this = len(records) + len(pending) < max_examples
            if record_this:
                tick = time.perf_counter_ns()
                teacher_action = teachers[actor].decide(game, legal)
                timings_ns["teacher_scoring"] += time.perf_counter_ns() - tick
                if teacher_action not in legal:
                    raise ValueError("Fair teacher selected an illegal action")
                record, digest, order = canonical_example(encoded, teacher_action, legal)
                if digest not in seen and digest not in pending_keys:
                    record.game_id = game_id
                    record.phase = game.state.current_prompt.name
                    record.behavior = behavior_name
                    record.actor_color = actor.value
                    canonical_visible = replace(
                        visible,
                        legal_actions=tuple(visible.legal_actions[index] for index in order),
                    )
                    raw_record = RawRecord(
                        observation=canonical_visible,
                        canonical_state_key=digest,
                        game_id=game_id,
                        board_seed=seed + game_id,
                        actor_color=actor.value,
                        actor_seat=int(game.state.color_to_index[actor]),
                        phase=record.phase,
                        behavior=behavior_name,
                        teacher="fair",
                        teacher_config_sha256=config_sha256,
                        teacher_source_sha256=teacher_source_sha256,
                        target_action_index=record.label,
                        outcome=-1,
                        final_vp=-1,
                        benchmark_protocol=PROTOCOL,
                        rules_manifest_sha256=rules_sha256,
                        engine_commit=engine["commit"],
                        feature_schema=FEATURE_SCHEMA,
                    )
                    pending.append((record, raw_record))
                    pending_keys.add(digest)
            tick = time.perf_counter_ns()
            action_index = policy.decide_visible(visible)
            timings_ns["behavior_decision"] += time.perf_counter_ns() - tick
            if type(action_index) is not int or not 0 <= action_index < len(legal):
                raise ValueError("Transformer behavior selected an illegal action index")
            tick = time.perf_counter_ns()
            game.execute(legal[action_index], validate_action=True)
            timings_ns["game_stepping"] += time.perf_counter_ns() - tick
            actions += 1

        winner = game.winning_color()
        if winner is not None:
            status = "completed"
            for record, raw_record in pending:
                actor_color = next(color for color in colors if color.value == record.actor_color)
                actual_vp = int(game.state.player_state[
                    f"P{game.state.color_to_index[actor_color]}_ACTUAL_VICTORY_POINTS"
                ])
                outcome = int(winner == actor_color)
                record.outcome, record.final_vp = outcome, actual_vp
                writer.write(replace(raw_record, outcome=outcome, final_vp=actual_vp))
                records.append(record)
                phase_counts[record.phase] = phase_counts.get(record.phase, 0) + 1
            seen.update(pending_keys)
        summaries.append({
            "game_id": game_id, "seed": seed + game_id,
            "status": status, "winner": winner.value if winner is not None else None,
            "actions": actions, "turns": int(game.state.num_turns),
            "stored_examples": len(pending) if status == "completed" else 0,
            "discarded_examples": len(pending) if status != "completed" else 0,
            "wall_seconds": round(time.perf_counter() - game_started, 3),
        })
        print(f"on-policy game {game_id + 1}/{games}: {status}, {actions} actions, {len(records)} stored", flush=True)
        if status == "wall_capped":
            break

    raw_sha256 = writer.close()
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("Behavior checkpoint changed during collection")
    if not records:
        raise RuntimeError("No complete game produced labeled examples")
    max_candidates = _pack(records, output)
    if writer.count != len(records):
        raise AssertionError("Raw and tensor row counts differ")
    manifest = {
        "schema": DATASET_SCHEMA, "dataset_schema": DATASET_SCHEMA,
        "feature_schema": FEATURE_SCHEMA, "raw_schema": RAW_SCHEMA,
        "collector_schema": COLLECTOR_SCHEMA,
        "dataset_sha256": _sha256(output),
        "raw_file": str(raw_path), "raw_records": writer.count,
        "raw_sha256": raw_sha256,
        "behavior": behavior_name,
        "smoke_only": smoke_stub,
        "behavior_checkpoint_used": not smoke_stub,
        "behavior_checkpoint": str(checkpoint_path),
        "behavior_checkpoint_sha256": checkpoint_sha256,
        "behavior_checkpoint_schema": checkpoint_metadata["schema"],
        "behavior_checkpoint_dataset_sha256": checkpoint_metadata.get("dataset_sha256"),
        "behavior_checkpoint_split_file_sha256": checkpoint_metadata.get("split_file_sha256"),
        "teacher": "fair", "teacher_config": str(fair_config_path),
        "teacher_config_sha256": config_sha256,
        "teacher_source_sha256": teacher_source_sha256,
        "raw_teacher_config_sha256": config_sha256,
        "raw_teacher_source_sha256": teacher_source_sha256,
        "teacher_information_boundary": "fair policy uses public state and acting player's private state",
        "teacher_action_contract": "Teacher labels are selected before the behavior action executes.",
        "outcome_contract": "Only completed games are stored; outcomes are analysis fields, not policy inputs.",
        "behavior_information_boundary": "Transformer receives only VisibleObservation.",
        "benchmark_protocol": PROTOCOL,
        "rules_manifest_sha256": rules_sha256,
        "engine": engine,
        "chance_contract": {"isolated_copy_rng": True, "forced_search_outcomes": True,
                            "canonical_action_order": True, "python_hash_seed": os.environ.get("PYTHONHASHSEED")},
        "collector_sha256": _sha256(Path(__file__)),
        "visible_snapshot_sha256": _sha256(WORKSPACE / "catanLabs/Transformer/visible_observation.py"),
        "encoder_sha256": _sha256(WORKSPACE / "catanLabs/Transformer/model.py"),
        "unique_examples": len(records),
        "unique_key": "player-relative visible state tokens + sorted legal actions",
        "feature_boundary": "own private cards; public board/opponents/bank; no opponent hands or unplayed card identities",
        "max_legal_actions": max_candidates,
        "games": summaries,
        "phases": phase_counts,
        "trajectory_policies": {behavior_name: len(records)},
        "collection_wall_seconds": round(time.perf_counter() - started, 3),
        "profile_seconds": {key: round(value / 1e9, 6) for key, value in timings_ns.items()},
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20273000)
    parser.add_argument("--wall-seconds", type=float, default=120.0)
    parser.add_argument("--max-examples", type=int, default=4000)
    parser.add_argument("--fair-config", type=Path,
                        default=WORKSPACE / "catanLabs/Competitive/configs/fair_army_finish.json")
    parser.add_argument("--output", type=Path,
                        default=WORKSPACE / "catanLabs/Transformer/runs/on_policy/dataset.npz")
    args = parser.parse_args()
    result = collect(args.checkpoint, args.output, args.games, args.seed,
                     args.wall_seconds, args.max_examples, args.fair_config.resolve(strict=True))
    print(json.dumps({"dataset": str(args.output), "completed_games": sum(
        game["status"] == "completed" for game in result["games"]),
        "examples": result["unique_examples"], "dataset_sha256": result["dataset_sha256"]}, indent=2))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
