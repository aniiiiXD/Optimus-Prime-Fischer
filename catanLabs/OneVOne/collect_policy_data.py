"""Collect visible-state/action/outcome trajectories from a teacher league."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

from catanLabs.Catanatron.engine import source_info  # noqa: E402

ENGINE = source_info()
sys.path.insert(0, ENGINE["python_root"])

from catanLabs.Catanatron.tournament_protocol import (  # noqa: E402
    canonical_action_key,
    install_v5_protocol,
)
from catanLabs.OneVOne.policy_features import (  # noqa: E402
    ACTION_FEATURE_DIM,
    FEATURES_VERSION,
    FAMILY_NAMES,
    encode_legal_actions,
    encode_state,
)

install_v5_protocol()


DEFAULT_TEACHERS = "onevone,fair,tactical,catanatron,alphabeta2,weighted"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_player(name: str, color: object, config: dict, seed: int):
    if name == "onevone":
        from catanLabs.OneVOne.agent import OneVOneAgent
        return OneVOneAgent(color, config)
    from catanLabs.OneVOne.evaluate import make_opponent
    return make_opponent(name, color, config)


def collect_game(seed: int, focal_seat: int, teacher_name: str,
                 opponent_name: str, config: dict, caps: dict) -> dict:
    from catanatron import Color, Game, RandomPlayer

    rules = config["rules"]
    colors = [Color.RED, Color.BLUE]
    game = Game(
        [RandomPlayer(color) for color in colors],
        seed=seed,
        vps_to_win=int(rules["target_vp"]),
        discard_limit=int(rules["discard_limit"]),
        friendly_robber=bool(rules["friendly_robber"]),
        number_placement=str(rules["number_placement"]),
    )
    game.playable_actions.sort(key=canonical_action_key)
    focal_color = game.state.colors[focal_seat]
    other_color = next(color for color in game.state.colors if color != focal_color)
    players = {
        focal_color: build_player(teacher_name, focal_color, config, seed),
        other_color: build_player(opponent_name, other_color, config, seed),
    }
    examples = []
    started = time.perf_counter()
    action_count = 0
    status = "capped"
    error = None

    try:
        while (action_count < caps["max_actions"]
               and game.state.num_turns < caps["max_turns"]):
            winner = game.winning_color()
            if winner is not None:
                status = "completed"
                break
            if time.perf_counter() - started > caps["max_game_seconds"]:
                break
            game.playable_actions.sort(key=canonical_action_key)
            actor = game.state.current_color()
            legal = game.playable_actions
            player = players[actor]

            state_features = action_features = families = None
            if actor == focal_color:
                state_features = encode_state(game, focal_color)
                action_features, families = encode_legal_actions(legal, game.state.board.map)

            chosen = player.decide(game, legal)
            if chosen not in legal:
                raise ValueError(f"{type(player).__name__} returned an illegal action: {chosen!r}")

            if actor == focal_color:
                chosen_index = next(i for i, action in enumerate(legal) if action == chosen)
                examples.append({
                    "state": state_features.tolist(),
                    "actions": action_features.tolist(),
                    "families": families.tolist(),
                    "chosen": int(chosen_index),
                    "chosen_action": f"{chosen.action_type.name}:{chosen.value!r}",
                    "decision": len(examples),
                })

            game.execute(chosen, validate_action=True)
            action_count += 1

        if game.winning_color() is not None:
            status = "completed"
    except Exception as exc:  # keep faulty episodes out of the training labels
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    winner = game.winning_color()
    completed = status == "completed"
    won = completed and winner == focal_color
    if completed:
        outcome = 1.0 if won else -1.0
        for example in examples:
            example.update({
                "seed": int(seed),
                "seat": int(focal_seat),
                "teacher": teacher_name,
                "opponent": opponent_name,
                "outcome": outcome,
            })
    else:
        examples = []

    own_prefix = f"P{game.state.color_to_index[focal_color]}_"
    enemy_prefix = f"P{game.state.color_to_index[other_color]}_"
    return {
        "seed": int(seed),
        "seat": int(focal_seat),
        "teacher": teacher_name,
        "opponent": opponent_name,
        "status": status,
        "won": bool(won),
        "candidate_vp": int(game.state.player_state[own_prefix + "ACTUAL_VICTORY_POINTS"]),
        "opponent_vp": int(game.state.player_state[enemy_prefix + "ACTUAL_VICTORY_POINTS"]),
        "turns": int(game.state.num_turns),
        "actions": int(action_count),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "examples": examples,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    parser.add_argument("--teachers", default=DEFAULT_TEACHERS)
    parser.add_argument("--boards", type=int, default=16,
                        help="Paired board seeds per teacher; each seed is played from both seats")
    parser.add_argument("--seed", type=int, default=20272000)
    parser.add_argument("--max-actions", type=int, default=15000)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--max-game-seconds", type=float, default=45.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    teacher_names = tuple(part.strip() for part in args.teachers.split(",") if part.strip())
    from catanLabs.OneVOne.evaluate import OPPONENTS
    allowed = set(OPPONENTS) | {"onevone"}
    if not teacher_names or any(name not in allowed for name in teacher_names):
        parser.error("teachers must be selected from " + ",".join(sorted(allowed)))
    if len(set(teacher_names)) != len(teacher_names):
        parser.error("teachers must not contain duplicates")
    if args.boards < 1 or min(args.max_actions, args.max_turns) < 1 or args.max_game_seconds <= 0:
        parser.error("boards and game caps must be positive")

    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text())
    if config.get("rules", {}).get("players") != 2:
        parser.error("this collector requires the provisional two-player rules")
    output = args.output or ROOT / "data" / f"bootstrap_{args.seed}_{args.boards}boards.jsonl.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest_path.exists():
        parser.error(f"output already exists: {output} (choose another path to avoid overwriting data)")

    caps = {
        "max_actions": args.max_actions,
        "max_turns": args.max_turns,
        "max_game_seconds": args.max_game_seconds,
    }
    games = []
    total_examples = 0
    example_counts = defaultdict(int)
    started = time.perf_counter()
    game_total = len(teacher_names) * args.boards * 2
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=6) as stream:
        for teacher_index, teacher in enumerate(teacher_names):
            opponents = [name for name in teacher_names if name != teacher]
            if not opponents:
                opponents = ["alphabeta2"]
            for board_offset in range(args.boards):
                seed = args.seed + board_offset
                opponent = opponents[(board_offset + teacher_index) % len(opponents)]
                for seat in range(2):
                    result = collect_game(
                        seed, seat, teacher, opponent, config, caps,
                    )
                    examples = result.pop("examples")
                    for example in examples:
                        stream.write(json.dumps(example, separators=(",", ":")) + "\n")
                    total_examples += len(examples)
                    example_counts[teacher] += len(examples)
                    games.append(result)
                    if len(games) % 10 == 0 or len(games) == game_total:
                        print(
                            f"games {len(games)}/{game_total}; rows {total_examples}; "
                            f"elapsed {time.perf_counter() - started:.1f}s",
                            flush=True,
                        )

    by_teacher = {}
    for name in teacher_names:
        rows = [game for game in games if game["teacher"] == name]
        by_teacher[name] = {
            "games": len(rows),
            "completed": sum(game["status"] == "completed" for game in rows),
            "wins": sum(game["won"] for game in rows),
            "examples": example_counts[name],
            "errors": [game for game in rows if game["status"] == "error"],
            "capped": sum(game["status"] == "capped" for game in rows),
        }
    manifest = {
        "format": "onevone-visible-state-action-outcome-jsonl-v1",
        "features_version": FEATURES_VERSION,
        "families": list(FAMILY_NAMES),
        "action_feature_dim": ACTION_FEATURE_DIM,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": ENGINE,
        "rules": config["rules"],
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "teachers": list(teacher_names),
        "boards_per_teacher": args.boards,
        "first_seed": args.seed,
        "seat_protocol": "same seed, focal teacher in both engine seats",
        "caps": caps,
        "games_total": len(games),
        "completed_games": sum(game["status"] == "completed" for game in games),
        "errors": sum(game["status"] == "error" for game in games),
        "capped_games": sum(game["status"] == "capped" for game in games),
        "examples": total_examples,
        "by_teacher": by_teacher,
        "game_records": games,
        "dataset_sha256": sha256(output),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "label_notes": (
            "Each sample contains the focal teacher's visible state, all legal candidate "
            "actions, its selected action, and the completed game's +/-1 outcome. "
            "Capped/error games are omitted. These are mixed-teacher behavior-cloning "
            "targets plus noisy terminal value labels, not expert or RL targets."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "dataset": str(output),
        "manifest": str(manifest_path),
        "games": len(games),
        "completed": manifest["completed_games"],
        "examples": total_examples,
        "errors": manifest["errors"],
        "capped": manifest["capped_games"],
        "elapsed_seconds": manifest["elapsed_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable, *sys.argv],
                   {**os.environ, "PYTHONHASHSEED": "0"})
    main()
