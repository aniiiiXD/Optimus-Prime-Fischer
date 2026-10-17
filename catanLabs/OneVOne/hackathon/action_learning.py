"""Learn action preferences from paired terminal rollouts, then run frozen matches."""

import argparse
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor/kit"))
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np
from catan_kit import _patch
from catan_kit import match
from catan_kit.obs import masked_copy
from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

from catanLabs.OneVOne.action_planner import ACTION_DIM, ActionPlanner
from catanLabs.OneVOne.learning import MODEL_VERSION, normalize_scores
from resource_experiments import paired_difference, write_json


class AuditedSeat(match.Seat):
    """Record accepted responses after the official clock/validation path finishes."""

    def __init__(self, color, agent, clocked=True):
        super().__init__(color, agent, clocked=False if os.environ.get("ONEVONE_UNCLOCKED") == "1" else clocked)
        self._audit_focal = color == match.COLORS[0]

    def decide(self, game, playable_actions):
        directory = os.environ.get("ONEVONE_AUDIT_DIR")
        if not directory or len(playable_actions) == 1:
            return super().decide(game, playable_actions)
        pid, sequence = self.agent.proc.pid, self.agent.seq
        errors, random_moves = self.errors, self.random_moves
        action = super().decide(game, playable_actions)
        requested = self.agent.seq > sequence
        if not hasattr(self, "_audit_requests"):
            self._audit_requests = {}
        request = None
        if requested:
            request = self._audit_requests[pid] = self._audit_requests.get(pid, 0) + 1
        accepted = bool(requested and not self.over and self.errors == errors and self.random_moves == random_moves)
        row = {"pid": pid, "request": request, "accepted": accepted, "focal": self._audit_focal,
               "action": f"{action.action_type.name}:{action.value!r}",
               "arm_agentpath": str(self.agent.path), "arm": os.environ.get("ONEVONE_AUDIT_ARM"),
               "fallback": self.random_moves > random_moves, "over": self.over,
               "errors": self.errors - errors}
        with (Path(directory) / f"{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        return action


# Spawn workers import this module too; the official implementation remains the superclass.
match.Seat = AuditedSeat


class FrozenF:
    """F's one-ply value policy; its guesses and search cannot consume game dice."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.value = get_value_fn("base_fn", DEFAULT_WEIGHTS)

    def choose(self, game, actions):
        if len(actions) == 1:
            return actions[0]
        color = game.state.current_color()
        state = random.getstate()
        try:
            random.seed(self.rng.getrandbits(64))
            world = masked_copy(game, color, self.rng)
            scores = []
            for action in actions:
                after = world.copy()
                after.execute(action)
                scores.append(self.value(after, color))
            return actions[int(np.argmax(scores))]
        finally:
            random.setstate(state)


def rollout(game, action, perspective, seed):
    """Replace hidden information and future randomness identically for each alternative."""
    state = random.getstate()
    try:
        world = masked_copy(game, perspective, random.Random(seed))
        random.seed(seed ^ 0x937BD)
        policy = FrozenF(seed ^ 0x64F2)
        world.execute(action)
        steps = 0
        while world.winning_color() is None and world.state.num_turns < 1000 and steps < 6000:
            world.execute(policy.choose(world, world.playable_actions))
            steps += 1
        winner = world.winning_color()
        return {"outcome": None if winner is None else int(winner == perspective),
                "turns": world.state.num_turns, "steps": steps, "failure": None}
    except Exception as error:
        return {"outcome": None, "failure": f"{type(error).__name__}: {error}"}
    finally:
        random.setstate(state)


def alternatives(plan, count, rng):
    prior = normalize_scores(plan.heuristic)
    baseline = int(np.argmax(prior))
    chosen = [baseline]
    # Training samples across action types; inference continues to score every legal root.
    groups = {}
    for i, action in enumerate(plan.actions):
        groups.setdefault(action.action_type.name, []).append(i)
    keys = list(groups)
    rng.shuffle(keys)
    for key in keys:
        if len(chosen) >= count:
            break
        options = [i for i in groups[key] if i not in chosen]
        if options:
            chosen.append(int(rng.choice(options)))
    remaining = [i for i in range(len(plan.actions)) if i not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[:max(0, count - len(chosen))])
    return chosen, prior


def collect(output, boards=6, positions=2, choices=3, repeats=1, seed0=2030100000,
            weights=None, online=False, replay=None, validation_boards=None):
    """Generate practice trajectories and action comparisons, grouped by board seed."""
    from catanLabs.OneVOne.agent import DEFAULT_CONFIG, TRAINED_CONFIG
    from catanLabs.OneVOne.learning import LearningPlayer
    config = json.loads((TRAINED_CONFIG if TRAINED_CONFIG.is_file() else DEFAULT_CONFIG).read_text())
    rows, decisions, games, updates = [], [], [], []
    replay = list(replay or [])
    validation_boards = set(validation_boards or [])
    losses = 0
    rng = random.Random(173)
    for board in range(boards):
        seed = seed0 + board
        game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=seed)
        random.seed(seed ^ 0x27483)
        policy = FrozenF(seed ^ 0x8A5)
        focal = Color.RED if board % 2 == 0 else Color.BLUE
        player = LearningPlayer(focal, config)
        player.weights = None if weights is None else np.asarray(weights).copy()
        last_turn, count = -20, 0
        row_start = len(rows)
        for step in range(6000):
            if game.winning_color() is not None or game.state.num_turns >= 1000:
                break
            actions = game.playable_actions
            turn = game.state.num_turns
            color = game.state.current_color()
            prefix = f"P{game.state.color_to_index[color]}_"
            eligible = (color == focal and count < positions and 8 <= turn <= 100 and turn - last_turn >= 8
                        and game.state.current_prompt.name == "PLAY_TURN"
                        and game.state.player_state[prefix + "HAS_ROLLED"]
                        and len({a.action_type for a in actions}) > 1)
            if eligible:
                state = random.getstate()
                world = masked_copy(game, color, random.Random(seed + turn))
                plan = ActionPlanner(color, config).plan(world, actions)
                random.setstate(state)
                selected, prior = alternatives(plan, choices, rng)
                outcomes = {}
                record = {"board_seed": seed, "turn": turn, "seat": color.value,
                          "legal_actions": len(actions), "alternatives": []}
                for index in selected:
                    trials = [rollout(game, actions[index], color, seed * 31 + turn * 7 + k)
                              for k in range(repeats)]
                    # Failed or unfinished branches provide no training label.
                    completed = [t["outcome"] for t in trials if t["outcome"] is not None]
                    if len(completed) == repeats:
                        outcomes[index] = float(np.mean(completed))
                    record["alternatives"].append({"index": index, "action": str(actions[index]),
                                                     "prior": float(prior[index]), "trials": trials,
                                                     "rollout_seeds": [seed * 31 + turn * 7 + k for k in range(repeats)]})
                for i, j in combinations(selected, 2):
                    if i in outcomes and j in outcomes:
                        rows.append({"board_seed": seed, "turn": turn,
                                     "features_i": plan.features[i].tolist(),
                                     "features_j": plan.features[j].tolist(),
                                     "prior_i": float(prior[i]), "prior_j": float(prior[j]),
                                     "outcome_i": outcomes[i], "outcome_j": outcomes[j]})
                decisions.append(record)
                count += 1
                last_turn = turn
                write_json(output / "practice_choices.json", {"decisions": decisions, "pairs": rows})
                print(f"practice board {board + 1}/{boards}: position {count}/{positions}, {len(rows)} pairs", flush=True)
            if color == focal and len(actions) > 1:
                state = random.getstate()
                try:
                    world = masked_copy(game, focal, random.Random(seed * 61 + step))
                    action = player.decide(world, actions)
                finally:
                    random.setstate(state)
            else:
                action = policy.choose(game, actions)
            game.execute(action)
        winner = game.winning_color()
        won = None if winner is None else winner == focal
        games.append({"board_seed": seed, "focal": focal.value, "won": won,
                      "turns": game.state.num_turns, "positions": count})
        replay.extend(rows[row_start:])
        if online:
            losses, weights, update = update_after_four_losses(losses, won, replay, validation_boards, weights)
            if update is not None:
                updates.append(dict(update, after_game=board + 1, losses_since_update=4))
                print(f"online update after game {board + 1}: four actual losses", flush=True)
        write_json(output / "practice_games.json", {"games": games, "updates": updates})
    if online:
        write_json(output / "updated_model.json", {"version": MODEL_VERSION,
                   "weights": np.asarray(weights).tolist(), "updates": updates})
    return rows, decisions


def fit(rows, validation_boards, previous=None):
    """Fit residual logits with the heuristic prior inside the pairwise objective."""
    usable = [r for r in rows if r["outcome_i"] != r["outcome_j"]]
    train = [r for r in usable if r["board_seed"] not in validation_boards]
    valid = [r for r in usable if r["board_seed"] in validation_boards]
    start = np.zeros(ACTION_DIM) if previous is None else np.asarray(previous, dtype=float).copy()
    if start.shape != (ACTION_DIM,) or not np.isfinite(start).all():
        raise ValueError("Invalid initial action weights")

    def arrays(batch):
        x = np.asarray([np.asarray(r["features_i"]) - r["features_j"] for r in batch])
        p = np.asarray([r["prior_i"] - r["prior_j"] for r in batch])
        y = np.asarray([np.sign(r["outcome_i"] - r["outcome_j"]) for r in batch])
        return x, p, y

    def loss(batch, weights):
        if not batch:
            return None
        x, p, y = arrays(batch)
        return float(np.logaddexp(0, -y * (p + x @ weights)).mean())

    candidates = []
    if train:
        x, p, y = arrays(train)
        for regularization in (0.01, 0.1, 1.0):
            weights = start.copy()
            for epoch in range(400):
                margin = np.clip(y * (p + x @ weights), -50, 50)
                gradient = x.T @ (-y / (1 + np.exp(margin))) / len(train)
                gradient += regularization * weights
                weights -= 0.1 * gradient
            candidates.append((loss(valid or train, weights), regularization, weights))
        _, regularization, weights = min(candidates, key=lambda candidate: candidate[0])
    else:
        weights, regularization = start, None
    return weights, {"pairs_total": len(rows), "ties_excluded": len(rows) - len(usable),
                     "train_pairs": len(train), "validation_pairs": len(valid),
                     "validation_boards": sorted(validation_boards), "l2": regularization,
                     "training_log_loss": loss(train, weights),
                     "validation_log_loss_prior": loss(valid, np.zeros(ACTION_DIM)),
                     "validation_log_loss_model": loss(valid, weights),
                     "weight_l2": float(np.linalg.norm(weights)),
                     "warning": None if train and valid else "Insufficient decisive training or validation pairs; no strength claim."}


def update_after_four_losses(losses, won, replay, validation_boards, weights):
    """Only completed match losses trigger a replay update before the next game."""
    losses += won is False
    if losses < 4:
        return losses, weights, None
    weights, report = fit(replay, validation_boards, previous=weights)
    return 0, weights, report


def profile(output, name, model=None):
    path = output / f"{name}_agent.py"
    source = Path(os.path.relpath(HERE, output)).as_posix()
    path.write_text("from pathlib import Path\nimport sys\n"
                    f"sys.path.insert(0, str((Path(__file__).parent / {source!r}).resolve()))\n"
                    "from learning_agent import Agent as LearningAgent\n"
                    "class Agent(LearningAgent):\n    def __init__(self):\n"
                    f"        super().__init__(model={('str(Path(__file__).parent / ' + repr(model.name) + ')') if model else 'None'}, "
                    f"trace_path=str(Path(__file__).parent / 'traces' / {name!r}))\n")
    return path


def trace_summary(output, arm):
    counts = {"decisions": 0, "proposed_decisions": 0, "rejected_proposals": 0,
              "unmatched_proposals": 0, "fallback_moves": 0,
              "accepted_harness_moves": 0, "ml_changed": 0, "cross_type": 0, "winning_override": 0}
    audits = {}
    focal_path = (HERE / "agent.py" if arm == "original" else output / f"{arm}_agent.py").resolve()
    path_matches = {}
    for path in (output / "audits").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("arm") != arm or not row.get("focal", True):
                continue
            agentpath = row["arm_agentpath"]
            if agentpath not in path_matches:
                path_matches[agentpath] = Path(agentpath).resolve() == focal_path
            if not path_matches[agentpath]:
                continue
            counts["fallback_moves"] += bool(row.get("fallback"))
            counts["accepted_harness_moves"] += bool(row["accepted"])
            if row["request"] is not None:
                audits[(row["pid"], row["request"])] = row
    for path in (output / "traces" / arm).glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            counts["proposed_decisions"] += 1
            audit = audits.get((row.get("pid"), row.get("request")))
            if audit is None:
                counts["unmatched_proposals"] += 1
                continue
            if not audit["accepted"] or audit["action"] != row["executed_action"]:
                counts["rejected_proposals"] += 1
                continue
            counts["decisions"] += 1
            for key in ("ml_changed", "cross_type", "winning_override"):
                counts[key] += bool(row.get(key, False))
    counts["move_change_rate"] = counts["ml_changed"] / max(1, counts["decisions"])
    return counts


def main():
    _patch.ensure_hash_seed()
    _patch.apply()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--practice-boards", type=int, default=6)
    parser.add_argument("--positions", type=int, default=2)
    parser.add_argument("--alternatives", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--online-games", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2030200000)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--unclocked", action="store_true", help="Disable cumulative CPU/wall budgets for both seats; retain the 3-second decision timeout")
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or min(args.workers, args.positions, args.rollouts) < 1 or args.practice_boards < 3 or args.alternatives < 2 or args.online_games < 0:
        parser.error("Use even positive games, >=3 practice boards, >=2 alternatives, and positive counts")
    if args.train_only and args.eval_only:
        parser.error("Choose at most one of --train-only and --eval-only")
    if max(args.seed, 2030100000) <= min(args.seed + args.games // 2 - 1, 2030100000 + args.practice_boards - 1):
        parser.error("Evaluation boards must be separate from practice boards")
    if args.online_games and max(args.seed, 2030110000) <= min(args.seed + args.games // 2 - 1, 2030110000 + args.online_games - 1):
        parser.error("Evaluation boards must be separate from online practice boards")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clock = "unclocked" if args.unclocked else "official"
    protocol_path = output / "protocol.json"
    records_exist = any((output / f"{arm}.json").exists() for arm in ("original", "heuristic", "learned"))
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol.get("clock") != clock:
            parser.error("Existing output uses a different clock mode; choose a fresh output directory")
        if records_exist and (protocol.get("fresh_seed0") != args.seed or protocol.get("games_per_arm") != args.games):
            parser.error("Existing records use different boards or game counts; choose a fresh output directory")
    elif records_exist:
        parser.error("Existing records have no protocol; choose a fresh output directory")
    os.environ["ONEVONE_UNCLOCKED"] = "1" if args.unclocked else "0"
    (output / "traces").mkdir(exist_ok=True)
    model_path = output / "action_model.json"
    if not args.eval_only and (model_path.exists() or (output / "practice_choices.json").exists()):
        parser.error("Practice artifacts already exist; choose a new output or use --eval-only")
    from catanLabs.OneVOne.agent import DEFAULT_CONFIG, TRAINED_CONFIG
    config_path = TRAINED_CONFIG if TRAINED_CONFIG.is_file() else DEFAULT_CONFIG
    sources = [HERE.parent / "action_planner.py", HERE.parent / "learning.py",
               HERE.parent / "agent.py", HERE / "agent.py", HERE / "learning_agent.py",
               Path(__file__), config_path, HERE / "critic_experiment.py",
               HERE.parents[1] / "Competitive/fair_value_agent.py",
               HERE.parents[1] / "Competitive/tactical_vp_agent.py"]
    hashes = {str(path.relative_to(HERE.parents[2])): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    write_json(output / ("evaluation_sources.json" if args.eval_only else "training_sources.json"),
               {"sha256": hashes, "config": json.loads(config_path.read_text()),
                "python": sys.version, "platform": sys.platform})
    started = time.perf_counter()
    if not args.eval_only:
        rows, decisions = collect(output, args.practice_boards, args.positions, args.alternatives, args.rollouts)
        validation = {2030100000 + args.practice_boards - 1}
        weights, report = fit(rows, validation)
        trials = [trial for decision in decisions for alternative in decision["alternatives"] for trial in alternative["trials"]]
        report["rollouts"] = {"attempted": len(trials),
                              "completed": sum(trial["outcome"] is not None for trial in trials),
                              "failures": sum(trial["failure"] is not None for trial in trials)}
        write_json(model_path, {"version": MODEL_VERSION, "weights": weights.tolist(),
                                "action_dim": ACTION_DIM, "training": report})
        write_json(output / "training.json", report)
        print(json.dumps(report), flush=True)
        if args.online_games:
            online_dir = output / "online_practice"
            online_dir.mkdir(exist_ok=False)
            collect(online_dir, args.online_games, args.positions, args.alternatives, args.rollouts,
                    seed0=2030110000, weights=weights, online=True, replay=rows,
                    validation_boards=validation)
            write_json(model_path, json.loads((online_dir / "updated_model.json").read_text()))
    for path in sources:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(HERE.parents[2]))]:
            raise RuntimeError(f"Source changed during training: {path}; rerun with frozen code")
    if args.train_only:
        return
    if not model_path.exists():
        parser.error("No action_model.json; train first")
    import critic_experiment as evaluation
    evaluation.WORKERS = args.workers
    audit_dir = output / "audits"
    audit_dir.mkdir(exist_ok=True)
    os.environ["ONEVONE_AUDIT_DIR"] = str(audit_dir)
    arms = {"original": HERE / "agent.py", "heuristic": profile(output, "heuristic"),
            "learned": profile(output, "learned", model_path)}
    write_json(output / "protocol.json", {"games_per_arm": args.games, "fresh_seed0": args.seed,
               "practice_seed0": 2030100000, "practice_boards": args.practice_boards,
               "online_practice_seed0": 2030110000, "online_games": args.online_games,
               "opponent": "official supplied F", "clock": clock, "both_seats": True,
               "decision_timeout_s": match.CALL_TIMEOUT_S,
               "learning_during_evaluation": False, "automatic_promotion": False,
               "policy": "same all-legal-root planner for heuristic and learned arms",
               "python": sys.version, "platform": sys.platform})
    results = {}
    for name, path in arms.items():
        os.environ["ONEVONE_AUDIT_ARM"] = name
        tick = time.perf_counter()
        result, _ = evaluation.run(output, name, path, args.games, args.seed)
        results[name] = {"summary": result["summary"], "wall_s": time.perf_counter() - tick,
                         "trace": trace_summary(output, name)}
        if name == "original":
            original = result
        elif name == "heuristic":
            heuristic = result
            results[name]["versus_original"] = paired_difference(result, original)
        else:
            results[name]["versus_heuristic"] = paired_difference(result, heuristic)
            results[name]["versus_original"] = paired_difference(result, original)
        write_json(output / "results.json", {"arms": results, "wall_s": time.perf_counter() - started,
                   "promoted": False, "limitation": "Small screening evaluation; no strength confirmation."})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
