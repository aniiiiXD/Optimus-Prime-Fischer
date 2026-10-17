"""Collect paired planner rollouts and fit a gated action-advantage ensemble."""

from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor/kit"))
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np
from catan_kit import _patch
from catan_kit.obs import Obs, masked_copy
from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer

from catanLabs.OneVOne.action_planner import ACTION_DIM
from catanLabs.OneVOne.learning import LearningPlayer, heuristic_fingerprint, predict_advantages
from action_learning import FrozenF, alternatives
from resource_experiments import write_json


MODEL_VERSION = "onevone-advantage-mlp-v1"
ENSEMBLE_SIZE = 8
HIDDEN_DIM = 32
L2_VALUES = (0.1, 1.0, 10.0)
GATE_MARGINS = (0.0, 0.05, 0.1, 0.2)
GATE_Z = 1.64
MIN_GATE_POSITIONS = 8
MIN_GATE_BOARDS = 3
OPPONENTS = ("F", "Nano", "OneVOne")


class _Opponent:
    """One persistent opponent policy, given only its masked official observation."""

    def __init__(self, name, color, seed):
        self.name, self.color = name, color
        self.rng = random.Random(seed)
        if name == "F":
            self.agent = FrozenF(seed)
        elif name == "Nano":
            from my_team.nano import Agent
            self.agent = Agent()
        elif name == "OneVOne":
            from catanLabs.OneVOne.hackathon.agent import Agent
            self.agent = Agent()
        else:
            raise ValueError(f"Unknown practice opponent: {name}")

    def choose(self, game, actions, seed):
        if len(actions) == 1:
            return actions[0]
        if self.name == "F":
            return self.agent.choose(game, actions)
        state = random.getstate()
        try:
            random.seed(seed)
            view = masked_copy(game, self.color, random.Random(seed))
            index = self.agent.decide(Obs(view, self.color), actions)
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(actions):
                raise ValueError(f"{self.name} returned invalid legal-action index {index!r}")
            return actions[int(index)]
        finally:
            random.setstate(state)


def _sample_nonterminal(game, perspective, seed):
    rng = random.Random(seed)
    for _ in range(256):
        world = masked_copy(game, perspective, rng)
        if world.winning_color() is None:
            return world
    raise ValueError("Could not sample hidden cards consistent with the live position")


def _rollout(game, action, perspective, config, rollout_seed, opponent_name="F"):
    """Fork one hidden-information world and preserve the engine's global RNG."""
    rng_state = random.getstate()
    try:
        random.seed(rollout_seed ^ 0x937BD)
        world = _sample_nonterminal(game, perspective, rollout_seed)
        world.execute(action)
        focal = LearningPlayer(perspective, config)
        enemy = next(color for color in world.state.colors if color != perspective)
        opponent = _Opponent(opponent_name, enemy, rollout_seed ^ 0x64F2)
        steps = 0
        while world.winning_color() is None and world.state.num_turns < 1000 and steps < 6000:
            color = world.state.current_color()
            actions = world.playable_actions
            if color == perspective:
                if len(actions) == 1:
                    selected = actions[0]
                else:
                    state = random.getstate()
                    try:
                        decision_world = masked_copy(world, perspective, random.Random(rollout_seed + steps + 1))
                        selected = focal.decide(decision_world, actions)
                    finally:
                        random.setstate(state)
            else:
                selected = opponent.choose(world, actions, rollout_seed + steps + 1)
            world.execute(selected)
            steps += 1
        winner = world.winning_color()
        return {"outcome": None if winner is None else int(winner == perspective),
                "turns": world.state.num_turns, "steps": steps, "failure": None,
                "rollout_seed": rollout_seed}
    except Exception as error:
        return {"outcome": None, "turns": None, "steps": None,
                "failure": f"{type(error).__name__}: {error}", "rollout_seed": rollout_seed}
    finally:
        random.setstate(rng_state)


def collect_one(board_seed, config, positions=4, choices=3, repeats=3, candidate_model=None,
                opponent_name="F"):
    """Run one actual heuristic practice game and sample paired counterfactuals."""
    _patch.apply()
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=int(board_seed))
    random.seed(int(board_seed) ^ 0x27483)
    focal = Color.RED if int(board_seed) % 2 == 0 else Color.BLUE
    enemy = Color.BLUE if focal == Color.RED else Color.RED
    policy = _Opponent(opponent_name, enemy, int(board_seed) ^ 0x8A5)
    player = LearningPlayer(focal, config)
    rng = random.Random(int(board_seed) ^ 0xAC71)
    position_rows, pair_rows = [], []
    last_turn, collected = -20, 0
    attempted = failures = 0

    for step in range(6000):
        if game.winning_color() is not None or game.state.num_turns >= 1000:
            break
        actions = list(game.playable_actions)
        color = game.state.current_color()
        turn = game.state.num_turns
        prefix = f"P{game.state.color_to_index[color]}_"
        eligible = (color == focal and collected < positions and 8 <= turn <= 100
                    and turn - last_turn >= 8 and game.state.current_prompt.name == "PLAY_TURN"
                    and game.state.player_state[prefix + "HAS_ROLLED"]
                    and len({action.action_type for action in actions}) > 1)

        world = None
        plan = None
        if color == focal and len(actions) > 1:
            state = random.getstate()
            try:
                world = masked_copy(game, focal, random.Random(int(board_seed) * 61 + step))
                actual_action = player.decide(world, actions)
                plan = player.last_plan
                actual_index = actions.index(actual_action)
                if eligible and not np.any(plan.winning):
                    selected, prior = alternatives(plan, choices, rng)
                    candidate_index = None
                    if candidate_model is not None:
                        candidate_index, _ = _candidate(
                            {"heuristic_index": actual_index,
                             "all_features": plan.features.tolist()},
                            candidate_model,
                            float(candidate_model.get("gate", {}).get("z", GATE_Z)),
                            0.0,
                        )
                        if candidate_index is not None and candidate_index not in selected:
                            selected = [actual_index, candidate_index]
                            selected.extend(index for index in alternatives(plan, choices, rng)[0]
                                            if index not in selected)
                            selected = selected[:max(1, min(int(choices), len(actions)))]
                    record = {"board_seed": int(board_seed), "turn": turn, "seat": focal.value,
                              "opponent": opponent_name,
                              "legal_actions": len(actions), "heuristic_index": actual_index,
                              "heuristic_features": plan.features[actual_index].tolist(),
                              "heuristic_prior": float(prior[actual_index]),
                              "all_features": plan.features.tolist(),
                              "candidate_index": candidate_index, "alternatives": []}
                    outcomes = {}
                    for index in selected:
                        seeds = [int(board_seed) * 31 + turn * 7 + k for k in range(repeats)]
                        trials = [_rollout(game, actions[index], focal, config, seed, opponent_name) for seed in seeds]
                        attempted += len(trials)
                        failures += sum(trial["outcome"] is None for trial in trials)
                        completed = [trial["outcome"] for trial in trials]
                        if len(completed) == repeats and all(value is not None for value in completed):
                            outcomes[index] = float(np.mean(completed))
                        record["alternatives"].append({
                            "index": index, "action": f"{actions[index].action_type.name}:{actions[index].value!r}",
                            "features": plan.features[index].tolist(), "prior": float(prior[index]),
                            "trials": trials,
                        })
                    for i, j in combinations(selected, 2):
                        if i in outcomes and j in outcomes:
                            pair_rows.append({"board_seed": int(board_seed), "turn": turn,
                                              "features_i": plan.features[i].tolist(),
                                              "features_j": plan.features[j].tolist(),
                                              "prior_i": float(prior[i]), "prior_j": float(prior[j]),
                                              "outcome_i": outcomes[i], "outcome_j": outcomes[j]})
                    position_rows.append(record)
                    collected += 1
                    last_turn = turn
                action = actual_action
            finally:
                random.setstate(state)
        elif color == focal:
            action = actions[0]
        else:
            action = policy.choose(game, actions, int(board_seed) * 61 + step)
        game.execute(action)

    winner = game.winning_color()
    practice_game = {"board_seed": int(board_seed), "focal": focal.value,
                     "opponent": opponent_name,
                     "won": None if winner is None else winner == focal,
                     "turns": game.state.num_turns, "positions": collected}
    return {"positions": position_rows, "pairs": pair_rows, "practice_game": practice_game,
            "rollouts_attempted": attempted, "rollout_errors": failures}


def _collect_worker(args):
    return collect_one(*args)


def _checkpoint_metadata(seed, config, positions, choices, repeats, candidate_model, opponent):
    model_hash = None if candidate_model is None else hashlib.sha256(
        json.dumps(candidate_model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    opponent_sources = {"F": [HERE / "action_learning.py"],
                        "Nano": [HERE.parents[2] / "my_team/nano.py"],
                        "OneVOne": [HERE / "agent.py", HERE.parent / "agent.py",
                                    HERE.parent / "learning.py", HERE.parent / "action_planner.py",
                                    HERE.parent / "configs/default.json"]}[opponent]
    opponent_hash = hashlib.sha256()
    for path in opponent_sources:
        opponent_hash.update(path.read_bytes())
    baseline_engine = _config_data(config).get("planner", "action_planner")
    baseline_sources = ([HERE.parent / "nano_planner.py", HERE.parent / "learning.py",
                         HERE.parents[2] / "my_team/nano.py"] if baseline_engine == "nano" else
                        [HERE.parent / "action_planner.py", HERE.parent / "learning.py"])
    baseline_hash = hashlib.sha256()
    baseline_hash.update(Path(__file__).read_bytes())
    for path in baseline_sources:
        baseline_hash.update(path.read_bytes())
    return {"board_seed": int(seed), "config_sha256": _fingerprint(config),
            "candidate_model_sha256": model_hash, "positions_per_board": positions,
            "choices": choices, "repeats": repeats, "opponent": opponent,
            "opponent_sha256": opponent_hash.hexdigest(), "baseline_engine": baseline_engine,
            "baseline_sha256": baseline_hash.hexdigest()}


def _save_board(output, args, result):
    seed, config, positions, choices, repeats, candidate_model, opponent = args
    path = output / f"board_{seed}.json"
    temporary = path.with_suffix(".json.tmp")
    write_json(temporary, {"metadata": _checkpoint_metadata(
        seed, config, positions, choices, repeats, candidate_model, opponent), "result": result})
    os.replace(temporary, path)


def collect(output, config, board_seeds, workers=4, positions=4, choices=3, repeats=3,
            candidate_model=None):
    """Collect independent board games in processes and persist JSON-only artifacts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in board_seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Practice board seeds must be unique")
    jobs = [(seed, config, positions, choices, repeats, candidate_model, OPPONENTS[i % len(OPPONENTS)])
            for i, seed in enumerate(seeds)]
    results_by_seed = {}
    errors = []
    pending = []
    for args in jobs:
        checkpoint = output / f"board_{args[0]}.json"
        if not checkpoint.exists():
            pending.append(args)
            continue
        saved = json.loads(checkpoint.read_text())
        expected = _checkpoint_metadata(*args)
        if saved.get("metadata") != expected:
            raise ValueError(f"Checkpoint settings mismatch for board {args[0]}; choose a fresh output directory")
        results_by_seed[args[0]] = saved["result"]
        print(f"resumed board {args[0]}; completed {len(results_by_seed)}/{len(seeds)}", flush=True)
    if workers <= 1:
        for args in pending:
            try:
                result = _collect_worker(args)
                _save_board(output, args, result)
                results_by_seed[args[0]] = result
            except Exception as error:
                errors.append({"board_seed": args[0], "error": f"{type(error).__name__}: {error}"})
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(_collect_worker, args): args for args in pending}
            for future in as_completed(futures):
                args = futures[future]
                try:
                    result = future.result()
                    _save_board(output, args, result)
                    results_by_seed[args[0]] = result
                except Exception as error:
                    errors.append({"board_seed": args[0],
                                   "error": f"{type(error).__name__}: {error}"})
                print(f"counterfactual boards {len(results_by_seed) + len(errors)}/{len(seeds)}; "
                      f"positions {sum(len(result['positions']) for result in results_by_seed.values())}; "
                      f"rollouts {sum(result['rollouts_attempted'] for result in results_by_seed.values())}", flush=True)
    results = [results_by_seed[seed] for seed in seeds if seed in results_by_seed]
    positions_out = [row for result in results for row in result["positions"]]
    pairs_out = [row for result in results for row in result["pairs"]]
    games = [result["practice_game"] for result in results]
    opponent_counts = {name: sum(args[-1] == name for args in jobs) for name in OPPONENTS}
    baseline_engine = _config_data(config).get("planner", "action_planner")
    report = {"practice_games": len(games), "opponent_counts": opponent_counts, "board_errors": errors,
              "baseline_engine": baseline_engine,
              "positions": len(positions_out), "pairs": len(pairs_out),
              "rollouts_attempted": sum(result["rollouts_attempted"] for result in results),
              "rollouts_completed": sum(result["rollouts_attempted"] - result["rollout_errors"] for result in results),
              "rollout_errors": sum(result["rollout_errors"] for result in results)}
    write_json(output / "practice_choices.json", {"positions": positions_out, "pairs": pairs_out,
                                                    "practice_games": games, "collection": report})
    print(f"practice games {len(games)}/{len(seeds)}; completed rollouts "
          f"{report['rollouts_completed']}/{report['rollouts_attempted']}; errors "
          f"{report['rollout_errors'] + len(errors)}", flush=True)
    return {"positions": positions_out, "pairs": pairs_out, "practice_games": games,
            "collection": report}


def _fit_mlp(x, y, scale, alpha, seed):
    from sklearn.neural_network import MLPRegressor

    model = MLPRegressor(hidden_layer_sizes=(HIDDEN_DIM,), activation="tanh", solver="adam",
                         alpha=alpha, max_iter=300, random_state=seed)
    inputs = np.concatenate((x, -x)) / scale
    targets = np.concatenate((y, -y))
    return model.fit(inputs, targets)


def _export_mlp(model):
    return {"w1": model.coefs_[0].tolist(), "b1": model.intercepts_[0].tolist(),
            "w2": model.coefs_[1][:, 0].tolist(), "b2": float(model.intercepts_[1][0])}


def _predict_sklearn(model, x, scale):
    positive = model.predict(x / scale)
    negative = model.predict(-x / scale)
    return 0.5 * (positive - negative)


def _fingerprint(config):
    return heuristic_fingerprint(_config_data(config))


def _config_data(config):
    if isinstance(config, os.PathLike):
        return json.loads(Path(config).read_text())
    elif isinstance(config, str):
        try:
            path = Path(config)
            return json.loads(path.read_text()) if path.is_file() else json.loads(config)
        except OSError:
            return json.loads(config)
    return config


def _board_split(rows, train_boards):
    boards = sorted({int(row["board_seed"]) for row in rows})
    if len(boards) < 2:
        return set(boards), set()
    if train_boards is not None:
        train = set(map(int, train_boards)) & set(boards)
        if len(train) == len(boards):
            train.remove(boards[-1])
        return train, set(boards) - train
    rng = random.Random(7341)
    rng.shuffle(boards)
    n_train = max(1, min(len(boards) - 1, int(round(len(boards) * 0.8))))
    return set(boards[:n_train]), set(boards[n_train:])


def _arrays(rows):
    x = np.asarray([np.asarray(row["features_i"], dtype=float) - row["features_j"] for row in rows])
    y = np.asarray([float(row["outcome_i"]) - float(row["outcome_j"]) for row in rows])
    if x.shape != (len(rows), ACTION_DIM) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid counterfactual pair rows")
    return x, y


def _bootstrap_lower(values, rng, samples=4000):
    """Board bootstrap including zero contribution from boards without overrides."""
    if not values:
        return None, None
    boards = sorted(values)
    board_scores = [float(np.mean(values[board])) if values[board] else 0.0 for board in boards]
    if len(boards) == 1:
        return float(board_scores[0]), float(board_scores[0])
    draws = np.asarray([rng.choices(board_scores, k=len(board_scores)) for _ in range(samples)])
    return float(np.mean(board_scores)), float(np.quantile(draws.mean(axis=1), 0.05))


def fit_advantage(rows, train_boards, calibration_boards, config, positions=None):
    """Fit board-bootstrap tanh MLP members and gate on held-out paired outcomes."""
    if isinstance(rows, dict):
        positions = list(rows.get("positions", [])) if positions is None else list(positions)
        rows = list(rows.get("pairs", []))
    else:
        positions = list(positions or [])
        rows = list(rows)
    calibration_boards = set(map(int, calibration_boards))
    if train_boards is None:
        train_boards, _ = _board_split(rows, None)
    train_boards = set(map(int, train_boards))
    train = [row for row in rows if int(row["board_seed"]) in train_boards]
    if train_boards & calibration_boards:
        raise ValueError("Training and calibration boards must be disjoint")
    if not train:
        raise ValueError("No pair rows belong to training boards")
    x, y = _arrays(train)

    # Select L2 with a board split drawn only from training boards.
    unique = sorted({int(row["board_seed"]) for row in train})
    if len(unique) >= 2:
        rng = random.Random(8237)
        shuffled = unique[:]
        rng.shuffle(shuffled)
        internal = set(shuffled[max(1, int(round(len(shuffled) * 0.8))):]) or {shuffled[-1]}
        fit_mask = np.asarray([int(row["board_seed"]) not in internal for row in train])
        val_mask = ~fit_mask
        if not fit_mask.any() or not val_mask.any():
            fit_mask[:] = True
            val_mask[:] = True
    else:
        fit_mask = val_mask = np.ones(len(train), dtype=bool)
    losses = {}
    for l2 in L2_VALUES:
        scale = np.maximum(np.std(x[fit_mask], axis=0), 0.1)
        estimator = _fit_mlp(x[fit_mask], y[fit_mask], scale, l2, 177 + int(l2 * 10))
        prediction = _predict_sklearn(estimator, x[val_mask], scale)
        losses[str(l2)] = float(np.mean((prediction - y[val_mask]) ** 2))
    best_l2 = min(L2_VALUES, key=lambda value: losses[str(value)])

    feature_scale = np.maximum(np.std(x, axis=0), 0.1)

    rng = random.Random(19683)
    by_board = {}
    for row in train:
        by_board.setdefault(int(row["board_seed"]), []).append(row)
    boards = sorted(by_board)
    members = []
    for member_index in range(ENSEMBLE_SIZE):
        sampled = rng.choices(boards, k=len(boards))
        sample_rows = [row for board in sampled for row in by_board[board]]
        sx, sy = _arrays(sample_rows)
        fitted = _fit_mlp(sx, sy, feature_scale, best_l2, 19683 + member_index)
        members.append(_export_mlp(fitted))
    model = {"version": MODEL_VERSION, "feature_scale": feature_scale.tolist(),
             "members": members, "heuristic_sha256": _fingerprint(config),
             "gate": {"enabled": False, "z": GATE_Z, "margin": 0.0}}

    calibration = [position for position in positions
                   if int(position["board_seed"]) in calibration_boards]
    calibration_seed_set = {int(position["board_seed"]) for position in calibration}
    gate_stats = {"positions": 0, "boards": len(calibration_seed_set), "overrides": 0,
                  "fallback": 0,
                  "mean_advantage": None, "board_bootstrap_lower95": None,
                  "harmful": 0, "beneficial": 0, "tied": 0, "reason": "insufficient calibration evidence"}
    # Rows alone cannot reconstruct actual trial distributions; callers may attach positions.
    if calibration:
        accepted = None
        threshold_stats = []
        for margin in GATE_MARGINS:
            deltas_by_board = {seed: [] for seed in calibration_boards}
            override_boards = set()
            harmful = beneficial = tied = overrides = 0
            for position in calibration:
                seed = int(position["board_seed"])
                deltas_by_board.setdefault(seed, []).append(0.0)
                candidate, lower = _candidate(position, model, GATE_Z)
                if candidate is None or lower <= margin:
                    continue
                baseline = int(position["heuristic_index"])
                delta = _paired_position_difference(position, candidate, baseline)
                if delta is None:
                    continue
                deltas_by_board[seed][-1] = delta
                override_boards.add(seed)
                overrides += 1
                harmful += delta < 0
                beneficial += delta > 0
                tied += delta == 0
            mean, lower95 = _bootstrap_lower(deltas_by_board, random.Random(822 + int(margin * 1000)))
            board_count = len(override_boards)
            stats = {"margin": margin, "overrides": overrides,
                     "fallback": len(calibration) - overrides, "override_boards": board_count,
                     "mean_advantage": mean, "board_bootstrap_lower95": lower95,
                     "harmful": harmful, "beneficial": beneficial, "tied": tied}
            threshold_stats.append(stats)
            if overrides >= MIN_GATE_POSITIONS and board_count >= MIN_GATE_BOARDS and lower95 is not None and lower95 > 0:
                accepted = (margin, {"positions": len(calibration), "boards": len(calibration_seed_set),
                                     "overrides": overrides, "fallback": len(calibration) - overrides,
                                     "override_boards": board_count,
                                     "mean_advantage": mean, "board_bootstrap_lower95": lower95,
                                     "harmful": harmful, "beneficial": beneficial, "tied": tied,
                                     "thresholds": threshold_stats, "reason": None})
                break
        if accepted:
            margin, gate_stats = accepted
            model["gate"] = {"enabled": True, "z": GATE_Z, "margin": margin}
        else:
            gate_stats.update({"positions": len(calibration), "boards": len(calibration_seed_set),
                               "fallback": len(calibration), "thresholds": threshold_stats})

    baseline_engine = _config_data(config).get("planner", "action_planner")
    report = {"baseline_engine": baseline_engine,
              "pairs_total": len(rows), "training_pairs": len(train),
              "training_boards": sorted(train_boards), "calibration_boards": sorted(calibration_boards),
              "calibration_positions": len(calibration), "l2_cv_mse": losses,
              "selected_l2": best_l2, "ensemble_size": len(members),
              "gate_calibration": gate_stats,
              "confidence_note": "Ensemble spread and gate are empirical heuristics, not rigorous confidence bounds."}
    model["baseline_engine"] = baseline_engine
    model["training"] = report
    return model, report


def _candidate(position, model, z, margin=0.0):
    baseline = int(position["heuristic_index"])
    matrix = np.asarray(position.get("all_features"), dtype=float)
    if (matrix.ndim != 2 or matrix.shape[1] != ACTION_DIM or baseline < 0
            or baseline >= len(matrix) or not np.isfinite(matrix).all()):
        return None, float("-inf")
    advantages = predict_advantages(model, matrix, baseline)
    lowers = advantages.mean(axis=1) - z * advantages.std(axis=1) - margin
    candidate = int(np.argmax(lowers))
    if candidate == baseline or lowers[candidate] <= 0:
        return None, float("-inf")
    return candidate, float(lowers[candidate])


def _paired_position_difference(position, candidate, baseline):
    by_index = {int(row["index"]): row for row in position["alternatives"]}
    if candidate not in by_index or baseline not in by_index:
        return None
    a = by_index[candidate]["trials"]
    b = by_index[baseline]["trials"]
    if len(a) < 2 or len(a) != len(b):
        return None
    paired = []
    for x, y in zip(a, b):
        if (x.get("outcome") is None or y.get("outcome") is None
                or x.get("rollout_seed") != y.get("rollout_seed")):
            return None
        paired.append((x["outcome"], y["outcome"]))
    return float(np.mean([x - y for x, y in paired]))
