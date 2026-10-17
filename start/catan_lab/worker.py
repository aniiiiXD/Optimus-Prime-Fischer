"""Repository-specific adapters for smoke evaluation, isolated per process."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
import time
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def stats(values: list[float]) -> dict:
    if not values:
        return {"samples": 0}
    x = np.asarray(values, dtype=np.float64)
    return {"samples": len(values), "median_ms": float(np.median(x)),
            "p95_ms": float(np.percentile(x, 95)), "p99_ms": float(np.percentile(x, 99)),
            "mean_ms": float(x.mean()), "max_ms": float(x.max())}


def wilson(wins: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.95996398454
    p = wins / total
    denom = 1 + z*z / total
    mid = (p + z*z / (2*total)) / denom
    half = z * ((p*(1-p)/total + z*z/(4*total*total))**0.5) / denom
    return [max(0.0, mid-half), min(1.0, mid+half)]


def finish(result: dict, games: list[dict], timings: list[float], started: float, cpu: float) -> dict:
    completed = [g for g in games if g["winner"] is not None]
    wins = sum(g["student_won"] for g in completed)
    seats = sorted({g["seat"] for g in games})
    wall = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.update(status="ok", games=games, completed_games=len(completed),
                  capped_games=len(games)-len(completed), wins=wins,
                  win_rate_completed=wins/len(completed) if completed else None,
                  win_rate_all_attempts=wins/len(games) if games else None,
                  win_rate_wilson95=wilson(wins, len(completed)),
                  by_seat={str(s): {"attempted": sum(g["seat"] == s for g in games),
                            "completed": sum(g["seat"] == s for g in completed),
                            "wins": sum(g["seat"] == s and g["student_won"] for g in completed)}
                           for s in seats},
                  decision_latency=stats(timings), timed_decision_ms=timings,
                  benchmark_wall_s=wall, process_cpu_s=time.process_time()-cpu,
                  average_busy_cores=(time.process_time()-cpu)/wall,
                  peak_process_rss_mib=peak/(1024**2 if sys.platform == "darwin" else 1024),
                  torch_version=torch.__version__, numpy_version=np.__version__)
    return result


def checkpoint(name: str) -> tuple[Path, str]:
    path = ROOT / "artifacts/checkpoints" / name
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def benjamin(args, deadline):
    sys.path.insert(0, str(ROOT / "vendor/benjamin/src"))
    sys.path.insert(0, str(ROOT / "artifacts/native"))
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.policy.network import CatanPolicy
    from catan_rl.policy.board_geometry import build_geometry
    from catan_rl.policy.obs_tensor import obs_to_torch, masks_to_torch
    from catan_rl.policy.obs_schema import ActionType
    policy = CatanPolicy().eval()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    env = CatanEnv(opponent_type="heuristic", max_turns=100, ruleset="R0")
    obs, _ = env.reset(seed=args.seed)
    fixed_obs = obs_to_torch(obs, torch.device("cpu"), add_batch=True)
    fixed_masks = masks_to_torch(env.get_action_masks(), torch.device("cpu"), add_batch=True)
    for _ in range(10):
        policy.sample(fixed_obs, fixed_masks)
    model_times = []
    for _ in range(100):
        t = time.perf_counter_ns()
        policy.sample(fixed_obs, fixed_masks)
        model_times.append((time.perf_counter_ns()-t)/1e6)
    timings, games = [], []
    started, cpu = time.perf_counter(), time.process_time()
    validated = 0
    phase_contract_mismatches = []
    # This is an UNTRAINED architecture smoke test, not a strength benchmark.
    for i in range(min(args.games, 2)):
        obs, _ = env.reset(seed=args.seed+i//2, options={"agent_seat": i % 2})
        done = truncated = False
        for _ in range(128):
            if time.perf_counter() >= deadline:
                break
            t = time.perf_counter_ns()
            masks = masks_to_torch(env.get_action_masks(), torch.device("cpu"), add_batch=True)
            sampled = policy.sample(obs_to_torch(obs, torch.device("cpu"), add_batch=True), masks)
            tensor_action = sampled["action"]
            action = tensor_action[0].numpy()
            timings.append((time.perf_counter_ns()-t)/1e6)
            heads = policy.action_heads
            kind, res1 = tensor_action[:, 0], tensor_action[:, 4]
            selected_masks = [masks["type"], heads._corner_mask(kind, masks), masks["edge"],
                              masks["tile"], heads._resource1_mask(kind, masks),
                              heads._resource2_mask(kind, res1, masks)]
            for head, mask in enumerate(selected_masks):
                if bool(heads.head_relevance[kind, head].item()):
                    legal = bool(mask[0, action[head]])
                    if not legal and head == 3 and int(kind.item()) == ActionType.PLAY_KNIGHT:
                        # In upstream CatanEnv._apply_main_action PLAY_KNIGHT
                        # ignores tile_idx and schedules a separate MOVE_ROBBER.
                        # Its shared training schema nevertheless marks tile as
                        # relevant here, under an all-False tile mask. Keep the
                        # mismatch visible; never silently call it schema-legal.
                        phase_contract_mismatches.append({"action": action.tolist(),
                                                          "tile_mask_legal_count": int(mask.sum())})
                    else:
                        assert legal, f"Illegal executed head {head}"
            obs, _, done, truncated, _ = env.step(action)
            validated += 1
            if done or truncated:
                break
        games.append({"seat": i % 2, "seed": args.seed+i//2, "winner": None,
                      "student_won": False, "smoke_only": True, "terminated": bool(done),
                      "truncated": bool(truncated)})
    env.close()
    return finish({"agent": "benjamin", "trained": False,
                   "notes": "No public checkpoint in pinned Git tree; untrained legal-action/latency smoke only.",
                   "parameters": policy.num_parameters(), "validated_actions": validated,
                   "validation_scope": "fields consumed by the environment in this phase",
                   "schema_contract_mismatches": phase_contract_mismatches,
                   "fixed_state_model_sample_latency": stats(model_times),
                   "rules": {"players": 2, "target_vp": 15, "epoch": "R0", "p2p_trade": False},
                   "latency_scope": "observation/mask conversion + full six-head policy sample; excludes environment step"},
                  games, timings, started, cpu)


def eli(args, deadline):
    sys.path.insert(0, str(ROOT / "artifacts/native"))
    sys.path.insert(0, str(ROOT / "vendor/eli/training"))
    import catan_py
    from ppo import PolicyValueNet
    path, sha = checkpoint("eli_best.pt")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert (ck["codec_version"], ck["obs_version"], ck["obs_dim"], ck["num_actions"]) == (
        catan_py.CODEC_VERSION, catan_py.OBS_VERSION, catan_py.OBS_DIM, catan_py.NUM_ACTIONS)
    model = PolicyValueNet(catan_py.OBS_DIM, catan_py.NUM_ACTIONS, ck["config"]["hidden"]).eval()
    model.load_state_dict(ck["model_state"], strict=True)
    results = []
    # Matched training visibility plus a separate honest-observation transfer test.
    for visibility in ("perfect", "realistic"):
        games, timings, model_times = [], [], []
        started, cpu = time.perf_counter(), time.process_time()
        validated = 0
        for i in range(args.games):
            if time.perf_counter() >= deadline:
                break
            seat = i % 4
            seats = ["heuristic"] * 4
            seats[seat] = "policy"
            seed = args.seed + i//4
            env = catan_py.VecEnv(1, victory_target=7, visibility=visibility, seed=seed,
                                  seats=seats, max_turns=500)
            obs, mask, current = env.observe()
            if i == 0:
                for _ in range(10):
                    model(torch.as_tensor(obs), torch.as_tensor(mask))
                for _ in range(100):
                    t = time.perf_counter_ns()
                    model(torch.as_tensor(obs), torch.as_tensor(mask))
                    model_times.append((time.perf_counter_ns()-t)/1e6)
            winner, cap, decisions = None, True, 0
            for _ in range(10000):
                if time.perf_counter() >= deadline:
                    break
                assert int(current[0]) == seat
                t = time.perf_counter_ns()
                logits, _ = model(torch.as_tensor(obs), torch.as_tensor(mask))
                action = logits.argmax(1).numpy().astype(np.uint32)
                timings.append((time.perf_counter_ns()-t)/1e6)
                assert bool(mask[0, int(action[0])]), "Illegal model action"
                obs, mask, current, _, _, _ = env.step(action)
                validated += 1
                decisions += 1
                episodes = env.take_episode_stats()
                if episodes:
                    _, win, _, cap = episodes[0]
                    winner = int(win) if win >= 0 else None
                    break
            games.append({"seat": seat, "seed": seed, "winner": winner,
                          "student_won": winner == seat, "capped": bool(cap), "decisions": decisions})
        results.append(finish({"agent": "eli", "trained": True, "checkpoint_sha256": sha,
                        "checkpoint_training_config": ck["config"],
                        "checkpoint_training_steps": ck["global_step"],
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "validated_actions": validated, "opponents": "three heuristic-v1 bots",
                        "rules": {"players": 4, "target_vp": 7, "visibility": visibility,
                                  "p2p_trade": "bounded offer menu"},
                        "fixed_state_model_forward_latency": stats(model_times),
                        "latency_scope": "tensor views + masked MLP + argmax; excludes environment and scripted opponents"},
                       games, timings, started, cpu))
    return {"agent": "eli", "status": "ok", "variants": results,
            "completed_games": sum(r["completed_games"] for r in results)}


def settlers(args, deadline):
    sys.path.insert(0, str(ROOT / "vendor/settlers"))
    # The upstream rules module imports its GUI unconditionally. Headless-only
    # shim prevents scipy/Tk/display requirements without changing game logic.
    gui = types.ModuleType("ui.display")
    class NoDisplay:
        def __init__(self, *a, **kw):
            raise RuntimeError("GUI cannot be used by headless benchmark")
    gui.Display = NoDisplay
    sys.modules["ui.display"] = gui
    # Upstream predates NumPy 1.24; np.int was an alias of the builtin int.
    np.int = int
    from env.wrapper import EnvWrapper
    from RL.models.build_agent_model import build_agent_model
    path, sha = checkpoint("settlers_3825.pt")
    model = build_agent_model(device="cpu").eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    uniform = copy.deepcopy(model).eval()
    for p in uniform.parameters():
        p.zero_()
    games, timings = [], []
    started, cpu = time.perf_counter(), time.process_time()
    validated = 0
    for i in range(args.games):
        if time.perf_counter() >= deadline:
            break
        seed, seat = args.seed + i//4, i % 4
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = EnvWrapper(max_proposed_trades_per_turn=0, validate_actions=True, max_victory_points=10)
        obs = env.reset()
        seat_order = list(env.game.player_order)
        student = seat_order[seat]
        hidden = (torch.zeros(1, model.lstm_size), torch.zeros(1, model.lstm_size))
        winner = None
        for _ in range(12000):
            if time.perf_counter() >= deadline:
                break
            actor = obs["player_id"]
            policy = model if actor == student else uniform
            t = time.perf_counter_ns()
            inputs = policy.obs_to_torch(obs)
            masks = policy.act_masks_to_torch(env.get_action_masks())
            _, action, _, _ = policy.act(inputs, hidden, torch.ones(1, 1), masks, deterministic=False)
            action = policy.torch_act_to_np(action)
            if actor == student:
                timings.append((time.perf_counter_ns()-t)/1e6)
            obs, _, done, _ = env.step(action)
            validated += 1
            if done:
                winner = seat_order.index(env.winner.id)
                break
        games.append({"seed": seed, "seat": seat, "winner": winner, "student_won": winner == seat})
    return finish({"agent": "settlers", "trained": True, "checkpoint_sha256": sha,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "validated_actions": validated, "opponents": "three zero-logit, uniformly sampled conditional-head policies",
                   "rules": {"players": 4, "target_vp": 10, "p2p_trade": False},
                   "compatibility": ["headless ui.display shim", "np.int alias restored within worker"],
                   "latency_scope": "observation/mask conversion + attention + all conditional heads + action decoding"},
                  games, timings, started, cpu)


def catanatron(args, deadline):
    repo = "catanatron_1v1" if args.agent == "catanatron_1v1" else "catanatron"
    sys.path.insert(0, str(ROOT / "vendor" / repo / "catanatron"))
    from catanatron import Game, Color, RandomPlayer
    from catanatron.players.weighted_random import WeightedRandomPlayer
    from catanatron.players.value import ValueFunctionPlayer
    from catanatron.players.minimax import AlphaBetaPlayer
    games, timings = [], []
    started, cpu = time.perf_counter(), time.process_time()
    validated = 0
    for i in range(args.games):
        if time.perf_counter() >= deadline:
            break
        seat = i % (2 if repo.endswith("1v1") else 4)
        seed = args.seed+i//(2 if repo.endswith("1v1") else 4)
        colors = list(Color)[:2 if repo.endswith("1v1") else 4]
        players = [RandomPlayer(c) for c in colors]
        if repo.endswith("1v1"):
            from catanatron.colonist_1v1 import create_colonist_1v1_game
            game = create_colonist_1v1_game(players, seed=seed)
        else:
            game = Game(players, seed=seed, vps_to_win=10)
        # The engine shuffles seats. Choose the focal player AFTER seating.
        student = game.state.colors[seat]
        opponent_cls = {"random": RandomPlayer, "weighted": WeightedRandomPlayer,
                        "value": ValueFunctionPlayer}[args.classical_opponents]
        policies = {c: opponent_cls(c) for c in colors}
        policy_kind = "value" if repo.endswith("1v1") else args.classical_policy
        if policy_kind == "value":
            policies[student] = ValueFunctionPlayer(student)
        elif policy_kind == "alphabeta":
            policies[student] = AlphaBetaPlayer(student, AlphaBetaPlayer.Params(depth=1))
        else:
            policies[student] = WeightedRandomPlayer(student)
        for _ in range(15000):
            if time.perf_counter() >= deadline or game.state.num_turns >= 500:
                break
            actor = game.state.current_color()
            t = time.perf_counter_ns()
            action = policies[actor].decide(game, game.playable_actions)
            if actor == student:
                timings.append((time.perf_counter_ns()-t)/1e6)
            assert action in game.playable_actions
            game.execute(action, validate_action=True)
            validated += 1
            if game.winning_color() is not None:
                break
        win = game.winning_color()
        winner = game.state.colors.index(win) if win is not None else None
        games.append({"seed": seed, "seat": seat, "winner": winner, "student_won": win == student})
    return finish({"agent": repo, "trained": False, "validated_actions": validated,
                   "policy": type(policies[student]).__name__,
                   "search_depth": 1 if policy_kind == "alphabeta" else None,
                   "opponents": args.classical_opponents + " legal bots",
                   "rules": {"players": len(colors), "target_vp": 15 if repo.endswith("1v1") else 10},
                   "latency_scope": "classical policy decide; excludes game execute"}, games, timings, started, cpu)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", required=True)
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--classical-policy", choices=["weighted", "value", "alphabeta"], default="weighted")
    p.add_argument("--classical-opponents", choices=["random", "weighted", "value"], default="random")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        fn = {"benjamin": benjamin, "eli": eli, "settlers": settlers,
              "catanatron": catanatron, "catanatron_1v1": catanatron}[args.agent]
        result = fn(args, time.perf_counter()+args.seconds)
    result["seed"] = args.seed
    result["torch_threads"] = args.threads
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
