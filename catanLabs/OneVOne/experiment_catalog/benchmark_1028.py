"""Resumable, paired-seat benchmark of 1v1 bot profiles on all dataset boards.

Every profile plays the pinned Catanatron web bot on the same 1,028 starting
seeds from both seats. The recorded teacher games are never replayed as moves.
Run with the main workspace's start/.venv/bin/python. The isolated experiment
worktree supplies the more recent StrategyOne ablations and planner.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


MAIN = Path(__file__).resolve().parents[3]
CATALOG = Path(__file__).resolve().parent
STRATEGY = MAIN / "catanLabs/FromScratch/StrategyOne"
WORKTREE = MAIN.parent / "Hyde-catan-experiments"
EXPERIMENT = WORKTREE / "catanLabs/FromScratch/StrategyOne"
META = STRATEGY / "data/catanatron_diverse.meta.json"
DEFAULT_OUTPUT = CATALOG / "reports/all_1028"

OWN_PROFILES = (
    "onevone", "onevone_mlp16", "onevone_mlp64", "strategy_raw",
    "strategy_ppo", "strategy_calibrated", "settlement4", "settlement7",
    "settlement12", "settlement_pips", "buydev", "state_control",
    "history", "same_turn", "invest_dynamic", "invest_static", "hybrid",
)
EXTERNAL_PROFILES = ("hex_published", "hex_approx", "alphabeta2_unpruned")
PROFILES = OWN_PROFILES + EXTERNAL_PROFILES


def load_benchmark():
    file = STRATEGY / "experiments/catanatron_80pct/run.py"
    spec = importlib.util.spec_from_file_location("catalog_catanatron_benchmark", file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def board_rows():
    data = json.loads(META.read_text())
    rows = [(int(game["game_id"]), int(game["seed"])) for game in data["games"]]
    if len(rows) != 1028 or len({seed for _, seed in rows}) != len(rows):
        raise RuntimeError("Expected exactly 1,028 unique dataset board seeds")
    return rows


def load_factory(profile: str):
    import torch

    torch.set_num_threads(1)
    sys.path[:0] = [str(MAIN), str(EXPERIMENT), str(STRATEGY)]
    from catanLabs.OneVOne.agent import create_agent as create_onevone

    if profile == "onevone":
        config = MAIN / "catanLabs/OneVOne/configs/default.json"
        return lambda color, seed: create_onevone(color, str(config), seed)

    if profile.startswith("onevone_mlp"):
        from catanLabs.OneVOne.mlp_policy import MLPPolicyAgent, load_policy

        run = ("mlp-hierarchical-bootstrap-20261016" if profile.endswith("16")
               else "mlp-hierarchical-64board-20261016")
        model = load_policy(MAIN / "catanLabs/OneVOne/runs" / run / "policy.pt")
        return lambda color, seed: MLPPolicyAgent(color, model)

    if profile in ("hex_published", "hex_approx"):
        name = ("exact_agent" if profile == "hex_published" else "agent")
        module = importlib.import_module(
            "catanLabs.FromScratch.StrategyOne.experiments.hexmachina_web." + name
        )
        return lambda color, seed: module.create_agent(color, seed=seed)

    if profile == "alphabeta2_unpruned":
        from catanatron.players.minimax import AlphaBetaPlayer

        return lambda color, seed: AlphaBetaPlayer(
            color, AlphaBetaPlayer.Params(depth=2, prunning=False)
        )

    evaluation = importlib.import_module("evaluate_win_rate")
    if profile == "strategy_raw":
        checkpoint_path = STRATEGY / "runs/legal_ranker_v1/model.pt"
    elif profile == "strategy_ppo":
        checkpoint_path = STRATEGY / "runs/ppo_actor_critic_preliminary/latest.pt"
    else:
        checkpoint_path = STRATEGY / "runs/calibrated_ranker_v1/model.pt"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = evaluation.LegalRanker(checkpoint["state_size"], checkpoint["action_size"])
    adapter = None
    if "base_model" in checkpoint:
        model.load_state_dict(checkpoint["base_model"])
        adapter = evaluation.ActionCalibrationAdapter(checkpoint["state_size"])
        adapter.load_state_dict(checkpoint["adapter"])
        adapter.eval()
    else:
        model.load_state_dict(checkpoint.get("model", checkpoint.get("policy")))
    model.eval()

    head = None
    heads = {
        "settlement4": STRATEGY / "runs/settlement_target_v1/epoch_4.pt",
        "settlement7": EXPERIMENT / "runs/settlement_target_long_v1/epoch_7.pt",
        "settlement12": EXPERIMENT / "runs/settlement_target_long_v1/epoch_12.pt",
    }
    if profile not in ("strategy_raw", "strategy_ppo", "strategy_calibrated"):
        head_path = heads.get(profile, heads["settlement4"])
        saved_head = torch.load(head_path, map_location="cpu", weights_only=True)
        head = evaluation.SettlementTargetHead(saved_head["input_size"])
        head.load_state_dict(saved_head["head"])
        head.eval()

    with (STRATEGY / "data/catanatron_diverse.jsonl").open() as handle:
        first = json.loads(next(handle))
    edge_index = evaluation.stable_edge_index(first["observation"])
    device = torch.device("cpu")

    if profile.startswith("invest_") or profile == "hybrid":
        planner = importlib.import_module("invest_convert")
        config_name = ("invest_convert_static.json" if profile == "invest_static"
                       else "invest_convert_dynamic.json")
        coefficients = planner.Coefficients(**json.loads(
            (EXPERIMENT / "configs" / config_name).read_text()
        ))

        def create_planner(color, seed):
            return planner.InvestConvertPlayer(
                color, model, edge_index, device, adapter, head,
                coefficients=coefficients,
            )

        if profile != "hybrid":
            return create_planner

        onevone_config = MAIN / "catanLabs/OneVOne/configs/default.json"

        class Hybrid:
            def __init__(self, color, seed):
                self.turn = create_planner(color, seed)
                self.other = create_onevone(color, str(onevone_config), seed)

            def decide(self, game, playable_actions):
                if game.state.current_prompt.name == "PLAY_TURN":
                    prefix = f"P{game.state.color_to_index[self.turn.color]}_"
                    if game.state.player_state[prefix + "HAS_ROLLED"]:
                        return self.turn.decide(game, playable_actions)
                return self.other.decide(game, playable_actions)

        return lambda color, seed: Hybrid(color, seed)

    residual = None
    if profile in ("state_control", "history"):
        history_path = EXPERIMENT / "runs/history_experiment" / (
            "state_control.pt" if profile == "state_control" else "history_transformer.pt"
        )
        saved = torch.load(history_path, map_location="cpu", weights_only=True)
        residual = evaluation.HistoryResidual(
            checkpoint["state_size"], checkpoint["action_size"], saved["use_history"]
        )
        residual.load_state_dict(saved["residual"])
        residual.eval()

    options = {
        "settlement_heuristic": "pips" if profile == "settlement_pips" else "none",
        "buy_dev_end_hand_threshold": 8 if profile == "buydev" else None,
        "history_residual": residual,
        "same_turn_search": profile == "same_turn",
        "same_turn_depth": 4,
    }

    return lambda color, seed: evaluation.MLPPlayer(
        color, model, edge_index, device, adapter, head, **options
    )


def run_profile(profile: str, output: Path, limit: int | None):
    benchmark = load_benchmark()
    benchmark.install_v5_protocol()
    factory = load_factory(profile)
    boards = board_rows()[:limit]
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{profile}.jsonl"
    existing = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                key = (row["seed"], row["seat"])
                if key in existing:
                    raise RuntimeError(f"Duplicate saved result: {profile} {key}")
                existing[key] = row
    allowed = {(seed, seat) for _, seed in boards for seat in (0, 1)}
    if not set(existing).issubset(allowed):
        raise RuntimeError(f"Saved {profile} results contain non-schedule games")
    with path.open("a", buffering=1) as stream:
        for game_id, seed in boards:
            for seat in (0, 1):
                if (seed, seat) in existing:
                    continue
                row = benchmark.run_game(
                    seed, seat, candidate_factory=factory, max_seconds=60.0,
                    max_actions=10000, max_turns=500, trace=False,
                )
                row["game_id"] = game_id
                row["profile"] = profile
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                existing[(seed, seat)] = row
                if len(existing) % 100 == 0:
                    wins = sum(g["ours_won"] for g in existing.values())
                    print(f"{profile}: {len(existing)}/{len(allowed)}, wins {wins}", flush=True)
    print(f"{profile}: complete {len(existing)}", flush=True)


def summarize(output: Path, profiles: tuple[str, ...], limit: int | None):
    boards = board_rows()[:limit]
    expected = {(seed, seat) for _, seed in boards for seat in (0, 1)}
    rows = []
    for profile in profiles:
        path = output / f"{profile}.jsonl"
        games = [json.loads(line) for line in path.read_text().splitlines() if line]
        keys = {(g["seed"], g["seat"]) for g in games}
        if len(keys) != len(games) or keys != expected:
            raise RuntimeError(f"Incomplete/duplicated {profile}: {len(keys)}/{len(expected)}")
        wins = sum(g["ours_won"] for g in games)
        caps = sum(g["status"] == "capped" for g in games)
        errors = sum(g["status"] == "error" for g in games)
        rows.append({"profile": profile, "wins": wins, "games": len(games),
                     "win_rate": wins / len(games), "caps": caps, "errors": errors,
                     "seat0_wins": sum(g["ours_won"] for g in games if g["seat"] == 0),
                     "seat1_wins": sum(g["ours_won"] for g in games if g["seat"] == 1)})
    summary = {
        "protocol": "Catanatron v5, two players, 10 VP, official spiral, discard 7",
        "opponent": "pinned Catanatron web bot, pruned AlphaBeta depth 2",
        "dataset_boards": len(boards), "seats_per_board": 2,
        "warning": "These are dataset board seeds, not an independent held-out test",
        "results": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PROFILE | WINS/GAMES | RATE | CAPS | ERRORS", flush=True)
    for row in sorted(rows, key=lambda row: -row["win_rate"]):
        print(f"{row['profile']} | {row['wins']}/{row['games']} | "
              f"{row['win_rate']:.1%} | {row['caps']} | {row['errors']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, help="Run one resumable profile")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES,
                        default=list(PROFILES), help="Profiles for an orchestrated run")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--limit-boards", type=int, help="Pilot: first N board seeds")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit_boards is not None and not 1 <= args.limit_boards <= 1028:
        parser.error("--limit-boards must be in 1..1028")
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output.resolve()
    if args.profile:
        run_profile(args.profile, output, args.limit_boards)
        return

    output.mkdir(parents=True, exist_ok=True)
    selected = tuple(args.profiles)

    def launch(profile):
        log = output / f"{profile}.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--profile", profile,
                   "--output", str(output)]
        if args.limit_boards is not None:
            command += ["--limit-boards", str(args.limit_boards)]
        with log.open("a") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env={**os.environ, "PYTHONHASHSEED": "0",
                                            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        return profile, completed.returncode

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(launch, profile) for profile in selected]
        for future in as_completed(futures):
            profile, code = future.result()
            print(f"{profile}: exit {code}, elapsed {time.monotonic()-started:.0f}s", flush=True)
            if code:
                raise RuntimeError(f"{profile} failed; see {output / (profile + '.log')}")
    summarize(output, selected, args.limit_boards)


if __name__ == "__main__":
    main()
