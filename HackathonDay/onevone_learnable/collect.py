"""Fresh official-rule trajectories and alternative-action outcome labels.

Training simulations run in process without competition clocks. Final games
must use evaluate.py and the unchanged official clocked harness.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
KIT = ROOT.parent/"official-kit"
sys.path.insert(0, str(KIT))
import numpy as np
from catan_kit import match
from catan_kit._patch import ensure_hash_seed
from catan_kit.obs import masked_copy
from catanatron.game import Game
from catanatron.models.player import RandomPlayer
from features import CANDIDATE_NAMES, SCHEMA, encode_board
from policy import LearnableOneVOne, load_config

spec = importlib.util.spec_from_file_location("training_f", KIT/"examples/f_agent.py")
F_MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F_MODULE)
from catan_kit.obs import Obs


class Context:
    def __init__(self, color, policy, seed, official=False):
        self.color, self.policy, self.official = color, policy, official
        self.mask_rng = random.Random(seed)
        self.sim_state = random.Random(seed ^ 0x13579).getstate()

    def decide(self, game, actions):
        if len(actions) == 1:
            return actions[0]
        # Hypothetical agent RNG must never consume the game's real dice RNG.
        actual_rng = random.getstate()
        random.setstate(self.sim_state)
        try:
            world = masked_copy(game, self.color, self.mask_rng)
            if self.official:
                index = self.policy.decide(Obs(world, self.color), actions)
            else:
                index = self.policy.decide(world, actions)
            self.sim_state = random.getstate()
            return actions[index]
        finally:
            random.setstate(actual_rng)


class CollectPolicy(LearnableOneVOne):
    def __init__(self, color, seed, epsilon, branch):
        config = load_config()
        config.update(scorer="heuristic", checkpoint=None)
        super().__init__(color, config)
        self.rng = random.Random(seed)
        self.epsilon, self.branch_enabled = epsilon, branch
        self.rows, self.branch = [], None

    def decide(self, game, actions, cpu_left=None):
        choice = super().decide(game, actions, cpu_left)
        record = self.last_decision
        candidates = record["candidates"]
        if not candidates:
            return choice
        if record["reason"] != "guaranteed_win" and self.rng.random() < self.epsilon:
            pool = sorted(candidates, key=lambda c: -c.rank)[:min(4, len(candidates))]
            choice = self.rng.choice(pool).index
        # Always log the actual executed action after exploration and guards.
        board = encode_board(game, self.color)
        self.rows.append((board, candidates[choice].features, game.state.num_turns))
        kinds = {c.action.action_type.name for c in candidates}
        if (self.branch_enabled and self.branch is None and 8 <= game.state.num_turns <= 50
                and len(kinds) >= 2 and record["reason"] != "guaranteed_win"
                and game.state.player_state[f"P{game.state.color_to_index[self.color]}_ACTUAL_VICTORY_POINTS"] < 7):
            options, selected_kinds = [], set()
            for c in sorted(candidates, key=lambda c: -c.rank):
                if c.action.action_type.name not in selected_kinds:
                    options.append((c.action, c.features))
                    selected_kinds.add(c.action.action_type.name)
                if len(options) == 3:
                    break
            # The retained world is already observation-masked; continuations
            # resample its hidden pile independently for every repeat.
            self.branch = (game.copy(), self.color, board, options)
        return choice


def play_to_end(game, contexts):
    steps = 0
    while game.winning_color() is None and game.state.num_turns < 1000:
        if steps >= 12000:
            raise RuntimeError("Training game action loop")
        color = game.state.current_color()
        game.execute(contexts[color].decide(game, game.playable_actions))
        steps += 1
    return game.winning_color()


def job(args):
    gi, seed0, branch_count, repeats = args
    seed = seed0+gi//2
    own, enemy = match.COLORS[gi%2], match.COLORS[1-gi%2]
    epsilon = (0, .06, .12, .06)[gi%4]
    policy = CollectPolicy(own, seed*19+gi, epsilon, gi < branch_count)
    game = Game([RandomPlayer(c) for c in match.COLORS], seed=seed)
    random.seed(match.dice_seed(seed, gi//2))
    opponent = F_MODULE.Agent()
    contexts = {own: Context(own, policy, seed*13+gi),
                enemy: Context(enemy, opponent, seed*17+gi, official=True)}
    winner = play_to_end(game, contexts)
    outcome = .5 if winner is None else float(winner == own)
    selected = policy.rows
    # Sample across the game rather than allowing long games to dominate.
    if len(selected) > 20:
        selected = policy.rng.sample(selected, 20)
    rows = [(b, c, outcome, 1., seed, gi, -1) for b, c, _ in selected]
    continuation_games = 0
    if policy.branch:
        base, color, board, options = policy.branch
        for action_index, (action, features) in enumerate(options):
            outcomes = []
            for repeat in range(repeats):
                continuation_seed = 410000003 + gi*1009 + repeat*103
                world = masked_copy(base, color, random.Random(continuation_seed))
                config = load_config()
                config.update(scorer="heuristic", checkpoint=None)
                fresh = LearnableOneVOne(color, config)
                contexts = {color: Context(color, fresh, continuation_seed ^ 12345),
                            enemy: Context(enemy, F_MODULE.Agent(), continuation_seed ^ 54321, official=True)}
                random.seed(continuation_seed ^ 0xABCDEF)
                world.execute(action)
                result = play_to_end(world, contexts)
                outcomes.append(.5 if result is None else float(result == color))
                continuation_games += 1
            rows.append((board, features, float(np.mean(outcomes)), float(repeats), seed, gi, gi))
    meta = {"game": gi, "board_seed": seed, "a_first": gi%2 == 0,
            "outcome": outcome, "epsilon": epsilon, "rows": len(rows),
            "continuation_games": continuation_games, "turns": game.state.num_turns}
    return rows, meta


def main():
    ensure_hash_seed()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--seed0", type=int, default=2026810000)
    p.add_argument("--branches", type=int, default=96)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.games < 4 or args.games%2 or min(args.workers, args.repeats) < 1:
        p.error("Use at least four games in complete board pairs and positive workers/repeats")
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = [*sorted(ROOT.glob("*.py")), ROOT/"config.json"]
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    start, all_rows, games = time.monotonic(), [], []
    jobs = [(i, args.seed0, args.branches, args.repeats) for i in range(args.games)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for rows, meta in pool.map(job, jobs):
            all_rows.extend(rows)
            games.append(meta)
            if len(games)%32 == 0:
                print(f"Training games {len(games)}/{args.games}", flush=True)
    board, candidate, outcome, weight, seed, game, group = zip(*all_rows)
    np.savez_compressed(args.output/"dataset.npz", board=np.stack(board), candidate=np.stack(candidate),
                        outcome=np.asarray(outcome, np.float32), weight=np.asarray(weight, np.float32),
                        seed=np.asarray(seed), game=np.asarray(game), group=np.asarray(group), schema=SCHEMA)
    manifest = {"schema": SCHEMA, "feature_names": CANDIDATE_NAMES,
                "games": games, "full_games": args.games,
                "continuation_games": sum(g["continuation_games"] for g in games),
                "rows": len(all_rows), "opponent": "official F", "clocks": "training unclocked",
                "started_seed": args.seed0, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "policy_source_hashes": source_hashes,
                "source_unchanged": source_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                "wall_seconds": time.monotonic()-start,
                "dataset_sha256": hashlib.sha256((args.output/"dataset.npz").read_bytes()).hexdigest()}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not manifest["source_unchanged"]:
        raise RuntimeError("Source changed during data generation; discard this dataset")
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("games", "feature_names")}, indent=2))


if __name__ == "__main__":
    main()
