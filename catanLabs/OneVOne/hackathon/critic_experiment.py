"""Warm-start a tiny value critic, update after four losses, and evaluate frozen."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor/kit"))

import numpy as np
import psutil
from catan_kit._patch import ensure_hash_seed
from catan_kit import match
from catan_kit.evaluate import summarise
from catan_kit.obs import masked_copy
from catanatron.game import Game
from resource_experiments import paired_difference, write_json
from value_critic import VERSION, encode


class Probe:
    def __init__(self, collect):
        self.collect, self.rows, self.metrics = collect, [], {}
        self.counts = {}

    def before(self, game):
        self.colors = game.state.colors

    def step(self, game, action):
        if any(getattr(player, "errors", 0) for player in game.state.players):
            raise RuntimeError("Agent exception: stopping the local iteration before random fallbacks waste more games")
        turn = game.state.num_turns
        if turn in (20, 40) and turn not in self.metrics:
            self.metrics[turn] = {
                c.value: {"settlements": len(game.state.buildings_by_color[c]["SETTLEMENT"]), "cities": len(game.state.buildings_by_color[c]["CITY"]), "public_vp": game.state.player_state[f"P{game.state.color_to_index[c]}_VICTORY_POINTS"]}
                for c in self.colors
            }
        if self.collect and turn <= 50 and game.state.current_prompt.name == "PLAY_TURN":
            color = game.state.current_color()
            key = (color, turn)
            prefix = f"P{game.state.color_to_index[color]}_"
            # Retain post-roll and post-build positions without long trade sequences dominating.
            if game.state.player_state[prefix + "HAS_ROLLED"] and self.counts.get(key, 0) < 2:
                for perspective in self.colors:
                    self.rows.append((perspective, encode(game, perspective)))
                self.counts[key] = self.counts.get(key, 0) + 1

    def after(self, game):
        self.winner = game.winning_color()


PROBES = []
COLLECT = False
WORKERS = 4


class ObservedGame(Game):
    def play(self, accumulators=None, decide_fn=None):
        probe = Probe(COLLECT)
        result = super().play(accumulators=[probe], decide_fn=decide_fn)
        PROBES.append(probe)
        if len(PROBES) % 25 == 0:
            print(f"worker {os.getpid()}: {len(PROBES)} games completed", flush=True)
        return result


def agent_file(output, name, checkpoint=None, exploration=0.0, blend=0.25):
    path = output / f"{name}.py"
    path.write_text(
        "import sys\n" + f"sys.path.insert(0, {str(HERE)!r})\n"
        "from critic_agent import Agent as CriticAgent\n"
        "class Agent(CriticAgent):\n    def __init__(self):\n"
        f"        super().__init__(checkpoint={str(checkpoint) if checkpoint else None!r}, exploration={exploration!r}, blend={blend!r})\n"
    )
    return path


def shard(path, indices, seed, collect, cpu, opponent=None):
    global COLLECT
    COLLECT = collect
    if cpu is not None:
        os.environ["CATAN_CPU"] = str(cpu)
        match.sandbox.CPU = str(cpu)
    match.Game = ObservedGame
    PROBES.clear()
    records, _ = match.play_chunk(str(path), str(opponent or HERE / "vendor/kit/examples/f_agent.py"), [seed + i // 2 for i in indices], indices)
    rows, metrics = [], []
    for record, probe in zip(records, PROBES):
        focal = match.COLORS[0 if record["a_first"] else 1]
        metrics.append({"game": record["game"], "snapshots": {str(t): sides.get(focal.value) for t, sides in probe.metrics.items()}})
        if probe.winner is not None:
            rows.extend((record["board_seed"], row, int(probe.winner == color)) for color, row in probe.rows)
    return records, rows, metrics


def run(output, name, path, games, seed, collect=False, game_start=0, opponent=None):
    records, all_rows, metrics = [], [], []
    saved = output / f"{name}.json"
    if saved.exists() and not collect:
        previous = json.loads(saved.read_text())
        records, metrics = previous["records"], previous["early_building"]
    completed = {r["game"] for r in records}
    indices = [i for i in range(game_start, game_start + games) if i not in completed]
    cpus, physical = [], set()
    for cpu in sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((topology / "physical_package_id").read_text(), (topology / "core_id").read_text()) if topology.exists() else cpu
        if key not in physical:
            cpus.append(cpu); physical.add(key)
    if not cpus:
        # Use the OS scheduler on platforms without CPU affinity.
        cpus = [None] * (psutil.cpu_count(logical=False) or 1)
    count = min(WORKERS, len(cpus), max(1, len(indices) // 2))
    if len(indices) <= 2:
        batches = [shard(path, indices, seed, collect, None, opponent)] if indices else []
    else:
        # Keep the two seats of every board together and give each worker its own core.
        paired = [indices[i:i + 2] for i in range(0, len(indices), 2)]
        groups = [[game for pair in paired[i::count] for game in pair] for i in range(count)]
        with ProcessPoolExecutor(max_workers=count, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(shard, path, group, seed, collect, cpus[i], opponent) for i, group in enumerate(groups)]
            batches = []
            for future in as_completed(futures):
                batches.append(future.result())
                done = len(records) + sum(len(b[0]) for b in batches)
                print(f"{name}: {done}/{games} games completed", flush=True)
    for batch, rows, snapshots in batches:
        records.extend(batch); all_rows.extend(rows); metrics.extend(snapshots)
        s = summarise(records)
        if s["errors"] or s["timeouts"] or s["crashes"]:
            write_json(output / f"{name}.json", {"summary": s, "records": records})
            raise RuntimeError(f"{name}: agent errors; inspect records")
    records.sort(key=lambda r: r["game"])
    metrics.sort(key=lambda r: r["game"])
    s = summarise(records)
    print(f"{name}: {s['wins']}/{s['games']} wins, {s['games_over_budget']} overruns", flush=True)
    write_json(saved, {"summary": s, "records": records, "early_building": metrics})
    return {"summary": summarise(records), "records": records, "early_building": metrics}, all_rows


def save_model(model, mean, scale, path):
    np.savez(path, version=VERSION, mean=mean, scale=scale, w1=model.coefs_[0].astype(np.float32), b1=model.intercepts_[0].astype(np.float32), w2=model.coefs_[1].astype(np.float32), b2=model.intercepts_[1].astype(np.float32))


def train(output, rows):
    from sklearn.neural_network import MLPClassifier
    seeds = np.asarray([r[0] for r in rows])
    x, y = np.stack([r[1] for r in rows]), np.asarray([r[2] for r in rows])
    valid = seeds % 5 == 0
    assert valid.any() and (~valid).any(), "Need disjoint board groups for training and validation"
    mean, scale = x[~valid].mean(0), np.maximum(x[~valid].std(0), 0.1)
    normalized = (x - mean) / scale
    model = MLPClassifier(hidden_layer_sizes=(32,), batch_size=256, alpha=0.01, learning_rate_init=0.001, random_state=17)
    best_loss, best_weights = float("inf"), None
    for epoch in range(20):
        model.partial_fit(normalized[~valid], y[~valid], classes=[0, 1])
        predictions = np.clip(model.predict_proba(normalized[valid])[:, 1], 1e-6, 1 - 1e-6)
        loss = float(-np.mean(y[valid] * np.log(predictions) + (1 - y[valid]) * np.log(1 - predictions)))
        if loss < best_loss:
            best_loss = loss
            best_weights = ([a.copy() for a in model.coefs_], [a.copy() for a in model.intercepts_])
        if epoch % 5 == 4:
            print(f"critic epoch {epoch + 1}: held-out log loss {loss:.4f}", flush=True)
    model.coefs_, model.intercepts_ = best_weights
    # Reset Adam after restoring the best validation checkpoint.
    if hasattr(model, "_optimizer"):
        del model._optimizer
    save_model(model, mean, scale, output / "warm_critic.npz")
    np.savez_compressed(output / "warm_data.npz", x=x, y=y, seeds=seeds)
    report = {"rows": len(rows), "train_boards": len(set(seeds[~valid].tolist())), "validation_boards": len(set(seeds[valid].tolist())), "best_validation_log_loss": best_loss, "features": x.shape[1], "hidden_width": 32}
    write_json(output / "training.json", report)
    return model, mean, scale, rows, set(seeds[valid].tolist())


def online(output, model, mean, scale, replay, held_out, games):
    checkpoint = output / "online_critic.npz"
    save_model(model, mean, scale, checkpoint)
    path = agent_file(output, "online_agent", checkpoint, exploration=0.15)
    losses, updates, outcomes = 0, [], []
    for i in range(games):
        result, rows = run(output, f"online_game_{i:03d}", path, 1, 1140000, collect=True, game_start=i)
        # Seats alternate on new paired boards; earlier results alone enter replay.
        replay.extend(rows)
        won = result["records"][0]["winner"] == "a"
        outcomes.append(won)
        losses += not won
        if losses == 4:
            permitted = [r for r in replay if r[0] not in held_out]
            rng = np.random.default_rng(17 + i)
            chosen = rng.choice(len(permitted), min(12000, len(permitted)), replace=False)
            x = (np.stack([permitted[j][1] for j in chosen]) - mean) / scale
            y = np.asarray([permitted[j][2] for j in chosen])
            for _ in range(3):
                model.partial_fit(x, y, classes=[0, 1])
            save_model(model, mean, scale, checkpoint)
            updates.append({"after_game": i + 1, "losses_since_update": 4, "replay_rows": len(permitted)})
            print(f"online update {len(updates)} after game {i + 1}; four losses observed", flush=True)
            losses = 0
        write_json(output / "online.json", {"games": i + 1, "wins": sum(outcomes), "updates": updates})
    return checkpoint


def main():
    global WORKERS
    ensure_hash_seed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "reports/critic_fast_20261017")
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--warm-games", type=int, default=40)
    parser.add_argument("--online-games", type=int, default=16)
    parser.add_argument("--screen-games", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.games, args.warm_games, args.online_games, args.screen_games, args.workers) < 1 or args.games % 2 or args.warm_games % 2 or args.screen_games % 2:
        parser.error("Positive game counts required; eval and warm games must be even")
    WORKERS = args.workers
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    write_json(output / "protocol.json", {"eval_games": args.games, "eval_seed": 1100000, "warm_games": args.warm_games, "warm_seed": 1120000, "online_games": args.online_games, "online_seed": 1140000, "update_after_losses": 4, "final_evaluation_frozen": True, "clock": "official", "parallel_workers": WORKERS})
    baseline, _ = run(output, "baseline", HERE / "agent.py", args.games, 1100000)
    clock_agent = agent_file(output, "clock_agent")
    data_path = output / "collected_data.npz"
    if data_path.exists():
        with np.load(data_path, allow_pickle=False) as data:
            rows = list(zip(data["seeds"].tolist(), data["x"], data["y"].tolist()))
    else:
        _, rows = run(output, "warm_collection", agent_file(output, "exploring_teacher", exploration=0.2), args.warm_games, 1120000, collect=True)
        np.savez_compressed(data_path, seeds=np.asarray([r[0] for r in rows]), x=np.stack([r[1] for r in rows]), y=np.asarray([r[2] for r in rows]))
    model, mean, scale, replay, held_out = train(output, rows)
    checkpoint = online(output, model, mean, scale, replay, held_out, args.online_games)
    clock, _ = run(output, "clock_control", clock_agent, args.screen_games, 1170000)
    warm, _ = run(output, "warm_screen", agent_file(output, "warm_agent", output / "warm_critic.npz"), args.screen_games, 1170000)
    adaptive, _ = run(output, "online_screen", agent_file(output, "frozen_online", checkpoint), args.screen_games, 1170000)
    selected = "online" if adaptive["summary"]["wins"] > warm["summary"]["wins"] else "warm"
    checkpoint = checkpoint if selected == "online" else output / "warm_critic.npz"
    frozen = agent_file(output, "frozen_agent", checkpoint)
    candidate, _ = run(output, "frozen_critic", frozen, args.games, 1100000)
    report = {"baseline": baseline["summary"], "clock_control_separate_screen": clock["summary"], "warm_screen": warm["summary"], "online_screen": adaptive["summary"], "selected": selected, "critic": candidate["summary"], **paired_difference(candidate, baseline)}
    report["promotable"] = report["board_bootstrap_ci95"][0] > 0 and candidate["summary"]["games_over_budget"] < baseline["summary"]["games_over_budget"]
    write_json(output / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
