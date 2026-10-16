"""Tune on the 80% training boards, then test one hybrid on 20% heldout boards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from catanLabs.OneVOne.agent import DEFAULT_CONFIG
from catanLabs.OneVOne.outcome_transformer import OneVOneOutcomeAgent, load_model
from catanLabs.FromScratch.StrategyOne.experiments.catanatron_80pct.run import run_game


MAIN_BASELINE = Path("/Users/anixd/Documents/Hyde/catanLabs/OneVOne/experiment_catalog/reports/all_1028/onevone.jsonl")


def run_series(seeds, beta, model, max_game_seconds, report_path, config):
    if report_path.exists():
        rows = [json.loads(line) for line in report_path.open()]
    else:
        rows = []
    done = {(row["seed"], row["seat"]) for row in rows}
    for seed in seeds:
        for seat in range(2):
            if (seed, seat) in done:
                continue
            created = []
            def factory(color, _seed):
                agent = OneVOneOutcomeAgent(color, model, beta, config)
                created.append(agent)
                return agent
            result = run_game(seed, seat, candidate_factory=factory,
                              max_seconds=max_game_seconds, max_actions=10000,
                              max_turns=500, trace=False)
            result.update(beta=beta, rerank_considered=created[0].considered,
                          rerank_changed=created[0].changed)
            rows.append(result)
            with report_path.open("a") as stream:
                stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            print(f"beta={beta} {len(rows)}/{len(seeds)*2} "
                  f"wins={sum(row['ours_won'] for row in rows)} "
                  f"changed={sum(row['rerank_changed'] for row in rows)}", flush=True)
    return rows


def summarize(rows):
    return {"games": len(rows), "wins": sum(row["ours_won"] for row in rows),
            "completed": sum(row["status"] == "completed" for row in rows),
            "capped": sum(row["status"] == "capped" for row in rows),
            "errors": [row.get("error") for row in rows if row["status"] == "error"],
            "rerank_considered": sum(row.get("rerank_considered", 0) for row in rows),
            "rerank_changed": sum(row.get("rerank_changed", 0) for row in rows),
            "mean_vp_margin": statistics.mean(row["ours_vp"]-row["catanatron_vp"]
                                          for row in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path(__file__).parent / "runs/outcome_1028")
    parser.add_argument("--screen-boards", type=int, default=12)
    parser.add_argument("--max-game-seconds", type=float, default=90)
    parser.add_argument("--max-test-boards", type=int, default=None,
                        help="Optional pilot cap; omit for the full 206-board test")
    args = parser.parse_args()
    torch.set_num_threads(4)
    split = json.loads((args.run / "split.json").read_text())
    model = load_model(args.run / "best.pt")
    config = json.loads(DEFAULT_CONFIG.read_text())
    train_seeds = split["train_seeds"][:args.screen_boards]
    test_seeds = split["test_seeds"]
    if args.max_test_boards is not None:
        test_seeds = test_seeds[:args.max_test_boards]
    started = time.perf_counter()
    reports = args.run / "games"
    reports.mkdir(parents=True, exist_ok=True)
    screens = {}
    for beta in (0.0, 0.25, 0.5, 1.0, 2.0):
        rows = run_series(train_seeds, beta, model, args.max_game_seconds,
                          reports / f"screen_beta_{beta:g}.jsonl", config)
        screens[str(beta)] = summarize(rows)
        print("screen", beta, screens[str(beta)], flush=True)
    # Select the best nonzero beta on training boards. Test data is untouched.
    selected = max((0.25, 0.5, 1.0, 2.0),
                   key=lambda b: (screens[str(b)]["wins"],
                                  screens[str(b)]["mean_vp_margin"], -b))
    rows = run_series(test_seeds, selected, model, args.max_game_seconds,
                      reports / f"holdout_beta_{selected:g}.jsonl", config)
    baseline = [json.loads(line) for line in MAIN_BASELINE.open()]
    baseline = [row for row in baseline if row["seed"] in set(test_seeds)]
    if len(baseline) != len(test_seeds)*2:
        raise AssertionError("Archived OneVOne baseline lacks heldout pairs")
    report = {"screen_boards": len(train_seeds), "test_boards": len(test_seeds),
              "selected_beta": selected, "screen": screens,
              "hybrid_test": summarize(rows), "onevone_baseline_test": summarize(baseline),
              "wall_seconds": round(time.perf_counter() - started, 2),
              "baseline_source": str(MAIN_BASELINE)}
    (args.run / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FINAL", report["hybrid_test"], report["onevone_baseline_test"], flush=True)


if __name__ == "__main__":
    main()
