# HexMachina strategy test

This experiment compares two interpretations of the [HexMachina paper](https://arxiv.org/html/2506.04651v2) under the workspace's 1v1 Catanatron rules:

1. **Published source:** The author's unedited `FooPlayer` from [their repository](https://github.com/nbelle1/strategy-game-agents/tree/b1f2f1c6d43afd8d5ce5ca71542b78d37173b2a1/supplementary/Evolutions/GPT%20Evolution%202), loaded by `exact_agent.py`. `published_adapter.py` maps the author's adapter surface to the pinned engine. The only player identity adjustment is assigning the actual seat color after construction because the published constructor hardcodes `BLUE`.
2. **Mechanism port:** `agent.py` implements phase priorities, action prefiltering, robber disruption, a blended base value, and short greedy rollouts against the pinned engine state. It is an *approximation*, not the published player. It normalizes the Catanatron base score before blending because that score uses a `3e14` VP coefficient. It evaluates at most 12 candidate actions and at most three chance branches per candidate. It does not reproduce the author's LLM evolution process.

## Reproduce

From the workspace root, with the pinned engine and `start/.venv` installed:

```sh
start/.venv/bin/python catanLabs/FromScratch/StrategyOne/experiments/hexmachina_web/fetch_published.py
start/.venv/bin/python catanLabs/OneVOne/evaluate.py --agent-file catanLabs/FromScratch/StrategyOne/experiments/hexmachina_web/exact_agent.py --config catanLabs/OneVOne/configs/default.json --opponents value,catanatron,alphabeta2,onevone --boards 16 --seed 20268100 --max-game-seconds 60 --report catanLabs/FromScratch/StrategyOne/experiments/hexmachina_web/reproduction.json
```

The fetch script pins commit `b1f2f1c6d43afd8d5ce5ca71542b78d37173b2a1` and verifies SHA-256 `b01c69386a7a362365085e4b6866a003c93068e6911d79a958c999ce72c7c692`. The source cache is ignored by Git. Run with `--agent-file .../agent.py` for the mechanism port.

## Results

Each opponent played 16 board seeds (`20268100` through `20268115`) from **both seats**, 32 games per row. Rules: two players, 10 VP, official spiral numbers, discard limit 7, friendly robber off. The shared `install_v5_protocol()` isolates copied RNG and forces hypothetical roll/card outcomes. All listed games completed without caps or errors.

| Candidate | Opponent | Wins / 32 | Median candidate decision |
| --- | --- | ---: | ---: |
| Published FooPlayer | Value Base | 13 | 2.36 ms |
| Published FooPlayer | Web Catanatron (pruned AlphaBeta depth 2) | 17 | 2.59 ms |
| Published FooPlayer | Unpruned AlphaBeta depth 2 | 12 | 3.65 ms |
| Published FooPlayer | Workspace OneVOne incumbent | 14 | 3.98 ms |
| Mechanism port | Value Base | 16 | 1.13 ms |
| Mechanism port | Web Catanatron | 16 | 1.14 ms |
| Mechanism port | Unpruned AlphaBeta depth 2 | 13 | 1.78 ms |

Raw per-game outcomes and latency distributions are in `exact_web_benchmark_32_each.json`, `exact_vs_alphabeta2_32.json`, `web_benchmark_32_each.json`, and `approx_vs_alphabeta2_32.json`. With only 32 games per matchup, none of these differences is a reliable ranking.

## Interpretation and limitations

- The author's `FooPlayer` asks `state.players` objects for `victory_points`, `resources`, `settlements`, and `cities`, but their [bundled Player class](https://github.com/nbelle1/strategy-game-agents/blob/b1f2f1c6d43afd8d5ce5ca71542b78d37173b2a1/catanatron/catanatron_core/catanatron/models/player.py) has none of these fields. Their game state stores values separately in `player_state`. The pinned engine has the same shape. After initial placement, the published `_heuristic_value()` returned `0.0` and `get_game_phase()` returned `EARLY`, while the Catanatron `base_fn()` was nonzero. Thus the paper's description of its active phase and resource heuristic is stronger than what this released artifact executes in this engine. This is a source-code issue, not caused by our adapter shim.
- The author's adapter uses `catanatron_experimental` helpers; this test uses the pinned core equivalents. Their base value weights match the pinned core weights. Tree helpers differ in action schema for discard and robber. The paper's own engine version, precise evaluation seeds, and complete protocol may differ; these numbers do not establish a failed reproduction of its reported 54.1% from 1,000 games.
- Both the published code's opponent-resource inspection and Catanatron's `execute_spectrum()` can access hidden hands or deck composition through the full engine state. In our pinned engine, the direct inspection path is inactive because player objects lack resource fields; the shared chance expansion still uses hidden state. These benchmarks are full-state simulator tests, not a fair observation-only tournament guarantee.
- The mechanism port's direct road-count term uses a compatibility fallback of zero on this pinned board class; road action phase priority and simulated successor scoring still run. Treat the mechanism-port figures as a bounded sketch, not a validated implementation of every paper heuristic.
