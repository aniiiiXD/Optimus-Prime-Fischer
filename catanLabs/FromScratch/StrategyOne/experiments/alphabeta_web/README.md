# Catanatron AlphaBeta reproduction

Two pinned upstream variants were tested: `AlphaBetaPlayer(depth=2, value_fn="base", prunning=False)` (the documented default) and `catanatron.web.players.Catanatron` (the web bot, same depth/value with `prunning=True`). No weights were tuned on these games.

## Protocol

- 16 board seeds, `20268100` through `20268115`, each played from both seats: **32 games per opponent and variant**.
- Pinned Catanatron engine, two players, 10 victory points, official spiral numbers, discard limit 7, friendly robber off. Domestic trade actions remain engine enabled.
- `install_v5_protocol()` isolates cloned search RNG and sorts legal actions. All moves were validated as legal. Caps: 120 seconds, 10,000 actions, 500 turns per game.
- `F_BASE` and `F_CONTENDER` are the pinned Catanatron `ValueFunctionPlayer` presets supplied by `StrategyOne/generate_dataset.py`.
- `OneVOne` means the available `catanLabs/OneVOne/agent.py` using `configs/default.json`; `configs/trained.json` was absent. This matchup used the existing `catanLabs/OneVOne/evaluate.py` harness with the adapter in `agent.py`.

| AlphaBeta variant | Opponent | Wins / games | Completed | Capped / errors | Candidate decision p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Web, pruned | Value Base | 8 / 32 | 32 | 0 / 0 | 10.8 ms |
| Web, pruned | Value Contender | 17 / 32 | 32 | 0 / 0 | 12.2 ms |
| Official, unpruned | Value Base | 14 / 32 | 32 | 0 / 0 | 41.1 ms |
| Official, unpruned | Value Contender | 16 / 32 | 32 | 0 / 0 | 38.2 ms |
| Official, unpruned | OneVOne default | 12 / 32 | 32 | 0 / 0 | 33.3 ms |

The official variant won 7/16 from each seat against Value Base, 8/16 from each seat against Value Contender, and 7/16 from seat 0 plus 5/16 from seat 1 against OneVOne. These small samples do not establish a robust ranking. The official documentation's 80% over ValueFunction (25 games) did not reproduce on this schedule. The web variant's pruning materially changed outcomes, especially against Value Base.

The decision latency distribution includes forced moves, which dominate the median; p95 better represents decisions requiring search. The upstream search uses internal state in its chance expansion, including actual opponent resource frequencies for robber outcomes and hidden development-card composition. These are *engine-internal comparisons*, not a proof of fair play with hidden information. Paired seeds give the same initial board and seat rotation, but action-dependent chance trajectories can diverge.

## Reproduce

```sh
start/.venv/bin/python catanLabs/FromScratch/StrategyOne/experiments/alphabeta_web/benchmark.py --variant official --boards 16 --seed 20268100 --report catanLabs/FromScratch/StrategyOne/experiments/alphabeta_web/official_results.json
start/.venv/bin/python catanLabs/FromScratch/StrategyOne/experiments/alphabeta_web/benchmark.py --variant web --boards 16 --seed 20268100 --report catanLabs/FromScratch/StrategyOne/experiments/alphabeta_web/results.json
start/.venv/bin/python catanLabs/OneVOne/evaluate.py --agent-file catanLabs/FromScratch/StrategyOne/experiments/alphabeta_web/agent.py --config catanLabs/OneVOne/configs/default.json --opponents onevone --boards 16 --seed 20268100 --max-game-seconds 120 --report catanLabs/FromScratch/StrategyOne/experiments/alphabeta_web/vs_onevone.json
```
