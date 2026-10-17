# All 1,028 dataset boards: 1v1 bot comparison

Each of 20 playable profiles started a new Catanatron v5 game from each of the
1,028 unique dataset seeds, once from each seat. All 2,056 matches per profile
were played against the same pinned Catanatron web bot (pruned depth-2
AlphaBeta). Rules: two players, 10 VP, official spiral, discard at seven cards.
The saved teacher moves were **not** replayed. These are dataset board seeds,
so this is a diagnostic comparison, **not** an independent held-out test.

| Profile | Wins / 2,056 | Win rate |
| --- | ---: | ---: |
| OneVOne | 1,474 | 71.7% |
| Hybrid | 1,411 | 68.6% |
| AlphaBeta depth 2, unpruned | 1,268 | 61.7% |
| HexMachina approximate | 1,253 | 60.9% |
| Invest/convert, fixed horizon | 1,200 | 58.4% |
| Invest/convert, dynamic horizon | 1,187 | 57.7% |
| HexMachina published reproduction | 1,110 | 54.0% |
| Buy development card trigger | 304 | 14.8% |
| OneVOne MLP, 64 boards | 274 | 13.3% |
| Same-turn search | 260 | 12.6% |
| Settlement head, epoch 7 | 235 | 11.4% |
| Settlement head, epoch 12 | 235 | 11.4% |
| Settlement head, epoch 4 | 226 | 11.0% |
| OneVOne MLP, 16 boards | 215 | 10.5% |
| Settlement pip heuristic | 187 | 9.1% |
| History Transformer residual | 161 | 7.8% |
| State-only residual control | 151 | 7.3% |
| Calibrated StrategyOne MLP | 134 | 6.5% |
| Raw StrategyOne MLP | 89 | 4.3% |
| PPO StrategyOne MLP | 82 | 4.0% |

All 41,120 matches completed with zero errors and zero capped games. Each
profile's results contain exactly one match for each seed and seat. The exact
counts, seat splits, and protocol are in
[`reports/all_1028/summary.json`](reports/all_1028/summary.json); individual
match records are in adjacent profile JSONL files. The 1,028 source games
contain 156,471 saved **teacher decisions**: visible observation, legal
actions, chosen action, and eventual result. They do not store a complete
two-player move replay.
