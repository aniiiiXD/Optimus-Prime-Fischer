# Invest/convert planner experiment

The calibrated StrategyOne MLP and settlement head stayed frozen. A bounded
planner now expands deterministic actions within the current turn, including
bank trade → settlement/city/road. It first takes a found immediate win. Its
score values VP, dice-weighted future resource production, accessible
settlement sites, the current hand, and cards likely to be discarded. The
production and access terms are multiplied by a horizon that shrinks as public
VP totals rise. The planner never simulates hidden card draws, dice, or theft.

The coefficients were selected by game outcomes, not by teacher-action match.
Four hand-specified coefficient profiles played 16 fresh board seeds in both
seats against pinned Catanatron web AlphaBeta depth 2. The `convert` profile
had the highest training wins (22/32). This is a small coefficient search;
**no neural weights were updated**. Its 68.8% training win rate should not be
used as a strength estimate.

## Fresh confirmation 1

Seeds `20269100`–`20269163`, 64 boards × both seats per arm, identical 2-player
10-VP rules, official spiral numbers, discard limit 7, friendly robber off.
All 512 games completed without caps or errors.

| Arm | Wins / 128 | Win rate | Mean terminal VP |
| --- | ---: | ---: | ---: |
| Frozen MLP | 12 | 9.38% | 4.72 |
| MLP + selected changing-horizon planner | 64 | 50.00% | 8.07 |
| MLP + fixed-horizon planner | 67 | 52.34% | 8.23 |
| Existing OneVOne heuristic | 93 | 72.66% | 9.16 |

On exactly matched seeds and seats, the selected planner flipped 53 MLP losses
to wins and reversed one MLP win. Fixed horizon beat changing horizon by three
wins (8 cases favoring fixed, 5 favoring changing). The measured gain is from
turn planning and conversion; this test does **not** establish that the
changing invest/convert boundary helps. The full [confirmation report](invest_convert_confirm.json)
records every game and move count.

The frozen MLP made 22 maritime trades and 4,985 individual resource discard
actions in this screen. The selected planner made 1,105 trades and 964 discard
actions; it also built 579 settlements and 296 cities, versus the MLP's 345
settlements and 154 cities. These are action counts, not independent events.

## Fresh confirmation 2: phase hybrid

Seeds `20269200`–`20269263`, again both seats, all 384 games completed.
The hybrid used OneVOne outside `PLAY_TURN` after the roll, and the selected
MLP planner inside that phase.

| Arm | Wins / 128 | Win rate | Mean terminal VP |
| --- | ---: | ---: | ---: |
| Hybrid | 85 | 66.41% | 8.91 |
| Existing OneVOne heuristic | 96 | 75.00% | 9.27 |
| MLP + fixed-horizon planner | 73 | 57.03% | 8.20 |

The hybrid lost 11 wins relative to OneVOne on matched games: nine outcomes
favored the hybrid and 20 favored OneVOne. The [hybrid report](invest_convert_hybrid_confirm.json)
has the paired results.

## Verdict

**The same-turn planner works as an MLP repair**: it sharply improves resource
conversion and beats Catanatron about half the time on two fresh 128-game
screens. **The changing horizon has not earned its place**, and **OneVOne
remains the strongest tested agent**. Keep this as an experiment; do not replace
OneVOne with the MLP or hybrid yet.

The reports test only this pinned Catanatron opponent and provisional two-player
rules. The local engine supplies a full game object to Catanatron's search,
while the MLP uses a visible observation; that information asymmetry remains.

The agent module exposes `create_agent(color, checkpoint, seed)`. Pass
[`invest_convert_dynamic.json`](../configs/invest_convert_dynamic.json) for the
selected changing-horizon profile or
[`invest_convert_static.json`](../configs/invest_convert_static.json) for the
fixed-horizon ablation. The latter scored slightly better in both fresh
screens; neither replaces OneVOne as the current best agent.

## Reproduce

From the `Hyde-catan-experiments` worktree, using the main workspace's Python
environment and checkpoints:

```bash
/Users/anixd/Documents/Hyde/start/.venv/bin/python catanLabs/FromScratch/StrategyOne/invest_convert_experiment.py --phase tune --report catanLabs/FromScratch/StrategyOne/reports/invest_convert_tune.json
/Users/anixd/Documents/Hyde/start/.venv/bin/python catanLabs/FromScratch/StrategyOne/invest_convert_experiment.py --phase confirm --arms mlp convert static onevone --report catanLabs/FromScratch/StrategyOne/reports/invest_convert_confirm.json
/Users/anixd/Documents/Hyde/start/.venv/bin/python catanLabs/FromScratch/StrategyOne/invest_convert_experiment.py --phase confirm --seed 20269200 --boards 64 --arms hybrid onevone static --report catanLabs/FromScratch/StrategyOne/reports/invest_convert_hybrid_confirm.json
```
