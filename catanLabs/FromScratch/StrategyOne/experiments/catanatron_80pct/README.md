# StrategyOne MLP vs Catanatron web bot

This benchmark plays the existing StrategyOne calibrated MLP, settlement target
head, and same-turn search against the pinned public Catanatron web bot. It does
not train either agent. The opponent is the locally installed Catanatron code,
not a call to the live website.

## How the opponent chooses moves

The web bot is an `AlphaBetaPlayer` with search depth 2 and action pruning
(`web/players.py`). On each decision, it considers legal actions, expands chance
outcomes such as dice and card draws, examines the next decision (including an
opponent reply when the turn passes), and
scores resulting positions with a hand-written value function
(`players/minimax.py` and `players/value.py`). Its value function heavily rewards
public victory points, own dice-weighted production, denying opponent production,
and reachable expansion spots. It also values a hand close to settlement or city
costs and penalizes keeping more than seven resources. Those scores, not a learned
policy, pick the action.

## Test protocol

- Source board seeds: `data/catanatron_diverse.meta.json` (1,028 unique seeds).
- Choose 822 seeds (80% rounded) with selection seed `20268116`.
- Play both seat orders on each seed, yielding 1,644 head-to-head games.
- Two players, 10 VP, official spiral number placement, discard limit 7.
- Validate each chosen action against the engine's current legal action list.
- Record wins, terminal status, move counts, latency, and a first-board trace.
- `run.py` writes resumable shard reports; `merge.py` validates complete coverage
  and produces `mlp_vs_web_80pct_merged.json`.

## Result

All 1,644 games finished, with zero caps or errors. StrategyOne won 213
(12.96%); Catanatron won 1,431 (87.04%). StrategyOne won 107/822 as seat 0 and
106/822 as seat 1. On the supervised training board subset it won 171/1,314;
on the `game_id % 5 == 0` board subset it won 42/330. It won in both seats on
21 boards, one seat on 171, and neither seat on 630.

| Move | StrategyOne | Catanatron |
| --- | ---: | ---: |
| Maritime trade | 241 | 14,665 |
| Build settlement | 4,552 | 8,480 |
| Build city | 2,031 | 4,284 |
| Discard a resource | 61,732 | 11,251 |

Per 100 own `END_TURN` actions, those are 0.36 vs 21.95 trades, 6.75 vs 12.69
settlements, 3.01 vs 6.41 cities, and 91.58 vs 16.84 discard actions. These
are observations, not a causal ablation. A replayed turn in `move_probe.json`
shows StrategyOne choosing `END_TURN` with eight cards, including six wheat,
over a legal 4:1 wheat-to-ore trade that would have made a development card
purchase legal.

This is a new tournament on board seeds present in the dataset, not a replay of
stored decisions or a fresh unseen-board test. A `game_id % 5 == 0` subset was
held out from the current supervised training split, although the older MLP has
some prior split overlap. The simulator hands the Catanatron search the full
game object; its search can inspect hidden information. The MLP encodes its own
observation. This matters when interpreting the result.

## Reproduce

From the repository root, using `start/.venv/bin/python`:

```bash
start/.venv/bin/python catanLabs/FromScratch/StrategyOne/experiments/catanatron_80pct/run.py --report catanLabs/FromScratch/StrategyOne/experiments/catanatron_80pct/mlp_vs_web_80pct.json
```

The run requires the existing MLP and settlement head checkpoints in the
adjacent `Hyde-catan-experiments` worktree. `move_probe.py` reproduces one
decision where the MLP ended its turn despite an available bank trade.
