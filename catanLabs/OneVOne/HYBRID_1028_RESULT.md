# OneVOne heuristic + outcome Transformer, 1,028-game split

## What was trained

This experiment keeps OneVOne's existing heuristic value function, robber
policy, action-family choice and tactical VP override. A small CPU Transformer
(100,673 parameters) reads the current visible state plus the last 16 public
actions and estimates the chance of winning. For settlement, city, road and
maritime-trade choices, the agent considers the legal moves **within the
heuristic's chosen action type** and combines normalized OneVOne value with
the model's predicted win probability. It does not replace OneVOne's overall
decision policy. No opponent private hand or development-deck order is read.

The source data is the existing 1,028-game Catanatron trajectory set at
`catanLabs/FromScratch/StrategyOne/data/catanatron_diverse.jsonl`. Its three
teacher bots are Catanatron ValueFunctionPlayer(base),
ValueFunctionPlayer(contender), and VictoryPointPlayer. **The Transformer was
trained on their game outcomes, not OneVOne imitation labels.**

Games, not moves, were split: 822 train and 206 held out. Of the 822 training
games, 740 fit model weights and 82 selected the checkpoint. Up to 32
positions/game were sampled: 23,680 fit, 2,624 validation, 6,592 held-out
positions. The outcome label is the teacher player's final win/loss. Model
selection used validation NLL. The best checkpoint was epoch 1; later epochs
overfit. Held-out position AUC was 0.8895 (0.8841 with history zeroed), but
position classification is only a surrogate for winning a new game.

## Online win test

The coefficient was screened on 12 training seeds, both seats. The best
nonzero value was `beta=0.5`; plain OneVOne actually led that small screen
18/24 versus 16/24. The sealed 206 test seeds were then played from scratch
in both seats against the pinned Catanatron web bot under the same 10-VP
protocol as the archived baseline. These are new games, not trajectory replay.

| Agent | Wins / 412 | Win rate | Complete / errors |
| --- | ---: | ---: | ---: |
| Plain OneVOne, archived paired baseline | 290 | 70.39% | 412 / 0 |
| OneVOne + outcome Transformer | 291 | 70.63% | 412 / 0 |

The hybrid considered 7,435 same-type choices and changed 745 moves. Of the
paired games, it won 76 that plain OneVOne lost, while losing 75 that plain
OneVOne won. Difference: **+1/412 game = +0.24 percentage points**. A
board-clustered bootstrap 95% interval is roughly **-5.34 to +5.58 points**.
This gives no credible evidence that the model improves wins. Keep the plain
OneVOne agent as incumbent.

## Reproduce / inspect

- `train_outcome_transformer.py`: split, encoding, CPU training and position metrics.
- `outcome_transformer.py`: public-history encoder and heuristic/model hybrid.
- `evaluate_outcome_hybrid.py`: coefficient screen and full paired-seat test.
- `runs/outcome_1028/split.json`: exact game IDs and seeds.
- `runs/outcome_1028/best.pt`: selected model weights.
- `runs/outcome_1028/training.json`, `evaluation.json`: full metrics.
- `runs/outcome_1028/games/`: per-game test results.

From this worktree, with the linked `start/.venv` runtime and the original
1,028-game data in the main workspace:

```sh
start/.venv/bin/python -m catanLabs.OneVOne.train_outcome_transformer
start/.venv/bin/python -m catanLabs.OneVOne.evaluate_outcome_hybrid --screen-boards 12
```

The 173 MB encoded feature cache is local and intentionally excluded from
Git; it can be rebuilt from the source JSONL.
