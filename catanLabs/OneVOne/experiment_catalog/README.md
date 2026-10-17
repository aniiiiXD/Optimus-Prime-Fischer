# 1v1 Catan experiment catalog

This is the single index for **our two-player Catan bots and experiments**.
It covers the main `Hyde` workspace and the isolated
`Hyde-catan-experiments` worktree. Four-player TacticalVP and external
four-player RL projects are outside this catalog.

## Read this first

| File | Contains |
| --- | --- |
| [LEARNED.md](LEARNED.md) | Both MLP tracks, PPO, calibration, settlement head, longer training, history Transformer, and small heuristics |
| [PLANNING.md](PLANNING.md) | OneVOne heuristic, parameter search, tactical search, backward analysis, invest/convert planner, and hybrid |
| [OUTSIDE_BOTS.md](OUTSIDE_BOTS.md) | Catanatron reference bots and the locally tested HexMachina reproductions |
| [RESULTS_1028.md](RESULTS_1028.md) | Uniform 1,028-board, both-seat head-to-head for all 20 playable profiles |

**Current playable 1v1 bot:** [`OneVOne/agent.py`](../agent.py) with
[`configs/default.json`](../configs/default.json). It is a heuristic plus
bounded tactical search; neither its neural imitation run nor the StrategyOne
MLP has replaced it. `OneVOne/play_web.py` loads this default because no
`configs/trained.json` was promoted.

## Uniform 1,028-board head-to-head

See [RESULTS_1028.md](RESULTS_1028.md) for all 20 playable profiles on the
same 1,028 seeds, both seats, against the same Catanatron web bot. OneVOne won
1,474/2,056 (71.7%); all matches completed. These are dataset board seeds, not
an independent held-out test.

## Earlier 128-match head-to-head

The table below is **one** fresh 64-board test, both seats per board, against
the same pinned Catanatron web bot (pruned AlphaBeta, depth 2). All arms used
the same seed and seat schedule. Two players raced to 10 VP.

| Our 1v1 profile | Wins / 128 | Reading |
| --- | ---: | --- |
| OneVOne heuristic/search | **93** | Strongest on this schedule |
| Frozen StrategyOne calibrated MLP + settlement head | 12 | Learns some choices, but fails to convert resources into VP |
| MLP + changing-horizon invest/convert planner | 64 | Large repair from trade/build sequencing; changing boundary itself unproven |
| MLP + fixed-horizon planner | 67 | Best MLP-based profile on this schedule |

A second, **different** 64-board head-to-head measured OneVOne **96/128**,
fixed-horizon planner **73/128**, and their phase hybrid **85/128**. The
hybrid did not beat OneVOne. Source and game records are in the isolated
worktree's `catanLabs/FromScratch/StrategyOne/reports/INVEST_CONVERT_RESULT.md`
and adjacent `invest_convert_*.json` files.

## Where the code actually is

```text
Hyde/catanLabs/OneVOne/
  agent.py, configs/, evaluate.py, play_web.py       current 1v1 agent
  mlp_policy.py, train_policy.py, runs/                separate 5-head MLP track

Hyde/catanLabs/FromScratch/StrategyOne/
  generate_dataset.py, train_legal_ranker.py          external-teacher dataset + MLP
  ppo_actor_critic.py, train_calibrated_ranker.py     learning experiments
  settlement_target_head.py, evaluate_win_rate.py    learned target head + evaluator
  experiments/                                       web-bot and outside-agent tests

Hyde-catan-experiments/catanLabs/FromScratch/StrategyOne/
  same_turn_search.py, history_experiment.py         isolated search/history studies
  heuristic_experiments.py, loss_counterfactual_search.py
  invest_convert.py, invest_convert_experiment.py    latest planner + match runner
  configs/, reports/                                 frozen arms and full outcomes
```

The invest/convert source, configs, and report are presently **untracked in
the experiment worktree**; they are not in the main workspace or pushed
repository. Its model checkpoints and dataset are read through local links
to the main workspace. This catalog is a shared map; it does not duplicate
large datasets or checkpoints.

## How to read the numbers

- Compare wins **within the same report/schedule**. Some runs use five mixed
  opponents; others use only the Catanatron web bot. A pooled mixed-bot win
  rate is not a web-bot win rate.
- All cited 1v1 local tests use provisional Catanatron rules, generally 10
  VP, official spiral numbers, seven-card discard limit, and both seats when
  a report says paired. The final event contract is still unconfirmed.
- A higher teacher-action match or lower training loss is not a game win.
  Reports below mark observed behavior separately from proposed causes.
- `Catanatron web bot` means its pruning-enabled depth-2 AlphaBeta player.
  The unpruned depth-2 AlphaBeta reference is a different variant. The
  simulator gives its internal search full game state, which matters when
  interpreting hidden-information fairness.
