# OneVOne training results

## Current policy

Neither training run found a change that improved held-out wins. No
`configs/trained.json` was created, so `agent.create_agent()` and the play UI
currently use [`configs/default.json`](configs/default.json). This is the
measured heuristic baseline, not a learned neural policy.

## Latest held-out screen

The second run selected the unchanged baseline after six parameter mutations.
Its separate validation set contained 32 board seeds, four frozen opponents,
and both candidate seats: 256 games total. Every game completed without an
agent error or cap.

| Opponent | Wins | Win rate | Wilson 95% interval | Mean VP margin |
| --- | ---: | ---: | ---: | ---: |
| AlphaBeta depth 1 | 43/64 | 67.2% | 55.0–77.4% | +1.38 |
| AlphaBeta depth 2 | 40/64 | 62.5% | 50.3–73.3% | +1.50 |
| Tactical VP | 33/64 | 51.6% | 39.6–63.4% | +0.03 |
| Value function | 34/64 | 53.1% | 41.1–64.8% | +0.36 |
| **Overall** | **150/256** | **58.6%** | **52.5–64.5%** | **+0.82** |

Seat 0 won 77/128 (60.2%); seat 1 won 73/128 (57.0%). These Wilson
intervals are descriptive per-game intervals. For promotion, the trainer uses
a paired bootstrap clustered by board seed. The selected baseline compared
with itself had a zero difference and a `[0, 0]` interval, so promotion was
correctly rejected.

## Head-to-head against the Catanatron web bot

The current default heuristic was tested against the same built-in bot used by
the local Catanatron web UI. It is the pruning-enabled AlphaBeta bot at depth
2. The run used 64 board seeds, with both engine seats tested for every seed
(128 games total). Hyde won 85/128 (66.4%), with a per-game Wilson 95%
interval of 57.9–74.0% and mean final VP margin of +1.84. It won 43/64 games
in seat 0 and 42/64 in seat 1. All games completed without caps or agent
errors.

The live browser match used the same two agents and ended with Hyde OneVOne
(RED) beating Catanatron (BLUE) 10–7. This is one illustrative game; the
paired tournament above is the more useful strength estimate.

Full game-by-game results and environment metadata are in
[`reports/vs_catanatron_64.json`](reports/vs_catanatron_64.json). This remains
a local benchmark under provisional 1v1 rules, not evidence against the final
competition API or human players.

## From-scratch MLP bootstrap

The first dataset contains 20,407 decisions from 192 completed games on 16
board seeds. To test whether that was too little board diversity, it was
expanded with 576 more games on 48 disjoint seeds. The merged dataset has
77,612 decisions from 768 completed games on 64 seeds; every game completed
without errors or caps. The feature encoder contains only information visible
to the acting player and the engine's current legal actions.

The policy is a shared MLP with five action-family scoring heads and a state
value output. Supervised loss imitates the teacher's chosen legal action;
choices from losing teacher games receive one-quarter weight. The value target
repeats each game's terminal win/loss result for every decision, so it is a
noisy value label and does not identify which move caused the outcome. The
hierarchical loss first selects a family, then a target within that family.
Forced one-action prompts do not contribute policy loss.

The 16-board hierarchical fit reached 43.6% joint accuracy on non-forced
held-out choices and 69.6% outcome-sign accuracy, but its screen was poor:

| Opponent | MLP wins | Win rate | Mean VP margin | Errors / caps |
| --- | ---: | ---: | ---: | ---: |
| Current OneVOne heuristic | 0/64 | 0.0% | -6.53 | 0 / 0 |
| Catanatron web bot | 1/64 | 1.6% | -5.88 | 0 / 0 |
| AlphaBeta depth 2 | 3/64 | 4.7% | -5.81 | 0 / 0 |

The 64-board fit used 47 seeds for training and 17 for validation. Its best
checkpoint was epoch 2 of 7, with 41.6% non-forced joint action accuracy,
86.8% family accuracy, 53.4% conditional target accuracy, and 73.1% value-sign
accuracy. Its fresh 32-board paired-seat screen also lost heavily:

| Opponent | MLP wins | Win rate | Mean VP margin | Errors / caps |
| --- | ---: | ---: | ---: | ---: |
| Current OneVOne heuristic | 3/64 | 4.7% | -5.63 | 0 / 0 |
| Catanatron web bot | 7/64 | 10.9% | -4.78 | 0 / 0 |
| AlphaBeta depth 2 | 3/64 | 4.7% | -5.55 | 0 / 0 |

More board seeds did not improve the learned policy. The current data problem
is better described as weak or ambiguous targets: the mixed teacher league
includes a weak teacher that lost all 96 focal games in the expanded corpus,
and terminal game outcomes repeat across every action in a trajectory. These
are plausible contributors, not individually proven causes. The MLP is not
promoted; the live agent remains the heuristic. The next useful experiment is
to produce action-specific search or rollout targets, or train through online
league self-play, then evaluate on another fresh seed range. Neither run is
PPO/GRPO or online self-play RL.

Raw outputs are in
[`reports/mlp_bootstrap_screen.json`](reports/mlp_bootstrap_screen.json),
[`reports/mlp_hierarchical_bootstrap_screen.json`](reports/mlp_hierarchical_bootstrap_screen.json),
[`reports/mlp_hierarchical_64board_screen.json`](reports/mlp_hierarchical_64board_screen.json),
and [`POLICY_DATA.md`](POLICY_DATA.md). The merged data and model run are
[`bootstrap_20272000_64boards.jsonl.gz`](data/bootstrap_20272000_64boards.jsonl.gz)
and [`runs/mlp-hierarchical-64board-20261016`](runs/mlp-hierarchical-64board-20261016/).

## Search runs

| Run | Search | Training score, baseline / best | Held-out score, baseline / selected | Decision |
| --- | --- | ---: | ---: | --- |
| [`evolution-20261016-full`](runs/evolution-20261016-full/training_report.json) | 8 mutations, 8 train boards, 24 validation boards | 60.9% / 60.9% | 57.3% / 57.3% | No promotion |
| [`evolution-20261016-tactical-search`](runs/evolution-20261016-tactical-search/training_report.json) | 6 mutations, 12 train boards, 32 validation boards; added tactical threshold and search-budget mutations | 58.3% / 58.3% | 58.6% / 58.6% | No promotion |

The runs used distinct board-seed ranges. The tuning objective was macro-average
win rate over fixed `value`, `alphabeta1`, `alphabeta2`, and tactical opponents.
Changes were promoted only if the candidate's paired held-out board interval
was entirely above zero and all candidate games completed cleanly.

## What to improve next

These runs validate the current baseline against local bots; they do not show
that it is strong against people or the competition's final rules. The current
optimizer changes a small number of heuristic parameters, and this search did
not improve them. The next training step should learn a state/action value
model from 1v1 game outcomes and test it as a legal-action ranker, then evaluate
any improvement on new board seeds before using it in play. Do not describe the
current agent as PPO, GRPO, or self-play trained.
