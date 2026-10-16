# Catan ideas and comparisons to revisit

Updated 16 October 2026. Keep external win rates separate from our local
benchmarks: player count, VP target, opponents, information access, and game
counts differ.

## BenjaminL1 and Eli6th

These are separate repositories and codebases. [BenjaminL1/catan_rl](https://github.com/BenjaminL1/catan_rl)
targets a Colonist-style 1v1 variant with a tile Transformer, graph encoder,
and factored action heads. Its inspected repository does not report a
comparable Catanatron win rate or game count. [Eli6th/catan-rl](https://github.com/Eli6th/catan-rl)
uses a Rust four-player engine, PPO policy/value MLP, and policy-guided
full-game rollouts.

| Eli6th reported result | Opponents and rules | Evaluation size |
| --- | --- | --- |
| PPO policy: about 65% | Three frozen Heuristic-v1 bots in its own four-player, first-to-7-VP, perfect-information engine | 192 games in the fixed-seed policy gate (65.1% for `overnight`; 65.6% for `runB2`) |
| AlphaBot search: 82.0% | The same heuristic opponents and engine | The [experiment ledger](https://github.com/Eli6th/catan-rl/blob/main/training/results/EXPERIMENTS.md) says 120–150 games per search-design iteration; it does not identify the exact denominator for the 82.0% run |

Neither headline percentage measures play against Catanatron. The published
AlphaBot search starts from the real hidden simulator state, so its result
also does not establish strength under a fair, restricted observation API.
Our local [reproduction and transfer notes](External/README.md) record what
we actually measured when running or porting this code.

## Where our agents stand locally

- **Four-player, 10 VP:** The current measured Catanatron v5 incumbent is
  [TacticalVP](Competitive/TACTICAL_VP_MIN0_CONFIRM.md). On 100 fresh boards
  with all four seats, it won 207/400 against Weighted + Value + AlphaBeta1
  and 183/400 against Value + two AlphaBeta1. Matched FairValue controls won
  168/400 and 142/400, respectively. This is a confirmed local improvement,
  not a published-opponent or event score.
- **1v1, 10 VP:** The playable [OneVOne default heuristic](OneVOne/TRAINING_RESULTS.md)
  won 85/128 against the pinned Catanatron web bot on 64 board seeds and both
  seats. A separate 50-board, both-seat run won 81/100. On those same 100
  seed-seat pairs, the StrategyOne learned policy won 12/100; this comparison
  is derived from the saved [heuristic report](FromScratch/StrategyOne/experiments/catanatron_80pct/onevone_vs_web_80pct.json)
  and [MLP report](FromScratch/StrategyOne/experiments/catanatron_80pct/mlp_vs_web_80pct_merged.json),
  with no new games run. Those boards come from the existing dataset schedule,
  so this is a diagnostic paired comparison rather than a sealed test.
- **Learned policies:** StrategyOne's calibrated MLP, settlement-target head,
  and same-turn search won 213/1,644 (13.0%) against that web bot across 822
  dataset boards and both seats. It made far fewer maritime trades and more
  discards than the opponent; this is an observed correlation, not proof of
  the cause. The separate four-player TacticalVP Transformer clone also
  [lost its fresh game screen](Transformer/runs/gate6_tactical_distill/DISTILLATION.md)
  despite high teacher-action agreement. The current winning local policies
  are still search/heuristic based.

## Practical reading

The strongest supported claim is that our heuristic/search agents beat their
own local baselines. The MLP has not yet translated imitation accuracy into
competitive win rate. Cross-project headline percentages cannot rank agents
until they play under one ruleset, observation boundary, seed/seat schedule,
and opponent pool. Revisit BenjaminL1's structured board representation and
Eli6th's policy-guided rollouts as design ideas when the competition contract
and a fair held-out test are available.
