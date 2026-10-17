# Learned 1v1 policies: results and failure modes

There are **two separate MLP lines**. `OneVOne/mlp_policy.py` is a five-head
imitation model trained from a mixed teacher league. `StrategyOne` is a legal
action ranker trained from existing Catanatron teachers, later fitted with a
PPO checkpoint, calibration adapter, and settlement target head. Their data,
checkpoints, and game results must not be merged as one model.

## A. OneVOne five-head MLP

Source: [`OneVOne/TRAINING_RESULTS.md`](../TRAINING_RESULTS.md) and
[`POLICY_DATA.md`](../POLICY_DATA.md). It scores five action families plus a
state value. Its larger corpus had 77,612 decisions from 768 games on 64
boards, with six teacher styles. The best fit reached 41.6% exact joint
agreement on non-forced held-out choices; family accuracy was 86.8%.

| Fresh opponent | Wins / 64 with larger fit |
| --- | ---: |
| OneVOne heuristic | 3 |
| Catanatron web bot | 7 |
| Unpruned AlphaBeta depth 2 | 3 |

**Observed failure:** choosing the broad action family did not reliably
choose the right legal target. Earlier 16-board training was weaker still:
0/64 against OneVOne and 1/64 against the web bot. More boards alone did
not solve it. The corpus mixed teachers of unequal strength, and the same
terminal result was repeated as a value label at every move. Those are
plausible training weaknesses, not isolated causal proofs. This checkpoint
was never promoted.

## B. StrategyOne legal-action MLP

Source: [`StrategyOne/README.md`](../../FromScratch/StrategyOne/README.md),
[`MLP_FAILURE_ANALYSIS.md`](../../FromScratch/StrategyOne/reports/MLP_FAILURE_ANALYSIS.md),
and the [web-bot test](../../FromScratch/StrategyOne/experiments/catanatron_80pct/README.md).
Its 1,028-game external-teacher corpus contains 156,471 decision rows.
Teachers were pinned Catanatron Value Base, Value Contender, and Victory
Point bots. The first MLP reached 49.9% teacher agreement on real choices
in an older 20% test; the 75.8% all-action figure included forced choices
and had partial split overlap. Neither number measured playing strength.

On 50 games versus Value Base, the frozen MLP won 2. It ended the turn while
the teacher preferred `BUY_DEVELOPMENT_CARD` 435 times; on 173 held-out
teacher Buy examples it got **zero** correct. Buy represented only 1.12% of
training labels. Late in those losses it held 3.76 more cards while trailing
by 3.11 VP and 22.84 production pips. Territory and robber targets also
disagreed sharply. These are observed deficits; the exact contribution of
each to lost games was not isolated.

The later calibrated MLP, settlement target head, and small same-turn search
won **213/1,644 (13.0%)** against the web bot across both seats on 822
dataset board seeds. This is a wide diagnostic on **dataset boards**, not an
independent sealed-board score. Its maritime trades were rare and its discards
frequent relative to the web bot. On 100 matching seed-seat pairs selected
from that diagnostic, OneVOne won 81 and StrategyOne won 12; this is a
same-schedule comparison derived from saved reports, still on dataset boards.
On the fresh planner confirmation schedule,
the frozen calibrated/settlement MLP won **12/128 (9.4%)**.

## C. PPO from the StrategyOne MLP

Source: [`SEALED_128_PPO_VERDICT.md`](../../FromScratch/StrategyOne/reports/SEALED_128_PPO_VERDICT.md).
The preliminary run used 64 complete training games and only terminal
win/loss reward. A preliminary 100-game mixed-bot check tied imitation at
39 wins each. On a separate sealed 128-game mixed-bot test, PPO won
**44/128** versus imitation **45/128**. It was not promoted.

**Observed failure:** the PPO model still ranked `END_TURN` above rare
development-card purchases and kept the midgame production/expansion gap.
The sparse delayed reward and small training run are plausible reasons it
did not repair that decision; no evidence says PPO as a method cannot work.
No GRPO training result exists in this workspace.

## D. Learned calibration adapter

Source: [`CALIBRATION_LAYER_RESULTS.md`](../../FromScratch/StrategyOne/reports/CALIBRATION_LAYER_RESULTS.md).
It freezes the MLP and learns action-type residuals plus an End Turn score
adjustment. Buy Development matching rose from **0/173 to 78/173** in the
reported 20% check. In fresh, matched **mixed-bot** games it raised wins
from **37/128 to 57/128**. Against Value Base specifically, it moved only
from 1/26 to 3/26. The adapter fixes a measured ranking bias, but does not
solve strong-opponent play or the rest of resource conversion.

## E. Learned settlement-location head

Source: [`SETTLEMENT_TARGET_EXPERIMENT.md`](../../FromScratch/StrategyOne/reports/SETTLEMENT_TARGET_EXPERIMENT.md).
This head acts only after the calibrated MLP has chosen to build a
settlement; it reranks legal nodes using local board features. Teacher-node
matching rose from **26.2% to 39.7%** on 546 held-out choices. Across two
fresh matched mixed-bot screens, wins rose from **113/256 to 126/256**.
Most of the extra wins were against Weighted and Random; Value Base was
unchanged at 3/52. It improves location choice without fixing the model's
decision about *when* to build, its roads, or robber targeting.

## F. Longer training and two small rules

Source in the isolated worktree:
`catanLabs/FromScratch/StrategyOne/reports/LONG_TRAINING_AND_HEURISTICS.md`.
All arms used two fresh, matched 128-game mixed-bot screens.

| Change to calibrated MLP + settlement head | Wins / 256 | Verdict |
| --- | ---: | --- |
| Four-epoch head control | 125 | Baseline for this schedule |
| Seven-epoch head | 125 | Better label match, zero game gain |
| Twelve-epoch head | 123 | Lower training loss but worse held-out match and games |
| Highest adjacent dice pips for settlement | 118 | Worse overall; local production alone missed other value |
| Buy Dev before End Turn with at least eight cards | 127 | Only +2 wins; insufficient evidence to adopt |

The twelve-epoch decline is consistent with overfitting this target head.
The pips and Buy Dev rows are **heuristic overrides**, not learned policies.

## G. Public-history Transformer residual

Source in the isolated worktree:
`catanLabs/FromScratch/StrategyOne/reports/HISTORY_AND_SEARCH_EXPERIMENTS.md`.
It read 16 recent public events and was trained against teacher actions.
On a fresh 64-game mixed-bot screen, the unchanged agent won **36**, a
state-only residual won **20**, and the history Transformer won **26**.
Teacher agreement improved slightly over the unchanged agent, but win rate
fell. This tests one small residual trained for teacher matching; it does
not establish that longer-term context or a win-trained critic is useless.
