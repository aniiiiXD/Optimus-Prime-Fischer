# Catan Transformer v0 — first CPU run

**Result: the policy runs legally and quickly, but it is not strong enough to
replace the existing value or alpha-beta bots.** No checkpoint promotion or
new training approach follows from this run.

This is a historical v1 tournament. The benchmark now uses corrected v5
chance handling, so these numbers are not directly comparable with the later
[competitive CPU policy](../../../Competitive/RESULTS.md).

## Data and training

- 4,000 exact-unique decision states from 14 completed games (collection seeds
  20261015–20261028).
- Mixed trajectory players: 816 random, 876 weighted, 1,123 value and 1,185
  alpha-beta depth-one states.
- Target action: Catanatron alpha-beta depth one.
- Coverage: initial settlement, initial road, regular turns, discards and
  robber movement. Player-to-player trade offers are outside this benchmark.
- Observation features include the acting player's private hand and cards plus
  public board, opponents and bank information. The encoder does not read
  opponent hands or unplayed development-card identities.
- Model: 77,569 parameters, width 64, two Transformer layers, four heads.
- CPU fit: 590.5 seconds (under the 600-second cap), 105 epochs. The best
  held-out-loss checkpoint was epoch 8; later epochs overfit.
- Split by whole game: 3,082 training examples and 918 examples from held-out
  games 11–13. Best validation loss was 0.773; exact teacher-action agreement
  was 41.6%.

## Held-out tournament

The evaluator used seeds 20261029–20261032, separate from all collection
seeds. It tested 16 games per tier, rotating through all four seats against
three identical bots of that tier. The Wilson intervals below are descriptive;
the sample is small, and four seat rotations on one map are related.

| Three opponents | Wins | Wilson 95% interval | Mean final VP | Median / p95 decision latency |
|---|---:|---:|---:|---:|
| Random | 4/16 (25.0%) | 10.2–49.5% | 5.31 | 0.784 / 0.899 ms |
| Weighted | 3/16 (18.8%) | 6.6–43.0% | 5.94 | 0.782 / 0.908 ms |
| Value | 0/16 (0%) | 0–19.4% | 3.31 | 0.798 / 0.963 ms |
| Alpha-beta depth 1 | 0/16 (0%) | 0–19.4% | 3.56 | 0.801 / 1.006 ms |

There were no illegal actions or agent errors. Three games hit the evaluator's
500-turn cap (two in the random tier, one in weighted); each stayed in the
attempted-game denominator as a non-win. In this four-player setup, a fair
share against equally strong opponents is 25%, but this small run does not
show a dependable edge over random or weighted. The model clearly failed to
beat the stronger value and alpha-beta tiers.

## What this teaches us

The first dataset and training path work end to end: legal observations were
collected, the model trained from random weights on CPU, and the saved policy
completed the tournament without an illegal choice. The result also shows that
copying alpha-beta's selected move is not enough to copy its strength with this
small corpus and compact representation. Training agreement is not playing
strength.

The candidate adapter still receives Catanatron's full game object through the
benchmark interface, although this encoder reads only the visible fields listed
above. Before a high-stakes claim, the harness should enforce that boundary and
the tournament should use more independent held-out seeds.

## Reproducibility artifacts

- [Dataset and collection code](../../../../Transformer/README.md)
- [Dataset manifest](dataset.json)
- [Compressed dataset](dataset.npz)
- [Training metrics](training.json)
- [Tournament report](tournament.json)
- [Best checkpoint](best.pt)
- [Model code](../../model.py) · [Agent adapter](../../agent.py) · [Evaluator](../../../Benchmarks%26Evals/run.py)

The next training approach is paused until the user gives the flag.
