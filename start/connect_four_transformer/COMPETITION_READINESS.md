# CPU Connect Four candidate: readiness review

## Current recommendation

Use the Pons-phase-v3 checkpoint with the Transformer-guided depth-5 search
agent. The Transformer orders candidate moves; alpha-beta searches the reply
tree; exact game rules handle immediate wins and blocks. Neural value calls in
the search tree are disabled because they did not improve the measured result.

From `start/`, play with the measured configuration:

```bash
.venv/bin/python -m connect_four_transformer.play_hybrid \
  --checkpoint connect_four_distillation/artifacts/checkpoints/candidate_pons_phases_v3_5min.pt \
  --human X --depth 5 --nodes 5000 --model-evals 1 --model-value-weight 0
```

The checkpoint is unchanged. This is inference-time search around the trained
Transformer, not new weights, RL, or outcome-only self-play.

## Evidence on CPU

The final frozen comparison used 90 paired random openings. Each opening was
played twice with colors swapped, so each agent played 180 games. Both
candidates used a 5,000-node cap; the internal depth-3 alpha-beta opponent
used the same cap. Search/game code raises on an illegal action, and none was
observed.

| Candidate vs depth-3 alpha-beta | W–D–L | Score rate (opening-bootstrap 95% CI) | Median move time | P95 move time |
| --- | ---: | ---: | ---: | ---: |
| Plain guarded Transformer | 47–13–120 | 29.7% on the prior 90-opening paired suite | 0.5 ms | 0.7 ms |
| Plain alpha-beta, depth 5 | 123–11–46 | 71.4% (65.8–77.2%) | 33.9 ms | 95.8 ms |
| Transformer-guided alpha-beta, depth 5 | 150–6–24 | 85.0% (80.0–89.7%) | 76.6 ms | 159.4 ms |

On those same 90 openings, the guided agent's paired score advantage over
plain depth-5 alpha-beta was **+13.6 percentage points** (95% paired bootstrap
interval **+6.9 to +20.0 points**). It visited a median 723 nodes versus 1,026
for plain depth-5 search, but took longer per move; the current tradeoff is
strength over latency.

On 481 held-out positions from the exact Pons solver, the v3 model's raw
best-move agreement was 65.3% with mean regret 0.209. The tactical guard chose
a valid move in 100% of 153 positions where an immediate win or genuinely safe
block existed. Another 24 positions had no one-move defense and are reported
as unavoidable losses rather than fake “missed blocks.”

## Training experiment retained, not promoted

I generated 2,000 unique positions from 734 games of the hybrid agent versus
depth-3 search, then labeled all positions with completed depth-5 alpha-beta
search (20,000-node cap). The five-minute CPU fine-tune selected epoch 38 from
662, used 5e-5 learning rate, reached 350 MiB peak RSS, and took 300 seconds
of fit time. The generated dataset and report are
[`dagger_search_v1_2000.jsonl`](artifacts/data/dagger_search_v1_2000.jsonl) and
[`dagger_search_v1_2000.report.json`](artifacts/data/dagger_search_v1_2000.report.json);
the weights/report are
[`candidate_dagger_search_v1_5min.pt`](artifacts/checkpoints/candidate_dagger_search_v1_5min.pt)
and [`training_dagger_search_v1_5min.json`](artifacts/reports/training_dagger_search_v1_5min.json).

This checkpoint was not promoted: its exact-solver holdout fell to 59.5%
agreement / 0.265 regret, compared with v5's 62.0% / 0.234 and v3's 65.3% /
0.209. Its plain Transformer did improve on its collected-game distribution,
but its hybrid scored 85.0% versus v3 hybrid's 82.5% over 90 paired openings;
the paired difference was only +2.5 points (95% interval −1.9 to +6.9), so
there is not enough evidence to replace the older weights.

The collector and five-minute fine-tune are reproducible with:

```bash
.venv/bin/python -m connect_four_transformer.generate_dagger_dataset \
  --checkpoint connect_four_distillation/artifacts/checkpoints/candidate_pons_phases_v3_5min.pt \
  --positions 2000 --seed 2048 --teacher-depth 5 --node-limit 20000 \
  --output connect_four_transformer/artifacts/data/dagger_search_v1_2000.jsonl

.venv/bin/python -m connect_four_transformer.train \
  --dataset connect_four_transformer/artifacts/data/dagger_search_v1_2000.jsonl \
  --init-checkpoint connect_four_transformer/artifacts/checkpoints/candidate_tactical_v5_5min.pt \
  --checkpoint connect_four_transformer/artifacts/checkpoints/candidate_dagger_search_v1_5min.pt \
  --report connect_four_transformer/artifacts/reports/training_dagger_search_v1_5min.json \
  --time-budget-seconds 300 --batch-size 64 --learning-rate 5e-5 \
  --threads 4 --hard-tactical-targets --tactical-weight 3 --selection-metric balanced
```

## Reproduce the final match

```bash
.venv/bin/python -m connect_four_transformer.benchmark_hybrid_search \
  --checkpoint connect_four_distillation/artifacts/checkpoints/candidate_pons_phases_v3_5min.pt \
  --only-candidates alphabeta_d5,hybrid_d5_no_model_value \
  --games-per-side 45 --seeds 43031,43032 \
  --report connect_four_transformer/artifacts/reports/competition_candidate_v1_90opening_pairs.json
```

The report saves every game and the paired-opening bootstrap comparison:
[`competition_candidate_v1_90opening_pairs.json`](artifacts/reports/competition_candidate_v1_90opening_pairs.json).
The benchmark took 385 seconds, used about 205 MB peak RSS on this Mac, and
ran with one Torch thread.

The exact-solver comparison after the fine-tune is saved at
[`pons_holdout_after_dagger_v1.json`](artifacts/reports/pons_holdout_after_dagger_v1.json).

Focused tests and rules tests:

```bash
.venv/bin/python -m unittest \
  connect_four_transformer.test_hybrid_search \
  connect_four_transformer.test_tactical_categories \
  connect_four_transformer.test_data_pipeline \
  connect_four_distillation.test_oracle \
  connect_four_engine.test_engine -v
```

## What “competition-ready” means here

This is a credible, reproducible **CPU Connect Four baseline** against our
local search opponents, not a verified winner against a public perfect-play
engine or an unknown hackathon game. The official task, hardware, time limit,
and opponent set still need to be supplied before anyone can certify actual
competition readiness. The reusable pieces are now in place to swap the game
adapter and rerun the same paired evaluation when those rules are known.
