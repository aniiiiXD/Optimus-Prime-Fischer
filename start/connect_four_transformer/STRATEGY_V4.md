# Strategy v4: search-guided hard-example replay

## Why this experiment

The current agent has a one-ply safety guard, so it can avoid a move that gives
the opponent an immediate win, but it still loses to setups that require looking
two or more moves ahead. Repeating self-play with the same weak policy risks
repeating the same bad habits. This experiment asks the existing v3 policy where
it is wrong, then teaches those positions with a search teacher.

This is supervised search distillation, not policy-gradient RL: on a small CPU
and with a still-weak policy, the teacher gives lower-variance action targets
than a sparse win/loss reward. Self-play can come later, once we can reliably
measure and improve a baseline.

## Data and labels

1. Generate varied legal games from random, center-biased, tactical and shallow
   search policies.
2. Keep new states only from training opening groups; omit every base board and
   old tactical board.
3. Categorize states as immediate win, forced block, multiple threats, a
   two-ply fork sequence, support-square risk, or ordinary play.
4. Score every legal action with depth-limited alpha-beta; query v3 and measure
   its teacher regret: `best teacher value - value of v3's top move`.
5. Keep a category-balanced mix of high-regret failures and varied examples.
   Mirror augmentation teaches left/right symmetry.

The fork label means a line can leave the mover two immediate winning choices
after one opponent reply; it is a motif label, not a claim that the line is
forced. Alpha-beta action values are the actual decision targets.

## Training

- Start from a copy of the v3 weights; v3 itself is never overwritten.
- Replay the original training data and v1 tactical examples alongside the new
  mined examples, rather than fine-tuning only on failures.
- Optimize policy cross-entropy plus search-Q and state-value regression.
- Upweight exact wins/blocks and give high-regret replay examples a modest
  policy weight; do not increase model size or add a new inference dependency.
- Select a checkpoint with a held-out composite: 45% critical-move accuracy,
  35% overall teacher-best agreement, and 20% normalized teacher-regret score.

## Evaluation and success criteria

Compare v4 and v3 on the exact same grouped test boards, split results by the
six categories, replay the saved vertical/diagonal human-game failure fixtures,
and report model-forward latency and CPU/memory use. A v4 checkpoint is a
candidate improvement only if it improves the tactical/fixture results without
a material regression in overall move quality; winning more games against a
search opponent remains the final test. No results are presumed in advance.

## Reproduce

From the workspace root:

```bash
.venv/bin/python -m connect_four_transformer.generate_curriculum \
  --candidate-pool 8000 --label-limit 2400 --keep-limit 1400 --depth 5

.venv/bin/python -m connect_four_transformer.train \
  --dataset connect_four_transformer/artifacts/data/positions_v0_2000.jsonl \
  --extra-train connect_four_transformer/artifacts/data/tactics_v1.jsonl \
  --extra-train connect_four_transformer/artifacts/data/curriculum_v4.jsonl \
  --init-checkpoint connect_four_transformer/artifacts/checkpoints/candidate_tactical_v3.pt \
  --hard-tactical-targets --tactical-weight 3 --selection-metric balanced \
  --epochs 100 --batch-size 64 --learning-rate 0.0001 --threads 4 \
  --checkpoint connect_four_transformer/artifacts/checkpoints/candidate_tactical_v4b.pt \
  --report connect_four_transformer/artifacts/reports/training_tactical_v4b.json

.venv/bin/python -m connect_four_transformer.arena_cpu \
  --checkpoint connect_four_transformer/artifacts/checkpoints/candidate_tactical_v4b.pt \
  --report connect_four_transformer/artifacts/reports/arena_cpu_tactical_v4b_guarded.json

.venv/bin/python -m connect_four_transformer.tournament \
  --checkpoint connect_four_transformer/artifacts/checkpoints/candidate_tactical_v4b.pt \
  --games-per-side 10 --seed 2026 \
  --report connect_four_transformer/artifacts/reports/tournament_cpu_tactical_v4b_10.json
```

## First corrected run

The first v4 run exposed a reporting bug: critical columns are real board
columns, while model outputs are packed by candidate slot. The training
metric compared those two index systems directly. `data.py` now packs the
critical mask in candidate order; the first run's checkpoint/report remain as
history, and **v4b is the corrected, recommended checkpoint**.

On the unchanged 300-position test split, v4b improves over v3's raw policy:

| Metric | v3 | v4b |
|---|---:|---:|
| Teacher-best move agreement | 39.0% | 49.3% |
| Mean teacher regret (lower is better) | 0.354 | 0.278 |
| Immediate forced-block accuracy | 51.3% | 64.1% |
| Immediate win accuracy | 51.9% | 65.4% |
| Critical moves after the tactical guard | 100% | 100% |

The existing diagonal-miss replay remains a raw-policy miss, but the tactical
guard changes it to the correct blocking column for both v3 and v4b. So the
learned policy improved, while the rules-based safety check still supplies the
hard guarantee for these one-ply wins and blocks. The two-ply fork training
examples have not yet been validated as a separate held-out category.

Generation took 86.1 seconds. Corrected v4b fine-tuning took 102.2 seconds wall
time (152.0 process-CPU seconds), at 4 threads and 376 MiB peak RSS. The 100
epochs selected epoch 91; no GPU was used. The arena forward pass remained about
0.26 ms median on this CPU, so this run targeted decision quality rather than
inference speed.

On the same ten generated openings per side, the game tournament changed from
v3 to v4b as follows (W-D-L):

| Opponent | Side | v3 | v4b |
|---|---|---:|---:|
| Tactical | X | 6-2-2 | 6-1-3 |
| Tactical | O | 1-4-5 | 4-0-6 |
| Depth-3 alpha-beta | X | 3-0-7 | 7-0-3 |
| Depth-3 alpha-beta | O | 0-0-10 | 0-2-8 |

Both versions beat random 10/10 from either side. Ten games is a tiny sample,
but it shows a real signal against depth-3 search when playing X and a modest
draw gain as O; it does **not** eliminate the first-player/second-player
imbalance or establish strong play.

Reports: [v4b training](artifacts/reports/training_tactical_v4b.json),
[v4b raw/guarded arena](artifacts/reports/arena_cpu_tactical_v4b_guarded.json),
[v3 tournament](artifacts/reports/tournament_cpu_tactical_v3_10.json), and
[v4b tournament](artifacts/reports/tournament_cpu_tactical_v4b_10.json).
