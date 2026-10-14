# Strategy v5: unique-state coverage + persistent search labels

## Goal

Add as many *new, strategically varied* Connect Four states as our CPU budget
allows, avoid paying twice to label the same state, then give the model a full
five-minute training window. This is still supervised search distillation,
not policy-gradient RL: depth-limited alpha-beta supplies stable policy/Q/value
targets while we improve coverage and measure regret.

## Data flow

```text
7 trajectory styles
random · center · tactical · depth-2 search · long-game · edge-biased · safe
                         ↓
sample all game phases, cap each trajectory's influence
                         ↓
canonicalize left/right mirrors; key = (board, player-to-move)
                         ↓
exclude base + tactical + earlier curriculum + held-out openings
                         ↓
balanced tactical selection → depth-5 alpha-beta teacher
                         ↓
SQLite label cache → keep every selected unique row in JSONL
                         ↓
deduplicate extras → grouped train/validation/test → CPU time-budget fit
```

The six tactical tags are immediate win, forced block, multiple threats, fork
sequence, support-square risk, and ordinary. Phase quotas and a category-aware
sampler prevent a dataset made only of openings, endings, or one easy motif.
Each selected candidate is unique under horizontal reflection, and no chosen
new state is already present in the excluded datasets.

## Safe caching and pruning

- Identity is `(min(board, mirror(board)), player_to_move)`: a mirrored board
  is the same strategic position, but the side to move is part of the state.
- SQLite caches the expensive teacher label by canonical board, side, search
  depth, node limit, policy temperature, and an explicit teacher-version tag.
  Change the tag whenever the heuristic/scoring recipe changes.
- Model regret and model-selected moves are recomputed; they depend on the
  current checkpoint and must not be read from the teacher cache.
- Exact duplicates are omitted from the training rows. Replaying the same
  examples over multiple epochs is still intentional gradient descent, not
  wasted duplicate-data work.
- Do not prune “similar-looking” but different boards: tactical differences
  such as a support square can change the winning move. Keep motif categories
  for balancing and analysis, not as proof two positions are equivalent.
- Alpha-beta's bound-based pruning and in-memory transposition table remain
  safe search optimizations. The SQLite cache reuses complete labels across
  generator runs; it does not cache gradients or model outputs.
- The base dataset is split by opening group before extras are added. New
  mining excludes its validation/test opening groups to limit leakage.

## Five-minute training contract

`--time-budget-seconds 300` runs complete training epochs until the fit-loop
wall clock has reached at least 300 seconds; the current epoch is completed,
so it can overrun slightly. Validation is included in this fit-loop time;
final test evaluation and the latency benchmark are reported separately.
`optimizer_step_wall_seconds` isolates time spent in training epochs, and the
report also records process CPU seconds, peak RSS, epoch count and validation
curve. With time-budget mode, omit `--epochs` so an epoch cap cannot end the
run early. The checkpoint is a new file; v4b remains untouched.

## Reproduce this run

From the workspace root (generation can take a few minutes before training):

```bash
.venv/bin/python -m connect_four_transformer.test_data_pipeline

.venv/bin/python -m connect_four_transformer.generate_curriculum \
  --candidate-pool 12000 --label-limit 7000 --keep-limit 7000 --depth 5 \
  --node-limit 5000 --seed 2046 \
  --output connect_four_transformer/artifacts/data/curriculum_v5_unique.jsonl \
  --cache connect_four_transformer/artifacts/cache/teacher_labels.sqlite3

.venv/bin/python -m connect_four_transformer.train \
  --dataset connect_four_transformer/artifacts/data/positions_v0_2000.jsonl \
  --extra-train connect_four_transformer/artifacts/data/tactics_v1.jsonl \
  --extra-train connect_four_transformer/artifacts/data/curriculum_v4.jsonl \
  --extra-train connect_four_transformer/artifacts/data/curriculum_v5_unique.jsonl \
  --init-checkpoint connect_four_transformer/artifacts/checkpoints/candidate_tactical_v4b.pt \
  --time-budget-seconds 300 --batch-size 64 --learning-rate 0.0001 \
  --threads 4 --hard-tactical-targets --tactical-weight 3 \
  --selection-metric balanced --log-every 1 \
  --checkpoint connect_four_transformer/artifacts/checkpoints/candidate_tactical_v5_5min.pt \
  --report connect_four_transformer/artifacts/reports/training_tactical_v5_5min.json
```

Then compare against v4b on the same arena and tournament seeds. A longer run is
not automatically a better model: the acceptance signal is lower held-out
teacher regret / better critical decisions without worse play in the arena.

## Results from the five-minute run

The run mined 12,000 candidate states, excluded base/tactical/previous
curriculum states and held-out openings, found 11,333 eligible new states,
then labeled and kept 7,000 unique positions. The kept category counts were
1,145 immediate wins, 1,420 forced blocks, 284 multiple-threat states, 1,088
fork motifs, 1,327 support-square-risk states, and 1,736 ordinary states.
Generation plus depth-5 search labeling took 299.2 seconds; this first pass
filled the persistent SQLite cache (7,000 new labels, zero cache hits).

Training started from v4b, used 11,200 unique total rows (10,600 train, 300
validation, 300 held-out test), and ran 99 epochs: 303.1 seconds fit-loop wall
time, including 300.6 seconds inside training epochs, 446.3 process-CPU seconds,
and 407 MiB peak RSS. Best balanced-validation checkpoint was epoch 95.

| Held-out metric | v4b | v5 | Read |
|---|---:|---:|---|
| Teacher-best move agreement | 49.3% | 58.3% | better |
| Top-2 includes a teacher-best move | 68.3% | 76.0% | better |
| Critical-move accuracy, raw policy | 64.8% | 68.1% | better |
| Mean teacher regret | 0.278 | 0.209 | better |
| Q-value MAE | 0.438 | 0.499 | worse |
| State-value MAE | 0.523 | 0.576 | worse |

On the same 100 generated openings per side (seed 41027), v5 beat the tactical
baseline 141 times with 29 draws across both sides, versus v4b's 99 wins and 6
draws. Against depth-3 alpha-beta, v5 scored 49 wins / 13 draws / 138 losses,
while v4b scored 56 / 12 / 132: nearly the same overall, but v5 is materially
better as O (20 wins vs 12) and weaker as X (29 vs 44 wins). The 300-position
arena also shows an uneven tactical shift: immediate-win critical accuracy
rose 65.4%→75.0%, while forced-block raw accuracy fell 64.1%→59.0%; the
existing one-ply guard still handled all 91 critical positions for both.

So v5 is a real improvement in teacher agreement and versus the tactical
baseline, but not a universal upgrade: first-player play against search needs
work, and Q/value calibration regressed. Keep v4b as the conservative default;
use v5 as the new research checkpoint while we mine first-player losses,
analyze their highest-regret moves, and improve Q/value targets. It remains
CPU-fast at inference (about 0.57 ms median full move in the 300-position arena).

Artifacts: [new unique dataset](artifacts/data/curriculum_v5_unique.jsonl),
[persistent teacher cache](artifacts/cache/teacher_labels.sqlite3),
[v5 checkpoint](artifacts/checkpoints/candidate_tactical_v5_5min.pt),
[training/profile report](artifacts/reports/training_tactical_v5_5min.json),
[arena](artifacts/reports/arena_cpu_tactical_v5_5min.json), and the
[same-seed v4b/v5 100-opening tournament reports](artifacts/reports/tournament_cpu_tactical_v4b_100_seed41027.json) /
[v5 report](artifacts/reports/tournament_cpu_tactical_v5_100_seed41027.json).
