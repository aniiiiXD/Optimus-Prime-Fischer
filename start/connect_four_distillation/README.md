# Strong CPU teacher experiment

This folder isolates a stronger public Connect Four teacher from our existing
`connect_four_transformer/` training runs. It now contains isolated v1–v3
distillation checkpoints and reports; the original v5 checkpoint is unchanged.

## What is downloaded

- `vendor/connect4/`: Pascal Pons's [Connect 4 Solver](https://github.com/PascalPons/connect4), pinned at commit `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`. This independent Git checkout is **not included** in the Hyde repository. Its C++ source is AGPL-3.0 licensed; see the upstream `LICENSE` after cloning.
- `vendor/connect4/c4solver`: a locally compiled CPU executable. The repository does **not** contain an opening book; none has been downloaded. This matters because exact scoring of opening positions can be very slow.

Measured on this machine after compiling: the vendor checkout and executable
together occupy about **404 KB** (`du -sh`). That excludes the local datasets,
reports, and Transformer checkpoints described below.

## Build and verify

From the workspace root:

```bash
git clone https://github.com/PascalPons/connect4.git connect_four_distillation/vendor/connect4
git -C connect_four_distillation/vendor/connect4 checkout d6ba50d8aaf2308c769d9bf2abd42d90f34baf41
make -C connect_four_distillation/vendor/connect4 c4solver
.venv/bin/python -m unittest connect_four_distillation.test_oracle
.venv/bin/python -m connect_four_distillation.oracle 135567125554756746421621316174
```

The move string uses **human columns 1–7**, in move order. The adapter checks
the sequence with our own `connect_four_engine`, sends it to the solver, and
checks that both engines agree on which columns are legal. The returned score
is from the player-to-move's perspective. Higher is better: positive means a
forced win, zero a draw, negative a forced loss. With `--weak`, the scores are
only win/draw/loss; without it, score magnitude also ranks faster wins and
slower losses. `-1000` means an illegal/full column.

`PonsOracle` keeps one solver subprocess alive across queries, reusing its
transposition table. Each query has a timeout; without an opening book, start
with middle/late-game positions. A timeout does **not** mean the position is
unwinnable—it only means exact analysis exceeded our CPU budget.

## Initial distillation result: late-game only

1. Generate varied legal games and retain the **move histories**, not just
   board snapshots; this solver's CLI accepts move histories.
2. Query exact scores for late/midgame positions under a measured time budget.
   Cache by board and player, and record solver version, score mode, and time.
3. Create train/validation/test splits by opening/trajectory before training.
   Never relabel validation/test positions into training through transpositions.
4. Train a *new* transformer checkpoint on best-move policy and action-value
   targets; keep v4b and v5 untouched.
5. Compare held-out teacher agreement, regret, full-game win rate against the
   solver and other baselines, and CPU latency. Inspect fork failures separately.

## Generate exact teacher labels

`generate_dataset.py` samples mixed-policy trajectories, canonicalizes mirrored
boards, queries Pons once per unique position, and writes rows in the schema
already consumed by `connect_four_transformer.train`. Exact labels are cached
in SQLite by board, side to move, and solver revision; a timed-out query is
skipped and the solver process is restarted for the next position.

Start with a small late-game pilot because exact opening analysis can be slow:

```bash
.venv/bin/python -m connect_four_distillation.generate_dataset \
  --positions 64 --candidate-pool 256 --min-ply 32 --max-ply 40 \
  --timeout 1 --seed 2047 \
  --output connect_four_distillation/artifacts/data/pons_exact_pilot.jsonl
```

After inspecting the report and examples, train a **new** checkpoint. The model
uses the same policy/Q/value objectives as before, but its labels now come from
the exact solver:

```bash
.venv/bin/python -m connect_four_transformer.train \
  --dataset connect_four_distillation/artifacts/data/pons_exact_pilot.jsonl \
  --init-checkpoint connect_four_transformer/artifacts/checkpoints/candidate_tactical_v5_5min.pt \
  --time-budget-seconds 300 --batch-size 64 --learning-rate 0.0001 --threads 4 \
  --selection-metric balanced \
  --checkpoint connect_four_distillation/artifacts/checkpoints/pons_distilled_pilot.pt \
  --report connect_four_distillation/artifacts/reports/pons_distilled_pilot.json
```

The generated data/cache are local ignored artifacts. Reports and the resulting
small checkpoint can be kept with the project. The 64-position command above
checks the pipeline; the first measured run used 2,000 positions.

For the later CPU optimization study, `generate_dataset.py` also accepts
`--phase-balanced` and `--workers N`. The phase-balanced 4,000-position run
is documented in [the CPU optimization report](../connect_four_transformer/CPU_OPTIMIZATION_REPORT.md).
The tested 1,000-position pilot was not faster with four solver workers than
with one; use the default single persistent process for these short queries.

### Run v1

- Sampled 5,000 candidates from mixed-policy games at plies 28–40; kept 2,000
  unique states after mirror canonicalization.
- Pons solved all 2,000 within the one-second per-query limit: zero timeouts,
  median query about 0.11 ms. These are late/endgame states, not opening states.
- Fine-tuned from v5 for a 300-second fit budget; the trainer selected epoch 12
  from 759 epochs because validation began to worsen. The new weights use about
  360 MiB peak process memory.

| Measure on the same held-out Pons positions/openings | v5 | Pons-distilled v1 |
| --- | ---: | ---: |
| Exact teacher best move | 74.3% | 76.0% |
| Mean teacher regret | 0.127 | 0.129 |
| Raw immediate win/block accuracy | 76.0% | 67.7% |
| Guarded immediate win/block accuracy | 99.0% | 95.8% |
| Against depth-3 search as X, 30 games | 5–2–23 | 10–0–20 |
| Against depth-3 search as O, 30 games | 11–1–18 | 6–2–22 |

Across both colors, that is a 29.2% score rate for v5 and 28.3% for Pons-v1
(win = 1, draw = 0.5). The small tournament is noisy, but it does not show an
overall strength gain.

This first distillation changed play but did **not** improve overall strength:
it helped as X against search and regressed as O; teacher agreement improved a
little while mean regret got slightly worse. It also reduced raw tactical
accuracy. That result motivated the phase-spread experiments below; keep v5 as
the general baseline and treat `pons_distilled_v1_5min` as an endgame study.

## Phase-spread distillation: v2 and v3

The follow-up kept v5's broad replay data and added exact Pons labels from
multiple game phases. The 2,400-position v2 set spans plies 16–40 and labeled
in 5.4 seconds with no solver timeouts. The 3,000-position v3 set spans plies
12–40; 168 difficult queries exceeded the 50 ms cap and were skipped. The
solver cache avoids paying again for repeated canonical positions.

The trainer now rejects extra training examples whose six-move opening group
is in the base validation or test split. The benchmark can also remove every
opening group present in the training corpora before scoring an external test
set. For this comparison, that left **492 of 1,000** Pons-labeled holdout
positions; all three checkpoints saw exactly those same positions and game
openings.

| Checkpoint | Pons best-move accuracy | Mean Pons regret | Raw / guarded win-block accuracy | vs tactical bot, X / O | vs depth-3 search, X / O |
| --- | ---: | ---: | ---: | ---: | ---: |
| v5 baseline | 62.4% | 0.232 | 74.8% / 98.2% | 85–0–15 / 61–26–13 | 23–5–72 / 32–6–62 |
| Pons phase v2 | 64.2% | 0.219 | 72.4% / 98.2% | 82–1–17 / 88–0–12 | 19–7–74 / 34–4–62 |
| Pons phase v3 | 65.4% | 0.209 | 76.1% / 97.5% | 82–0–18 / 89–1–10 | 17–5–78 / 26–11–63 |

Game records are `wins-draws-losses`, 100 games per color against each bot,
using the same opening seed (43027). Distillation clearly improves exact
teacher agreement and the tactical-bot result, but **does not improve results
against depth-3 search in this sample**. v3 is best at imitating exact Pons
moves; v2 and v3 are both useful tactical-play candidates, with their tactical
score rates nearly tied (85.3% and 85.8%). Keep v5 as the current choice against
the search opponent (30.3% score rate versus v2's 29.3% and v3's 25.5% in this
seed). These are small experiments, not a claim of solved or tournament-winning
play.

The Transformer architecture remains small and unchanged: width 64, four heads,
two blocks, 114,819 parameters. v2 fine-tuned from v5 for 301 seconds at
learning rate `1e-4` and used about 430 MiB peak RSS. v3 fine-tuned from v2 for
303 seconds at `5e-5` and used about 436 MiB peak RSS. This is supervised
distillation, **not RL**.

Artifacts:

- v2 checkpoint: `artifacts/checkpoints/candidate_pons_phases_v2_5min.pt`
- v3 checkpoint: `artifacts/checkpoints/candidate_pons_phases_v3_5min.pt`
- v2 training report: `artifacts/reports/training_pons_phases_v2_5min.json`
- v3 training report: `artifacts/reports/training_pons_phases_v3_5min.json`
- paired v2 benchmark: `artifacts/reports/benchmark_pons_phases_v2_100.json`
- paired v3 benchmark: `artifacts/reports/benchmark_pons_phases_v3_100.json`

Play against v2 (the tactical-play candidate) from `start/`:

```bash
.venv/bin/python -m connect_four_transformer.play --human X \
  --checkpoint connect_four_distillation/artifacts/checkpoints/candidate_pons_phases_v2_5min.pt
```

Artifacts on this machine:

- Checkpoint: `artifacts/checkpoints/pons_distilled_v1_5min.pt`
- Training report: `artifacts/reports/pons_distilled_v1_5min.json`
- Paired benchmark: `artifacts/reports/benchmark_pons_v1_vs_v5.json`
- Generated rows and solver cache are ignored local files under `artifacts/data/`
  and `artifacts/cache/`.

To play against the experimental checkpoint from `start/`:

```bash
.venv/bin/python -m connect_four_transformer.play --human X \
  --checkpoint connect_four_distillation/artifacts/checkpoints/pons_distilled_v1_5min.pt
```

Do not equate the solver's exact scores with the existing depth-5 teacher's
heuristic scores without a documented target conversion. Do not call a model
perfect merely because its teacher is perfect on labeled positions.
