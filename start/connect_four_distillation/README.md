# Strong CPU teacher experiment

This folder isolates a stronger public Connect Four teacher from our existing
`connect_four_transformer/` training runs. No transformer weights have been
changed and no large training job has been started.

## What is downloaded

- `vendor/connect4/`: Pascal Pons's [Connect 4 Solver](https://github.com/PascalPons/connect4), pinned at commit `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`. This independent Git checkout is **not included** in the Hyde repository. Its C++ source is AGPL-3.0 licensed; see the upstream `LICENSE` after cloning.
- `vendor/connect4/c4solver`: a locally compiled CPU executable. The repository does **not** contain an opening book; none has been downloaded. This matters because exact scoring of opening positions can be very slow.

Measured on this machine after compiling: the vendor checkout and executable
together occupy about **404 KB** (`du -sh`). No pretrained model or dataset is
included in that figure. A future labeled dataset will grow with the number of
positions we choose to save.

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

## Distillation experiment (not run yet)

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

Do not equate the solver's exact scores with the existing depth-5 teacher's
heuristic scores without a documented target conversion. Do not call a model
perfect merely because its teacher is perfect on labeled positions.
