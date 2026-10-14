# Classical vs shared candidate Transformer

## Bottom line

The candidate-action interface works and is more reusable, but it did **not**
beat the classical Tic-Tac-Toe policy across the board. With this training
recipe it helped slightly at medium dataset sizes, matched closely at full
data, cost more CPU, and the extra multi-task losses sometimes distracted the
policy learner.

That negative result is useful: explicit candidates are an interface advantage,
not a free accuracy boost.

## Protocol

- Source data: all 4,520 reachable nonterminal positions labelled by exact Minimax.
- Split: 3,162 train / 678 validation / 680 test.
- Rotations and reflections are kept in one split, preventing symmetry leakage.
- Optimizer: AdamW, learning rate `1e-3`, batch size 128, 30 epochs.
- Repetitions: seeds 13, 42, and 73.
- CPU: Apple arm64, PyTorch 2.7.1, four Torch threads.
- Checkpoint choice: highest validation optimal-move accuracy, then lowest validation policy loss.

“10% data” means 316 unique training positions seen for 30 passes. It therefore
uses fewer examples **and** fewer optimizer steps than the full-data run.

## Held-out test result

Mean optimal-move accuracy across three seeds; `±` is sample standard deviation.

| Training positions | Classical fixed policy | Candidate policy | Shared multi-task |
|---:|---:|---:|---:|
| 316 (10%) | **59.46% ± 0.45%** | 59.12% ± 0.15% | 59.36% ± 0.17% |
| 796 (25%) | 64.02% ± 1.48% | **66.13% ± 3.82%** | 60.64% ± 2.63% |
| 1,588 (50%) | 81.03% ± 5.83% | **81.52% ± 4.06%** | 74.46% ± 4.05% |
| 3,162 (100%) | **93.82% ± 2.14%** | 92.16% ± 3.09% | 92.25% ± 2.10% |

The candidate-policy advantage at 25% and 50% is small relative to seed
variation, so we should call it a promising signal rather than a proven win.
At full data the simpler classical head is 1.67 percentage points better.

## Dangerous mistakes

This is the rate at which a model chose a losing move when a draw or win was
available.

| Training positions | Classical | Candidate | Shared multi-task |
|---:|---:|---:|---:|
| 316 | **27.21%** | 30.93% | 30.59% |
| 796 | 22.79% | **21.62%** | 27.84% |
| 1,588 | 9.71% | **9.66%** | 15.29% |
| 3,162 | **1.86%** | 3.38% | 2.40% |

The full-data shared seed-13 checkpoint was unusually safe at **0.74%** losing
blunders and learned 99.95% legality, 77.88% legal-action outcome accuracy, and
76.62% board-value accuracy. Across all three seeds, however, classical was
still safer on average.

## CPU cost

| Model | Parameters | Batch-1 median | Batch-128 median | Mean full-data training time |
|---|---:|---:|---:|---:|
| Classical | 17,601 | **0.261 ms** | **2.870 ms** | **10.34 s** |
| Candidate policy | 27,752 | 0.374 ms | 3.418 ms | 12.81 s |
| Shared multi-task | 27,752 | 0.376 ms | 3.679 ms | 12.91 s |

The candidate model has 57.7% more parameters, is about 43% slower at batch 1,
and about 19% slower at batch 128. All are still far below one millisecond per
single Tic-Tac-Toe decision on this CPU.

## Whole-game behavior

These are seed-13 checkpoints. Random matchups contain 1,000 games per side;
Minimax matchups contain one deterministic game per side.

| Model | As X vs random | As O vs random | As X vs Minimax | As O vs Minimax |
|---|---:|---:|---:|---:|
| Classical | 990 W / 0 L / 10 D | 824 W / 24 L / 152 D | draw | draw |
| Candidate policy | 996 W / 0 L / 4 D | 802 W / 29 L / 169 D | draw | **loss** |
| Shared multi-task | 980 W / 0 L / 20 D | 820 W / 8 L / 172 D | draw | draw |

A perfect Tic-Tac-Toe agent cannot always win against perfect play; the best
guarantee is never losing, because Minimax versus Minimax is a draw.

## What we learned

1. **Candidate scoring is valid:** a variable legal-action list can replace a game-specific output layer.
2. **It is not automatically more accurate:** the classical square head is an excellent inductive bias for a fixed 3×3 game.
3. **Auxiliary tasks really learned:** legality and values became meaningful, but their gradients competed with policy learning.
4. **The architecture pays off later:** variable actions and `Q/V` outputs matter more for Connect Four, Othello, chess, and search than for nine fixed squares.
5. **The next training experiment should be staged:** pretrain `Q/V/legality`, decay their weights, then finish with policy-only training instead of optimizing every objective equally for all 30 epochs.

## Reproduce or inspect

- Raw measurements: [`artifacts/reports/comparison.json`](artifacts/reports/comparison.json)
- Matchups: [`artifacts/reports/tournaments.json`](artifacts/reports/tournaments.json)
- Model implementation: [`model.py`](model.py)
- Experiment harness: [`experiment.py`](experiment.py)
- Interactive game: [`play.py`](play.py)
