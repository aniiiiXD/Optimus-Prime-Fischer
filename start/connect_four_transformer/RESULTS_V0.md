# Connect Four Transformer v0 results

## Dataset

| Measurement | Result |
|---|---:|
| Search-labelled positions | 2,000 |
| Source games | 489 |
| Distinct opening groups | 389 |
| Opening / early-mid / late-mid / late | 400 / 600 / 700 / 300 |
| Random / center / tactical / shallow-search / long-game sources | 434 / 329 / 492 / 567 / 178 |
| Teacher | alpha-beta, completed depth 4 for every position |
| Mean searched nodes per position | 503 |
| Median label time | 14.02 ms |
| p95 label time | 41.18 ms |
| End-to-end labeling throughput | 58.0 positions/s |
| Dataset size | 1.8 MiB JSONL |

The dataset is deduplicated by board plus side-to-move under horizontal
reflection. Training rows are mirrored randomly with probability 0.5; policy,
Q, candidates, and landing rows are reflected together. Opening groups are
kept inside one split.

## Model and training

| Measurement | Result |
|---|---:|
| Train / validation / test | 1,400 / 300 / 300 |
| Transformer | 2 blocks, width 64, 4 heads, MLP width 128 |
| Parameters | 114,819 |
| FP32 checkpoint size | 468 KiB |
| Epochs / selected epoch | 30 / 28 |
| Training wall time | 12.86 s |
| Process CPU time | 19.90 s |
| Peak process RSS | 333.7 MiB |

## Held-out position quality

| Metric | Result |
|---|---:|
| Best searched move accuracy | 43.0% |
| Best move in top two | 67.3% |
| Mean teacher regret | 0.339 |
| Legal-action Q MAE | 0.445 |
| State-value MAE | 0.520 |

Horizontal mirror augmentation improved the earlier unaugmented run from
40.7% to 43.0% best-move accuracy, from 60.3% to 67.3% top-two recall, and
reduced mean teacher regret from 0.396 to 0.339.

## CPU inference

| Batch | Median | p95 | Throughput at median |
|---:|---:|---:|---:|
| 1 | 0.387 ms | 0.438 ms | 2,584 positions/s |
| 64 | 3.606 ms | 4.079 ms | 17,746 positions/s |

## Tournament

One hundred games per side, using the same fixed set of random 0–4 move
opening prefixes for each opponent.

| Opponent | Model side | Win | Draw | Loss |
|---|---:|---:|---:|---:|
| Random | X | 91 | 0 | 9 |
| Random | O | 87 | 0 | 13 |
| Tactical win/block/center | X | 10 | 1 | 89 |
| Tactical win/block/center | O | 7 | 1 | 92 |
| Alpha-beta depth 3 | X | 5 | 0 | 95 |
| Alpha-beta depth 3 | O | 1 | 3 | 96 |

## Diagnosis

The infrastructure is fast; model inference is already cheaper than the
depth-3 search opponent. Playing strength is the problem. The 2,000-state,
depth-4 dataset contains broad positions but too little repeated tactical
supervision for the model to learn forced wins and blocks reliably. The value
targets are also shallow heuristic estimates, which explains the large Q and
state-value errors.

The next controlled run should add a tactical curriculum: generate immediate
wins, forced blocks, double threats and support-square traps exhaustively from
sampled legal states; label those at depth 6 or to terminal; mix them with the
existing broad dataset; then use hard-example replay on tournament losses.
Only after that comparison should model width or depth increase.

Raw reports:

- [`artifacts/reports/training_v0.json`](artifacts/reports/training_v0.json)
- [`artifacts/reports/tournament_v0.json`](artifacts/reports/tournament_v0.json)
