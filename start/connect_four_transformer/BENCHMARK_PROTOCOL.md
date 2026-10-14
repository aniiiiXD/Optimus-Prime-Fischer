# Checkpoint benchmark protocol

The project now has a single repeatable comparison runner:

```bash
.venv/bin/python -m connect_four_transformer.benchmark_models \
  --games-per-side 30 --split-seed 2026 --opening-seed 41027 \
  --report connect_four_transformer/artifacts/reports/model_benchmark_v1.json
```

By default it loads every `.pt` file in `artifacts/checkpoints/`. Each model is
measured on the same grouped 300-position test split and the same set of
openings, then (unless `--skip-tournament`) played as X and O against random,
tactical, and depth-3 alpha-beta opponents. `--checkpoint NAME=PATH` can select
and name a smaller set; repeat that option for each checkpoint.

## What it reports

- **Teacher-best rate:** how often the raw Transformer picks a depth-5
  alpha-beta best move on held-out positions.
- **Teacher regret:** teacher value of its best move minus the value of the
  model's raw pick; lower is better.
- **Critical-move rate:** fraction of immediate wins/forced blocks found by the
  raw policy, and separately after the one-ply safety guard.
- **Tournament W/D/L and score rate:** game results split by opponent and side;
  score rate is `(wins + 0.5 × draws) / games`.
- **Latency:** forward-pass and full guarded decision median/p95.
- **Weight fingerprint:** SHA-256 over the model tensors, to reveal accidental
  duplicate checkpoint weights. The report also includes the training metadata
  saved inside each checkpoint.

All checkpoints are evaluated sequentially on CPU to avoid cross-model timing
interference. The report's peak RSS is for the benchmark process as a whole,
not a per-model allocation measurement. Use the same seeds when comparing later
experiments. Treat position metrics, tactical fixtures, and games together:
one score alone can hide a failure such as strong teacher agreement but weak
play as one side.

## Existing isolated weights

The weights are separate `.pt` files, not one model overwritten each run:

| Checkpoint | Training history | What it represents |
|---|---|---|
| `candidate_v0.pt` | `training_v0.json` | Original broad search-supervised baseline |
| `candidate_tactical_v1.pt` | `training_tactical_v1.json` | First tactical-data experiment |
| `candidate_tactical_v2.pt` | `training_tactical_v2.json` | Tactical-only follow-up |
| `candidate_tactical_v3.pt` | `training_tactical_v3.json` | Tactical data with critical-move selection |
| `candidate_tactical_v4.pt` | `training_tactical_v4.json` | Earlier search-guided curriculum run |
| `candidate_tactical_v4b.pt` | `training_tactical_v4b.json` | Corrected v4 run; conservative reference |
| `candidate_tactical_v5_5min.pt` | `training_tactical_v5_5min.json` | Unique-state curriculum + five-minute fit |

v1–v3 are separate from-scratch runs; v4/v4b are independently saved
fine-tunes from v3; v5 is a separate fine-tune from v4b. The benchmark computes
the tensor hashes rather than assuming different filenames mean different
weights. Current working recommendation: keep v4b as the conservative
reference, and compare v5 as an experimental candidate until its search-opponent
side imbalance improves.

## First all-checkpoint comparison

Run on the CPU with seed 2026 for the held-out split and seed 41027 for 30
common openings per side. No two checkpoints had the same tensor hash. Each
W-D-L entry below is 30 games, with the one-ply safety guard enabled for
tournament play.

| Checkpoint | Teacher-best | Regret ↓ | Critical raw | Tactical X / O | Depth-3 X / O |
|---|---:|---:|---:|---:|---:|
| v0 | 43.0% | 0.339 | 35.2% | 24-1-5 / 17-5-8 | 3-3-24 / 4-1-25 |
| tactical v1 | 38.7% | 0.397 | 38.5% | 20-4-6 / 11-2-17 | 13-0-17 / 16-0-14 |
| tactical v2 | 38.7% | 0.385 | 34.1% | 16-4-10 / 8-0-22 | 3-0-27 / 4-0-26 |
| tactical v3 | 39.0% | 0.354 | 51.6% | 17-8-5 / 5-13-12 | 3-0-27 / 1-0-29 |
| tactical v4 | 48.0% | 0.294 | 68.1% | 27-0-3 / 15-1-14 | 15-2-13 / 3-9-18 |
| tactical v4b | 49.3% | 0.278 | 64.8% | 21-0-9 / 14-1-15 | 14-1-15 / 3-3-24 |
| tactical v5 (5 min) | **58.3%** | **0.209** | 68.1% | 27-0-3 / 15-9-6 | 9-1-20 / 7-2-21 |

v5 leads on teacher agreement/regret and performed strongly against the
tactical baseline in the separate 100-opening comparison. It does not dominate
the depth-3 results in every seed/side combination; keep the side split visible
and avoid picking a winner from a single headline metric. The complete report
contains all per-category position metrics, both sides against all opponents,
latency percentiles, per-checkpoint metadata, and weight fingerprints:
[`model_benchmark_v1.json`](artifacts/reports/model_benchmark_v1.json).
