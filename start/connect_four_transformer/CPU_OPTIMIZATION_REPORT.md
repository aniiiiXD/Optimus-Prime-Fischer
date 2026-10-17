# Connect Four CPU experiment — 15 October 2026

## Decision

Keep the **Pons phase-v3 checkpoint + depth-5 Transformer-guided alpha-beta**
as the current candidate. Promote the search hotpath optimization: it makes
the same decisions with fewer Python allocations. Do **not** promote any of
the three new weights yet; all trail v3 on the independent, lineage-filtered
Pons holdout. The new exact-label dataset is useful experimental material,
not proof that the player has improved.

## What we changed and measured

| Workload | Change | Measured result | Decision |
| --- | --- | --- | --- |
| Live search | Precomputed 69 four-cell windows; winning-drop checks no longer call `step/undo` | 46.6 → 18.9 ms median, 299.1 → 121.6 ms P95 on 16 fixed boards; **2.46× median speedup**; identical moves, reasons, depth and nodes | Keep |
| Single-position inference | PyTorch intra-op thread sweep | Batch 1: 0.296 ms with 1 thread; 0.364 ms with 2; 0.389 ms with 4 | Use 1 thread for interactive inference |
| Batched inference | Same model, batch 128 | 11.8k positions/s with 1 thread; 20.0k/s with 4 | Batch and use 4 threads for offline scoring |
| Optimizer step | Batch 128, policy/Q/value losses + AdamW | 3.49k positions/s with 1 thread; 4.28k/s with 4 | Use 4 threads for this training batch; gain is only 23% |
| Pons dataset | Mirror deduplication, phase interleaving, persistent SQLite labels and batched commits | 4,000 unique exact-label positions: 1,394 early, 1,396 middle, 1,210 late; 13.0 s labeling, 3 timeouts | Keep as supplementary data |
| Pons workers | 1 vs 4 solver processes on the same 1,000-position pilot | 2.879 vs 2.934 s | Default to 1; parallel startup/IPC loses on these cheap queries |

Search timing is a **whole hybrid decision**, not a neural-only forward pass.
An independent second 16-board seed (952) showed **28.1 → 12.9 ms median,
2.18× faster**, again with identical moves and nodes. These are small fixed
samples, not a device-wide performance guarantee.
The two timing tables use different workloads and should not be compared as
though they were the same measurement. The search benchmark uses 16 seeded
nonterminal boards (seed 951), depth 5, 5,000-node cap, one Torch thread,
the same v3 weights, and one root neural evaluation per decision. It compares
the old Python functions with their optimized equivalents in a standalone
process. The single-position and batch figures are fixed-tensor kernel
microbenchmarks; they exclude data loading and game simulation. These are
measurements on this arm64 Mac (model `Mac17,3`, PyTorch 2.7.1), not portable
hardware promises.

## Model-size and extra-data experiment

Both fresh models used the **same 5,167-position filtered corpus**, split into
4,478 train / 345 validation / 344 internal test; the external holdout's
board and opening groups were excluded from every current training input.
Each fit ran for at least 90 s on CPU with batch 128, four Torch threads,
learning rate `5e-4`, seed 5021, and the same tactical target/selection
settings. A third run continued v3 for at least 60 s at `5e-5`.

| Weights | Parameters | Fit time | Peak RSS | Internal best move | External best move | External regret | External raw critical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing Pons v3 | 114,819 | earlier 5 min run | — | — | **65.4%** | **0.209** | **77.3%** |
| Fresh width 64, 2 blocks | 114,819 | 90.3 s | 423 MiB | 53.5% | 57.1% | 0.293 | 56.5% |
| Fresh width 96, 3 blocks | 329,955 | 91.0 s | 539 MiB | 58.7% | 53.9% | 0.308 | 53.9% |
| v3 + balanced-data fine-tune | 114,819 | 60.3 s | 420 MiB | 64.2% | 63.8% | 0.213 | 76.0% |

External numbers use **the same 492 Pons-labeled positions** for all four
weights. From a 1,000-position holdout, 508 positions were removed because
their opening group occurs in **any known v3 ancestor training dataset**:
base search, tactical, v4/v5 curriculum, and Pons v1/v2/v3. The new 4k set
was excluded from the new runs wherever it shared an external holdout board
or opening group. The guard scored 100% on this set's measured one-ply
critical positions for all models; that demonstrates the guard works here,
not that the neural policies are equivalent or that deeper tactics are solved.

The 96-wide model did better than the 64-wide model on their internal test,
but worse on the independent holdout. More parameters or more examples do not
automatically improve generalization, especially under a short fit budget.
The v3 checkpoint also inherits earlier training runs, so this is a promotion
comparison, **not an equal-total-compute architecture contest**. The fine-tune
also failed to beat v3. No full-game tournament was needed to
reject them as promotions on this gate; future candidates still need paired
games before being called stronger players.

## CPU strategy and next gate

- **Python search:** keep precomputed windows and allocation-free tactical
  probes. The randomized differential tests compare the optimized probes
  against the rules engine and the heuristic against a reference loop.
- **SIMD:** matrix multiplication already goes through PyTorch's CPU vector
  kernels; a hand-written NEON dot-product demo elsewhere in the workspace is
  *not* wired into Connect Four. The branchy seven-action tactical code did
  not warrant a custom native SIMD boundary yet. Try bitboards/native kernels
  only if a profile shows search still dominates after this 2.46× gain.
- **Parallelism:** do not use more threads for one tiny forward pass. Batch
  offline inference and training. Pons worker count is configurable, but the
  tested short queries favor one long-lived process and persistent cache.
- **Context:** the current 42 board tokens plus side-to-move encode the full
  Connect Four state. Increasing width is model capacity, **not** longer game
  history. Additional history tokens need a separate hypothesis and test.
- **Promotion gate:** independent teacher agreement/regret and critical
  accuracy first, then paired full games with colors swapped, then latency and
  memory. A local validation gain alone is insufficient.

## Reproduce from `start/`

```bash
.venv/bin/python -m unittest connect_four_transformer.test_hybrid_search connect_four_transformer.test_data_pipeline connect_four_transformer.test_tactical_categories connect_four_distillation.test_oracle connect_four_distillation.test_generate_dataset

.venv/bin/python -m connect_four_transformer.cpu_search_bench \
  --output connect_four_transformer/artifacts/reports/cpu_search_hotpath_v1.json

.venv/bin/python -m connect_four_transformer.cpu_sweep \
  --threads 1,2,4 --batch-sizes 1,32,128 --repeats 12 \
  --output connect_four_transformer/artifacts/reports/cpu_sweep_v3.json

.venv/bin/python -m connect_four_distillation.generate_dataset \
  --positions 4000 --candidate-pool 8000 --min-ply 12 --max-ply 40 \
  --seed 4109 --timeout 0.2 --workers 1 --phase-balanced \
  --output connect_four_distillation/artifacts/data/pons_balanced_cpu_v1_4000.jsonl
```

The three training configurations and complete raw comparison are in the saved
JSON reports below. Data and solver cache are ignored local artifacts; the
checkpoint and report names distinguish each experiment. Do not overwrite
the v3 champion checkpoint with a failed candidate.

## Artifacts

- Search comparison: `artifacts/reports/cpu_search_hotpath_v1.json`
- Second search seed: `artifacts/reports/cpu_search_hotpath_seed952.json`
- CPU thread/batch sweep: `artifacts/reports/cpu_sweep_v3.json`
- External four-model holdout: `../connect_four_distillation/artifacts/reports/benchmark_balanced_cpu_all_full_lineage_holdout.json`
- New exact data: `../connect_four_distillation/artifacts/data/pons_balanced_cpu_v1_4000.jsonl` (local, ignored)
- Fresh model reports: `../connect_four_distillation/artifacts/reports/training_balanced_cpu_64_90s.json` and `training_balanced_cpu_96_90s.json`
- Fine-tune report: `../connect_four_distillation/artifacts/reports/training_balanced_cpu_v3_finetune_60s.json`
- New weights: `../connect_four_distillation/artifacts/checkpoints/candidate_balanced_cpu_{64_90s,96_90s,v3_finetune_60s}.pt`
