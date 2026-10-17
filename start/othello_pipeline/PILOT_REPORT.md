# Othello teacher → Transformer: first CPU pilot

**15 October 2026. Status: pipeline works; playing strength is not yet good.**

## Reproducibility and checks

- Neural Reversi engine `v6.3.0` at commit `2d5e4306`, with the two `e6bbc4f6` weight files; local SHA-256 matched `teacher_manifest.json`. Built native arm64 CLI in ignored `vendor/`.
- Live GTP opening returned one of our four legal moves. Batched `solve --all-moves` returned all four opening actions. A forced-pass case agreed, and all four exact late-game move scores matched an independent tiny Python solver, including sign.
- 600 **unique D4-canonical** positions sampled from 140 random legal games: 200 early, 200 middle, 200 late; 136 canonical six-ply opening groups. Teacher level 8, one teacher thread, 64 MB hash. Labeling took **30.5 s**, with 12 persistent-cache hits.
- Same split for both students: 472 train / 128 validation positions; no canonical overlap removed from validation. The input dataset's SHA-256 is `594b0c2ff81d04867a9dcd2919f4da6d30da7979c83f0f57437affb90c3dbc3f`.

## What the network learned

Architecture: 64 relative board tokens + state token, two bidirectional attention blocks, width 64, four heads, shared policy/Q/legality heads, state value head; **110,212 parameters**. Five FP32 AdamW epochs, batch 64, two Torch threads. The dataset has **no outcome labels**, so the value head is not trained.

| Checkpoint | Fit time | Held-out best-score move rate | Mean teacher regret | Raw legal recall | Raw illegal false-positive rate |
|---|---:|---:|---:|---:|---:|
| Legal BCE weight 1 | 2.7 s | **27.3%** | **6.18** | 0.0% | 0.0% |
| Legal BCE weight 8 | 5.9 s | 26.6% | **6.18** | 99.4% | 37.5% |
| Random legal choice | — | 15.4% | 10.33 | — | — |

“Best-score move” counts ties as correct. The stricter exact-action top-1 was 22.7% for weight 1. The balanced legality head learns to flag legal moves, but flags many illegal moves too and did not improve policy regret. **Both checkpoints still require the engine's legal mask.** The longer second fit time is not a controlled CPU-performance comparison; another Rust build was active on the machine.

## Whole-game test of the first checkpoint

Paired-color games from the **same standard starting board**; baselines are intentionally simple. This is a smoke tournament, not an independent Elo estimate.

| Opponent | Total student W/D/L | Student as black | Student as white | Median student move |
|---|---:|---:|---:|---:|
| Random, 20 games | 18/0/2 | 8/0/2 | 10/0/0 | 0.44 ms |
| Greedy, 20 games | 10/0/10 | **0/0/10** | **10/0/0** | 0.41 ms |
| Minimax depth 2, 10 games | 5/0/5 | 4/0/1 | 1/0/4 | 0.73 ms |

The greedy 50% aggregate is misleading: the model lost *every* black game. This is a clear regression target. There are too few games and too little opening diversity to generalize a strength claim. No Edax or strong-teacher matches have been run.

## CPU observations

The untrained model's forward-only, 12-sample pilot measured legal-candidate batch-1 p50 **0.463 ms at one thread** versus **0.664 ms at four**. Batch 32: **7.29 ms at one thread** versus **5.00 ms at four**. This supports separate interactive and offline thread settings; it does not establish a stable roofline or prove specific NEON/Apple matrix-kernel dispatch. The one-thread batch-128 p95 had a 103 ms outlier, so all CPU numbers need fresh-process repeats on an idle machine. PyTorch process peak RSS during the first fit was 359 MB; this includes runtime/optimizer overhead, not just the 0.44 MB of FP32 model parameters.

## Decision

Keep both checkpoints as **pilot artifacts only**. Do not promote either. The class-balanced legality change is useful diagnostically, not a winning-policy improvement.

Next quality gate: produce a larger, stronger-label dataset with controlled opening-family holdout and tactical/pass/endgame sampling; add genuine outcome/value targets; then compare direct student, student+fixed-budget search, and Edax on paired diverse openings. Before making CPU optimization claims, profile full decisions and rerun 1/2/4-thread sweeps without unrelated builds.

Local ignored artifacts: `artifacts/data/teacher_l8_600.jsonl`, `artifacts/cache/teacher_labels.sqlite`, `artifacts/checkpoints/student_l8_600*.pt`, and `artifacts/reports/student_l8_600_*.json`. They are not stored in Git; the manifest, code, tests, and this report are.
