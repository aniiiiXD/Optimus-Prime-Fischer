# Othello pipeline — teacher-first restart

This is the clean workspace for the next game experiment. It **reuses** the
existing rules in `../transformer_games_lab/gamebench/games/othello.py`; it does
not copy the old OthelloGPT project or the Connect Four training code.

The complete multi-perspective plan, math, dataset split, and CPU promotion
gates are in [DESIGN.md](DESIGN.md).
The first bounded run and its honest limitations are in [PILOT_REPORT.md](PILOT_REPORT.md).

## Model choice

**Primary teacher:** [Neural Reversi](https://github.com/natsutteatsuiyone/neural-reversi), pinned to engine `v6.3.0` and the published `eval-e6bbc4f6.zst` / `eval_sm-e6bbc4f6.zst` weights. It combines a trained neural position evaluator with CPU search. Its CLI supports GTP (`play`, `genmove`) and a `solve --all-moves` mode, which are useful for producing policy targets. The author's Hard-30 benchmark reports 80.3% best-move agreement at depth 15; this is **not** an independently measured Elo or a guarantee it beats Edax.

**Independent opponent:** [Edax](https://github.com/abulmo/edax-reversi). Its CPU search gives us an outside check on whether distilling Neural Reversi actually improves our model. It is an engine, not a Transformer.

**Existing research baseline:** `../othello_world/` contains OthelloGPT, which predicts moves from history. That is valuable for representation studies, but next-move prediction alone does not establish strong play. Our student will be a small Transformer trained to choose among *legal actions*.

The engine is GPL-3.0; check the separately published weights' terms before redistributing a combined product. Details and fixed download hashes are in [teacher_manifest.json](teacher_manifest.json).

## What exists now

```text
othello_pipeline/
  README.md                experiment design and commands
  teacher_manifest.json   fixed engine/weight sources and SHA-256
  encoding.py             board and action conversion for the teacher CLI
  smoke.py                checks conversion and a complete legal game
  model.py                board/candidate Transformer (policy, Q, V, legality)
  dataset.py              strict teacher-label schema and leakage-aware split
  teacher.py              batched all-move score parser and CLI wrapper
  verify_teacher.py       opening, pass, and exact late-score parity
  generate_dataset.py     phase-balanced positions and persistent label cache
  train.py                CPU distillation trainer (requires verified JSONL)
  assess.py               compare checkpoints against held-out teacher scores
  evaluate.py             paired-color games against baseline agents
  profile_cpu.py          forward latency and batch-throughput sweep
  test_*.py               correctness and one-step gradient tests
  scripts/setup_teacher.sh pinned source/weight download, hash check, build

../transformer_games_lab/gamebench/games/othello.py  shared game rules
../othello_world/                              prior OthelloGPT research repo
```

From the workspace root:

```bash
PYTHONPATH=transformer_games_lab ./.venv/bin/python -m othello_pipeline.smoke
```

The pinned teacher source is built locally under ignored `vendor/`; its two
checksum-verified weight files are under ignored `artifacts/teacher_weights/`.
The pilot dataset, checkpoints, and reports are also ignored local artifacts,
so a fresh clone must reinstall them. This setup currently uses about 1 GiB
of Rust build output plus 274 MiB of local artifacts.

To reproduce the teacher on another arm64 Mac with Rust installed:

```bash
bash othello_pipeline/scripts/setup_teacher.sh
```

It downloads about 258 MB of compressed weights and verifies both hashes;
the first native build also downloads Rust dependencies and takes longer.

Run the model/training checks and an untrained CPU profile:

```bash
PYTHONPATH=transformer_games_lab ./.venv/bin/python -m unittest \
  othello_pipeline.test_model othello_pipeline.test_pipeline -v
PYTHONPATH=transformer_games_lab ./.venv/bin/python -m othello_pipeline.verify_teacher
PYTHONPATH=transformer_games_lab ./.venv/bin/python -m othello_pipeline.profile_cpu \
  --threads 1 --batch-sizes 1,32,128
```

The default student has **110,212 parameters**. A brief forward-only pilot on
this arm64 Mac measured legal-only batch-1 median **0.463 ms with one Torch
thread**, versus **0.664 ms with four**; batch-32 medians were 7.29 ms and
5.00 ms respectively. These are **random weights**, 12 samples per setting,
not game strength or whole-move latency. The batch-128 one-thread p95 was noisy,
so rerun with more samples before making systems claims.

Reproduce a fresh small dataset and fit (choose **new** artifact names;
generation refuses to overwrite existing data):

```bash
PYTHONPATH=transformer_games_lab ./.venv/bin/python -m othello_pipeline.generate_dataset \
  --positions 600 --candidate-games 140 --level 8 \
  --output othello_pipeline/artifacts/data/teacher_l8_600_new.jsonl
PYTHONPATH=transformer_games_lab ./.venv/bin/python -m othello_pipeline.train \
  --dataset othello_pipeline/artifacts/data/teacher_l8_600_new.jsonl \
  --output othello_pipeline/artifacts/checkpoints/student_l8_600_new.pt \
  --epochs 5 --threads 2 --batch-size 64
```

The first 600-position pilot was deliberately small and **not competitive**.
Its best checkpoint beat random legal move selection on held-out teacher
regret, but only split 10/20 versus greedy and 5/10 versus minimax depth 2.
Do not confuse its untrained value head or hard legality mask with learned
winning strategy.

## Verified stage and next quality gate

1. **Done:** pinned teacher build, GTP opening parity, exact late score-sign parity, forced-pass parity. Our board indices are row-major; `encoding.py` converts to GTP's column-letter/row-number squares.
2. **Pilot done:** all-legal-action teacher scores cached by canonical board, side, and teacher settings; a 600-position phase-balanced dataset and two student checkpoints.
3. **Next:** better/harder labels, genuine outcome targets for `V`, independent opening-family tests, and paired games from diverse starting positions. Current aggregate games hide a severe black/white split.
4. **Promotion:** independent Edax matches and full CPU decision profiling. A high training-set move agreement or fast forward alone is not enough.

Neural Reversi is a **teacher engine** (neural evaluator + search), not the Transformer student we are training. Long dataset/training runs should wait until the pilot's color split and value-label plan are reviewed.

## Source notes

- [Neural Reversi README and reported benchmarks](https://github.com/natsutteatsuiyone/neural-reversi/tree/v6.3.0)
- [CLI/GTP and `solve --all-moves` documentation](https://github.com/natsutteatsuiyone/neural-reversi/blob/v6.3.0/crates/cli/README.md)
- [Published weight files](https://github.com/natsutteatsuiyone/neural-reversi-weights/releases/tag/v7.0.0-wasm)
- [Edax source and build instructions](https://github.com/abulmo/edax-reversi)

