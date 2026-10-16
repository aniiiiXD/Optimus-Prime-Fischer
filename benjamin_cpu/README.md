# Benjamin 1v1 RL pipeline on CPU

This worktree runs the public [BenjaminL1/catan_rl](https://github.com/BenjaminL1/catan_rl) pipeline at commit `8c3bace` entirely on CPU. The upstream checkout is at `external/benjamin_catan_rl/`; it is ignored by this worktree's Git because the upstream source is fetched separately. The virtual environment is local to this worktree (`.venv/`).

## What was run

1. Built the upstream Rust extension and Python package in the local virtual environment.
2. Generated **256** games from Benjamin's heuristic with its 30% perturbation setting: **73,903** decisions after forced moves were filtered. This took **53.2 seconds** on CPU.
3. Trained Benjamin's Transformer/GNN behavior-cloning policy for three CPU epochs. Best validation NLL: **4.2668** at optimizer step 700; training took **83.9 seconds**. Bridged its checkpoint into Benjamin's PPO checkpoint format.
4. Warm-started Benjamin's PPO against his fixed heuristic for **8,192 transitions**, then warm-started league self-play for **4,096 transitions** with 50% fixed heuristic and 50% snapshots. Both ran with `device: cpu`; the league actually created three snapshots.
5. Evaluated all three checkpoints with the **same 40 R1-rule games** (20 per seat), on CPU, against Benjamin's fixed heuristic.

| Checkpoint | Wins / 40 | Truncated games | CPU evaluation time |
|---|---:|---:|---:|
| Behavior clone | 7 | 2 | 22.03 s |
| After fixed-heuristic PPO | 3 | 0 | 22.21 s |
| After snapshot self-play | 2 | 1 | 22.38 s |

Evaluation JSON: `reports/bc256_eval.json`, `reports/fixed256_eval.json`, `reports/selfplay256_eval.json`. Checkpoints and dataset stay in `external/benjamin_catan_rl/runs/benjamin_cpu_pilot/` in this worktree.

## Verdict and limits

- **CPU execution works end to end**, including native engine build, heuristic data generation, imitation, PPO, league snapshots, and game evaluation. No GPU/MPS was used for these runs.
- **The short run did not improve wins.** It has only 256 teacher games and 12,288 PPO transitions total. Benjamin's published default calls for 30,000 teacher games and about 50 million PPO transitions. Our pilot is a pipeline/throughput check, not a reproduction of his full strength result.
- The rules are Benjamin's **15-VP Colonist-style 1v1 variant**, not our pinned **10-VP Catanatron** rules. These 40-game win rates cannot be compared with OneVOne's win rates or used as its replacement.
- The 8,192-transition fixed PPO stage took about 23 wall-clock seconds including one 20-game in-training evaluation. A simple linear extrapolation suggests tens of hours for 50 million CPU transitions; it is only a rough throughput estimate because environment count, rollout size, eval cadence and self-play overhead differ at scale.

## Reproduction

Start in this worktree. Fetch the upstream source at the pinned commit, then install it into this worktree's own environment:

```sh
git clone https://github.com/BenjaminL1/catan_rl external/benjamin_catan_rl
git -C external/benjamin_catan_rl checkout 8c3bace
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e external/benjamin_catan_rl
```

From `external/benjamin_catan_rl/`, the stages are:

```sh
../../.venv/bin/python -m catan_rl.cli.generate_bc_dataset \
  --out runs/benjamin_cpu_pilot/bc_data_256 --n-games 256 --shard-size 128 \
  --seed 20261018 --progress-every 64
OMP_NUM_THREADS=4 ../../.venv/bin/python -m catan_rl.cli.train_bc \
  --data runs/benjamin_cpu_pilot/bc_data_256 \
  --out runs/benjamin_cpu_pilot/bc_model_256 --device cpu \
  --max-epochs 3 --batch-size 256 --warmup-steps 20 \
  --val-every-steps 100 --chunk-rows 20000 --max-steps 900 --aug-prob 0
../../.venv/bin/python scripts/bc_to_checkpoint.py \
  --in runs/benjamin_cpu_pilot/bc_model_256/best.pt \
  --out runs/benjamin_cpu_pilot/bc_model_256/best_ckpt.pt
OMP_NUM_THREADS=4 ../../.venv/bin/python -m catan_rl.cli.train \
  --config ../../benjamin_cpu/configs/cpu_256_fixed.yaml \
  --run-dir runs/benjamin_cpu_pilot/fixed_cpu_256
OMP_NUM_THREADS=4 ../../.venv/bin/python -m catan_rl.cli.train \
  --config ../../benjamin_cpu/configs/cpu_256_selfplay.yaml \
  --run-dir runs/benjamin_cpu_pilot/selfplay_cpu_256
```

`benjamin_cpu/evaluate.py` loads the saved checkpoints and runs the paired CPU gate. The original workspace's shared Python environment was used briefly during setup, then the `catan-rl` package was removed from it; this worktree now has its own environment.
