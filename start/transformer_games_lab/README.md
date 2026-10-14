# CPU GameBench

A small, reusable game-intelligence engine for hackathon experiments. New
games implement one adapter; tournaments, search, imitation training, and CPU
profiling stay unchanged.

This is infrastructure version **0.1**, not a claim of strong game play. The
first goal is to make experiments comparable and failures visible.

Our research target is to train strategy into a Transformer and measure whether
it improves winning outcomes. See [PROJECT_TARGETS.md](PROJECT_TARGETS.md) for
the pretraining/post-training plan, dataset requirements, and evaluation
criteria. The trainer currently implements teacher imitation as a baseline;
outcome-driven post-training is a planned milestone.

## What works now

```text
Game adapter
  reset · current_player · legal_actions · apply_action
  terminal · returns · observation_tensor · history_tokens
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Search agents   Transformer    New game plugin
     minimax/MCTS    policy model   same contract
          │              │
          └───────┬──────┘
                  ▼
       Tournament + CPU profiler
       W/D/L · Elo · latency · RSS
       FLOPs · FLOP/s · intensity
```

- Games: Othello and Connect Four.
- Baselines: random, generic one-ply greedy, depth-limited alpha-beta, UCT MCTS.
- Model: configurable causal Transformer policy.
- Training: minimal next-action imitation from any registered teacher agent.
- Evaluation: deterministic round robin, swapped seats, W/D/L, score rate,
  sequential Elo, and per-agent move latency.
- Systems: forward, forward+backward, and full train-step CPU workloads.
- Native systems lab: CPU feature report, scalar-vs-SIMD dot product, and
  isolated matrix-multiplication sweeps.

`PROJECT_DESIGN.md` preserves the research background. The runnable browser
experience lives only in the Next.js app under `web/`.

## Environment

From the workspace root, run `bash scripts/setup_env.sh`, then `cd
transformer_games_lab`. The Python environment is shared at `../.venv/`;
`bash scripts/setup.sh` also sets up the web app's npm dependencies.

PyTorch is the only large dependency. The game adapters, search, and tournament
code themselves use the Python standard library.

## Run a tournament

```bash
../.venv/bin/python tournament.py \
  --game othello \
  --agents random,greedy,minimax:2,mcts:50 \
  --games 20 \
  --output reports/othello_baselines.json
```

Swap only the adapter:

```bash
../.venv/bin/python tournament.py \
  --game connect4 \
  --agents random,greedy,minimax:3 \
  --games 30
```

`--games` means games per pair. Seats alternate so first-player advantage is
not silently assigned to one agent.

After training, make the saved policy an ordinary tournament participant:

```bash
../.venv/bin/python tournament.py \
  --game othello \
  --agents random,greedy,transformer:artifacts/latest_transformer.pt \
  --games 20
```

The checkpoint records its game and model shape. Loading it for a different
game fails early instead of silently comparing incompatible action spaces.

## Profile a Transformer

Fast smoke run:

```bash
../.venv/bin/python benchmark.py \
  --game othello \
  --preset micro \
  --batch-sizes 1,8 \
  --sequence-lengths 8,32 \
  --modes forward,forward_backward \
  --threads 1,4 \
  --output reports/micro_cpu.json
```

Architecture-shaped comparison with the public OthelloGPT dimensions (this
creates fresh random weights; it does not load that checkpoint):

```bash
../.venv/bin/python benchmark.py \
  --preset othello-gpt-shape \
  --batch-sizes 1 \
  --sequence-lengths 16,32,59 \
  --modes forward \
  --threads 1,4,8 \
  --min-seconds 1.0 \
  --output reports/othello_shape_cpu.json
```

### Parameters that make the benchmark trustworthy

| Parameter | Default | Why it exists |
|---|---:|---|
| warmup calls | 10 | removes first-dispatch and allocator startup from steady-state latency |
| target sample | 8 ms | repeats very fast calls inside one timer window, reducing timer noise |
| minimum samples | 25 | prevents a result from resting on a few lucky timings |
| minimum timed duration | 0.5 s | makes the run long enough to observe scheduler noise |
| robust CV target | 5% | continues sampling until median absolute deviation is acceptably small |
| maximum samples | 80 | prevents a noisy machine from benchmarking forever |
| threads | explicit | thread count can matter more than architecture for small CPU models |
| batch and sequence | explicit grid | exposes throughput/latency and linear/quadratic scaling separately |
| dtype | explicit | float32 and bfloat16 have different kernels and memory footprints |

Latency reports mean, median, p90, p99, min, max, ordinary CV, and robust CV.
Use **median** for the headline and p90/p99 for tail behavior. A `noisy` result
did not reach the requested robust-CV threshold and should be rerun on an idle
machine.

Memory is sampled in a separate phase because a sampler thread perturbs short
latency measurements. FLOPs and bytes are labeled estimates: dense
multiply-add accounting and one streaming pass over parameter bytes. They are
not hardware performance-counter readings. Arithmetic intensity therefore
means estimated FLOPs per estimated weight byte, not an AI quality score.

For a competition report, repeat the whole command in at least three fresh
processes, keep power mode and background load constant, and report the raw
JSON—not only the best run.

### Plug in another workload

The profiler is not coupled to PyTorch. Implement the `BenchmarkWorkload`
protocol from `gamebench/systems/benchmark.py`: `run`, `synchronize`,
`metadata`, `work_items_per_call`, `estimated_flops_per_call`, and
`estimated_bytes_per_call`. Then pass the object to `run_benchmark`. Returning
`None` for FLOPs or bytes is valid and preferable to inventing a number.

`TransformerWorkload` is the first plugin. It exposes three boundaries that
matter during optimization:

- `forward`: deployment/inference cost;
- `forward_backward`: gradient-computation cost without an optimizer update;
- `train_step`: forward, backward, and AdamW update together.

The analytical FLOP count covers the dense matrix multiplications in attention,
the MLP, and policy head. Softmax, normalization, activation, indexing, and
optimizer arithmetic are deliberately omitted, so measured time remains the
source of truth.

## SIMD and matrix acceleration

Start with the machine report, then separate vector and matrix behavior:

```bash
../.venv/bin/python hardware_report.py
../.venv/bin/python simd_benchmark.py --output reports/simd.json
../.venv/bin/python matrix_benchmark.py \
  --sizes 128,512,1024 \
  --dtypes float32,bfloat16 \
  --threads 1,4 \
  --output reports/matrix.json
```

On Apple Silicon, NEON is the relevant SIMD ISA; AVX belongs to x86. Matrix
hardware terminology is more subtle, so the tools report Arm SME/SME2 and the
PyTorch/BLAS backend without claiming that a private kernel was definitely used.
The worked explanation is in `SYSTEMS_LAB.md`.

## One web app

The browser workspace is now one Next.js App Router project under `web/`:

- `/profiler` — interactive measurements and explanations;
- `/learn` — SIMD, matrix, roofline, and Transformer theory.

Install it once:

```bash
cd web
npm install
cd ..
```

Then launch it alongside the Python tools:

```bash
../.venv/bin/python dashboard.py
```

Or run Next directly with `cd web && npm run dev`. The profiler page reads the
M5 baseline JSON files from `reports/` on the server and accepts additional
reports through its file picker. It connects each result to the underlying idea:

- interactive instruction → kernel → operator → model path;
- scalar-versus-SIMD comparison with numerical error checking;
- matrix throughput curves by shape, dtype, and requested thread count;
- Transformer forward/backward/train-step comparisons;
- educational roofline plot based on explicitly labelled byte estimates;
- measurement-method walkthrough.

The launcher prints the exact URL and chooses a free port if `8765` is already
occupied. Python owns experiments and JSON generation; Next owns every browser
view. There is no second frontend stack to keep synchronized.

## Minimal training path

This proves that the adapter, tokenization, model, dataset, loss, and checkpoint
path connect. It is imitation of a baseline teacher, not reinforcement learning:

```bash
../.venv/bin/python train.py \
  --game othello \
  --model transformer \
  --device cpu \
  --teacher greedy \
  --games 100 \
  --epochs 4
```

The report separates teacher-action accuracy from raw legal-action rate. A
model can copy the teacher poorly yet still make legal moves, or match one
teacher action while failing to become strategically strong. Tournament play
is the next evaluation layer.

## Add a new game

Implement `GameState` and `GameAdapter` in `gamebench/games/`, then add one
factory entry to `gamebench/registry.py`. Declare these facts explicitly:

- discrete action-space size;
- maximum game length;
- observation tensor shape and meaning;
- action-to-token mapping;
- pass/chance/information rules;
- terminal utilities for every player.

Do not force a hidden-information or stochastic game into this two-player,
perfect-information contract. That needs a deliberate extension for chance
nodes, information states, and legal observations.

## Tests

```bash
../.venv/bin/python -m unittest discover -s tests -v
```

The tests cover rule transitions, terminal rewards, adapter tokenization,
tournament determinism, and the benchmark's statistical/result schema.
