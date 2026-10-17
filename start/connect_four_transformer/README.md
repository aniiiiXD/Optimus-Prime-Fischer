# Connect Four Transformer

This is the first complete CPU-only loop built on
[`../connect_four_engine`](../connect_four_engine):

```text
mixed legal games
      ↓
alpha-beta labels every legal column
      ↓
2,000-position search dataset
      ↓
42 board tokens → 2 Transformer blocks → candidate policy + Q + value
      ↓
held-out metrics, CPU profile, tournament, playable checkpoint
```

## What exists

| Path | Purpose |
|---|---|
| [`search.py`](search.py) | depth-limited alpha-beta teacher with mirror cache and CPU counters |
| [`generate_dataset.py`](generate_dataset.py) | mixed-policy games, phase balancing, deduplication and search labels |
| [`data.py`](data.py) | player-relative tokens, mirror augmentation and grouped splits |
| [`model.py`](model.py) | 114,819-parameter candidate-action Transformer |
| [`train.py`](train.py) | policy/Q/value training and CPU profiling |
| [`agent.py`](agent.py) | checkpoint loading and top-move inspection |
| [`play.py`](play.py) | terminal game against the model |
| [`tournament.py`](tournament.py) | games against random, tactical and depth-3 search agents |
| [`DATASET_PLAN.md`](DATASET_PLAN.md) | detailed CPU dataset design |
| [`RESULTS_V0.md`](RESULTS_V0.md) | measured first-run results and next diagnosis |
| [`CPU_ARENA_V1.md`](CPU_ARENA_V1.md) | Game Arena-inspired CPU diagnostic and tactical-training comparison |
| [`arena_cpu.py`](arena_cpu.py) | held-out tactical suite, human-game replay, text observation and CPU timing |
| [`generate_tactics.py`](generate_tactics.py) | additional one-ply win/block examples, search-labelled on CPU |
| [`generate_curriculum.py`](generate_curriculum.py) | mine v3's high-regret mistakes plus fork/support-square motifs for search labels |
| [`STRATEGY_V4.md`](STRATEGY_V4.md) | hard-example replay design, safeguards, and corrected v4b results |
| [`STRATEGY_V5_UNIQUE_5MIN.md`](STRATEGY_V5_UNIQUE_5MIN.md) | canonical unique-state mining, persistent teacher-label cache, and timed training |
| [`teacher_cache.py`](teacher_cache.py) | SQLite cache for deterministic alpha-beta labels, keyed by canonical state and search settings |
| [`test_data_pipeline.py`](test_data_pipeline.py) | regression checks for mirror identity, move-history canonicalization, and cache invalidation |
| [`benchmark_models.py`](benchmark_models.py) | same-test-set and same-opening comparison across saved checkpoints |
| [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) | benchmark metrics, reproducible seeds, and checkpoint inventory |
| [`cpu_search_bench.py`](cpu_search_bench.py) | fixed-board before/after benchmark for the hybrid search hotpath |
| [`cpu_sweep.py`](cpu_sweep.py) | batch/thread CPU forward and optimizer-step sweep |
| [`CPU_OPTIMIZATION_REPORT.md`](CPU_OPTIMIZATION_REPORT.md) | measured search, data, model-size and holdout results from 15 October |

## Move selection with one-ply simulation

```text
board → Transformer scores legal columns → simulate each candidate drop
      → check for an immediate opponent win → choose the highest-scored safe move
```

The simulator runs in `agent.py` after the Transformer forward pass. It does
not add logic or weights to an attention block. If every move permits an
immediate win, it chooses the move that leaves the fewest winning replies.
It catches moves that hand the opponent a win on their very next turn, but it
does not solve deeper forks or multi-turn support-square traps. The Transformer's
original probabilities remain visible for comparison.

## Reproduce v0

Use the shared Python environment from the workspace root:

```bash
.venv/bin/python -m connect_four_transformer.generate_dataset \
  --positions 2000 --depth 4 --node-limit 5000 \
  --output connect_four_transformer/artifacts/data/positions_v0_2000.jsonl

.venv/bin/python -m connect_four_transformer.train \
  --dataset connect_four_transformer/artifacts/data/positions_v0_2000.jsonl \
  --epochs 30 --batch-size 64 --threads 4

.venv/bin/python -m connect_four_transformer.tournament
```

Play against the current checkpoint:

```bash
.venv/bin/python -m connect_four_transformer.play --human X
```

During the model's turn the interface prints its state value, top three column
probabilities, per-column Q estimates, forward-pass latency, and a small rules
simulation. By default, the simulator checks each proposed drop for an
immediate winning reply and avoids moves that allow one when a safer legal move
exists. The Transformer's scores and weights are unchanged; use `--no-lookahead`
to compare against its raw top-probability move.

To play against the corrected hard-example checkpoint, pass
`--checkpoint connect_four_transformer/artifacts/checkpoints/candidate_tactical_v4b.pt`.
See [`CPU_ARENA_V1.md`](CPU_ARENA_V1.md) for earlier tactical experiments and
[`STRATEGY_V4.md`](STRATEGY_V4.md) for the v4b strategy, run profile, and limits.

## Play in Chrome

Start the local browser game from the workspace root:

```bash
.venv/bin/python -m connect_four_transformer.web_server
```

Then open <http://127.0.0.1:8765> in Chrome. By default, you play X and the
saved v5 checkpoint plays O. Choose X/O, use the arrows above the board to drop
a piece, and select **New game** to restart. The decision panel shows the
Transformer's top three choices, Q estimates, one-ply tactical result, and
separate forward/simulation timings. Stop the server with Ctrl-C in its
terminal. Use `--checkpoint PATH` to swap checkpoints or `--port 8766` to use
another local port.

## Current boundary

The v0 model learned a center-opening preference and usually beats random play,
but it is weak against immediate tactical reasoning and search. It is a valid
trained Transformer and a reproducible baseline. The next gain should come
from targeted tactical/deeper labels and hard-example replay before increasing
the 115k-parameter model.
