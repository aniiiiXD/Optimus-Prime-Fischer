# Connect Four Transformer plan (CPU only)

## Goal

Build a small model that reads a Connect Four board, scores the legal columns,
and learns from search-generated examples. The question is not whether it can
copy a teacher's move on a random split; the question is whether it makes fewer
strategic mistakes in real games for a measured CPU budget.

The rules engine already exists in [`../connect_four_engine`](../connect_four_engine).
There are no Connect Four model weights or training examples in our project yet.

## What the knowledge corpus says

| Reference | What it demonstrates | What we take from it |
|---|---|---|
| [`repos/connect-4/REPORT.md`](../game_transformer_knowledge_corpus/repos/connect-4/REPORT.md) | A 42-cell board Transformer with 6 blocks, width 128, 8 heads, policy/value heads; its report describes roughly 700k MCTS positions and ~77% teacher-move accuracy | Board-state input, policy plus value, varied opening coverage, mirror augmentation, and search-derived targets work as a recipe; its Colab/GPU scale is not our CPU target |
| [`repos/connect-4/data_generation/data_generation_no_guide.ipynb`](../game_transformer_knowledge_corpus/repos/connect-4/data_generation/data_generation_no_guide.ipynb) | UCB-style tree selection, random rollouts, root visit counts and Q estimates; 2,000 simulations per queried position in the notebook defaults | Keep visits and per-action values in our records; benchmark teacher cost before generating at scale |
| [`repos/connect-4/mirroring/mirror.py`](../game_transformer_knowledge_corpus/repos/connect-4/mirroring/mirror.py) | Mirrors the board and every action-indexed target together | Horizontal reflection must also reverse legal columns, policy, Q and visits |
| [`notes/06_mav_takeaway.md`](../game_transformer_knowledge_corpus/notes/06_mav_takeaway.md) | Connect Four uses a solver teacher in the MAV work; one model can produce policy, action values and state value | Copy the multi-output decision interface, not the billion-parameter scale or enormous corpus |
| [`notes/03_cpu_first_architecture.md`](../game_transformer_knowledge_corpus/notes/03_cpu_first_architecture.md) | Recommends separating game adapter, state encoder, candidate scorer, search and evaluator | Keep our existing engine as the rules authority and make the model consume a variable legal-action list |
| [`notes/04_experiments_and_metrics.md`](../game_transformer_knowledge_corpus/notes/04_experiments_and_metrics.md) | Gives policy/value/search ablations and CPU metrics | Change one factor at a time; measure win rate, regret, calibration and latency |
| [`repos/laya`](../game_transformer_knowledge_corpus/repos/laya) | Typed, non-autoregressive candidate decisions | Score legal action candidates in one forward pass instead of generating move text |

The downloaded Connect Four project's reported accuracy is agreement with its
MCTS teacher, not a guarantee of beating strong opponents. The report used a
GPU, so its training time is not a CPU comparison for us.

## Proposed model v0

Use the current board, rather than a move-history language model. Connect Four
has a compact state and rules depend on the current geometry.

```text
42 board-cell tokens + one state token
              |
  2 bidirectional Transformer blocks
  width 64 · 4 heads · MLP width 128
              |
  shared state representation
       |                 |
 legal column candidates  state summary
       |                 `--> V(s): expected result
 landing-row + column
 + column board features
       |
 shared candidate scorer, reused for each legal column
       |--> policy score P(a | s)
       `--> Q(s,a): searched outcome estimate
```

Each board token means empty, current player's piece, or opponent's piece. The
input is normalized to the player-to-move perspective, with learned row and
column embeddings. A `[STATE]` token pools global information. For a candidate
column, the scorer sees the state summary, that column's pooled board features,
the cell where the piece would land, and an embedding of the column. Full
columns are omitted from the candidate list and masked by the engine.

Initial settings are deliberately provisional: 2 blocks, width 64, 4 heads,
MLP width 128, FP32. At only 43 state tokens and at most 7 candidates, this is
small enough to train and profile on CPU. We increase model size only if a
capacity experiment shows a repeatable strength gain. With the shared candidate
fusion head, expect roughly 120k parameters (about 0.5 MB of FP32 weights); we
will record the actual count when implementing it.

Training objectives are introduced in stages:

1. **Policy:** cross-entropy against search visit probabilities or a soft target over best moves.
2. **Action value:** Huber loss against each legal move's search score, normalized to `[-1, 1]`.
3. **State value:** Huber loss against search value and, separately, terminal game result.

Start with policy-only versus policy-plus-value ablations. Add the Q loss once
we have reliable per-action labels. Legal masking stays in the game engine;
the model never gets to make an illegal move.

## CPU workflow

1. Keep all inference and training on CPU, FP32, and pin Torch to 2–4 threads.
2. Build and profile a search teacher before asking it for a large dataset.
3. Start with 1,000–2,000 unique states. Time game generation, labeling, and training separately.
4. Increase the teacher node budget or dataset size only after measuring positions/second and label quality.
5. Train batches of 32 or 64 first; compare 1, 2, and 4 Torch threads. Small workloads can get slower with extra threads.
6. Save compact compressed shards, model checkpoints, configs, seeds, and a report. Do not keep a huge in-memory dictionary of Python objects.
7. Profile batch-1 move latency separately from batch evaluation used by search.

The model itself should be inexpensive here. Search-based label generation is
likely to consume most of the CPU budget, so its node count and cache behavior
must be logged.

## Implementation phases and exit checks

| Phase | Work | Exit check |
|---|---|---|
| 0. Freeze the rules boundary | Use `ConnectFour.step/undo`, legal actions, snapshots, and state keys | Search can apply/undo moves without changing the caller's board |
| 1. CPU teacher | Add negamax alpha-beta, center-first move order, tactical win/block checks, a transposition table, and a measured node limit | Beats the tactical heuristic in paired games; outputs per-action values; reports nodes/s and cache hits |
| 2. Dataset pilot | Generate 1k–2k unique canonical positions from mixed policies and label every legal action | Inspectable examples, no invalid positions, reproducible with a seed; label-generation time is known |
| 3. Data scale-up | Grow toward 10k–20k unique positions if the pilot shows useful targets and affordable CPU time | Balanced by game phase and tactical difficulty; held-out opening groups remain untouched |
| 4. Policy/value models | Train the fixed seven-logit baseline and candidate scorer on the same split and budget | Compare top-1/top-3, teacher regret, outcome calibration, game results and CPU use |
| 5. Post-training | Replay tactical misses; optionally add MCTS visit targets or self-play outcomes | Improvement survives at least three seeds and fixed opponent budgets |
| 6. Search-assisted inference | Use policy to order moves and value to score leaves in a fixed alpha-beta budget | Beats direct policy and plain search at the same CPU-time budget |

## Evaluation opponents

- Random legal player: checks basic competence and color balance.
- Tactical player: always wins immediately, blocks immediate wins, then prefers the center.
- Shallow alpha-beta: measures whether the learned model adds anything to cheap search.
- Deeper alpha-beta: strong reference; report draws and losses, not a claim that the Transformer must beat it.
- Earlier model checkpoint: regression opponent for each training change.

Play at least both colors and use a fixed set of opening prefixes. Track win,
draw and loss separately. A perfect Connect Four player should not be judged by
policy accuracy alone; teacher agreement, tactical puzzle accuracy and game
outcomes answer different questions.

## Primary metrics

- Data: unique canonical states, positions by ply, games, source policy, teacher nodes, labels/second, duplicate/cache rate.
- Learning: policy loss, top-1/top-3 teacher agreement, best-move recall, action-value error and state-value calibration.
- Tactics: immediate-win detection, immediate-block detection, double-threat / support-square traps, late-game cases.
- Games: win/draw/loss by color against each fixed opponent and opening set.
- CPU: training wall/CPU seconds, inference p50/p95, examples/s, peak RSS, thread scaling, search nodes/s, cache hit rate.

The canonical experiment changes one item at a time: target quality, data
amount, model head, auxiliary loss, symmetry augmentation, search integration,
then quantization. Keep a separate test set untouched until model choices are
finished.

## Implemented v0 files

| File | Role |
|---|---|
| [`search.py`](search.py) | iterative-deepening alpha-beta teacher, heuristic, mirror-aware cache, CPU counters |
| [`generate_dataset.py`](generate_dataset.py) | mixed-policy position collection and search labeling |
| [`data.py`](data.py) | relative tokenization and opening-group train/validation/test split |
| [`model.py`](model.py) | 42-cell state encoder and shared legal-column policy/Q/value scorer |
| [`train.py`](train.py) | CPU training, held-out metrics, latency and memory report |
| [`agent.py`](agent.py) | checkpoint inference with inspectable top moves |
| [`play.py`](play.py) | human-versus-Transformer terminal game |
| [`tournament.py`](tournament.py) | fixed-opening games against random, tactical and alpha-beta agents |
