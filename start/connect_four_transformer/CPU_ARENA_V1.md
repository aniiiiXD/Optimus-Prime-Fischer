# CPU Arena experiment: tactical supervision

This is a local, CPU-only experiment inspired by Google DeepMind's public
[Game Arena harness](https://github.com/google-deepmind/game_arena) and Kaggle's
[Four in a Row harness](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/open_spiel_env/games/connect_four/harness.py).
The public harness supplies a text board, player to move, legal columns, and a
`Final Answer: <zero-based column>` response convention. Our local adapter
serializes the board to text, parses it back, asks the existing model, and
parses its answer. The model itself still consumes **42 structured board
tokens**, so its results are not Game Arena leaderboard results or a test of
language understanding.

## What we built

- `arena_cpu.py` evaluates the frozen grouped test split, categorizes immediate
  wins and unique forced blocks, replays both errors from the human match, and
  reports CPU model and full-move latency.
- `generate_tactics.py` produces new win/block positions with exact one-ply
  policy targets and depth-4 search values. It excludes every base board and
  all validation/test opening groups.
- `train.py` now accepts an extra training-only dataset, hard tactical targets,
  tactical sample weighting, and selection by held-out critical-move accuracy.
  The default command and v0 checkpoint retain their original behavior.

## Reproduce

From the workspace root, with the existing CPU PyTorch environment:

```bash
.venv/bin/python -m connect_four_transformer.arena_cpu \
  --checkpoint connect_four_transformer/artifacts/checkpoints/candidate_v0.pt \
  --report connect_four_transformer/artifacts/reports/arena_cpu_v0.json

.venv/bin/python -m connect_four_transformer.generate_tactics \
  --per-kind 400 --candidate-pool 6000 --depth 4 --node-limit 5000

.venv/bin/python -m connect_four_transformer.train \
  --dataset connect_four_transformer/artifacts/data/positions_v0_2000.jsonl \
  --extra-train connect_four_transformer/artifacts/data/tactics_v1.jsonl \
  --hard-tactical-targets --tactical-weight 3 \
  --selection-metric critical_accuracy --epochs 30 --batch-size 64 --threads 4 \
  --checkpoint connect_four_transformer/artifacts/checkpoints/candidate_tactical_v3.pt \
  --report connect_four_transformer/artifacts/reports/training_tactical_v3.json

.venv/bin/python -m connect_four_transformer.arena_cpu \
  --checkpoint connect_four_transformer/artifacts/checkpoints/candidate_tactical_v3.pt \
  --report connect_four_transformer/artifacts/reports/arena_cpu_tactical_v3.json

.venv/bin/python -m connect_four_transformer.play \
  --checkpoint connect_four_transformer/artifacts/checkpoints/candidate_tactical_v3.pt
```

The test split is the same 300 positions for both checkpoints. Its tactical
subsets contain 39 unique forced blocks and 52 immediate wins. The two named
human-game fixtures are extra diagnostics, not part of the held-out set.

## Measured result

| Metric | Original v0 | Tactical v3 |
|---|---:|---:|
| Training positions | 1,400 | 2,200 (including 800 new tactics) |
| Best training epoch | 28 | 18 |
| CPU training wall time | 12.86 s | 20.05 s |
| Held-out forced blocks chosen | 13/39 (33.3%) | 20/39 (51.3%) |
| Held-out immediate wins chosen | 19/52 (36.5%) | 27/52 (51.9%) |
| Overall teacher-best agreement | 43.0% | 39.0% |
| Mean teacher regret | 0.339 | 0.354 |
| Human vertical block fixture | Fail | Pass |
| Human diagonal block fixture | Fail | Fail |

In 100 fixed-opening games per side, tactical-v3 versus the tactical opponent
went from **10W/1D/89L to 38W/16D/46L as X**, but only from **7W/1D/92L to
9W/1D/90L as O**. Against depth-3 alpha-beta, v3 won 6/100 as X and 0/100
as O. This is a meaningful tactical gain with a serious side imbalance, not a
strong game-playing model yet.

Threat direction matters. On the held-out set, horizontal-block accuracy
improved from **2/18 to 11/18**, while diagonal-block accuracy fell from
**8/17 to 6/17**. Vertical blocks were 3/4 in both runs; that group is too
small for a stable conclusion. The exact diagonal trap from the human game
remains a failure. The CPU runner reports warm model-forward and full
text-to-action median/p95 latency, process CPU time, thread count and peak RSS
in its JSON. Individual runs ranged from roughly 0.3 ms to more than 1 ms
median under changing background load, so these samples do not establish a
speed difference between checkpoints.

The first tactical-only ablation increased blocks to 41.0% but lowered overall
teacher agreement. Adding 800 new positions while still selecting by soft-label
validation loss chose epoch 1 and performed worse on blocks (28.2%). Selecting
on held-out critical-move accuracy chose epoch 18 and produced the v3 result.
This shows that the checkpoint-selection metric materially affects outcomes.

## Next experiment

The diagonal fixture and O-side tournament losses remain open. Before
increasing model width, generate targeted diagonal and two-ply fork positions,
then report separate vertical/horizontal/diagonal and X/O results. A small
search-assisted player can be measured as another variant, with its extra CPU
cost visible rather than mixed into direct-model results.

Raw reports:

- [v0 CPU diagnostic](artifacts/reports/arena_cpu_v0.json)
- [v3 CPU diagnostic](artifacts/reports/arena_cpu_tactical_v3.json)
- [v3 training](artifacts/reports/training_tactical_v3.json)
- [v3 tournament](artifacts/reports/tournament_cpu_tactical_v3_100.json)
