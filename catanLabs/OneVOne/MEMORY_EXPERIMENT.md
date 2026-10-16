# OneVOne → tiny Transformer, CPU pilot

## Question

Can a small history-aware Transformer distilled from OneVOne improve 1v1 play?

## Setup

- Teacher: unchanged `OneVOneAgent` decisions from `bootstrap_20272000_64boards.jsonl.gz`. Filtered to **OneVOne teacher only**: 128 complete games across 64 boards. The games contain five opponent types.
- Inputs: the existing player-visible state and encoded legal actions. The memory model also sees its four prior *own visible observations*. It sees no opponent hand or future random events.
- Architecture: two Transformer encoder layers, width 64, four attention heads, legal-action scorer; **168,897 parameters**. Both versions have identical architecture and parameter count. The control receives the current observation only.
- Split: 52 board seeds for training, 12 held out board seeds for move agreement. Both seats of a board stay in the same split. Only decisions with multiple legal actions contribute to the loss.
- Online gate: six additional fresh board seeds, both seats, against the pinned Catanatron web bot. The same 12 games were played by each candidate and unchanged OneVOne.

## Measured result

| Agent | Held out teacher move agreement | Fresh wins vs Catanatron |
|---|---:|---:|
| OneVOne teacher | — | 11/12 |
| Tiny Transformer, no history | 53.17% (1,232 choices) | 0/12 |
| Tiny Transformer, four past observations | 53.41% (1,232 choices) | 1/12 |

Training used 5,565 nonforced choices, eight CPU epochs, and a separate checkpoint per variant. All 36 online games completed; no caps or errors. Full run data and checkpoints are in `runs/memory_v2/`.

The model with history bought four development cards and built 13 cities across its 12 games. OneVOne bought 32 and built 27. This is descriptive: the games have different lengths. It is consistent with weak VP conversion and shows why move agreement does not imply game strength.

## Interpretation

History improved action imitation by **0.24 percentage points** on this holdout, too small to claim a useful effect. Direct cloning discarded OneVOne's deterministic tactical search and value evaluation. The teacher itself chooses from the current observation, so its labels provide little training signal for using history to infer hidden information. To test whether memory improves *winning*, retain OneVOne as a tactical scaffold and train a history-aware critic from game outcomes or search targets. Keep the untouched OneVOne agent as the promotion gate.

## Reproduce

```sh
cd /Users/anixd/Documents/Hyde-onevone-transformer
start/.venv/bin/python -m catanLabs.OneVOne.run_memory_experiment \
  --epochs 8 --game-seeds 6 --out catanLabs/OneVOne/runs/memory_v2
```

The worktree's `start/.venv` and `start/catan_lab` symlinks point to the shared local Python environment and pinned Catanatron checkout. The dataset path is passed read-only from the original workspace by default.
