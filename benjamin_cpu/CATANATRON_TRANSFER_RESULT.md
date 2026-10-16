# Benjamin CPU-policy transfer to the Catanatron 1v1 baseline

## What was actually tested

The three checkpoints trained in this worktree (256 Benjamin-heuristic games,
then 8,192 fixed-opponent PPO transitions and 4,096 league self-play
transitions) were run directly on CPU against the pinned Catanatron web bot.
This is a **new cross-engine policy transfer**. It is separate from the native
Benjamin R1-rule evaluation in `README.md`.

The adapter in `catanatron_adapter.py` maps all 19 hexes, 54 vertices and 72
edges by graph isomorphism. The chosen mapping aligns all 18 port vertices.
It translates the public Catanatron board and own private hand into
Benjamin's observation schema, maps Catanatron legal moves into Benjamin's
six action heads, and ranks **only moves that Catanatron lists as legal**.
Rival resource identities are estimated from publicly visible production and
total hand count; they are never read from Catanatron's hidden resource state.
No image input or hidden deck order is used.

Before evaluation, a separate random-game probe checked observation encoding
and legal-action mapping on 9,045 Catanatron states, covering all 13 action
types observed under this 1v1 protocol. All evaluation games completed with
zero illegal moves, errors or caps.

## Same 20 held-out board seeds, both seats

These 20 seeds come from the OneVOne experiment's 206-game held-out split.
The opponent is the same pinned Catanatron web bot; the target is 10 VP,
official spiral board, discard at 7, no friendly robber. OneVOne's row is the
archived result on these exact seeds and seats.

| Agent | Wins / 40 | Mean own VP | Median per-game decision latency on CPU |
| --- | ---: | ---: | ---: |
| Benjamin behavior clone, 256-game pilot | 0 | 3.43 | 1.64 ms |
| Benjamin after fixed-opponent PPO | 0 | 3.00 | 1.86 ms |
| Benjamin after league self-play | 0 | 3.43 | 2.01 ms |
| OneVOne heuristic baseline | 27 | — | — |

The concrete finding is that **these small CPU-trained Benjamin checkpoints
run legally and quickly, but do not beat the Catanatron web bot in this
40-game transfer test**. This does **not** establish that the Benjamin project
or a fully trained checkpoint is weak. The source repo's own rules target 15
VP and differ in discard and pre-roll development-card rules. Our adapter also
has imperfect opponent hand estimates because Catanatron does not expose the
same broadcast tracker. Those domain differences, along with the very short
training run, limit what can be inferred about the original architecture.

## Direct Benjamin vs OneVOne: 50 fresh games

Using the latest CPU self-play checkpoint, we sampled 25 fresh random board
seeds and swapped seats on each board. Under the same Catanatron 10-VP rules:

| Agent | Wins / 50 | Mean VP |
| --- | ---: | ---: |
| Benjamin CPU self-play checkpoint | **0** | 3.16 |
| Plain OneVOne | **50** | 10.20 |

All 50 games completed, with no errors or caps. Benjamin won 0/25 from each
seat. Across all games it built exactly 100 settlements (the two setup
settlements in every game) and zero cities; OneVOne built 248 settlements and
102 cities in total. This identifies stalled expansion in these transferred
games, but does not isolate whether limited training or the cross-engine
observation/rule differences caused it. Exact seeds and per-game moves are
saved in `reports/benjamin_vs_onevone_50.meta.json` and `.jsonl`.

## Artifacts

- `catanatron_adapter.py`: geometry, public observation and legal action bridge.
- `evaluate_catanatron.py`: reproducible paired-seat runner.
- `reports/heldout_206_seeds.json`: exact held-out split; first 20 used here.
- `reports/catanatron_{bc,fixed,selfplay}_v3.jsonl`: per-game results.
- `evaluate_vs_onevone.py`: paired-seat direct head-to-head runner.
- `reports/benjamin_vs_onevone_50.{meta,summary}.json` and `.jsonl`: direct
  head-to-head seed list, summary and per-game results.
- The checkpoints remain in the ignored upstream source checkout under
  `external/benjamin_catan_rl/runs/benjamin_cpu_pilot/`.

Example command from this worktree:

```sh
/Users/anixd/Documents/Hyde/start/.venv/bin/python -m benjamin_cpu.evaluate_catanatron \
  --boards 20 \
  --checkpoint external/benjamin_catan_rl/runs/benjamin_cpu_pilot/bc_model_256/best_ckpt.pt \
  --out benjamin_cpu/reports/catanatron_bc_v3.jsonl
```
