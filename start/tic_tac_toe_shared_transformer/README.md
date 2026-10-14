# Tic-Tac-Toe shared candidate Transformer

This folder is a controlled experiment, not a claim that a larger head always
beats a smaller one. It compares three models under the same CPU training
protocol:

| Name | Decision interface | Supervision |
|---|---|---|
| `classic_policy` | fixed logits for squares 1–9 | optimal policy |
| `candidate_policy` | explicit shared scorer for `(state, candidate move)` | optimal policy |
| `shared_multitask` | same candidate scorer plus `Q`, `V`, legality heads | policy + exact Minimax outcomes + legality |

## Architecture

```text
10 state tokens (9 cells + turn)
            |
 token + row + column embeddings
            |
 2 bidirectional Transformer blocks
            |-------------------------------|
 contextual square for move a        turn/state summary
            |                               |
 action ID + action row/column              +--> V(s): loss/draw/win
            |
 concatenate [square, state, action] -> shared fusion MLP
                                      |-> policy score
                                      |-> Q(s,a): loss/draw/win
                                      `-> legality score
```

The word **shared** has two meanings here: the board is encoded once for every
decision head, and the same candidate scorer is reused for every move.

## Run the controlled comparison

From the workspace root, using the already installed Othello virtual environment:

```bash
.venv/bin/python -m tic_tac_toe_shared_transformer.experiment
```

A quick smoke experiment is:

```bash
.venv/bin/python -m tic_tac_toe_shared_transformer.experiment \
  --fractions 0.25,1.0 --seeds 13 --epochs 5
```

Outputs:

- `artifacts/reports/comparison.json`: raw runs, curves, aggregates, and CPU latency;
- `artifacts/checkpoints/*.pt`: best-validation checkpoints for full-data runs;
- `COMPARISON.md`: the interpreted result from the checked-in experiment.

The architectures and auxiliary-loss weights are pluggable, for example:

```bash
.venv/bin/python -m tic_tac_toe_shared_transformer.experiment \
  --architectures candidate_policy,shared_multitask \
  --action-value-weight 0.2 --state-value-weight 0.1 --legality-weight 0.02
```

Run real-game matchups for the saved full-data checkpoints with:

```bash
.venv/bin/python -m tic_tac_toe_shared_transformer.tournament_compare
```

## Play and inspect top decisions

```bash
.venv/bin/python -m tic_tac_toe_shared_transformer.play --human X
```

On each model turn, the terminal prints its board-value prediction and top three
moves as `move=policy probability/predicted outcome`.

## Reproducibility rules

- Train/validation/test are separated by complete rotation/reflection families.
- Training fractions are nested and contain complete symmetry families.
- Every architecture sees the same examples, split, epochs, batch size, learning
  rate, optimizer, and CPU thread count.
- Checkpoints are selected on validation data; test metrics are reported afterward.
- The multi-task model gets richer labels, so compare it with `candidate_policy`
  to isolate the benefit of those labels.

The broader model taxonomy and design rationale are in
[`../model_styles_catalog`](../model_styles_catalog).
