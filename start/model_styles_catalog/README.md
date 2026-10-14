# Game-model styles catalog

This folder separates two choices that are often mixed together:

1. **Backbone:** how a model represents a game state (Transformer, CNN, GNN, Mamba, and so on).
2. **Decision head:** what the model predicts (next token, action score, value, transition, or a ranking).

Our current experiment uses a **bidirectional Transformer backbone** with a
**candidate-action, multi-task decision head**. It is implemented in
[`../tic_tac_toe_shared_transformer`](../tic_tac_toe_shared_transformer).

## Catalog map

| Document | Question it answers |
|---|---|
| [01-decision-models.md](01-decision-models.md) | What can replace plain token-to-token prediction? |
| [02-backbones.md](02-backbones.md) | What can process the board or history? |
| [03-learning-styles.md](03-learning-styles.md) | Where do labels and rewards come from? |
| [04-what-we-have.md](04-what-we-have.md) | Which styles already exist in this workspace? |
| [05-shared-transformer-design.md](05-shared-transformer-design.md) | What exactly are we building and testing? |

## The key idea

Instead of asking only “which token comes next?”, present a state and a set of
candidate actions, then reuse one scorer for every candidate:

```text
state tokens -> shared Transformer state encoder
                              |
candidate action tokens ------+-> shared candidate representation
                                   |-> policy score  P(a | s)
                                   |-> action value  Q(s, a)
                                   |-> legality      L(s, a)
state summary -------------------->|-> state value   V(s)
```

This interface works with nine Tic-Tac-Toe squares, seven Connect Four columns,
or a variable list of chess moves without rebuilding the final layer.
