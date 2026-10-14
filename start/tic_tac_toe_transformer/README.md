# Tic-Tac-Toe Transformer — architecture, step by step

This folder contains tokenization, learned token/spatial embeddings,
multi-head attention, a stack of two Transformer blocks, a move policy head,
and a supervised training loop. It does not yet contain a value head or a
separate game engine. The existing game engine is in
`../tic_tac_toe_lab/tic_tac_toe.py`.

## Position encoding

The board is read in row-major order: squares 1 through 9, left-to-right and
top-to-bottom. Each square becomes one categorical token, followed by a token
for whose turn it is.

| Token ID | Meaning |
|---:|---|
| 0 | Reserved padding token; not emitted for a board |
| 1 | Empty square |
| 2 | X on square |
| 3 | O on square |
| 4 | It is X's turn |
| 5 | It is O's turn |

Example board:

```text
X |   |
--+---+--
  | O |
--+---+--
  |   | X
```

With O to move, the token sequence is:

```text
[2, 1, 1, 1, 3, 1, 1, 1, 2, 5]
```

The first nine positions correspond to squares 1–9; the last position tells
the model whose turn it is. Token 0 is reserved for padding and is not used in
this fixed-length input. The IDs are labels, not numeric measurements. The
embedding layer below turns each ID into a learned vector and adds learned row
and column vectors for each square.

## Learned embeddings

`TicTacToeEmbeddings(d_model=32)` maps each token into a learned 32-number
vector. For each square, it adds learned row and column vectors to the token
vector. This gives the model a 2D coordinate signal while keeping the same
token meanings reusable in every square. The turn token gets its own learned
slot vector.

For a batch of boards, the output has shape `(batch, 10, d_model)`. For one
board, that is a `10 × d_model` matrix: nine square vectors and one turn vector.
These vectors are learned by gradient descent once we add the training code.

```python
import torch
from tic_tac_toe_transformer.embeddings import TicTacToeEmbeddings
from tic_tac_toe_transformer.input import tokenize_position

tokens = tokenize_position(
    ["X", " ", " ", " ", "O", " ", " ", " ", "X"],
    current_player="O",
)
embed = TicTacToeEmbeddings(d_model=32)
vectors = embed(torch.tensor([tokens], dtype=torch.long))
print(vectors.shape)  # torch.Size([1, 10, 32])
```

## Attention design

The initial model dimensions are deliberately small:

| Setting | Choice | Meaning |
|---|---:|---|
| Transformer blocks | 2 | Repeat attention + MLP twice in the eventual model |
| Embedding width | 32 | Each square is represented by 32 numbers |
| Attention heads | 4 | Four learned ways to compare the 10 tokens |
| Width per head | 8 | `32 / 4` numbers per head |
| MLP width | 64 | Planned inner width after attention in each block |

`MultiHeadSelfAttention` implements one attention layer. The model dimensions
are collected in `config.py`; the attention layer returns `(batch, 10, 32)`.
Use `return_weights=True` to also inspect the attention matrix with shape
`(batch, 4, 10, 10)`. Each row of that matrix describes where one token looks.

The computation is:

```text
input vectors → Q, K, V → QKᵀ / √8 → softmax → weights × V → output vectors
```

All 10 tokens can attend to one another because the full board is available at
once. A causal mask is for predicting future sequence tokens; we are predicting
a move from a complete current board.

Each block does:

```text
tokens → LayerNorm → self-attention → add original tokens (residual)
       → LayerNorm → 32 → 64 → 32 MLP → add previous tokens (residual)
```

`TicTacToeTransformerBlocks` stacks the configured two blocks and keeps the
shape `(batch, 10, 32)`. It takes embedded vectors as input; embeddings are a
separate component.

```python
from tic_tac_toe_transformer.blocks import TicTacToeTransformerBlocks

context = TicTacToeTransformerBlocks()(vectors)
print(context.shape)  # torch.Size([1, 10, 32])
```

## Move policy head

`MovePolicyHead` takes the nine contextual square vectors from the Transformer
and applies the same learned `32 → 1` scorer to each. This returns nine logits:
one score for each square. These are scores, not probabilities yet.

The game rules supply a boolean legal-action mask. `mask_illegal_moves` sets
illegal square scores to negative infinity, so they receive zero probability
after softmax and cannot be selected by `choose_move`.

```python
from tic_tac_toe_transformer.policy import (
    MovePolicyHead,
    choose_move,
    policy_probabilities,
)

logits = MovePolicyHead()(context)  # (batch, 9)
legal = torch.tensor(
    [[False, True, True, True, False, True, True, True, False]],
    dtype=torch.bool,
)
probabilities = policy_probabilities(logits, legal)
square_index = choose_move(logits, legal)  # 0-based square index
```

The mask is external because the rules engine decides legality; the learned
policy only ranks the moves it is allowed to play.

## Minimax-labelled dataset

Generate one example for every distinct legal, nonterminal position:

```bash
python -m tic_tac_toe_transformer.dataset
```

This writes `artifacts/minimax_positions.jsonl`. Minimax supplies equal target
probability to every move that has the best perfect-play outcome. Each JSONL
row contains the board and player, 10 input tokens, a 9-square legal mask, a
9-value target policy, optimal moves, and the outcome value from the current
player's perspective. The generator reuses the existing Tic-Tac-Toe rules in
`tic_tac_toe_lab/` and does not train the Transformer.

For example, a target policy such as `[0, 0.5, 0, 0, 0.5, 0, 0, 0, 0]` means
Minimax considers squares 2 and 5 equally good. The training loop passes
batches of these token rows through the Transformer and compares its move
scores with these target probabilities.

## Train the policy

```bash
python -m tic_tac_toe_transformer.train --epochs 100 --log-every 5
```

Run this with Python and PyTorch installed (in this workspace, the existing
environment is `./.venv/bin/python`).

This starts all model weights randomly and trains them with AdamW using soft
target cross-entropy against the Minimax move distribution. The default split
is 80% training and 20% validation. Validation reports the fraction of boards
where the model's top legal move is among Minimax's optimal moves; it does not
penalize the model for choosing a different move that is equally optimal.
The best validation checkpoint is written to
`artifacts/minimax_policy.pt`.

Each printed training update also reports CPU process time, wall time, average
CPU utilization (which may exceed 100% when multiple cores are used), estimated
cores occupied, and peak resident memory (RSS). These measurements use Python's
standard library and do not add a profiling dependency. Change how often
updates print with `--log-every 1` (every epoch) or a larger number.

## Play against the trained Transformer

After training, run:

```bash
./.venv/bin/python -m tic_tac_toe_transformer.play
```

You play as X by default; the Transformer plays as O. It prints its top three
legal move choices and their probabilities before playing the best one. To let
the Transformer play first, add `--human O` to the command.
The game's rules reject occupied squares and stop when someone wins or the board
is full.

## Strength and CPU baseline

Run the complete baseline before changing the dataset, loss, model, or policy:

```bash
./.venv/bin/python -m tic_tac_toe_transformer.evaluation.run_baseline
```

The report measures the model on every reachable nonterminal position, plays
100 games from each side against a seeded random agent, plays once from each
side against deterministic Minimax, and benchmarks warmed-up CPU inference at
batch sizes 1 and 128. It saves the full machine-readable result to
`artifacts/reports/baseline_v0.json`.

The position report distinguishes three ideas:

- **Optimal accuracy:** the selected move has the same game-theoretic value as
  Minimax's best move.
- **Regret:** best move value minus selected move value. Values are -1, 0, and
  +1, so a move that changes a forced win into a loss has regret 2.
- **Losing blunder:** the model had a win or draw available but chose a move
  that allows a forced loss.

It also breaks errors down by move number and player, records chosen-square
counts, reports p50/p95 move latency, estimated matmul FLOPs, throughput,
average CPU cores, parameter memory, and peak process memory. The all-position
score is a diagnostic because these positions supplied the original training
data; it is not presented as unseen-test accuracy.

## Counterexample-guided optimal training

Refine the baseline by training primarily on its mistakes:

```bash
./.venv/bin/python -m tic_tac_toe_transformer.training.hard_mining
```

Each round exhaustively verifies the greedy policy, collects only suboptimal
positions, adds their rotations and reflections to a persistent hard buffer,
and includes a small replay sample to prevent forgetting. The loss rewards the
entire set of equally optimal actions and applies a margin between the best
optimal and best suboptimal action. Verification stops only when every one of
the 4,520 reachable nonterminal positions selects an optimal move.

The completed run reduced 184 counterexamples to zero in seven rounds. Its
largest training round used 1,799 selected positions, took 12.47 seconds of CPU
wall time overall, and saved the certified weights to
`artifacts/checkpoints/optimal_policy.pt`. The independent report is stored at
`artifacts/reports/optimal_policy.json`: zero blunders, draws against Minimax
from both sides, and no losses in 200 seeded games against random play.

The terminal game now loads this optimal checkpoint by default.

## Tiny forward pass (untrained)

```python
import torch
from tic_tac_toe_transformer.input import tokenize_position
from tic_tac_toe_transformer.embeddings import TicTacToeEmbeddings
from tic_tac_toe_transformer.blocks import TicTacToeTransformerBlocks
from tic_tac_toe_transformer.policy import MovePolicyHead, choose_move

board = ["X", " ", " ", " ", "O", " ", " ", " ", "X"]
tokens = tokenize_position(board, current_player="O")
vectors = TicTacToeEmbeddings()(torch.tensor([tokens], dtype=torch.long))
context = TicTacToeTransformerBlocks()(vectors)
logits = MovePolicyHead()(context)
legal = torch.tensor(
    [[False, True, True, True, False, True, True, True, False]],
    dtype=torch.bool,
)
move = choose_move(logits, legal)
print(context.shape)  # torch.Size([1, 10, 32])
print(move)  # zero-based square index
```

A newly constructed model has random initial values. The code above
demonstrates the path to a legal move; load `artifacts/minimax_policy.pt` to
use the weights learned by the training script.
