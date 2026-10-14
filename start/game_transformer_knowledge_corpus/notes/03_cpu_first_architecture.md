# CPU-first multi-game architecture

## Stable boundary

```text
GameAdapter
  reset(seed) → state
  legal_actions(state) → [action]
  step(state, action) → next_state
  terminal(state) → bool
  reward(state, player) → float
  encode_state(state) → state tokens
  encode_action(action) → action tokens
            │
            ▼
CandidateActionTransformer
  state encoder → shared representation
  candidate encoder → one vector/action
  cross/candidate scoring → policy logits
  pooled state → scalar value
            │
            ▼
legal mask → direct policy / alpha-beta / MCTS
```

The adapter owns rules. The model owns learned scoring. Search owns lookahead. The trainer and evaluator depend only on these contracts.

## Small starting model

| Parameter | Start value | Reason |
|---|---:|---|
| blocks | 4 | enough depth to test representation learning |
| model width | 128 | small CPU matrix multiplications |
| attention heads | 4 | head dimension 32, cleanly divisible |
| MLP width | 512 | conventional 4× expansion |
| context | game-dependent, capped at 128 | covers Othello history with room for markers |
| dropout | 0.1 during training, 0 at inference | basic regularization |
| policy/value heads | one each | separates action ranking from outcome prediction |
| precision | FP32 baseline, then INT8 dynamic/weight-only tests | correctness first, CPU speed second |

Approximate Transformer-block parameters, ignoring small biases and norms:

\[
P_{\text{block}}\approx 4d^2+2d(4d)=12d^2.
\]

For (L=4,d=128), blocks contain about (4\cdot12\cdot128^2=786{,}432) parameters. Embeddings and heads keep the whole model comfortably around one million parameters for small action vocabularies.

## Two tokenization modes

### History tokens

Use `[GAME] [SIDE] move_1 move_2 ... move_t`. This matches OthelloGPT and makes causal next-move training trivial. Its weakness is that the network must reconstruct the current board from history.

### Board plus candidates

Use `[GAME] [SIDE] square_1 ... square_N [SEP] candidate_1 ... candidate_k`. Add square/row/column/type embeddings. This makes current state explicit and supports variable candidate actions.

Start with board-plus-candidates for competition; retain history mode as the scientific interpretability experiment.

## CPU-sensitive decisions

- Small widths and short sequences matter more than shaving a few Python instructions.
- Batch 1 optimizes interactive latency; batched leaf evaluation optimizes MCTS throughput.
- Cache encoded states by a canonical state hash; cache search results with a transposition table.
- Apply board symmetries before hashing so equivalent states can share data and cache entries.
- Quantization helps only after validating policy ordering and value drift on a golden test set.
- Cap thread counts explicitly; more CPU threads can increase latency for tiny matrices.
- Measure warm and cold inference separately.

## What not to hard-code

- board dimensions;
- fixed action vocabulary size;
- perfect-information assumptions;
- exactly two players;
- history-only versus board-state input;
- legal actions derivable from logits;
- terminal rewards always in `{-1, 0, 1}`.
