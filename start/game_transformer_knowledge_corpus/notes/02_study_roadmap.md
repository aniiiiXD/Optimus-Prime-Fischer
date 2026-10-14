# Study roadmap

## Phase 1 — understand the model, one tensor at a time

1. Trace one Othello history through `othello_world/mingpt/model.py`; write every tensor shape on paper.
2. Calculate one attention head by hand for three toy tokens: (QK^\top/\sqrt{d_h}), causal mask, softmax, weighted values.
3. Run a public OthelloGPT checkpoint on CPU and print its top three legal and illegal logits.
4. Train a linear probe on one layer and ask whether board-square state is decodable.

**Exit test:** explain why a history-only model can represent a board it never receives directly.

## Phase 2 — separate legality from winning

1. Generate random legal Othello trajectories.
2. Train next-move/legal-move prediction.
3. Evaluate illegal-move rate separately from win rate.
4. Add expert targets from a heuristic, shallow minimax, or MCTS teacher.
5. Add a value head trained on final outcomes.

**Exit test:** show a model whose legal-move accuracy is high but whose win rate is weak, then improve win rate without hiding the distinction.

## Phase 3 — candidate-action interface

1. Define the game adapter: `reset`, `legal_actions`, `step`, `terminal`, `reward`, `encode`.
2. Encode the state once and score a variable set of legal action candidates.
3. Mask illegal actions in the engine even if the model appears to understand legality.
4. Return top-k actions, probabilities, value, confidence, and forward-pass timing.

**Exit test:** swap Tic-Tac-Toe for Othello without changing the trainer or evaluator.

## Phase 4 — post-training experiments

Run one controlled ablation at a time:

- imitation data quality and curriculum;
- winner-only filtering versus all positions;
- value-loss weight;
- hard-example replay;
- symmetry augmentation;
- shallow search targets;
- outcome-weighted behavioral cloning;
- optional self-play only after the supervised baseline is stable.

**Exit test:** every claimed improvement has confidence intervals over fixed seeds and opponents.

## Phase 5 — CPU systems work

1. Establish eager FP32 correctness and latency.
2. Sweep batch, sequence length, width, depth, heads, and threads.
3. Compare PyTorch eager/compile, ONNX Runtime, and quantized inference.
4. Add state/transposition caching and search tree reuse.
5. Measure time by tokenization, model forward, legal masking, search, and environment transition.

**Exit test:** report strength per millisecond and memory, not only raw win rate.

## Paper order

1. `02_emergent_world_representations_othello_gpt.pdf`
2. `07_word_play_othello.pdf`
3. `03_chess_transformer.pdf`
4. `04_go_transformer.pdf`
5. `01_mastering_board_games_external_internal_planning.pdf`
6. `06_mastering_chess_transformer.pdf`
7. `05_chessformer_unified_architecture.pdf`
