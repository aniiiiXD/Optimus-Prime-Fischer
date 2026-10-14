# Repository guide

## Read in this order

| Repository | What it teaches | First files to inspect | Practical warning |
|---|---|---|---|
| `othello_world` | canonical causal OthelloGPT and probes | `data/othello.py`, `mingpt/model.py`, `mingpt/dataset.py`, `train_gpt_othello.ipynb` | original default training is GPU-heavy; use its small public checkpoint for study |
| `othello_mamba` | clean Transformer/Mamba/Jamba comparison | `othello.py`, `data.py`, `models/lm.py`, `train.py` | data and pretrained models are separate downloads |
| `laya` | typed one-pass decisions, batching, confidence, CPU paths | `README.md`, `laya/common.py`, `laya/agent.py`, `laya/train.py`, `docs/compile-and-fast-path.md` | Laya is a general decision encoder, not a trained board-game engine |
| `ChessTransformer` | modern policy/value network plus MCTS and alpha-beta | `doc/transformer.md`, `doc/tokenization.md`, `src/chesstransformer/models/`, `src/chesstransformer/bots/`, `scripts/bench_inference.py` | primarily GPU-oriented; bundled tracked assets make this snapshot large |
| `chessformer` | compact, legible chess training/inference package | `chessformer/model.py`, `dataset.py`, `tokenizer.py`, `trainer.py`, `inference.py` | verify the paper/repo checkpoint format before mixing revisions |
| `chess-transformers` | configurable chess encoders and encoder-decoders | `chess_transformers/models/`, `configs/models/`, `train/train.py`, `evaluate/evaluate.py` | full datasets/checkpoints are external |
| `chessformers` | small decoder-only chess model and console inference | `chessformers/model.py`, `dataset.py`, `tokenizer.py`, `trainer.py`, `inference.py` | older project; useful for simplicity, not as a current performance baseline |
| `connect-4` | end-to-end game deployment with CNN/Transformer alternatives | `README.md`, `model_wrappers.py`, `connect4_engine.py` | deployment-centric and TensorFlow-based; do not copy architecture blindly |
| `Emergent-World-Representations-Othello` | independent Othello world-representation/probing work | inspect the Python/notebook files and `Documents/` | large checkpoints are cataloged but intentionally excluded from source-only download |

## Concrete OthelloGPT code path

In `repos/othello_world/mingpt/model.py`:

1. `GPT.__init__` creates `tok_emb`, learned `pos_emb`, a stack of `Block`s, final layer norm, and vocabulary `head`.
2. `CausalSelfAttention.forward` projects (Q,K,V), splits the width into heads, applies the triangular causal mask, softmaxes, combines values, and projects back.
3. `Block.forward` applies attention and the 4×-wide MLP through residual connections.
4. `GPT.forward` maps a move prefix to logits and optionally computes cross-entropy against the next moves.
5. `GPTforProbing` exposes an internal layer so a probe can decode board state.

In `repos/othello_world/data/othello.py`, the game engine converts move sequences into legal board transitions. This file is the bridge between token correctness and actual Othello correctness.

## Concrete Laya code path

In `repos/laya/laya/common.py`:

- `QTYPES` assigns integer identities to `choice`, `score`, and `noul`.
- question rendering converts structured criteria into the option-bearing model input.
- the encoder is loaded through Hugging Face `AutoModel`.
- output probabilities are interpreted according to question type.

In `repos/laya/laya/train.py`, supervised examples are converted into targets, including soft label distributions. In `laya/confidence.py`, confidence gating is separated from the raw decision. In `laya/shortlist.py`, large candidate sets can be narrowed by embeddings or tournament elimination.

The game-engine lesson is architectural, not a promise of weight transfer: keep state encoding, candidate enumeration, policy scoring, calibration, and abstention as explicit modules.

## Which code should we borrow?

- Borrow **Othello rules and sequence representation** from `othello_world` or the cleaner `othello_mamba` reimplementation.
- Borrow **candidate-oriented API ideas and batching/evaluation discipline** from `laya`.
- Borrow **policy/value separation, MCTS, tree reuse, and inference regression tests** from `ChessTransformer`.
- Borrow **small decoder simplicity** from `chessformers` when teaching or making a minimal baseline.
- Do not merge entire repositories. Reimplement the small interfaces we need and keep provenance in comments.
