# Checkpoint catalog

This separates **source code present locally**, **weights present locally**, and **weights merely linked by a repository**.

| Project | Local weight status | Upstream availability | Recommendation |
|---|---|---|---|
| Othello World | small linear probe checkpoint is included; base GPT checkpoint is not | README links Google Drive base/probe checkpoints; TransformerLens also exposes an OthelloGPT workflow | use the TransformerLens/public checkpoint for the first CPU forward-pass study |
| Othello Mamba | none | README says to contact the author for pretrained models | train a tiny config or use it only as architecture reference |
| Laya | none; source only | Hugging Face: `convaiinnovations/laya`, `laya-multilingual`, and `laya-typed-decisions` | do not fetch all three casually: upstream documents roughly 2.3 GB total |
| ChessTransformer | two ~45 MB `ema_state.pt` files are locally present; several `.safetensors`/ONNX paths are Git LFS pointer text, not weights | repository/Docker setup resolves model assets | inspect `ema_state.pt` format before inference; never mistake a 130-byte LFS pointer for a checkpoint |
| chess-transformers | none | four checkpoint URLs are in the README and auto-download paths | choose one architecture only after checking memory requirements |
| chessformer | none | Hugging Face model at `edarsem/chessformer` | best clean external checkpoint candidate for its matching code revision |
| chessformers | none | README links a Google Drive checkpoint | older/simple study option |
| Connect 4 | `network_training/Transformer/transformer.weights.h5` is present (~10 MB) | tracked directly in repository | smallest immediately local trained Transformer artifact in this corpus |
| Emergent World Representations Othello | a partial set of small probe checkpoints is present; large base-model checkpoints are excluded | upstream repository tracks many duplicated 13–101 MB checkpoints and datasets | use as probing reference; avoid downloading the entire binary history |

## Safety check before loading

```bash
file path/to/checkpoint
head -n 3 path/to/checkpoint   # an LFS pointer begins with “version https://git-lfs...”
```

Only load checkpoints from sources you trust. PyTorch pickle-based `.pt`, `.pth`, and `.ckpt` files can execute code during unsafe deserialization; prefer `safetensors` where available and use framework safe/weights-only loading modes.
