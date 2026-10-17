# Audited validation: MLP versus Transformer

Both checkpoints are evaluated on the **same validation rows** from the audited whole-game split. Final games remain sealed. This is teacher-action imitation, **not a win-rate comparison**.

Rows: **3,060** from **10** games in the 10-game validation partition. Choice rows (more than one legal action): **1,723**.

| Phase | Rows | Transformer exact | MLP exact | Transformer choice rows | MLP choice rows |
|---|---:|---:|---:|---:|---:|
| Initial settlement | 80 | 20.0% | 16.2% | 20.0% | 16.2% |
| Initial road | 80 | 51.2% | 51.2% | 51.2% | 51.2% |
| Main turn | 2,275 | 85.7% | 81.9% | 65.2% | 56.2% |
| Discard | 460 | 74.1% | 41.7% | 74.1% | 41.7% |
| Move robber | 165 | 16.4% | 20.6% | 16.4% | 20.6% |

Overall exact agreement: Transformer **77.6%**, MLP **70.1%**. On choice rows: Transformer **60.2%**, MLP **46.8%**.

Model-forward CPU latency excludes snapshot construction and feature encoding. Transformer p50/p95: **0.331/0.386 ms**; MLP p50/p95: **0.133/0.150 ms**.

The JSON report includes paired correctness overlap, teacher-action-type tables, checkpoint hashes, and descriptive disagreement examples. Agreement differences are diagnostic; the final split must remain untouched until model selection is frozen.

The v2 Transformer checkpoint's `teacher` metadata incorrectly says AlphaBeta1 because the old trainer hard-coded that field. Its audited dataset manifest and raw records identify **fair** as the teacher. The trainer now reads and verifies the dataset manifest when writing new checkpoints; the frozen v2 checkpoint has not been modified.

## Reproduce

```bash
start/.venv/bin/python catanLabs/Transformer/compare_diagnostics.py \
  --dataset catanLabs/Transformer/runs/gate2_native_v2/dataset.npz \
  --split-file catanLabs/Transformer/runs/gate2_native_v2/dataset.split.json \
  --transformer-checkpoint catanLabs/Transformer/runs/gate3_matched/transformer/best.pt \
  --mlp-checkpoint catanLabs/Transformer/runs/gate3_matched/mlp/best.pt \
  --latency-samples 200 \
  --output catanLabs/Transformer/runs/gate3_matched/model_comparison.json
```

Dataset SHA-256: `bc558d4bbbd75d5f4d82fdbc80ad3884bf2eb4c140f89833275f9e84f1d03280`  
Split SHA-256: `f9e151e0481cbf146f05b1feb4a4d5a03658fdc8e8e1080228521e383d95c6fb`  
Transformer checkpoint SHA-256: `7a28240e136bbcf62e597a4131c4598241e451a803889b8d28e396e77cbf6081`  
MLP checkpoint SHA-256: `1957cce9e60f6121198aeb93cccdb9caf8b707804be1965480404548fa565681`
