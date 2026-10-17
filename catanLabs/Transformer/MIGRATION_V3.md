# Visible feature v3 migration

The v2 raw snapshots already contain each opponent's public resource-card and
development-card counts. V3 adds those two counts to player token columns 32
and 33. It keeps the raw snapshots, labels, legal-action order, outcomes, games,
and board split unchanged. The v3 dataset and checkpoint schemas prevent v2
weights or tensors from being silently interpreted as v3.

The immutable source is `runs/gate2_native_v2/`. The migrated dataset is
`runs/gate2_native_v3/dataset.npz`; its manifest is `dataset.json` and its
frozen split is `dataset.split.json`. The manifest identifies the raw file as
**source v2 visible snapshots** and records source dataset, audit, split, raw,
and v3 encoder hashes. The v3 split has the same board seeds and game IDs as
v2, while its dataset hash is updated to bind it to the new NPZ.

Run from the workspace root:

```bash
start/.venv/bin/python catanLabs/Transformer/migrate_v3.py \
  --output catanLabs/Transformer/runs/gate2_native_v3/dataset.npz
start/.venv/bin/python catanLabs/Transformer/audit_dataset.py \
  --dataset catanLabs/Transformer/runs/gate2_native_v3/dataset.npz \
  --raw catanLabs/Transformer/runs/gate2_native_v2/dataset.raw.jsonl.gz \
  --report catanLabs/Transformer/runs/gate2_native_v3/dataset.audit.json \
  --sample-size 64
```

The completed migration has 100 finished games and 29,899 decision rows.
Its audit passed with zero exact raw-observation duplicates, zero hidden-field
or metadata errors, and 64/64 sampled re-encodings matching the NPZ. The
train/validation/final partitions contain 23,932/3,060/2,907 rows and retain
the v2 membership SHA-256
`8bec87c4392193f49cc5df435412d503ed13eb1d03ebc6ea1a1b47aa4d1597bc`.

The full audit checks raw metadata and duplicates across every row. Tensor
reconstruction checks a deterministic 64-row sample. The final split remains
sealed for model selection.
