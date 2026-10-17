# Merge audited base and on-policy fair labels

`merge_on_policy.py` combines the audited 100-game v3 base corpus with the
supplemental Transformer self-play corpus. The supplemental corpus is checked
as a separate source role; it does not claim to pass the base 100-game Gate 2
threshold. This run has 98 completed supplemental games and 60,000 rows.

Every raw visible observation is re-encoded with the v3 encoder. The base row
wins if the same exact encoded visible state occurs again. The new raw file
stores v3 state keys, labels, and shifted game IDs in NPZ row order. Source
dataset, raw, manifest, audit, split, teacher, rules, and checkpoint hashes are
recorded in the merged manifest.

The base's train, validation, and final game assignments are preserved.
On-policy game IDs start after the largest base game ID. With at least ten
distinct supplemental boards, a deterministic 10% enters validation; the
other supplemental boards enter train. No supplemental board enters final.
The ten base final boards remain sealed for final evaluation.

Run from the workspace root:

```bash
start/.venv/bin/python catanLabs/Transformer/merge_on_policy.py \
  --base catanLabs/Transformer/runs/gate2_native_v3/dataset.npz \
  --on-policy catanLabs/Transformer/runs/gate4_dagger/dataset.npz \
  --output catanLabs/Transformer/runs/gate4_merged_v3/dataset.npz
start/.venv/bin/python catanLabs/Transformer/audit_merged.py \
  --dataset catanLabs/Transformer/runs/gate4_merged_v3/dataset.npz \
  --sample-size 64
```

The resulting `dataset.npz`, `dataset.json`, `dataset.raw.jsonl.gz`, and
`dataset.split.json` are accepted by `train.py`. The completed merge has
89,899 rows, with no exact encoded-state duplicates across sources. The split
contains 78,479 train, 8,513 validation, and 2,907 sealed final rows.
`dataset.audit.json` passes: all 89,899 raw records align to NPZ metadata,
64/64 sampled tensor reconstructions match, and no hidden opponent-card
fields or duplicate encoded keys were found. No training was started by the
merge or audit commands.
