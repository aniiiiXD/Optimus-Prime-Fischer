# Raw visible decision records

`collect.py` now writes a compressed JSONL file beside each tensor dataset:
`dataset.npz` and `dataset.raw.jsonl.gz`. Use `--raw-output` to select another
path. The raw file contains one line per unique decision, in the same order as
the NPZ rows. Its `observation.legal_actions` are in the same canonical order
as the NPZ candidate axis, and `target_action_index` equals the NPZ label.

Each line includes the immutable, player-relative visible board snapshot;
canonical legal actions; game and board seed; actor color and absolute seat; phase; behavior and
teacher IDs; teacher configuration and source hashes; the chosen legal index;
final outcome/VP if the game completed; and rules, protocol, engine and feature
schema identifiers. Opponent card identities, development-deck contents and
simulator RNG are absent. Incomplete games retain outcome and final VP as `-1`.
The format is gzip-compressed UTF-8 JSON, with no pickle. The dataset manifest
records the raw path, row count and SHA256. The writer streams completed games
so raw snapshots do not accumulate across the full collection run.

The saved [two-game smoke dataset](runs/gate2_raw_smoke/dataset.json) has 555
rows from two completed games. Its raw file is 172,084 bytes; the NPZ tensor
file is 272,451 bytes. `test_raw_storage.py` reloads the JSONL and checks that
all seven re-encoded arrays, labels, outcomes and rule identifiers agree with
the NPZ and manifest.

Run the focused checks from the workspace root:

```bash
start/.venv/bin/python -m unittest \
  catanLabs.Transformer.test_collect_protocol \
  catanLabs.Transformer.test_visible_observation \
  catanLabs.Transformer.test_raw_storage -v
```
