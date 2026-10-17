# On-policy fair-label collector

`collect_on_policy.py` prepares a DAgger-style dataset. One v3 Transformer
checkpoint plays all four seats using only player-relative
`VisibleObservation` snapshots. At each visited decision, the incumbent fair
teacher supplies a legal-action label before the Transformer acts. The game
continues with the Transformer's choice. Records are committed only when the
game finishes; a wall-capped or turn-capped game contributes no rows.

The collector uses the provisional v5 rules, canonical action ordering,
isolated search-copy RNG, and forced hypothetical outcomes. Output includes
v3 NPZ tensors, raw visible JSONL v2 records, and a manifest with the behavior
checkpoint hash, teacher source/config hashes, engine commit, rules hash,
completion status, and timing counters. Outcomes are analysis fields and are
never inputs to the policy. The optional injected policy in the unit test is
marked `smoke_visible_stub` and `smoke_only=true` so its rows cannot be
mistaken for Transformer self-play.

Example command after choosing a frozen v3 checkpoint:

```bash
start/.venv/bin/python catanLabs/Transformer/collect_on_policy.py \
  --checkpoint catanLabs/Transformer/runs/gate3_v3_choice/best.pt \
  --games 10 --seed 20273000 --wall-seconds 120 --max-examples 4000 \
  --output catanLabs/Transformer/runs/on_policy/dataset.npz
```

Only the bounded unit smoke has been run so far; no full on-policy corpus has
been collected or used for training.

```bash
start/.venv/bin/python -m unittest catanLabs.Transformer.test_collect_on_policy -v
```
