# Action-conditioned MLP baseline

[`mlp.py`](mlp.py) scores every legal action from the same v2 state and action tensors used by the Transformer. Each state token passes through a shared MLP; a pooled state summary and the action's tile, node, or edge token feed a small scoring head. It has no attention or search. Padded actions receive a score of `-1e9` and cannot be selected.

At width 32, it has **13,729 parameters**, within 1% of the width-32, one-layer Transformer (**13,857 parameters**). This supports a roughly parameter-matched architecture comparison. It does not itself imply equal training speed or playing strength.

[`train_mlp.py`](train_mlp.py) uses the Transformer trainer's v2 schema validation, held-out metrics, audited whole-game split helper, hard CPU optimizer budget, masked cross-entropy, AdamW settings, seed, and best-validation checkpoint selection. Both trainers stop after six complete epochs without meaningful validation-loss improvement by default, and write the best checkpoint as soon as it improves. With `--split-file`, the MLP trains only on `game_ids.train`, validates on `game_ids.validation`, and leaves `game_ids.final` sealed. [`mlp_agent.py`](mlp_agent.py) loads this checkpoint through the visible-observation adapter for the same Catanatron benchmark interface.

```bash
start/.venv/bin/python catanLabs/Transformer/train_mlp.py \
  --dataset catanLabs/Transformer/runs/gate2_native_v2/dataset.npz \
  --split-file catanLabs/Transformer/runs/gate2_native_v2/dataset.split.json \
  --output-dir catanLabs/Transformer/runs/gate3_mlp \
  --width 32 --batch-size 24 --threads 4 --seconds 600
```

The matched [Gate 3 run](runs/gate3_matched/model_comparison.md) trained on the
audited 100-game corpus. The MLP stopped at its validation plateau after 55.5
CPU seconds and reached 46.8% teacher agreement on validation decisions with
more than one legal action; the Transformer reached 60.2% within its 590.9
second cap. In the subsequent [fresh-board game screen](runs/gate4_screen/SCREEN.md),
both learned agents were far weaker than the fair-value incumbent. The focused
MLP test is:

```bash
start/.venv/bin/python -m unittest catanLabs.Transformer.tests.test_mlp -v
```
