# CPU Catan policy lab

The current pipeline is described in the [gated execution plan](../NEXT_AGENT_PIPELINE.md).
It collects native Catanatron games and trains a policy from scratch. The
trajectory set rotates random, weighted, value, and depth-one alpha-beta
opponents. New datasets use our visibility-checked fair-value agent for
target actions by default. AlphaBeta1 labels require `--teacher alphabeta1`
and are marked as having an unverified full-game information boundary.

The [visible snapshot](visible_observation.py) gives the policy its own private
hand, public board and opponent counts, public bank, and legal actions. It does
not expose opponent hand identities, the development deck, or the RNG. The compact policy
uses a Transformer over player-relative board tokens and legal-action tokens;
the output is one score per legal action.

The current audited v3 dataset re-encodes 29,899 saved player-visible decisions
from 100 complete games. It adds public opponent card counts that v2 omitted;
labels and the sealed board split are unchanged. The [migration audit](runs/gate2_native_v3/dataset.audit.json)
records source and target hashes. The frozen v2 checkpoint can be replayed only
through [the explicit v2 adapter](legacy_v2_agent.py).

From the workspace root, using the pinned Catanatron source and shared runtime:

```bash
start/.venv/bin/python catanLabs/Transformer/train.py \
  --dataset catanLabs/Transformer/runs/gate2_native_v3/dataset.npz \
  --split-file catanLabs/Transformer/runs/gate2_native_v3/dataset.split.json \
  --output-dir catanLabs/Transformer/runs/gate3_v3_choice \
  --seconds 590 --width 32 --layers 1 --heads 4 --batch-size 24 --threads 4
start/.venv/bin/python catanLabs/Competitive/mixed_eval.py \
  --agent-file catanLabs/Transformer/agent.py \
  --checkpoint catanLabs/Transformer/runs/gate3_v3_choice/best.pt \
  --boards 8 --seed 20271400 \
  --report catanLabs/Transformer/runs/gate4_v3_screen/transformer.json
```

The optimizer script reserves a small shutdown margin under the ten-minute
limit, omits forced one-action rows from optimization, and writes a training
summary and checkpoint. The validation and final board splits remain disjoint;
the first tournament is a screen, not enough to claim a stable improvement.

The saved `runs/pilot` artifacts use schema v1 and the historical v1 evaluator.
The current encoder and trainer reject v1/v2 training artifacts. The current
benchmark uses the corrected v5 protocol, so its results remain separate from
the old pilot tournament.

## First run

The pilot used 4,000 unique observations from 14 completed games. Training ran
for 590.5 seconds on CPU and retained the epoch-8 checkpoint from a 105-epoch
run because later epochs overfit the held-out games. On four disjoint board
seeds with all seats rotated, v0 won 4/16 against random, 3/16 against weighted,
0/16 against value, and 0/16 against alpha-beta depth one. It is fast and legal,
but it is not competitive with the stronger bots yet. See the [full results](runs/pilot/RESULTS.md).
