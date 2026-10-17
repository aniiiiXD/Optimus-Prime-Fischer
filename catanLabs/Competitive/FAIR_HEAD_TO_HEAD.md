# Direct fair-opponent evaluation

[`fair_head_to_head_eval.py`](fair_head_to_head_eval.py) gives the candidate exactly three opponents in every game: one `FairValueAgent` using `fair_army_finish.json`, one Catanatron `ValueFunctionPlayer`, and one depth-one `AlphaBetaPlayer`. It runs the v5 first-to-10 rules with isolated copy RNG, forced hypothetical outcomes, canonical action order, no domestic trade offers, four candidate seats per board, and mixed-eval's opponent rotation. The tactical and fair candidates must share identical FairValue settings; the tactical configuration may add only `tactical_*` keys.

From the workspace root, run paired reports with the **same** seed, board count, and time cap:

```bash
start/.venv/bin/python catanLabs/Competitive/fair_head_to_head_eval.py \
  --candidate tactical --boards 24 --seed 20272000 --max-game-seconds 90 \
  --report catanLabs/Competitive/runs/fair_head_to_head/tactical.json
start/.venv/bin/python catanLabs/Competitive/fair_head_to_head_eval.py \
  --candidate fair --boards 24 --seed 20272000 --max-game-seconds 90 \
  --report catanLabs/Competitive/runs/fair_head_to_head/fair.json
start/.venv/bin/python catanLabs/Competitive/compare.py \
  catanLabs/Competitive/runs/fair_head_to_head/tactical.json \
  catanLabs/Competitive/runs/fair_head_to_head/fair.json \
  --output catanLabs/Competitive/runs/fair_head_to_head/comparison.json
```

The JSON format is accepted by `compare.py`; each game also records the exact opponent-to-color assignment. This direct table has its own protocol name, so it cannot be accidentally compared against a different mixed-opponent table.

## One-board smoke

Seed `20271960` was run with four seats and a 20-second per-game cap for both candidates. All eight games completed with zero errors, and the `(seed, seat, candidate color, opponent assignments)` tuples matched exactly. Tactical and fair each won 0/4; tactical VP by seat was `[5, 9, 7, 6]` and fair VP was `[5, 8, 7, 6]`. The [paired smoke comparison](runs/fair_head_to_head_smoke/comparison.json) is a format and legality check, **not a strength estimate**. A multi-board held-out run is needed for any tactical-policy decision.
