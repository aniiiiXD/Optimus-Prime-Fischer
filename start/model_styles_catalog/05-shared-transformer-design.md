# Shared candidate Transformer design

## One state, many decisions

For a state `s` and candidate move `a`, the model constructs

`h(s,a) = MLP([h_square(s,a); h_turn(s); e_action(a)])`.

The same small heads are reused for every candidate:

- policy logit: `z_a = w_p^T h(s,a)`;
- outcome logits: `q_a = W_q h(s,a)` for loss/draw/win;
- legality logit: `l_a = w_l^T h(s,a)`;
- board value: `v = W_v h_turn(s)` for loss/draw/win.

The input candidate list may be `[0,...,8]` during training or only the legal
moves during inference. Since the scorer is shared, rearranging candidates only
rearranges the outputs.

## Controlled comparison

| Variant | Backbone | Candidate head | Training signals |
|---|---|---|---|
| `classic_policy` | existing board Transformer | fixed score at each square token | policy only |
| `candidate_policy` | same-size board Transformer | explicit shared `(state, action)` scorer | policy only |
| `shared_multitask` | same as candidate variant | same scorer plus auxiliary heads | policy + `Q` + `V` + legality |

All variants use a symmetry-grouped train/validation/test split, identical
training subsets, AdamW settings, epoch budgets, and CPU thread settings.
This avoids putting a rotated copy of a test board into training.

## What the experiment can and cannot prove

- If `candidate_policy` beats `classic_policy`, the explicit action interface helped.
- If `shared_multitask` beats `candidate_policy`, richer supervision helped.
- A small three-seed result estimates variance; it is not a universal theorem.
- Tic-Tac-Toe is tiny, so the reusable interface matters more than raw latency.
