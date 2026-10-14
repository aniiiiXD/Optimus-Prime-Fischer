# Decision-model styles

These are output designs. Any of them can sit on top of a Transformer, CNN,
GNN, recurrent model, or state-space model.

| Style | Output | Typical objective | Strength | Main limitation |
|---|---|---|---|---|
| Autoregressive next-token model | next token or move | `-log P(a_t | a_<t)` | Natural for move histories and generation | Must learn rules and strategy indirectly |
| Fixed policy head | one logit per global action | cross-entropy to expert policy | Very small and fast | Output size is tied to one game |
| Candidate-action scorer | one score per supplied action | listwise cross-entropy | Variable action spaces; one shared scorer | Needs action features/tokens |
| Action-value model | `Q(s,a)` for each action | regression or outcome classification | Directly compares consequences | Targets may require search or rollouts |
| Policy-value model | `P(a|s)` and `V(s)` | policy loss + value loss | Useful alone and inside MCTS | Shared losses must be balanced |
| Multi-task decision model | policy, `Q`, `V`, legality/confidence | weighted sum of losses | More supervision per position | Gains are not architecture-only gains |
| Pairwise ranker | probability that action A beats B | pairwise logistic/ranking loss | Learns preferences from comparisons | Inference may need many pairs |
| Energy-based model | low energy for compatible `(s,a)` | contrastive/ranking loss | Flexible and naturally candidate-based | Negative sampling matters |
| World/transition model | predicted next state and reward | state/reward prediction loss | Supports planning in imagination | Small prediction errors can compound |
| Actor-critic | actor policy plus learned critic | policy gradient + value loss | Learns from outcomes without expert moves | Higher variance and harder debugging |
| Search-distilled policy | move distribution found by search | KL/cross-entropy to search visits | Makes expensive search cheap at runtime | Generating search targets costs CPU |

## Equations

For candidate actions `A(s)`, a shared scorer `f_theta` produces

`z_a = f_theta(s, a)` and `P_theta(a|s) = exp(z_a) / sum_{b in A(s)} exp(z_b)`.

A policy-value model also predicts `V_theta(s)`, while an action-value model
predicts `Q_theta(s,a)`. Our multi-task experiment minimizes

`L = L_policy + 0.5 L_Q + 0.25 L_V + 0.1 L_legal`.

The game engine still applies a legal-action mask. Learned legality is an
auxiliary representation test, not permission to make illegal moves.
