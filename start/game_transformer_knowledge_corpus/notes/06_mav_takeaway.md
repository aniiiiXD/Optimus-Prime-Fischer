# Multi-Action-Value Transformer: Hackathon Takeaway

Source: *Mastering Board Games by External and Internal Planning with Language Models*, arXiv:2412.12119v3 (2025).

## Page 1 — the paper in one picture

MAV is a decoder-only Transformer trained across Chess, Chess960, Connect Four, and Hex. One formatted request can ask it to reconstruct the next state, enumerate legal moves, assign a value to several moves, and emit the best move. It therefore acts as a world model, action-value function, and policy.

The paper used 1B and 2.7B-parameter models. We should copy the interface and objectives, not the scale.

## Page 2 — what must be learned

For a state `s` and legal candidate action `a`, learn:

- transition: `T(s,a) -> s_next`;
- legality: `A_legal(s)`;
- action value: `Q(s,a) = expected outcome after action a`;
- policy/tie-breaker: choose the strongest action among similarly valued moves;
- terminal result: identify win, draw, or loss.

The paper discretizes values into 64 tokens and reports that taking the expectation over the predicted bucket distribution works better than using only its mode. Candidate order and requested `top_k` are randomized during training to reduce slot/order bias and hallucination. Loss on input tokens is masked; the model learns the requested outputs.

Our smaller model can replace textual generation with candidate tokens and explicit heads:

`board + candidate actions -> shared Transformer -> policy logits + Q values + state value`.

## Page 3 — data strategy

The paper uses strong teachers: Stockfish for Chess/Chess960, Fhourstones for Connect Four, and neurobenzene for Hex. The reported corpus contains 3.1B Chess positions, 1.2B Chess960 positions, 21.8M Connect Four positions, and 125.6M Hex positions. Connect Four uses 2,458,293 epsilon-greedy games with epsilon 0.4.

We should begin far smaller:

1. sample diverse legal states;
2. enumerate every legal action;
3. label each action with solver/minimax/MCTS value;
4. add the chosen best action and final outcome;
5. augment with valid board symmetries;
6. oversample mistakes, tactical positions, and rare terminal states;
7. cache canonical state labels across runs.

## Page 4 — planning

Direct policy is cheapest. External MCTS repeatedly queries the model and reliably improves strength, but costs more latency. Internal search fine-tunes the model on linearized minimax traces and executes the learned search in one long generation.

The paper's external-search prior softmaxes the top-k action values and mixes it with a uniform legal-action policy. Reported tuned values are temperature 0.1, top-k 5, exploration mixture 0.05, asynchronous batch size 16. Tree reuse saved 5-10% of model calls. Mean value-bucket scoring outperformed max/mode scoring.

For our CPU engine: first use direct policy, then shallow alpha-beta or MCTS with batched leaves and tree reuse. Internal search is a later experiment because long autoregressive traces are CPU-expensive.

## Page 5 — objectives and milestones

Suggested objective:

`L = lambda_policy * CE(policy,target) + lambda_q * value_loss + lambda_v * outcome_loss + lambda_legal * BCE(legal_mask)`.

Start with `(lambda_policy, lambda_q, lambda_v, lambda_legal) = (1.0, 1.0, 0.25, 0.25)` and ablate each term. If Q uses buckets, use cross-entropy and decode with the expected bucket value; if it uses a scalar, use Huber loss.

Milestones:

1. zero illegal moves after masking and high raw legality recall;
2. low action-value error and calibrated outcomes;
3. stronger direct policy than heuristic/minimax baselines at equal time;
4. measurable strength gain from a fixed search budget;
5. stable batch-1 CPU latency, memory, and optimized/reference parity.

Run the ablation ladder: policy only; policy plus Q; add symmetry; add hard-example replay; add search; quantize. Judge every change using win rate/Elo, regret, calibration, p50/p95 latency, memory, and strength per second.
