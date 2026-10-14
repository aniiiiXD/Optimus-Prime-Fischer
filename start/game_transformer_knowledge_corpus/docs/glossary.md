# Minimal glossary

- **Model:** the whole parameterized function that maps game information to predictions.
- **Block:** one repeated residual unit containing attention, normalization, and an MLP.
- **Width (`d_model`):** the number of learned features carried by every token vector.
- **Head:** one attention subspace with its own query/key/value projections; multiple heads inspect different relationships in parallel.
- **Attention:** data-dependent weighted mixing of token vectors using `softmax(QKᵀ/√d)V`.
- **MLP width:** the hidden size inside each block's per-token feed-forward network; “64-wide” means 64 hidden numbers there.
- **Vocabulary:** the complete set of discrete token IDs the model can read or predict.
- **Policy:** a probability distribution over actions, `π(a|s)`.
- **Value:** predicted eventual outcome from a position, commonly in `[-1, 1]`.
- **Logit:** an unnormalized score before softmax.
- **Legal mask:** the rule-engine filter that forces impossible actions to probability zero.
- **Causal mask:** the attention filter that prevents a token from looking at future tokens.
- **Pretraining:** broad initial learning from a large dataset, commonly move imitation.
- **Post-training:** targeted updates that improve outcomes, preferences, calibration, or a specific strategy.
- **Minimax:** choose the move with the best worst-case result assuming an optimal opponent.
- **MCTS:** repeatedly simulate/select/expand/evaluate game branches and choose using visit statistics.
- **Probe:** a small classifier trained on frozen activations to test what information the model represents.
- **Checkpoint:** saved model/optimizer parameters at a particular training step.
