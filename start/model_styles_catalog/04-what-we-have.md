# What exists in this workspace

| Location | Model or system | Style already represented |
|---|---|---|
| [`../tic_tac_toe_transformer`](../tic_tac_toe_transformer) | 10-token, 2-block, 32-wide board Transformer | fixed nine-action policy; Minimax imitation; hard-example mining |
| [`../tic_tac_toe_transformer/artifacts/checkpoints/optimal_policy.pt`](../tic_tac_toe_transformer/artifacts/checkpoints/optimal_policy.pt) | final 17,601-parameter Tic-Tac-Toe checkpoint | 100% diagnostic Minimax-optimal move selection after hard mining |
| [`../tic_tac_toe_lab`](../tic_tac_toe_lab) | earlier agents and rules | tabular/Q-learning and MLP baselines |
| [`../othello_world`](../othello_world) | OthelloGPT repository and local lab | causal move-history Transformer / next-token policy |
| [`../transformer_games_lab`](../transformer_games_lab) | reusable systems harness | game adapters, evaluation, profiling, web experiments |
| [`../game_transformer_knowledge_corpus/repos/laya`](../game_transformer_knowledge_corpus/repos/laya) | locally runnable typed-decision project | encoder plus typed/candidate-style decisions |
| `game_transformer_knowledge_corpus/repos/*` | Othello, chess, and Connect Four references | policy-value, MCTS, board classifiers, Transformer/Mamba comparisons |

## What was missing

We did not yet have one small, controlled experiment that holds the board
encoder and data split constant while comparing:

1. fixed square logits;
2. explicit candidate-action scoring;
3. candidate scoring with shared policy, action-value, state-value, and legality tasks.

That controlled experiment now lives in
[`../tic_tac_toe_shared_transformer`](../tic_tac_toe_shared_transformer).
