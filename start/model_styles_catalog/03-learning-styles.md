# Learning styles

| Learning style | Training signal | What changes the weights | Best use |
|---|---|---|---|
| Behavioral cloning | expert or Minimax move | cross-entropy gradient | fast supervised pretraining |
| Outcome supervision | final win/draw/loss | value or `Q` error | learning what leads to winning |
| Temporal-difference learning | reward plus bootstrapped next value | TD error `r + gamma V(s') - V(s)` | repeated self-play |
| Policy gradient | sampled return | `-R log P(a|s)` | optimizing outcomes directly |
| Search distillation | Minimax/MCTS action distribution | KL or cross-entropy | compressing search into one pass |
| Self-play | games against current/older policies | any of the above targets | continuously improving opponents |
| Curriculum | easier positions before harder ones | ordinary loss on staged data | stabilizing difficult games |
| Hard-example mining | positions the model currently gets wrong | extra weight or replay frequency | eliminating rare tactical blunders |
| Symmetry augmentation | rotated/reflected equivalent positions | ordinary loss on transformed data | encoding known board invariances |

For the current controlled Tic-Tac-Toe study, all labels come from exact
Minimax. No self-play is needed and no search runs at inference time.
