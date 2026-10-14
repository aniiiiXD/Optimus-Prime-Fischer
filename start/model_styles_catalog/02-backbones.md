# State-encoding backbones

| Backbone | How information moves | Good fit | CPU concern |
|---|---|---|---|
| Bidirectional Transformer encoder | every token attends to every board token | complete board positions | attention cost grows as `O(T^2 d)` |
| Causal Transformer decoder | each history token sees only earlier moves | next-move prediction from histories | KV caching helps autoregressive inference |
| CNN / ResNet | local filters spread information across layers | regular 2-D boards | highly optimized convolutions |
| Graph neural network | messages follow board/game relations | irregular boards or relational actions | scatter operations can be CPU-unfriendly |
| RNN / GRU / LSTM | one recurrent state advances per move | compact move histories | sequential dependency limits batching |
| Mamba / state-space model | recurrent state with parallel training form | long histories | kernel quality determines real CPU gains |
| Set Transformer | attention over an unordered candidate set | variable legal-action lists | candidate count can dominate cost |
| Hybrid board + action encoder | one state encoder plus shared candidate scorer | reusable multi-game decision engine | extra candidate projection is small |

## Important distinction

“Transformer” describes the representation backbone; “policy”, “value”, and
“candidate scoring” describe its outputs. A model can therefore be both a
Transformer and a policy-value or world model.
