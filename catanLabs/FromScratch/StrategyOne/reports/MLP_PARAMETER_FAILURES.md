# MLP parameter failure analysis

## Method

This inspected the original MLP on the clean game-level holdout: 99 positions where the external teacher selected `BUY_DEVELOPMENT_CARD` and `END_TURN` was legal. It compared the two raw legal-action scores and ablated each encoded state feature group while holding the exact same legal action pair fixed.

This identifies behaviourally failing parameter paths. A neural decision is distributed across many weights, so a single weight norm is not a causal explanation.

## Exact observed failure

| Signal | Original MLP | PPO MLP |
| --- | ---: | ---: |
| Correct `BUY_DEVELOPMENT_CARD` choice | 0 / 99 | 0 / 99 |
| Predicted `END_TURN` | 98 / 99 | 98 / 99 |
| Mean Buy rank among legal actions | 3.54 | 2.79 |
| Mean `Buy score − End score` | -3.260 | -1.505 |

The action-score path for `BUY_DEVELOPMENT_CARD` is therefore broken at the decision boundary: Buy improves slightly after PPO but never becomes the top legal action.

## Failing input interaction

The important state-group test is the score change after masking one group. A value near zero means that group has almost no effect on the Buy-versus-End decision.

| State input group | Original score change when masked | Interpretation |
| --- | ---: | --- |
| Own resource counts | **-0.001** | The model effectively ignores its own resources when deciding Buy versus End. |
| Own development cards | -0.014 | Almost ignored. |
| Own VP | -0.023 | Almost ignored. |
| Building layout | +0.282 | Observed building state slightly suppresses Buy. |
| Tiles and robber | -1.011 | The board state boosts Buy, but cannot overcome the action-score failure. |
| Ports | -0.431 | Port state boosts Buy, but cannot overcome the action-score failure. |
| Opponent public state | -0.255 | Opponent context has some influence. |

Negative means that removing the observed group made Buy relatively worse; positive means masking it made Buy relatively better.

## What is *not* the culprit

The first-layer weight norm for the `BUY_DEVELOPMENT_CARD` action bit is **0.487**, larger than the `END_TURN` bit's **0.387**. So this is not simply a zeroed or tiny Buy-action column. The error emerges in the later nonlinear interaction between action type and state.

## Most likely training cause

The dataset has 1,758 `BUY_DEVELOPMENT_CARD` labels out of 156,471 decisions (1.12%), compared with 53,906 `END_TURN` labels (34.45%). The ranker is strongly calibrated toward ending turns, and it has failed to learn resource-sensitive conditions that overcome that prior.

PPO reduced the mean Buy-versus-End deficit from 3.260 to 1.505 but did not produce a single correct Buy choice. Terminal reward alone is too weak to repair this rare-action supervised failure.

## Repair target

Train the same legal-action MLP with a balanced/pairwise Buy-versus-End loss and an imitation anchor during RL. This is a change to the learning objective and sampling, not a hand-written gameplay heuristic.
