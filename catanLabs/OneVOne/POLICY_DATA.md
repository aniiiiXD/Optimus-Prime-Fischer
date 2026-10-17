# MLP policy bootstrap data

## What the first dataset contains

`data/bootstrap_20272000_16boards.jsonl.gz` contains 20,407 decision examples
from 192 completed games. Six teachers contributed examples: `onevone`,
`fair`, `tactical`, `catanatron`, `alphabeta2`, and `weighted`. Each teacher
played 16 board seeds from both seats. The paired seats reduce turn-order bias;
the validation split later groups all examples from the same board seed
together.

Every JSONL row stores:

- the focal player's fixed-length visible-state vector;
- feature vectors and action-family IDs for every legal action;
- the teacher's selected legal-action index;
- the teacher, opponent, seed, seat, and final game outcome (`+1` win, `-1`
  loss).

The manifest records the rules, engine revision, hashes, teacher results, game
caps, and dataset checksum. Games that error or hit a cap are excluded from the
training rows.

The expanded set in `data/bootstrap_20272000_64boards.jsonl.gz` merges that
starter set with 57,205 examples from 576 additional completed games. It has
77,612 rows total, 768 games, and 64 distinct board seeds. The second seed
range (20272016–20272063) does not overlap the starter range (20272000–20272015).
All 768 games completed without errors or caps. `merge_policy_data.py` checks
feature/rule compatibility and rejects overlapping board seed ranges.

## Information boundary

The state encoder includes the public board and scores, current prompt, the
acting player's private resource and development-card hand, the opponent's
public resource-card count, and any current public trade offer. It does not
include the opponent's resource identities, opponent hidden victory points,
development deck order, future dice, or theft results. The legal-action list
comes directly from the engine and is the action mask: the policy can rank only
actions that the rules currently permit.

The v1 encoder assumes the fixed 54-node, 19-hex, 72-edge BASE board. Change
the feature schema and collect a new versioned dataset if the event uses a
different map or rules interface.

## Policy model and targets

`FiveHeadPolicy` has a shared state encoder, a shared candidate-action encoder,
five action-family score heads, and a state value head. The action heads cover
builds, robber/discard, trades, development cards, and turn flow. For one
position, it scores all legal candidates and selects the highest scoring one.

The first fit is teacher behavior cloning plus terminal outcome regression.
Moves from teacher wins get full policy weight; moves from losses get 0.25
weight. The value head predicts the final `+1/-1` result from each focal state.
That result is repeated across the game's decisions, so it is a noisy state
label and does not identify which move caused the win or loss. The value head
is not yet used for search or action selection.

Because teacher identity is metadata rather than a model input, the model is
asked to imitate a mixture of styles. Conflicting choices can blur the policy,
and copying teachers alone cannot surpass their information. A stronger next
dataset should add search-ranked alternatives or same-state rollout
comparisons, then train with online league self-play and retain frozen
opponents.

## Bootstrap fits and tournament result

The data split groups all games from each board seed together. In the starter
fit, that gave 14,447 training examples and 5,960 validation examples from 12
and 4 seeds. The first fit's 75.1% action accuracy counted forced one-action
prompts and overstated policy quality. The corrected hierarchical fit reached
43.6% joint accuracy on non-forced held-out choices and 69.6% outcome-sign
accuracy. On a fresh 32-board paired-seat screen, it won 0/64 against the
incumbent, 1/64 against Catanatron, and 3/64 against AlphaBeta depth 2.

To check whether board diversity was the limiting factor, the expanded fit
used 47 training and 17 validation seeds (57,107 training rows and 20,505
validation rows). It stopped at epoch 7 and kept the best checkpoint from
epoch 2. Held-out non-forced metrics were 41.6% joint action accuracy, 86.8%
action-family accuracy, 53.4% conditional target accuracy, and 73.1%
outcome-sign accuracy. These remain imitation/value-prediction metrics, not
direct evidence of winning strength.

On a fresh 32-board paired-seat screen, all 192 games completed without errors
or caps, but the larger policy still lost heavily:

| Opponent | Wins | Win rate | Mean VP margin |
| --- | ---: | ---: | ---: |
| Current OneVOne heuristic | 3/64 | 4.7% | -5.63 |
| Catanatron web bot | 7/64 | 10.9% | -4.78 |
| AlphaBeta depth 2 | 3/64 | 4.7% | -5.55 |

More board seeds did not rescue the policy. That makes target quality a more
useful next focus than blindly collecting more of the same: teachers disagree,
the league includes a weak teacher that lost all 96 focal games, and each state
inherits the same terminal outcome label regardless of which decision caused
it. Search-ranked action alternatives, same-state rollouts, or online league
self-play are better candidates for the next dataset. This experiment does not
establish which one will work. The MLP is not promoted; the playable agent
remains the heuristic.

The v1 and hierarchical checkpoints and reports are in
[`runs/mlp-bootstrap-20261016`](runs/mlp-bootstrap-20261016/) and
[`runs/mlp-hierarchical-bootstrap-20261016`](runs/mlp-hierarchical-bootstrap-20261016/).
The expanded checkpoint and its training report are in
[`runs/mlp-hierarchical-64board-20261016`](runs/mlp-hierarchical-64board-20261016/).
Raw match results are in
[`reports/mlp_bootstrap_screen.json`](reports/mlp_bootstrap_screen.json) and
[`reports/mlp_hierarchical_bootstrap_screen.json`](reports/mlp_hierarchical_bootstrap_screen.json),
and [`reports/mlp_hierarchical_64board_screen.json`](reports/mlp_hierarchical_64board_screen.json).
