# Learnable OneVOne

An official-interface 1v1 policy with interchangeable heuristic, MLP and
transformer scorers. Both neural scorers have trained checkpoints. The default
is the MLP, selected by validation outcome loss before the final game tests.

```
official observation + legal actions
    -> visible state and deterministic action features
    -> expert prior + learned outcome score for every legal action
    -> bounded guaranteed-win guard
    -> index of the selected legal action
```

The engine still owns the rules, resources, turns, build legality, card rules,
scoring and victory condition. No learned policy can return an action outside
the supplied list. Strategic preferences are soft: learned scores can change
placements, builds, trades, development purchases, robber choices and discards.
There is no mandatory tactical-VP override for ordinary nonwinning positions.
Known wins have a separate guard, including bounded sequences of up to three
deterministic own actions. Search does not roll dice, draw development cards,
resolve theft, or inspect opponent private victory points.

## Run it

Use the official kit and pinned game environment described in
`../LOGISTICS.md`. From the repository root:

```sh
HackathonDay/.venv/bin/python HackathonDay/onevone_learnable/evaluate.py \
  --games 200 --seed0 2026900000 --label my_f_200
```

The recorder enforces official clocks, plays 100 boards from both seats, checks
the kit/engine versions, and writes source hashes, game records and fault
counters to `reports/my_f_200.json`. Labels cannot overwrite existing records.
Change the seed range and label for each independent evaluation.

Play against the original OneVOne:

```sh
HackathonDay/.venv/bin/python HackathonDay/onevone_learnable/evaluate.py \
  --opponent HackathonDay/onevone_baseline/agent.py \
  --games 200 --seed0 2026910000 --label my_head_to_head_200
```

Choose a supplied scorer without editing the default configuration:

```sh
# Trained transformer
HackathonDay/.venv/bin/python HackathonDay/onevone_learnable/evaluate.py \
  --agent HackathonDay/onevone_learnable/variants/transformer/agent.py \
  --games 200 --seed0 2026920000 --label transformer_f_200

# Same redesigned policy with learning disabled, for a controlled comparison
HackathonDay/.venv/bin/python HackathonDay/onevone_learnable/evaluate.py \
  --agent HackathonDay/onevone_learnable/variants/heuristic/agent.py \
  --games 200 --seed0 2026920000 --label heuristic_f_200
```

## How heuristic and learning combine

`expert.py` exposes the original value function as named, inspectable components:
VP, production, opponent production, expansion, hand readiness, discard risk,
roads, development count and knights. Monopoly uses the opponent resources
visible under the official two-player rules. Development purchases pay the
known cost and add an unknown-card count, without pretending the hidden card
is a known Knight. Robber scoring models denial and expected theft utility.

The original VP coefficient is `3e14`. Feeding raw scores into a neural blend
would effectively silence the network. Instead, each decision converts expert
scores into ranks in `[0,1]`, preserving equal scores and the expert ordering.

```
combined = prior_weight * expert_rank
         + learning_weight * tanh(win_logit - median_legal_win_logit)
```

`config.json` controls these weights and the checkpoint. Defaults are
`prior_weight=1` and `learning_weight=0.35`. They are experimental settings,
not learned constants. Increasing learning influence allows larger departures
from the expert. Set `prior_weight=0` for selection entirely by the learned
score, while retaining legality and the win guard. Setting `scorer=heuristic`
uses the expert alone. Switching networks does not change candidate generation.

The default heuristic coefficients retain the previous expert's preferences;
the redesign makes these preferences replaceable rather than claiming that
the coefficients themselves have been solved. The learned value predicts
outcomes under the data-generating policy and can be wrong when deployed with
a different future policy. Completed games are the performance gate.

## Add another learning method

`LearnableOneVOne.prepare_candidates(game, legal)` returns a current visible
state vector and one `Candidate` per legal action, preserving legal indices.
Each candidate includes its prior score/rank, versioned 171-entry features,
immediate known VP gain and deterministic successor where available.

Implement this scorer interface and pass it to `LearnableOneVOne(..., scorer=x)`
or `Agent(scorer=x)`:

```python
class MyScorer:
    def predict(self, board, candidates):
        # board: float32 [55, 32]; candidates: float32 [legal_count, 171]
        return my_model(board, candidates)  # finite logits [legal_count]
```

The 55 tokens describe 54 board nodes and a global visible-state token. The MLP
uses the candidate features plus pooled board/global context. The transformer
uses attention with a graph adjacency bias and markers of the affected nodes.
Training uses PyTorch; playing uses equivalent NumPy exports, with no PyTorch
import or neural-framework startup in the game process.

For a policy-gradient method, override `select_candidate` to sample from
masked legal logits during data collection; log its final executed action,
probability and outcome after the win guard. For a learned position evaluator,
use the deterministic successor fields and supply expected features for random
outcomes. Neither extension requires replacing the rules engine. PPO, online
opponent adaptation and action-history memory are not implemented here.

`policy.last_decision` exposes the candidates, logits, combined scores, final
choice, prior choice, guard reason and model error. `policy.stats` counts actual
model-caused changes after the guard. Invalid logits use the expert and record
a model error. An absent configured checkpoint fails visibly at load time.
The clock reserve uses a deterministic legal fallback before exhausting the
official allowance. This is distinct from the official harness's random moves
after an overrun; those remain failures in the result.

## Generate data and train

```sh
HackathonDay/.venv/bin/python HackathonDay/onevone_learnable/collect.py \
  --games 512 --seed0 2026930000 --branches 64 --repeats 3 --workers 4 \
  --output HackathonDay/onevone_learnable/data/my_dataset

# Run in an environment with the game dependencies, NumPy and torch==2.7.1.
python HackathonDay/onevone_learnable/train.py \
  --data HackathonDay/onevone_learnable/data/my_dataset/dataset.npz \
  --architecture mlp --epochs 15 \
  --output HackathonDay/onevone_learnable/models/my_mlp

python HackathonDay/onevone_learnable/train.py \
  --data HackathonDay/onevone_learnable/data/my_dataset/dataset.npz \
  --architecture transformer --epochs 15 \
  --output HackathonDay/onevone_learnable/models/my_transformer
```

Training uses final outcomes, plus repeated continuation outcomes for alternative
legal action families. It does not simply imitate the expert's chosen moves.
The collector mixes modest exploration with the prior, logs the actual chosen
action, and isolates agent simulation RNG from game chance RNG. Whole board
seeds, both seats and continuation siblings stay in the same train/validation
split. Keep final evaluation seeds outside all training/validation data. The
data folder is ignored; source, manifests and small checkpoints are tracked.

This is an offline supervised value learner. New self-play datasets can be
collected with updated policies by changing the collector's policy factory;
automatic model promotion and iterative self-play are not implemented.

## Verify

```sh
HackathonDay/.venv/bin/python HackathonDay/onevone_learnable/tests/test_policy.py
HackathonDay/.venv/bin/python HackathonDay/onevone_learnable/audit.py \
  --games 8 --seed0 2026940000 \
  --output HackathonDay/onevone_learnable/reports/my_activity.json
```

The tests exercise visibility invariance, legal candidate coverage, input
preservation, learned overrides, guaranteed-win protection, invalid-model
fallback, clock reserve and missing-checkpoint errors. Training verifies
Torch/NumPy logits. The activity audit requires the model to alter actual
executed decisions and records examples; its unclocked games do not replace
the official evaluation.

This is a local diagnostic redesign derived from pre-event OneVOne code.
It retains that provenance and is not certified as an event-eligible submission.
The existing private opponent O is not part of these local tests.

The board-token layout and compact graph-transformer structure follow the
parallel `experiment/goal-economy-transformer` experiment; the runtime interface,
all-action expert, MLP, soft blend, training data and final tests are specific
to this redesign. No code or checkpoint is loaded from that worktree at runtime.
