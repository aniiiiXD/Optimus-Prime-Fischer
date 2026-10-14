# Project targets: train a Transformer to win games

## Mission

Build a reusable, CPU-first system that can take a game's rules, learn useful
playing strategies in a Transformer, and demonstrate that those strategies
increase the chance of winning. The main research question is not whether the
model predicts familiar moves; it is whether training changes its decisions in
ways that improve results against controlled opponents.

The current OthelloGPT checkpoint is a pretrained reference for studying
inference and internal representations. It is not the model we claim to have
trained in this project. Our trainable GameBench Transformer and its checkpoints
are separate experiments.

## Learning plan

### 1. Pretrain a game model

Start from randomly initialized weights for the selected game. Generate legal
game trajectories from a mixture of random play, simple policies, and available
strong teachers. Train the model to predict the next move from the preceding
history, while applying the game's legal-action mask at decision time.

This stage should give the model a useful representation of the game's move
sequences and common positions. Next-move loss and legal-move rate are
diagnostics; they do not by themselves show that the model has learned to win.

### 2. Post-train toward winning strategies

Turn strategic ideas into controlled training variants. A strategy can provide
preferred actions, search-ranked actions, rewards, or a value target describing
the eventual result. Fine-tune a copy of the same pretrained checkpoint for
each strategy so we can compare what that strategy changes.

Use terminal outcomes as the ground-truth objective wherever possible:

```text
win = +1
draw =  0
loss = -1
```

Intermediate rewards may help learning, but keep them separate and check that
they improve final outcomes instead of teaching the model to exploit the reward
definition. Self-play can create more trajectories; a fixed set of opponents
and positions must remain available for evaluation.

### 3. Study what training changes

For each run, retain the starting checkpoint, final checkpoint, data manifest,
training configuration, and evaluation results. Compare parameter deltas,
gradient/update sizes, layer activations, policy probabilities, and selected
moves on the same saved positions. Use controlled ablations—change one strategy
or data source at a time—to test which change caused a behavioral difference.

Weight differences show where values changed, but are not by themselves an
explanation of why the model got stronger. Connect any weight or activation
finding to changed decisions and tournament outcomes.

## Dataset contract

Every generated or imported training example should be traceable to:

- game identifier and rules/version hash;
- random seed and full legal move history;
- player to move and legal actions at the target position;
- target action or target policy, if supplied by a teacher;
- terminal outcome and optional value target;
- source policy/search method and its configuration;
- split assignment at the whole-game level.

Keep training, validation, and final evaluation games separate. Apply board
rotations/reflections only when they preserve the game's rules, and only after
the whole-game split, to prevent transformed copies leaking across splits.

For a game not known before the event, there is no honest game-specific
pretraining dataset in advance. Implement its rules, generate legal trajectories
with the adapter, and create teacher/self-play data after the rules are
revealed. The reusable asset is the data-generation and experiment pipeline.

## Evaluation: winning is the target

Evaluate every checkpoint against the same frozen opponent pool, with seat or
starting-player swaps and recorded seeds. Include simple baselines and the
strongest practical search/teacher baseline for that game. Report uncertainty
when the match count is small; do not promote a strategy based on one lucky
self-play run.

| Measure | What it tells us |
|---|---|
| Win/draw/loss and score rate | Whether the model is actually getting better results |
| Elo or paired win-rate change | How policy versions compare across opponents |
| Legal-action rate | Whether the policy respects the game's action rules |
| Policy loss / teacher agreement | Whether optimization is learning its specified target |
| Value calibration, if used | Whether predicted win chances match observed outcomes |
| Move latency, throughput, peak memory | Whether the chosen policy fits the runtime budget |

Training loss is an optimization signal. Win rate against fixed opponents is
the primary success measure. A strategy is useful if it improves results
reliably; speed is a separate constraint and becomes a priority when the actual
game or evaluation budget requires it.

## Experiment discipline

Each experiment should change one meaningful factor at a time: data mixture,
strategy, objective, search teacher, model setting, or training schedule. Use
the same evaluation pool and seat swaps for the comparison. Save raw results,
not just the best run.

Record at minimum:

```text
experiment ID, parent checkpoint, game/rules version, code revision
dataset sources/counts/splits, strategy and teacher configuration, seed
model shape, optimizer/objective, training steps and wall time
train/validation loss, legal rate, win/draw/loss, score rate, Elo
inference latency, peak process memory, checkpoint path
```

## Current position and next milestones

The repository already has game adapters, a small causal Transformer, an
imitation trainer, search baselines, tournaments, and CPU benchmarks. The
existing trainer is a starting point: it imitates a teacher and does not yet
optimize terminal win outcomes.

1. Establish reproducible baseline tournaments for each supported game.
2. Make datasets and training runs versioned and inspectable.
3. Add a strategy-comparison experiment that can generate teacher-labelled
   games and compare the resulting checkpoints.
4. Add outcome/value targets and post-training only after the supervised
   baseline is measured.
5. Track policy/activation changes alongside paired tournament outcomes.
6. Revisit inference optimization only if the revealed task has a speed or
   memory limit that the measured model violates.

## Success condition

For a given game, we can reproduce the full path from rules to generated data,
pretraining, strategy-specific post-training, and paired evaluation—and show
which change improved the model's probability of winning, with runtime and
memory costs reported alongside it.
