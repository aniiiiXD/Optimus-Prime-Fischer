# Strategy One: Catanatron-labelled dataset

This folder collects training examples only from the pinned external
[Catanatron](https://github.com/bcollazo/catanatron) implementation. The
teachers are its existing Value Function Base, Value Function Contender, and
Victory Point players. No locally invented heuristic chooses any saved label.

The collector records only the teacher's decisions. It rotates the teacher
between both seats and cycles every combination of the three teachers and five
existing opponents: Value Base, Value Contender, Victory Point, Weighted
Random, and Random. Each JSONL row contains a player-visible observation,
legal actions, the exact teacher action, and the terminal result.

The local rules are provisional: two players, 10 VP, and Catanatron's official
spiral number placement. Confirm the final rules and action API at the event.

Run from the workspace root:

```bash
start/.venv/bin/python catanLabs/FromScratch/StrategyOne/generate_dataset.py \
  --games 16 --seed 20261016
```

## PPO actor-critic fine-tuning

`ppo_actor_critic.py` reuses the trained legal-action MLP as the policy and
adds an MLP critic that predicts eventual win value. It runs complete 1v1
games through Catanatron. Each policy decision is scored only against the
legal actions supplied by the engine; the only learning reward is `+1` for a
win, `-1` for a loss, and `0` for a turn-cap draw. No Catan heuristic reward is
added.

The opponent pool mixes frozen older copies of the policy (`SELF`) with the
external Catanatron bots, so self-play does not train solely against the latest
version of itself.

```bash
start/.venv/bin/python catanLabs/FromScratch/StrategyOne/ppo_actor_critic.py \
  --updates 4 --episodes-per-update 16 --ppo-epochs 4 \
  --opponents SELF,F_BASE,VP,W,R \
  --output catanLabs/FromScratch/StrategyOne/runs/ppo_actor_critic_v1
```

This command writes `latest.pt` and `metrics.json`. Its win rate is a training
signal against a changing opponent mixture; use the fixed win-rate evaluator
for comparable model selection.

## Learned action calibration

`train_calibrated_ranker.py` freezes the base MLP and trains two learned
calibration layers: state-conditioned residuals for every action type and a
state-conditioned adjustment to `END_TURN`. It balances teacher cases where
`BUY_DEVELOPMENT_CARD` must outrank `END_TURN`; it does not implement a fixed
gameplay rule.

```bash
start/.venv/bin/python catanLabs/FromScratch/StrategyOne/train_calibrated_ranker.py \
  --epochs 1 --output catanLabs/FromScratch/StrategyOne/runs/calibrated_ranker_v1
```

The ordinary win-rate evaluator automatically accepts either an original MLP,
a PPO checkpoint, or a calibrated checkpoint.

## Settlement location classifier

`settlement_target_head.py` learns local board features for each legal
settlement node from external Catanatron choices. The calibrated MLP continues
to decide whether to build; this classifier only chooses the location when
multiple settlement nodes are legal.

```bash
start/.venv/bin/python catanLabs/FromScratch/StrategyOne/settlement_target_head.py \
  --epochs 4 --output catanLabs/FromScratch/StrategyOne/runs/settlement_target_v1

start/.venv/bin/python catanLabs/FromScratch/StrategyOne/evaluate_win_rate.py \
  --checkpoint catanLabs/FromScratch/StrategyOne/runs/calibrated_ranker_v1/model.pt \
  --settlement-target-checkpoint catanLabs/FromScratch/StrategyOne/runs/settlement_target_v1/epoch_4.pt \
  --games-total 128 --seed 20266400
```

The target head represents adjacent tile resources and dice numbers, ports,
roads, nearby buildings, current production, hand, and VP. It uses no fixed
weights for preferred locations. Results are in
`reports/SETTLEMENT_TARGET_EXPERIMENT.md`.

## Isolated experiment branch

This branch adds optional heuristic switches to `evaluate_win_rate.py` and a
12-epoch target-head training run. It uses symlinks to read the original
dataset and calibrated checkpoints; generated files stay in this worktree.

```bash
/Users/anixd/Documents/Hyde/start/.venv/bin/python \
  catanLabs/FromScratch/StrategyOne/evaluate_win_rate.py \
  --checkpoint catanLabs/FromScratch/StrategyOne/runs/calibrated_ranker_v1/model.pt \
  --settlement-target-checkpoint catanLabs/FromScratch/StrategyOne/runs/settlement_target_v1/epoch_4.pt \
  --settlement-heuristic pips --games-total 128 --seed 20266900
```

Alternatively, use `--buy-dev-end-hand-threshold 8` with no settlement
heuristic. These are experiment switches and are disabled by default. See
`reports/LONG_TRAINING_AND_HEURISTICS.md` for matched results.

## History and search experiments

`history_experiment.py` trains a public-history Transformer residual and a
state-only residual control from the same external teacher games. The frozen
calibrated agent supplies the base scores. Both variants are evaluated on
fresh games with `evaluate_win_rate.py --history-checkpoint PATH`.

`--same-turn-search` enables a bounded search for a winning sequence of
deterministic builds and maritime trades before ending the current turn.
`--same-turn-depth` controls its depth (default 4). The search uses the live
player's own information and the public board; it does not simulate unknown
opponent cards or future dice. The switch is disabled by default.

`loss_counterfactual_search.py` replays lost games from earlier positions. It
uses root UCB sampling, then evaluates candidates under separate matched
chance seeds. These copied simulator states contain the opponent's actual
hidden cards, so the resulting scores are **offline oracle diagnostics**, not
valid live-agent win rates or training labels for a deployed policy.
Results and limitations are in `reports/HISTORY_AND_SEARCH_EXPERIMENTS.md`.

```bash
/Users/anixd/Documents/Hyde/start/.venv/bin/python \
  catanLabs/FromScratch/StrategyOne/history_experiment.py

/Users/anixd/Documents/Hyde/start/.venv/bin/python \
  catanLabs/FromScratch/StrategyOne/evaluate_win_rate.py \
  --checkpoint catanLabs/FromScratch/StrategyOne/runs/calibrated_ranker_v1/model.pt \
  --settlement-target-checkpoint catanLabs/FromScratch/StrategyOne/runs/settlement_target_v1/epoch_4.pt \
  --same-turn-search --games-total 64 --seed 20267300

/Users/anixd/Documents/Hyde/start/.venv/bin/python \
  catanLabs/FromScratch/StrategyOne/loss_counterfactual_search.py \
  --games 6 --simulations 12 --validation-rollouts 8
```
