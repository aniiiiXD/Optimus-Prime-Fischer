# Official hackathon evaluation

Quick parallel evaluation, after setup:

```bash
catanLabs/OneVOne/hackathon/.venv/bin/python catanLabs/OneVOne/hackathon/quick_eval.py --games 8 --workers 4
```

Use `--games 500 --seed 1260000` for a full paired-seat run, and `--a` to
select another agent or a saved report profile. Keep the checkout, interpreter,
and environment on the native filesystem. Results record the runtime, source
hashes, seeds, CPU failures, and individual games. Linux uses physical-core
pinning; other platforms use the OS scheduler.

Nano is the local evaluation default. Quick evaluation is uncapped; add
`--clocked` to enforce the official clock. Learning strategies remain experiments.
See [experiment results](../reports/experiment_summary_20261017.json)
for what helped and what failed. `critic_agent.py` uses its cached heuristic by
default; the saved `warm_agent.py` and `online_agent.py` profiles under
`reports/critic_fast_20261017` load the respective learned checkpoints.

For learning or tuning, install `requirements-train.txt` into the same environment,
then run `critic_experiment.py`, `tune_cmaes.py`, `priority_experiments.py`, or
`resource_experiments.py` from this directory. Use a new output directory when
rerunning an experiment. Scores from different runtimes are separate experiments.

For action learning from the agent's own games:

```bash
PYTHONHASHSEED=0 catanLabs/OneVOne/hackathon/.venv/bin/python \
  catanLabs/OneVOne/hackathon/action_learning.py \
  --output catanLabs/OneVOne/reports/action_learning_new --games 40 --online-games 8 --unclocked
```

This compares the original, a shared planner with economy features, and that
same planner with learned residual action scores. Every legal move remains
selectable, both variants use identical lookahead, and verified wins are protected.
Training compares alternative actions through completed practice rollouts; no
human labels are needed. A replay update runs after four actual practice losses.
Learning stops before evaluation on separate paired boards. `--unclocked`
disables both seats' game CPU budgets; omit it to use the official clock.
Reports count only ML changes accepted by the harness and preserve
failures, CPU usage, source hashes and checkpoints. This is an experiment;
the default OneVOne agent stays unchanged.

For CMA-ES tuning followed by a confidence-gated MLP ensemble:

```bash
PYTHONHASHSEED=0 catanLabs/OneVOne/hackathon/.venv/bin/python \
  catanLabs/OneVOne/hackathon/tune_then_learn.py \
  --output catanLabs/OneVOne/reports/mlp_league_new --games 40
```

Nano is the baseline. This uses F, Nano and OneVOne opponents, separate boards for tuning, training,
gate calibration and final testing, and uncapped games. Counterfactuals continue
with our actual planner. The MLP overrides it only when the calibrated gate
permits; its checkpoint must match the frozen heuristic config. Every legal
root and Nano's full purchase search remain available to both variants.
The pipeline also screens a four-action horizon (`nano.search_depth: 4`) against
native search before tuning. Pruning limits extra lookahead, not legal actions.

Run from the workspace root. Requires `git` and `uv`; setup downloads Python
3.11 and installs an isolated runtime. It does not need Torch or the old local
Catanatron checkout.

```bash
./catanLabs/OneVOne/hackathon/setup.sh
./catanLabs/OneVOne/hackathon/evaluate.sh
```

Setup pins the [official kit](https://github.com/lyric-hyde/catan-hackathon-agent)
to commit `04fffac90e00610f60e0b06d919f262f4b3ed9db`, installs its pinned
engine commit `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`, and runs the kit's
self-check. Only the dependencies used by this heuristic and evaluator are
installed; the kit's optional neural-policy packages are omitted.

The default evaluation is 20 games against the supplied F bot, beginning at
seed 990000. Each board is played from both seats with the same dice. Choose an
even game count to keep every board paired:

```bash
./catanLabs/OneVOne/hackathon/evaluate.sh --games 100 --seed0 991000
./catanLabs/OneVOne/hackathon/evaluate.sh --b examples/random_agent.py --games 20
```

Additional arguments go directly to the unmodified official evaluator. Opponent
paths supplied with `--b` are relative to `hackathon/vendor/kit`; absolute paths
also work. The kit supplies F and random, not O.

`agent.py` retains the legacy OneVOne strategy: sample a world through the
official observation API, choose a move, return its index. It uses the existing
default/promoted config and preserves the strategy and search budgets. The
official harness enforces masking, process isolation, decision timeouts, and
the CPU clock (300 ms + 12.5 ms per decision, at most 1.5 s per game). Exhausting
the clock makes the harness play random moves for the rest of that game; check
the reported over-budget and random-move counts alongside wins.

This is a local evaluation adapter. It imports this workspace's OneVOne and
Competitive modules, so this folder alone is not a standalone submission zip.
The downloaded kit, interpreter, caches, and virtual environment are ignored
by Git. Older OneVOne tournament scores use a different protocol and should
not be compared directly with these results.

## Initial check

The [20-game run against F](../reports/hackathon_vs_f_20.txt), using seed 990000
on Python 3.11.15, won 11/20. It reported zero errors, timeouts, or crashes,
508.5 ms mean CPU per game, and 1,060 ms worst CPU. Two games exhausted the
incremental allowance, causing 51 random fallback moves. Staying below the
1.5-second ceiling alone does not ensure staying within the earlier allowance.
This is a small compatibility screen, not a reliable strength estimate.

## Resource-awareness and online learning experiment

```bash
PYTHONHASHSEED=0 catanLabs/OneVOne/hackathon/.venv/bin/python \
  catanLabs/OneVOne/hackathon/test_resource_policy.py
PYTHONHASHSEED=0 catanLabs/OneVOne/hackathon/.venv/bin/python -u \
  catanLabs/OneVOne/hackathon/resource_experiments.py
```

The kit exposes both resource hands. `resource_policy.py` uses that information
to simulate Monopoly exactly and to give the robber a bonus for blocking a
resource that brings the opponent closer to a city, settlement, or development
card. Opponent development-card identities remain hidden. The recipe estimate
does not check whether the opponent has a legal building site; it is a threat
proxy, not an exact opponent planner.

The learner starts from the existing heuristic weights, with the exact-resource
Monopoly correction enabled. It trains three parameters: log scale of hand
synergy, log scale of reachable production/buildable sites, and robber resource
need weight. Each successive batch plays new games against F. It compares the
current weights with positive and negative changes to one coordinate, retaining
the profile with most wins; ties keep the current weights. This is online
episodic parameter optimization, not a neural model, PPO, or learning during
scored evaluation. Its small training sample can easily select noisy changes.

Defaults use 72 training games across 12 boards (three profiles per batch),
then 32 games each for baseline, exact Monopoly, hand-aware robber, and the
frozen learned policy. The strongest non-baseline screen arm receives a fresh
64-game confirmation against a matched baseline. All games use the official
clock and masked observation interface. Seed ranges are separate: training
993000 onward, screen 995000 onward, confirmation 997000 onward.

Reports save every match, summaries, runnable frozen profiles, parameter
updates, and source hashes under `../reports/resource_learning_20261017/`.
Confirmation uncertainty resamples whole boards, retaining both seats together.
No default agent or promoted config is changed. `promotable` requires a
positive lower bound on the paired board-bootstrap interval and no candidate
CPU overruns, errors, timeouts, or crashes. Use `--output` with a new directory
to rerun; existing experiment directories are never overwritten.
