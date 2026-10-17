# OneVOne Catan agent

For the official hackathon scoring harness, use the
[evaluation setup](hackathon/README.md):

```bash
./catanLabs/OneVOne/hackathon/setup.sh
./catanLabs/OneVOne/hackathon/evaluate.sh --games 20
```

The [1v1 experiment catalog](experiment_catalog/README.md) indexes every
two-player bot and research arm, including the separate StrategyOne and
invest/convert worktree experiments.

The goal is to maximize wins. The evaluator pairs each board seed with both
player seats, tests reference opponents, and records win rate, final VP, seat
split, errors/caps, and decision latency. The current 32-board held-out screen
completed 256 games with no caps or errors; the baseline won 150/256 (58.6%)
against the four-opponent league. Both search runs and per-opponent results are
summarized in [`TRAINING_RESULTS.md`](TRAINING_RESULTS.md).

This folder targets the local Catanatron engine in `start/catan_lab/vendor/catanatron`. It currently assumes provisional base-game 1v1 rules: 10 VP, discard limit 7, official number placement, normal robber. The hackathon observation/action contract is still unconfirmed; update `configs/default.json` after receiving it.

## Train the strategy

Training uses seeded evolutionary search over the heuristic's resource, robber,
and development-card parameters. Each candidate plays both seats against fixed
`value`, `alphabeta1`, `alphabeta2`, and tactical opponents. It selects by
average win rate, then checks the candidate on a separate set of boards. The
optimizer writes a run report and promotes to `configs/trained.json` only when
the paired held-out win-rate interval is above zero and every game completes
without errors or caps. This is direct game-outcome optimization, not a claim
that PPO or GRPO has been trained.

```bash
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/OneVOne/train.py \
  --trials 8 --train-boards 8 --validation-boards 24 \
  --budget-seconds 480
```

Training creates a new timestamped run directory. The report records the seed
ranges, config hashes, trial scores, held-out comparison, and promotion decision.
`agent.create_agent()` and the play UI load the promoted config automatically;
otherwise they use `configs/default.json`.

Two parameter searches completed without a promotable improvement. The current
playable agent remains the heuristic baseline; detailed scores and raw game
records are in `TRAINING_RESULTS.md` and the linked run directories.

## Bootstrap an MLP policy

`collect_policy_data.py` records the acting player's visible features, all
legal action candidates, the teacher's chosen action, and the completed game's
win/loss. It uses six teacher styles, swaps the focal player between seats, and
omits capped or failed games. The current training set combines two disjoint
seed ranges: 77,612 decision examples from 768 completed games on 64 boards.
It remains bootstrap data, and the trained policy lost badly on unseen boards.

```bash
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/OneVOne/collect_policy_data.py \
  --boards 16 --teachers onevone,fair,tactical,catanatron,alphabeta2,weighted \
  --seed 20272000 \
  --output catanLabs/OneVOne/data/bootstrap_20272000_16boards.jsonl.gz

PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/OneVOne/collect_policy_data.py \
  --boards 48 --teachers onevone,fair,tactical,catanatron,alphabeta2,weighted \
  --seed 20272016 \
  --output catanLabs/OneVOne/data/bootstrap_20272016_48boards.jsonl.gz

PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/OneVOne/merge_policy_data.py \
  --output catanLabs/OneVOne/data/bootstrap_20272000_64boards.jsonl.gz \
  catanLabs/OneVOne/data/bootstrap_20272000_16boards.jsonl.gz \
  catanLabs/OneVOne/data/bootstrap_20272016_48boards.jsonl.gz

PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/OneVOne/train_policy.py \
  --data catanLabs/OneVOne/data/bootstrap_20272000_64boards.jsonl.gz \
  --config catanLabs/OneVOne/configs/default.json \
  --output-dir catanLabs/OneVOne/runs/mlp-hierarchical-64board-20261016
```

The MLP has five legal-action scoring heads (build, robber/discard, trade,
development, and turn flow) plus a state value head trained on terminal
outcomes. It imitates teacher choices, gives losing-game choices less weight,
and learns the action family separately from the target within that family.
This is supervised bootstrapping, not PPO/GRPO or online self-play RL. The
expanded fit's best epoch reached 41.6% joint accuracy on non-forced held-out
choices, but its tournament was poor: 3/64 wins against the incumbent, 7/64
against Catanatron, and 3/64 against AlphaBeta depth 2. More board seeds did
not make imitation competitive. Keep the MLP as an experiment; the playable
agent remains the heuristic. The next useful data should give action-specific
search or rollout feedback, since repeating a game's final win/loss on every
decision does not show which move caused the result.

See [`POLICY_DATA.md`](POLICY_DATA.md) for feature and label details. Evaluate
the candidate against the incumbent and frozen bots with:

```bash
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/OneVOne/evaluate.py \
  --agent-file catanLabs/OneVOne/mlp_policy.py \
  --config catanLabs/OneVOne/runs/mlp-hierarchical-64board-20261016/policy_config.json \
  --boards 32 --seed 20273300 --opponents onevone,catanatron,alphabeta2 \
  --report catanLabs/OneVOne/reports/mlp_hierarchical_64board_screen.json
```

## Run it

From the workspace root:

```bash
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/OneVOne/evaluate.py \
  --boards 1 --opponents tactical \
  --report catanLabs/OneVOne/reports/smoke.json
```

That runs two games on one board seed: one game per seat. If clean, run a small screen:

```bash
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/OneVOne/evaluate.py \
  --boards 24 --opponents value,alphabeta1,tactical \
  --report catanLabs/OneVOne/reports/screen.json
```

Then freeze the agent/config and use a fresh seed range for confirmation:

```bash
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/OneVOne/evaluate.py \
  --boards 100 --seed 20271000 --opponents value,alphabeta1,tactical \
  --report catanLabs/OneVOne/reports/confirm.json
```

Available opponents: `random`, `weighted`, `value`, `alphabeta1`, `alphabeta2`, `catanatron`, `fair`, `tactical`, and `onevone`. `catanatron` is the pinned web UI's own depth-2, pruning-enabled bot; `onevone` is the current heuristic incumbent. `--max-game-seconds`, `--max-actions`, and `--max-turns` cap individual games. A cap counts as a non-win and appears in the report.

To pit our agent against Catanatron's built-in opponent across paired seeds and
both seats:

```bash
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/OneVOne/evaluate.py \
  --boards 64 --opponents catanatron --seed 20271700 \
  --report catanLabs/OneVOne/reports/vs_catanatron_64.json
```

The current default policy scored 85/128 wins (66.4%) in this paired-seat
run. See [`TRAINING_RESULTS.md`](TRAINING_RESULTS.md) and the JSON report for
the full results and limits.

## Play against it

The pinned Catanatron browser UI and its dependencies are already available in
this workspace. Start the game servers with:

```bash
./catanLabs/OneVOne/play.sh
```

Open [http://127.0.0.1:3000](http://127.0.0.1:3000), choose **Human** for one
seat and **Hyde OneVOne** for the other, and set the game to the
provisional local benchmark rules: BASE map, 10 VP, discard limit 7, and
Friendly Robber off. If training did not clear the held-out promotion gate,
the bot label still works and loads the default strategy. Stop the servers
with Ctrl+C in the terminal.

## Files

- `DESIGN.md`: win objective, rules walls, resources-to-VP routes, the search/learning plan, evidence gates, and internet references.
- `agent.py`: Catanatron-compatible player factory and 1v1 policy.
- `evaluate.py`: paired-seat 1v1 tournament runner with reproducible JSON output.
- `train.py`: held-out-gated parameter training loop.
- `play_web.py` and `play.sh`: launch the human-versus-agent browser game.
- `configs/default.json`: provisional local rules and heuristic parameters.

The implementation reuses the local FairValue/TacticalVP research policy in `catanLabs/Competitive`. It is meant to run inside this workspace and needs its pinned Catanatron source; it is not yet an event submission adapter.
