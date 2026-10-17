# Reproducing the published CPU Catan policy + search agent

We cloned [Eli6th/catan-rl](https://github.com/Eli6th/catan-rl) at commit
`021279c56834b6203480e5292e1de7246e47bd68`. It includes a fast Rust
four-player Catan engine, a 1,108,268-parameter PPO policy/value MLP, its
trained checkpoint, and `AlphaBot`: policy-ranked candidate actions followed
by random full-game rollouts. The source is MIT licensed. The upstream
[experiment ledger](catan-rl/training/results/EXPERIMENTS.md) explains the
reported 82% score and the much longer CPU training history behind the
shipped checkpoint.

This local copy is built on an Apple M5 CPU. [`upstream_local.patch`](upstream_local.patch)
records the simulator target setting, a raw-policy ablation, an optional
trade-enabled AlphaBot, and an optimizer-resume fix that applies the requested
learning rate after restoring Adam state. The Python wrappers implement seat
rotation and reproducible JSON reports.

We also [ported the fine-tuned policy into the Catanatron evaluator](TRANSFER_EXPERIMENT.md)
with visible-only features and legal-action translation. On 16 matched boards
rotated through all four seats, it won 8/64 against Value and 8/64 against
AlphaBeta1; our fair-value incumbent won 21/64 and 19/64. The upstream search
win rate does not transfer directly to this arena.

## Rules boundary

The upstream 82% claim used **four players, first to 7 VP, perfect
information, three frozen Heuristic-v1 opponents**. Its AlphaBot encodes
opponents' private hands and searches from clones of the real hidden state.
The upstream simulator also has its own board layout and 299-action codec.
These weights cannot be plugged directly into Catanatron, and its win rate
cannot be compared with our 10-point Catanatron scores. This is a runnable
CPU search and training experiment, not a fair Catanatron submission agent.

## Reproduce locally

From the workspace root, with the pinned source folder present:

```bash
cd catanLabs/External/catan-rl/rust
cargo build --release -p catan-sim
cd ../../../../
start/.venv/bin/python catanLabs/External/evaluate_upstream.py \
  --target 7 --opponent H --games 48 --seed 777 \
  --output catanLabs/External/reports/pretrained_7_h.json
```

The evaluator plays the same seed range with AlphaBot in each of four seats.
`H` means upstream Heuristic-v1; `V` means its evolved Heuristic-v2. The
default search parameters `8,96,300` mean eight policy-selected root actions,
96 rollouts per action, and a 300-turn horizon. Each report stores individual
outcomes, a 95% Wilson interval, model/binary hashes, and per-seat results.

[`evaluate_ppo_policy.py`](evaluate_ppo_policy.py) tests a `.pt` checkpoint
without search. [`compare_upstream.py`](compare_upstream.py) compares two
search reports with a board-seed bootstrap interval;
[`compare_upstream_series.py`](compare_upstream_series.py) pools independent
matched board sets.

## Measured pretrained search results

| Rules and opponents | Wins, all four seats | 95% Wilson interval |
|---|---:|---:|
| First to 7, 3× Heuristic-v1 | 145/192 (75.5%) | 69.0–81.1% |
| First to 10, 3× Heuristic-v1 | 153/192 (79.7%) | 73.4–84.8% |
| First to 10, 3× evolved Heuristic-v2 | 88/128 (68.8%) | 60.3–76.1% |

These use 48 seed indices per seat, beginning at seed 777. Individual seat
rates vary; the 7-point seat-0 rate was 38/48 (79.2%). The published 82%
is a different sample and seat protocol. Reports:
[`7-point`](reports/pretrained_7_h.json),
[`10-point`](reports/pretrained_10_h.json),
[`evolved opponent`](reports/pretrained_10_v2.json).

At seven points, the shipped network **without search** won 119/192 (62.0%)
when allowed all legal actions; the shipped search won 145/192 (75.5%).
That reproduces the direction of the published 65%-to-82% story on this CPU,
though the exact rates differ with seeds and seat rotation. The [raw-policy
report](reports/policy_all_actions_7_h.json) and [paired comparison](reports/search_vs_policy_all_7_h.json)
are saved.

## What the network and search each contribute

The simulator now has two raw-policy modes using the **same network** as
AlphaBot: `P` excludes trade proposals to match AlphaBot's original root
actions; `Q` permits all legal moves, as upstream PPO evaluation does. This
matters because the network often chooses useful trade offers, while original
AlphaBot filters them out. `T` is an experimental AlphaBot that permits trade
proposals; `A` is the unchanged original search agent.

| 10-point opponent | Shipped policy Q | Shipped search A | CPU-fine-tuned policy Q | CPU-fine-tuned search A |
|---|---:|---:|---:|---:|
| 3× Heuristic-v1, 192 games | 90 (46.9%) | **153 (79.7%)** | 126 (65.6%) | 148 (77.1%) |
| 3× Heuristic-v2, 128 games | 52 (40.6%) | **88 (68.8%)** | 81 (63.3%) | 87 (68.0%) |

All columns within a row use the same seed indices and all four seats. The
search-vs-policy comparison changes both planning and the trade-proposal
filter. Against the *filtered* policy P, shipped search A won 153 vs 74 on
the Heuristic-v1 boards and 88 vs 43 on the Heuristic-v2 boards. Thus search
has a large effect even with an identical root action set. The Q-versus-A
paired board-bootstrap difference was +32.8 points, 95% interval +23.4 to
+42.2 against Heuristic-v1, and +28.1 points, interval +17.2 to +39.1
against Heuristic-v2. See the JSON [H1 comparison](reports/search_vs_policy_all_10_h.json)
and [H2 comparison](reports/search_vs_policy_all_10_v2.json).

## CPU fine-tuning

We resumed the shipped 512-hidden checkpoint on 10-point games with mixed
policy, Heuristic-v1, and Heuristic-v2 seats. The saved source checkpoint
already records 46.5 million steps and was resumed from an earlier long run.
This adaptation is a CPU fine-tune, not a from-scratch 10-minute reproduction
of the published 82% result. We stopped an initial short run after finding
that Adam resume restored the old learning rate; its time counts against the
10-minute CPU training allowance. The corrected run uses `--minutes 8`.

The completed corrected run added **9,510,912 decisions in about 482 seconds**
of optimizer time. The stopped first run used about 19 seconds, so total
optimizer time was roughly 501 seconds, below the 600-second allowance.
The new [PyTorch checkpoint](models/cpu10_finetuned.pt) and
[Rust inference weights](models/cpu10_finetuned.ctnn) are saved separately.
Source logs are [here](reports/cpu10_finetune_fixed.log).
The [training summary](reports/cpu10_training_summary.json) records the CPU,
steps, time budget, configuration, and checkpoint hashes.

The standalone full-action policy **did improve**: on matched 10-point boards
it gained +18.8 points (95% paired board-bootstrap interval +7.3 to +30.7)
against Heuristic-v1 and +22.7 points (+13.3 to +32.0) against Heuristic-v2.
The original trade-filtered search did **not** gain: 148 vs 153 and 87 vs 88,
with paired intervals spanning zero. We retain the shipped weights for the
original trade-filtered AlphaBot. A stronger standalone policy does not guarantee a
stronger top-eight search prior, especially while useful trade actions are
filtered from the search roots.

## Trade-enabled search experiment

The optional `T` mode lets AlphaBot consider legal trade offers among its
policy-ranked search roots. It still uses eight roots, 96 rollouts per root,
and the same 300-turn horizon. The code change is a single candidate filter
switch; training weights and the rest of search are unchanged.

Against three evolved Heuristic-v2 opponents in 10-point games, all four
seats rotated:

| Board set | Fine-tuned search with trades T | Same network, original filter A | Shipped search A |
|---|---:|---:|---:|
| 16-board screen, seed 20263200 | 57/64 | 53/64 | 50/64 |
| 32-board fresh confirmation, seed 20263300 | 107/128 | 86/128 | 96/128 |
| **Pooled** | **164/192 (85.4%)** | 139/192 (72.4%) | 146/192 (76.0%) |

Pooled paired board-bootstrap differences were **+13.0 points** versus the
same fine-tuned network with trade filtering (95% interval +6.8 to +19.8)
and **+9.4 points** versus the shipped search (interval +2.6 to +16.2).
The fresh confirmation alone was +16.4 points versus the same-network
filtered search (interval +7.8 to +25.0), but +8.6 versus shipped search
with an interval starting at zero. Pooling includes the exploratory screen,
so the latter gain deserves another independent opponent-mix test before
general promotion.

Runtime is material: the 128-game confirmation took **458 seconds** for T,
**165 seconds** for the fine-tuned filtered search, and **245 seconds** for
the shipped search on this Mac. The simulator runs games in parallel, so
these are tournament wall times, not per-move latency. Trade-enabled T is
about 2.8 times slower than its same-network control on this set. The evolved
heuristic accepts trades using a simple surplus threshold, which may be
exploitable. The result does not establish strength against human or other
trade policies.

Read the [pooled comparison with shipped search](reports/trade_search_vs_pretrained_pooled_10_v2.json),
[same-network comparison](reports/trade_search_vs_filtered_pooled_10_v2.json),
and [fresh confirmation report](reports/finetuned_search_trades_confirm_10_v2.json).

On an additional 16-board Heuristic-v1 holdout (seed 20263450), trade-enabled
search won **59/64 (92.2%)** versus **53/64 (82.8%)** for shipped search. The
paired difference was +9.4 points with a 95% interval from 0 to +18.8;
this small set supports the direction but does not prove a separate H1 gain.
The [H1 report](reports/finetuned_search_trades_holdout_10_h.json) and
[paired comparison](reports/trade_search_vs_pretrained_holdout_10_h.json)
are saved.

The fine-tuned raw policy was also evaluated with its opponent-private
observation block hidden: 116/192 (60.4%) against evolved bots on one
10-point test, versus 118/192 (61.5%) with that block visible. This is a
preliminary observation-only check for the raw policy. Search still clones
the full hidden state, so neither AlphaBot variant is a fair-information
submission agent.

Run the selected upstream experiment from the workspace root:

```bash
start/.venv/bin/python catanLabs/External/evaluate_upstream.py \
  --candidate T --net catanLabs/External/models/cpu10_finetuned.ctnn \
  --target 10 --opponent V --games 32 --seed 20263300 \
  --output catanLabs/External/reports/reproduced_trade_search.json
```

To rebuild the Python binding and repeat that fine-tune, run from the
workspace root:

```bash
cd catanLabs/External/catan-rl/rust/catan-py
VIRTUAL_ENV=/Users/anixd/Documents/Hyde/start/.venv \
  /Users/anixd/Documents/Hyde/start/.venv/bin/maturin develop --release
cd ../..
/Users/anixd/Documents/Hyde/start/.venv/bin/python training/ppo.py \
  --name cpu10_finetune --minutes 8 --num-envs 256 --rollout 96 \
  --victory-target 10 --vp-delta 0 --visibility perfect \
  --lr 0.0001 --epochs 4 --minibatch 4096 --hidden 512 \
  --device cpu --seed 61018 --eval-every 999999 \
  --resume models/catan-512-best.pt --entropy-coef 0.01 \
  --train-seats policy,heuristic,policy,heuristic_v2
```

The resumed model uses PPO with legal-action masking and terminal win/loss
reward. The Rust environment runs many games at once; PyTorch computes the
policy and gradient updates on CPU. The training wall-clock cap applies to
the optimizer loop; final evaluation follows it.
