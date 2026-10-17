# Catan source audit and CPU pilot — 15 October 2026

The useful starting point is **Catanatron for a standard-game simulator and
classical teacher**, **BenjaminL1 for architecture ideas**, and **Charlesworth
for a runnable pretrained attention policy**. Eli's native engine is an
excellent performance reference, but its shipped model is an MLP trained with
privileged information and a shorter victory target. No cross-repository
strength ranking is justified by these pilots.

## What was actually inspected

All five repositories are downloaded under `vendor/` at the commits recorded in
[sources.json](sources.json). Source checkouts are clean. Only three selected
weight files were extracted; checksums and Git blob IDs are in the evidence.
No training or checkpoint modification ran.

| Project | Source findings | Readiness and constraint |
| --- | --- | --- |
| [BenjaminL1/catan_rl](https://github.com/BenjaminL1/catan_rl) | Tile Transformer, tripartite GNN, pointer heads, PPO and league | Built native dice/engine dependency; policy smoke works; no checkpoint files in the pinned Git tree |
| [bcollazo/catanatron](https://github.com/bcollazo/catanatron) | Python rules, legal actions, random/value/search players, Gym interface | Standard four-player base-game core works; Gym defaults to one opponent and its fixed action array omits player-to-player trade offers |
| [Eli6th/catan-rl](https://github.com/Eli6th/catan-rl) | Rust engine, native Python environment, MLP policy, root rollout search | Pretrained MLP runs; trained at 7 VP with perfect information; search code also explicitly uses perfect information |
| [Catan-RL/settlers](https://github.com/Catan-RL/settlers) | Fork of Henry Charlesworth's attention/PPO project | Pretrained `3825` checkpoint runs after two headless/NumPy compatibility shims |
| [PeterLP123/catanatron-1v1](https://github.com/PeterLP123/catanatron-1v1) | Behavioral cloning, DAgger, optional MaskablePPO, benchmark tooling | Core and value-function player run; specific 2-player 15-VP preset; no PPO training launched |

These are source-level findings, not a certification that every engine exactly
matches the organizer rules or official Catan. Trading, robber behavior, number
placement, development-card timing and visibility must be pinned separately.

## BenjaminL1 architecture: actual code, not README dimensions

`vendor/benjamin/src/catan_rl/policy/encoders.py`:

- `TileEncoder`: 19 tile vectors of width 79, projected to width 96;
  2 encoder layers, 4 attention heads, feed-forward width 192, axial positions.
- `GraphEncoder`: hex/vertex/edge message passing, hidden width 64, 2 rounds;
  preserves node representations for pointer action heads.
- `CountDevEncoder` and player encoders: development-card counts and player
  information. Current source schema uses 67 current-player features and
  69 opponent features, plus a 14-feature public global block. The README's
  54/61 table is stale for this pinned implementation.

`policy/network.py`, `CatanPolicy._encode`, `forward`, `sample`:

```text
19 tile vectors ── Transformer ──────────────┐
19 hex / 54 corner / 72 edge nodes ── GNN ──┤
own information + opponent public features ┤
development counts + public global block ─┤
                                           ↓
                                  fused 512-wide representation
                                           ↓
               six conditional action heads + value + belief/auxiliary value
```

Total instantiated parameter count: **1,380,763**. Geometry must be initialized
with `policy/board_geometry.py:build_geometry` and `CatanPolicy.set_board_geometry`.
Using the constructor's placeholder adjacency would invalidate a board benchmark.

`policy/heads.py:CatanActionHeads.sample` produces action type, corner, edge,
tile, first resource and second resource. It encodes the board once and then
conditions later heads on earlier choices. This is autoregression over fields
of one structured decision, not generation of a long text answer:

\[
\pi(a\mid o)=\prod_{h\in\text{relevant heads}}\pi_h(a_h\mid o,a_{<h}),
\qquad \log\pi(a\mid o)=\sum_h\log\pi_h(a_h\mid o,a_{<h}).
\]

`ppo/trainer.py:PPOTrainer.update` performs policy optimization;
`selfplay/league.py:League` manages frozen opponents and snapshots. Merely
having league code does not establish how a particular model was trained.

**A concrete interface mismatch found:** the shared schema marks the tile
head relevant for `PLAY_KNIGHT`, but the live environment supplies an empty
tile mask for that main-turn action and ignores its tile field. It schedules a
separate `MOVE_ROBBER` step. The policy falls back to a uniform tile sample under
the empty mask. In 256 smoke decisions, 2 such actions failed the strict schema
mask contract while the fields actually executed by the environment remained
valid. The harness records both examples; upstream code is unmodified. Resolve
this contract before reusing the PPO/BC loss and claiming every relevant head
is correctly supervised at inference time.

## Other models and action encodings

Charlesworth fork: `RL/models/build_agent_model.py:build_agent_model` builds a
**1,928,995-parameter** attention policy; tile input width 60, tile hidden
width 64, 2 tile encoder layers, 4 heads, fused observation width 512. The
tile encoder and development-card attention feed conditional action heads.
`SettlersAgentPolicy.act` selects actions; `env/wrapper.py:EnvWrapper` supplies
observations, masks, translations and validation. The selected checkpoint is
`RL/results/default_after_update_3825.pt` (7,805,063 bytes).

Eli: `training/ppo.py:PolicyValueNet` is **1350 → 512 → 512**, ReLU, with a
299-action policy and scalar value head: **1,108,268 parameters**. The shipped
checkpoint records 46,497,792 global steps for its run, resumed from another
checkpoint; this is not its total lifetime training exposure.
`rust/catan-env/src/codec.rs` defines 299 action IDs and schema version 1.
`obs.rs:encode_obs` defines a 1,350-float view and explicit `Perfect`/`Realistic`
modes. `alpha.rs:AlphaBot.choose_action` ranks root candidates with the policy,
runs sampled rollouts, and evaluates their leaves. Both root and leaf encodings
explicitly use `Visibility::Perfect`; full-state rollouts also retain hidden
information. Changing a neural observation alone would not make this search
honest under hidden-information rules.

The native simulator CLI creates `CatanGame::new`, whose default target is now
10 VP. The training headline is for 7 VP. Its documented search command therefore
does not by itself reproduce that headline at this commit. This lab did not
claim or reproduce the reported 82% search win rate.

## Measured CPU pilot

Machine: arm64 Mac, 10 physical/logical cores, 24 GiB RAM; shared Python 3.11.15,
PyTorch 2.7.1, NumPy 2.2.6. Model/BLAS/Rayon thread budgets are set to 1 for these
workers. CPU only; no CUDA device is exposed locally. Native builds are release
mode and do not run concurrently with the final measured games. Normal desktop
applications were active, so latency is a local observation rather than a
hardware-isolated benchmark.

| Experiment | Rules/opponents | Result | Median / p95 decision latency |
| --- | --- | --- | --- |
| Benjamin Transformer/GNN | 1v1, R0, 15 VP; untrained live smoke | 256 executed-field mask checks; 2 schema mismatches above; no strength score | 0.950 / 1.017 ms |
| Charlesworth pretrained attention | 4 players, 10 VP, player trading off; 3 uniform conditional-head policies | 32/32 wins | 1.844 / 1.923 ms |
| Eli pretrained MLP, training visibility | 4 players, 7 VP, perfect information; 3 heuristic-v1 bots | 151/256 wins (59.0%) | 0.0299 / 0.0315 ms |
| Eli pretrained MLP, visibility transfer | Same opponents/target, hidden opponent private block | 156/256 wins (60.9%) | 0.0295 / 0.0314 ms |
| Catanatron weighted baseline | 4 players, 10 VP; 3 weighted bots | 16/64 wins (25.0%) | 0.0005 / 0.0023 ms |
| Catanatron value heuristic | Same rules and weighted opponents | 61/64 wins (95.3%) | 0.349 / 3.990 ms |
| Catanatron alpha-beta depth 1 | Same rules and weighted opponents | 64/64 wins | 0.416 / 5.146 ms |

Separate 1v1-fork smoke: its value heuristic won 32/32 against a random legal
opponent under its own 15-VP variant. This is not evidence of PPO/Transformer
strength or a comparison with the 4-player policies.

Peak process RSS was about 196–223 MiB across these Python workers; this includes
the framework and imports, not just model weights. Full per-seat results, raw
latencies, checksums and caps are in local `artifacts/reports/`; the compact
archive retains per-game outcomes and aggregate metrics.

Interpretation limits:

- 32/32 against a weak uniform policy establishes runnable competent behavior,
  not competitive human-level strength. Its approximate Wilson interval is
  still below certainty at the lower end.
- The Eli confidence intervals overlap. These observations do not show that
  hiding information improves a policy trained with perfect information.
- Games reuse a seed across seat rotations. Wilson intervals are descriptive
  approximations; stronger claims need seed-group resampling and several
  independent evaluation suites.
- Four equally strong players have a 25% fair-share win rate; two have 50%.
  Comparing those raw percentages without their opponents/rules is misleading.
- Latencies cover different computations and include different forced-decision
  handling. Eli auto-resolves forced choices; the other adapters can include
  them. These timings do not isolate the architectural cost of attention.
- Alpha-beta depth 1 is effectively shallow action evaluation. Its 64/64 versus
  weighted bots does not prove it stronger than the value heuristic or a human.

## Recommended next experiment and 48-hour priorities

1. Obtain the organizer's exact rules, visibility, action protocol, timing budget
   and weight/training eligibility. Preserve all existing variants as studies.
2. For standard Catan, start with Catanatron's core and value-function teacher.
   It is cheap enough for broad data collection; selectively compare a deeper
   search teacher on difficult states after measuring the budget. Explicitly
   add negotiation if required: the stock Gym action array does not cover it.
3. For a Colonist-style 1v1 variant, study Benjamin's representation and the
   1v1 fork's BC/DAgger pipeline. Resolve the knight contract and use current
   schema dimensions; do not assume public pretrained weights exist.
4. Collect `(observation, legal candidates, teacher choice, final outcome,
   seat, map seed, rules version, teacher budget)` records. Split by game/map
   seed and related trajectory groups; keep public/private information separate.
   A heuristic choice is an imitation target, not a perfect-play label.
5. Train an MLP control and small Transformer on the same training examples,
   splits, losses and compute allocation. Start with 2 encoder blocks,
   width 96–128 and 4 heads; a GNN is an ablation, subject to competition rules.
6. Only after that baseline passes evaluation, test PPO fine-tuning against a
   mixed opponent pool and separately test bounded search. Keep checkpoints
   and evaluate each change before promotion.
7. Finish with CPU latency distributions, full game-time budgets, peak RSS,
   illegal-action/timeout checks, seat fairness and cold-start protocol tests.

This order turns the available projects into concrete evidence before spending
the training budget. It does not assume Catan is the confirmed hackathon game.
