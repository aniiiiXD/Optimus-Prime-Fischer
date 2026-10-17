# Catan AI: execution plan for the next agents

**Status:** active execution record, 16 October 2026. Work through the gates
in order and record evidence before moving to the next one.

## Goal and current target

Build a four-player Catan agent that wins more often than our current
player-visible [`TacticalVPAgent`](Competitive/tactical_vp_agent.py) under the
same rules, seats, boards, and opponents. Keep the agent legal, hidden-state
safe, and fast enough for the eventual competition interface. The **working**
rules are the [Catanatron v5 benchmark](Benchmarks%26Evals/README.md): first to
10 victory points, standard seven-card discard limit, official spiral map,
no friendly robber, and no domestic trade offers. The hackathon rules will be
revealed on the day. We are using real Catan as the main research target and
must test which decisions survive related Catan rule families, rather than
assume this exact local configuration will be the competition game.

**Promotion means measured playing strength.** Teacher-action accuracy,
training loss, and a strong result in another Catan engine are diagnostics,
not a reason to replace the incumbent.

## Agent quick start

**Current gate: 6 (self-play RL and measured speed improvements).** G0–G3 and the G4 learned
policy diagnosis are complete. The corrected Transformer and its on-policy
fine-tune still lose substantially to FairValue, so no learned checkpoint was
promoted. A deterministic late-turn tactical scanner showed a small gain on
100 new boards in two mixed-opponent tables, but failed the predeclared
promotion gate; a direct FairValue-opponent confirmation was inconclusive.
Road Building continuation search fixed a real one-ply omission but made
negligible difference in fresh games. Robber victim and last-seat opening
changes also failed their fresh promotion screens. A selective fair-rescue
diagnostic shows opening errors cost the fine-tuned Transformer about one VP,
and isolated main-turn rescue adds about 1.3 VP, yet neither closes the game
gap. Phase-gated guided search improved the diverse opponent mix but regressed
against the strong mix. A frozen early exact-VP tactical search then passed
the 100-board mixed, direct, and uniform confirmation tables, and is the new
local v5 incumbent. Its scalar Catan-family drill is complete: all variants
remain legal, but only friendly robber has a preliminary gain and stricter
discards may regress. The next work is safely distilling the proven teacher
and preparing measured self-play. Keep every
training run within **600 CPU seconds** and every
long-running evaluation bounded with a saved report.

## Goal ledger

This is the handoff record for future agents. A goal is **done** only when its
pass condition below has evidence linked here. Keep unfinished work marked as
active or queued; do not infer completion from a started run.

| ID | Deliverable | Status | Evidence / next action |
|---|---|---|---|
| G0.1 | Player-visible observation contract and legal-action boundary | **done (2026-10-15)** | [Snapshot and adapter](Transformer/visible_observation.py), [all-seat visibility tests](Transformer/test_visible_observation.py): four seeded games and hidden-card/deck/RNG perturbation checks pass. Encoder integration continues under G0.4. |
| G0.2 | Feature schema v2, ORE-port fix, artifact compatibility guard | **done (2026-10-15)** | [Schema test](Transformer/tests/test_feature_schema.py): 3/3 pass; old v1 NPZ is rejected before training. Snapshot-to-encoder wiring continues under G0.4. |
| G0.3 | v5 isolated RNG and forced-outcome teacher collection | **done (2026-10-15)** | [Shared protocol](Catanatron/tournament_protocol.py) and [collection tests](Transformer/test_collect_protocol.py): 5/5 pass; benchmark refactor matched 16/16 smoke games. Fair-value teacher is default; AlphaBeta1 is explicit and marked unverified. |
| G0.4 | Reproducibility, visibility, seeded all-seat smoke, manifest hashes | **done (2026-10-15)** | [Gate 0 handoff](Transformer/GATE0_HANDOFF.md): 13 tests pass; 100/100 games completed with 29,676 v2 decisions; sampled encoded rows and legal labels inspected; four-seat model smoke completed. |
| G1 | Phase failure table and component/latency profiling report | **done (2026-10-15)** | [Profile and phase diagnostics](Transformer/runs/gate1_profile/DIAGNOSTICS.md): teacher 22.6 s of 55.3 s collection; opening settlement 18.8% and robber 15.5% teacher agreement on a 20 s probe. A padded-action reporting bug was found and fixed. Disagreement is not proof of bad play. |
| G2 | Versioned, varied native-game dataset | **done (2026-10-16)** | [Audit](Transformer/runs/gate2_native_v2/dataset.audit.json) passed: 100 complete games, 29,899 unique visible decisions, 0 duplicates, 64/64 raw reconstructions, 80/10/10 [frozen board split](Transformer/runs/gate2_native_v2/dataset.split.json). |
| G3 | Matched CPU-budget MLP and Transformer checkpoints | **done (2026-10-16)** | [Matched comparison](Transformer/runs/gate3_matched/model_comparison.md): 13,729-param MLP, 55.5 s to plateau; 13,857-param Transformer, 590.9 s cap. Same dataset/split. On 1,723 validation decisions with a choice, teacher agreement was 46.8% vs 60.2%. Playing strength remains untested. |
| G4 | Held-out tournament and promotion decision | **done; no promotion (2026-10-16)** | [Eight-board all-seat screen](Transformer/runs/gate4_screen/SCREEN.md): learned models 2/32 and 0/32 in the two mixes versus incumbent 12/32 and 15/32. Confirmation is not justified. |
| G4.D | Diagnose the learned-policy game collapse | **done; no learned-policy promotion (2026-10-16)** | [On-policy trace](Transformer/runs/gate4_onpolicy/diagnostic.md), [phase-swap ablation](Competitive/PHASE_SWAP_ABLATION.md), and [representation audit](Transformer/runs/gate4_screen/REPRESENTATION_AUDIT.md) found multi-phase errors and missing public counts. The [v3 screen](Transformer/runs/gate4_v3_screen/SCREEN.md) was 1/32 in both mixes. [On-policy diagnosis](Transformer/runs/gate4_dagger/DIAGNOSTIC.md), [89,899-row merge audit](Transformer/runs/gate4_merged_v3/dataset.audit.json), and [fine-tune screen](Transformer/runs/gate5_dagger_screen/SCREEN.md) found 1/32 in both mixes versus fair 12/32 and 9/32. A [fresh selective-rescue screen](Competitive/SELECTIVE_RESCUE.md) found fair openings add about 1 VP to the fine-tuned model; [isolated main-turn rescue](Competitive/PHASE_RESCUE.md) added about 1.3 VP and +7/32 wins in each mix. Discard rescue was opponent-dependent; robber and trade/end rescues did not consistently improve wins. Fair still leads; no promotion. |
| G4.F | Catan-family robustness and adaptation drill | **done; limited scalar drill (2026-10-16)** | [Adaptation map and result](FAMILY_READINESS.md) records fresh all-seat screens for target 8/12 VP, discard 5, friendly robber, and random number placement. All 320 candidate/control games completed legally. Friendly robber was +15.6 points with interval above zero; other variants were inconclusive except a possible −3.1-point discard-5 regression. No event-family transfer claim. |
| G5 | Fair search and search-derived teacher improvements | **done; TacticalVP promoted (2026-10-16)** | Earlier late-only tactical search and several structural candidates failed their promotion screens. A new frozen [early tactical VP search](Competitive/TACTICAL_VP_MIN0_CONFIRM.md), selected on eight boards then confirmed only on new seed ranges, beat FairValue by +9.75 points diverse and +10.25 strong on 100 mixed boards; direct FairValue-opponent and uniform Value/AlphaBeta tables also had lower paired 95% bounds above zero. All 4,000 games completed legally without caps. `TacticalVPAgent` with [`tactical_vp_min0.json`](Competitive/configs/tactical_vp_min0.json) is now the local v5 incumbent. |
| G6 | Self-play RL and measured speed improvements | **active; two distillation baselines rejected (2026-10-16)** | A [30,929-row, 102-board TacticalVP self-play corpus](Transformer/runs/gate6_tactical_distill/tactical_selfplay_100.audit.json) passed raw reconstruction and split audit. Its 77,569-parameter Transformer trained for 590 CPU seconds to 81.0% held-out agreement but lost by −21.9 points diverse/−15.6 strong in a fresh screen versus TacticalVP. [Phase diagnostics](Transformer/runs/gate6_tactical_distill/DISTILLATION.md) identify opening settlement (13.8%), robber (16.3%), and roads (28.7%) as the actual weak decisions. A 6× phase-weighted run raised settlement agreement to 25.0% but collapsed nonforced agreement and lost by −53.1/−28.1 points. Next: a narrow model-assisted search component with TacticalVP fallback; do not start unconstrained RL or replace the incumbent with a raw clone. |

**Current execution boundary:** keep TacticalVP as the acting policy and test
a narrow model-assisted search component, beginning with an auditable root
ranking or action-type prior. Agents must update this table with exact evidence paths and
dates at handoff. Later goals remain conditional on earlier pass criteria.

## Facts that must stay attached to the plan

| Evidence | What it tells us |
|---|---|
| The fair-value incumbent won 32/96 against three Value bots and 28/96 against three AlphaBeta1 bots in its original v5 held-out set; it also won 38/96 in a Weighted + Value + AlphaBeta1 mix and 31/96 in a Value + two AlphaBeta1 mix. [Results](Competitive/RESULTS.md) | This is our current general playing-strength reference, with uncertainty and opponent dependence. |
| Two-action fair search won 88/192 against three AlphaBeta1 bots versus 64/192 for the incumbent on matched boards, but 36/96 versus 38/96 in the stronger mixed table. [Search experiment](Competitive/SEARCH_EXPERIMENT.md) | Search can help, but the existing version is not a general upgrade. |
| The pilot Transformer used 4,000 unique positions from 14 games, collected in 7.5 seconds. Training took 590.5 CPU seconds; its best checkpoint was epoch 8 of 105 and it won 0/16 against Value and 0/16 against AlphaBeta1 in the historical v1 screen. [Dataset](Transformer/runs/pilot/dataset.json), [training](Transformer/runs/pilot/training.json) | Dataset coverage, label quality, phase balance, and CPU update cost deserve attention before an engine rewrite. The old v1 games are not numerically comparable with v5. |
| The pilot's node encoder computes ORE-port category 6, then writes `_unit(6, 6)`, which is all zeros. [Encoder](Transformer/model.py) | The ORE-port feature is absent. Fix the schema, regenerate tensors, and retrain; old weights cannot be repaired by changing the encoder alone. This bug has **not** been shown to explain all losses. |
| The old collector uses AlphaBeta1 labels without the evaluator's isolated-copy RNG and forced chance-outcome corrections. [Collector](Transformer/collect.py), [v5 runner](Benchmarks%26Evals/run.py) | Regenerate teacher labels under the corrected chance semantics. Do not infer the size of this bias without a comparison. |
| Catanatron AlphaBeta1 receives the full simulator game while labeling, and its information boundary has not been proven fair. | New training data should default to a teacher with checked hidden-state invariance; keep AlphaBeta1 labels explicitly marked as unverified/privileged until audited. |
| The fine-tuned external PPO policy, transferred without its hidden-state rollout search, won 8/64 against Value and 8/64 against AlphaBeta1. The incumbent won 21/64 and 19/64 on those same boards. [Transfer report](External/TRANSFER_EXPERIMENT.md) | Direct cross-engine weights are not our competitive agent. The upstream 85.4% result used another engine, trade-enabled search, and access to real hidden state. |

We **do not know a single exact cause** of poor learned play. The confirmed
defects and the search/engine differences are hypotheses to isolate in native,
matched games.

## Information required before a competition-ready claim

Get these facts from the actual hackathon/game contract and save them in a
versioned [rules manifest](rules_manifest.json) before final training or final
evaluation. The current manifest marks the local v5 target as provisional and
leaves unknown competition fields empty:

1. Player count, victory target, setup order, map/port/number generation,
   robber and discard rules, development-card rules, and whether domestic
   offers and responses exist.
2. Exactly what the agent observes: own hand and cards, opponent card **counts**,
   public action history, bank counts, legal moves, and any hidden fields.
3. Action format, invalid-action behavior, whether the agent can keep memory
   between decisions, and whether a call may inspect or clone game state.
4. Hardware, per-decision time and memory limits, model format, and submission
   packaging. Record measured cold-start and warm-decision latency separately.
5. Evaluation opponent mix, seat policy, board distribution, and whether the
   final scoring uses four-player wins, points, or another metric.

Until these are supplied, use the current v5 rules **as a local research
target** and label every result accordingly. If competition rules differ,
update the manifest, dataset, and benchmark together; do not carry over a win
rate as though the games were identical.

When the event specification arrives, the first agent should make a short
**contract diff** against this manifest: changed rules, visible fields, legal
actions, player count, scoring, and runtime. Then run legal-action and latency
smokes before adapting data or training. Keep the encoder and action adapter
versioned so a new family cannot silently reuse incompatible weights.

## Dependency pipeline

```text
rules + observation contract
         ↓
correct visible encoder and chance semantics
         ↓
profile + phase-specific failure report
         ↓
versioned, varied native-game dataset
         ↓
same-data MLP / Transformer training
         ↓
held-out game tournament and promotion gate
         ↓
fair search, search distillation, then self-play RL if evidence supports them
         ↓
speed engineering at the measured bottleneck
```

### Gate 0 — Lock the interface and measurement

**Make:** a typed, player-visible observation object plus legal actions. The
collector, trainer, agent, and evaluator must use one schema version. The
candidate should receive this view, not the full Catanatron `Game`. Own
private cards are allowed; opponent resource types, opponent unplayed-card
identities, deck order, and the future RNG stream are excluded. If a public
opponent card count must be computed from simulator internals, isolate that
operation in the observation builder and expose only the count.

Correct the ORE-port encoding with a **new** feature-schema version; keep old
artifacts identified as v1. Make teacher collection use the same isolated RNG
and hypothetical chance handling as benchmark v5. Freeze a manifest with
engine commit, rules, schema hash, seed ranges, and opponent versions.

**Pass when:** a visible-only agent can finish seeded games in every seat;
legal-action and hidden-hand/deck/RNG perturbation tests pass; replaying the
same seed and agent reproduces game outcomes; a small sample of encoded
observations and action IDs is manually inspected. Save test output and hashes.

### Gate 1 — Diagnose and profile before scaling

**Make:** one bounded failure report. Time game stepping, teacher scoring,
observation encoding, model forward pass, and optimizer updates separately.
Measure p50/p95 decision latency. On held-out native-game positions, break
teacher agreement and action-score gap down by opening settlement, opening
road, expansion, city/development choice, robber/discard, and late game.
Include port type and seat. Compare raw transferred PPO, pilot Transformer,
and the incumbent on **the same** native observations and legal actions where
possible. Any search comparison must use the same engine and seed/seat grid.

The old pilot had 3,005/4,000 `PLAY_TURN` rows, only 112 opening settlement
rows and 112 opening road rows. Overall action accuracy hides these failures.
Record counts and uncertainty per phase. Also measure how many examples are
from distinct games and boards, not just unique tensor hashes.

**Pass when:** the report names the largest measured time cost and at least
two concrete, phase-specific policy disagreements or game errors. It labels
teacher disagreement separately from proven bad play, and distinguishes facts
from possible causes. No simulator rewrite follows from a headline win rate
alone.

### Gate 2 — Build a versioned native dataset

**Make:** at least **20,000 unique player-visible decisions from at least 100
completed Catanatron games** for the first useful comparison. Rotate seats,
boards, and mixed opponent lineups. Collect from the corrected incumbent,
Value, AlphaBeta1 as an explicitly marked unverified behavior policy,
exploratory agents, and later our own checkpoint so the
model sees positions it will actually reach. Oversample rare openings,
robber/discard, city-vs-development, bank-trade, and finish-race decisions;
report both the raw distribution and training weights.

Store **raw visible snapshots and canonical legal actions** before generating
tensors. Each record needs rules/schema version, game and board seed, seat,
phase, legal set, chosen action, teacher identity/version, optional scores for
all legal actions, action history available to the player, and completed-game
outcome. Deduplicate exact player-visible states. Split train/validation/final
test by whole game **and board seed before augmentation or tuning**. Keep final
test seeds sealed from model selection. Label dice and unknown-card decisions
with corrected chance expectations; never use actual future draws.

**Pass when:** a manifest reports games, unique observations, phase/pattern
counts, duplicate rate, teacher mix, hidden-field audit, split hashes, and
collection throughput. A small random sample must reconstruct the same legal
actions and labels from the saved record.

### Gate 3 — Run a fair model comparison

Train an **action-conditioned MLP** and a small **board/action Transformer** on
the same dataset, split, label targets, and hardware. Start with corrected
teacher-action labels. Add teacher action scores or search-derived outcomes
only when their semantics are audited; one noisy terminal result is not a
perfect label for every move. Log phase-specific validation metrics.

The existing user budget is **at most 600 CPU seconds per training run**.
Early-stop when held-out loss stops improving, rather than using the whole
budget automatically. Report parameter count, effective optimizer steps,
examples seen, wall time, thread count, checkpoint hash, and p50/p95 inference
time. A Transformer earns its cost only if it improves held-out games over
the MLP at matched data and compute.

**Pass when:** both model families have reproducible checkpoints and the
validation split is independent by game and board. Advance to the game screen
even if one has higher action accuracy; let playing results decide.

### Gate 4 — Screen, confirm, and promote

Use the v5 legal-action tournament and the **same** boards and all four seats
for candidate and incumbent. Begin with 8–16 untouched boards per strong
opponent table. Only promising candidates receive a confirmation set of at
least **100 new boards × four seats** against Value, AlphaBeta1, and mixed
strong opponents. Include a direct composition containing both candidate
and incumbent. Report wins, final VP, illegal actions, capped games,
decision latency, and board-clustered paired 95% intervals.

**Provisional promotion gate:** at least +5 percentage points in the main
strong mixed table versus the incumbent, paired interval lower bound above
zero, no more than a 5-point observed regression in either uniform strong
tier, zero illegal actions, and p95 latency within the actual competition
limit. If the interval is wide, add games rather than declaring victory.
Keep the incumbent if the gate fails. Frozen evaluation seeds are never used
for training, parameter tuning, or picking checkpoints.

**Family robustness probe (G4.F):** after a candidate is viable under the
main rules, test supported variations such as board/number layouts, victory
target, discard/robber settings, and opponent styles. Use fresh seeds and all
seats, report legality and win rate separately by variant, and mark unsupported
variants explicitly. This is a preparation drill; it does not replace the
event's eventual rules or guarantee transfer to an unknown game family.

### Gate 5 — Add fair search and improve the teacher

Start here only if Gate 1 identifies planning errors or a Gate 4 model makes
a useful action prior. Compare policy-only, incumbent-prior search, and
learned-prior search with **identical root count, rollout budget, rules, and
leaf evaluator**. Try cheap same-turn lookahead first. For deeper search,
sample hidden hands and cards from beliefs consistent with public counts and
history, and draw fresh chance outcomes. Never clone the real private state
into a decision branch. Spend extra search on opening placements and close
endgames only if the measured gain justifies the latency.

If search beats the incumbent across mixed opponents, store action visit
counts and rollout win estimates. Train a separate outcome predictor from
these results, then distill the improved search choices into the next policy.
Do not assume a PPO critic is a calibrated rollout replacement. [Deep Catan](https://www.lamsade.dauphine.fr/~cazenave/papers/DeepCatan.pdf)
supports iterative search teaching; [Eli6th's experiment ledger](https://github.com/Eli6th/catan-rl/blob/main/training/results/EXPERIMENTS.md)
records poor results when its PPO value head replaced full rollouts.

### Gate 6 — Self-play and speed work, conditional on evidence

Use league PPO or another RL update only after a legal native supervised
policy can finish games reliably. Mix frozen incumbent, search teacher, and
prior checkpoints; test against opponents outside that league. Keep terminal
win rate as the final metric and watch for exploitation of a single trade or
opponent rule.

Optimize whichever component the profiler identifies:

- If **training updates** dominate: compare batch size, CPU thread count,
  cached tensor encoding, MLP size, and early stopping. Benchmark compilation
  or quantisation only after a warm-up and include its overhead.
- If **game stepping/collection** dominates more than half of wall time after
  those changes: consider batched workers or a fast native engine. Require
  transition, legal-action, and complete-game parity tests against the target
  rules before using its data. The existing Rust engine differs in ports,
  actions, and trade rules; a faster wrong engine creates worse training data.
- If **search** dominates: narrow roots by action type, share/copy public
  features efficiently, batch leaf inference, and allocate rollouts where
  extra search changes choices. Measure strength at equal decision time.

The 10-minute CPU limit stays the default for **each training comparison**.
Request a larger compute budget only if held-out learning curves are still
improving at the cap or the promoted model is demonstrably search limited.
More compute is an experiment, not a promise of a stronger agent.

## Bounded-agent working rules

Each future agent should take one gate or one sharply defined experiment,
then hand off a report. A run must specify **input hashes, command, seed
range, wall-time cap, output paths, and success/failure rule before launch**.
Use a short smoke test (two boards or a small data slice) before a long run.
Save checkpoints and manifests incrementally during collection. Training
runs stop at 600 CPU seconds; a first tournament screen stops after 8–16
boards per table. Run the 100-board confirmation only after a screen passes.
Do not repeat a failed run with more compute until its failure mode is named.

At handoff, write five lines: **what changed; what was run; measured result;
remaining uncertainty; exact next gate**. Preserve incumbent weights,
configs, and reports. Keep provisional screens separate from untouched
confirmation results. A negative result is a completed experiment.

## First implementation ticket

**Gate 0 plus a small Gate 1 pilot.** Create the visible-only API, version the
corrected port feature schema, apply v5 chance semantics to collection, and
prove reproducibility/hidden-field invariance. Then profile a 100-game
collection and a short model update without spending a full 10-minute run.
Deliver the contract, tests, a profiling JSON report, and a one-page failure
table. That evidence decides whether Gate 2 data collection or a different
bottleneck receives the next bounded task.
