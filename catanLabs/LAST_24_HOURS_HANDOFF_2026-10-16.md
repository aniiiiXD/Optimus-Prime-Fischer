# Catan lab: last 24 hours and next-agent handoff

**As of 16 October 2026, approximately 08:15 IST.** This covers the Catan
work done since the morning of 15 October. It is an evidence map for the next
agent, not a claim that we are ready for the unreleased hackathon interface.
Read this alongside the [gated execution plan](NEXT_AGENT_PIPELINE.md),
[rules manifest](rules_manifest.json), and [hackathon deck strategy](HACKATHON_DECK_STRATEGY.md).

## Executive status

We have a runnable Catan playground, a repeatable four-player benchmark,
several CPU-trained model baselines, and a stronger player-visible search
policy. **The current local Catanatron v5 incumbent is `TacticalVPAgent` with
`tactical_min_vp=0`**, frozen in
[`tactical_vp_min0.json`](Competitive/configs/tactical_vp_min0.json). It beat
the former FairValue incumbent by **+9.75 percentage points** in a diverse
mixed-opponent table and **+10.25 points** in the main strong mixed table,
using 100 fresh boards and all four seats per table. Those are local
four-player, first-to-10, no-domestic-trade results, not event results.

The best completed from-scratch Transformer imitation run is legal and fast,
but far weaker than TacticalVP in games. Its 81.0% overall held-out teacher
agreement hides serious opening, robber, and road errors. A simple 6×
rare-phase sampling intervention made playing results worse. **Do not replace
TacticalVP with a raw Transformer clone or infer playing strength from action
accuracy.** The next ML experiment should give a model a narrow job inside
the proven policy, with TacticalVP as fallback.

The hackathon announcement confirms Catan, bot competition, limited compute,
and judging by wins plus ability to explain strategy. The linked 11-slide
deck shows a **two-player** introductory game, but the full problem statement,
rules, bot interface, hardware limit, and evaluator arrive on hackathon day.
Our four-player benchmark is a research target. The first event-day action is
a contract diff, not a claim that local win rates transfer.

## What now exists

| Area | Completed artifact | Why it matters |
|---|---|---|
| Engine | [Pinned Catanatron checkout and adapter](Catanatron/README.md), upstream commit `ecf931181b9a65bb4116a2153fb78c16f1438e00` | Seeded legal Catan games and established Random, Weighted, Value, and AlphaBeta1 bots. |
| Tournament | [Benchmark v5](Benchmarks%26Evals/README.md), [mixed evaluator](Competitive/mixed_eval.py), [paired comparator](Competitive/compare.py) | Same boards, all four candidate seats, opponent rotation, legality/cap checks, per-game JSON, VP and latency, board-clustered intervals. |
| Visibility | [Visible observation](Transformer/visible_observation.py), [feature encoder](Transformer/model.py), [shared chance protocol](Catanatron/tournament_protocol.py) | Model input has own private cards and public opponent counts, board, bank, and legal actions; hidden opponent card identities, deck order, and future RNG are excluded. |
| Native data | [Audited 100-game v3 corpus](Transformer/runs/gate2_native_v3/dataset.audit.json) and [sealed split](Transformer/runs/gate2_native_v3/dataset.split.json) | 29,899 unique visible decisions from 100 completed games, with raw reconstruction checks and board-disjoint train/validation/final partitions. |
| CPU models | [Matched MLP/Transformer](Transformer/runs/gate3_matched/model_comparison.md), [v3](Transformer/runs/gate4_v3_screen/SCREEN.md), [on-policy fine-tune](Transformer/runs/gate5_dagger_screen/SCREEN.md), [TacticalVP distillation](Transformer/runs/gate6_tactical_distill/DISTILLATION.md) | Reproducible training and game screens show what learned policies currently fail at. |
| Search incumbent | [TacticalVP confirmation](Competitive/TACTICAL_VP_MIN0_CONFIRM.md) | Actual local improvement, frozen config, fresh matched confirmation, legal play, measured latency. |
| Family drill | [Family readiness map and scalar screens](FAMILY_READINESS.md), [family evaluator](Competitive/family_eval.py) | Tests changed VP targets, discard rule, robber rule, and number layout; identifies adapters needed for larger rule changes. |
| Other engine | [CPU PPO/search reproduction](External/README.md) and [transfer experiment](External/TRANSFER_EXPERIMENT.md) | Demonstrates fast CPU training/search can work, while direct cross-engine weight transfer failed in our arena. |
| Play UI | [Catanatron web launcher](Competitive/PLAY.md) | A human can play locally against FairValue and built-in bots. The UI has **not** been switched to TacticalVP. |
| Event planning | [Deck-derived strategy](HACKATHON_DECK_STRATEGY.md) and [public strategy resources](HACKATHON_DECK_STRATEGY.md#similar-public-strategy-resources) | Converts rules and human heuristics into hypotheses to test after the contract arrives. |

The [visual status page](index.html) is useful for presentation, but the
dated reports above are the evidence for exact results. Some older READMEs
still call FairValue the incumbent; the confirmation report and this handoff
supersede that wording.

## Chronological research record

### 1. Playground and benchmark

The Catanatron source was downloaded and pinned. The benchmark plays
four-player base Catan to 10 VP, standard discard at more than seven cards,
official spiral number placement, no friendly robber, and **no direct
player-to-player trade offers**. It allows bank and port trades. Search-copy
RNG is isolated so trying a hypothetical move cannot advance the real game's
dice. Hypothetical roll/card outcomes are forced correctly; legal actions are
canonically sorted and `PYTHONHASHSEED=0` is fixed. The upstream engine code
was not changed for these protocol fixes. Old v1–v4 reports cannot be pooled
with v5 results. The benchmark rejects an action outside the provided legal
list and keeps errors or capped games in the denominator.

The basic opponent ladder was established: Random, Weighted Random, Value,
and depth-one AlphaBeta. In an early weak-opponent pilot, Value won 61/64
and AlphaBeta1 64/64 against three Weighted bots. That only shows Weighted is
weak; it does not create a universal ranking of Value versus AlphaBeta1.
The first model/teacher pilot used an older protocol and must remain separate
from the corrected v5 tables. See [benchmark instructions](Benchmarks%26Evals/README.md)
and the [historical pilot](Transformer/runs/pilot/RESULTS.md).

### 2. FairValue teacher and search experiments

FairValue added fair chance handling and a score for points, production,
reachable sites, hand synergy, army, development-card timing, and public-board
robber pressure. Its original v5 24-board all-seat results were **32/96**
against three Value bots, **28/96** against three AlphaBeta1 bots, **38/96**
against Weighted + Value + AlphaBeta1, and **31/96** against Value + two
AlphaBeta1. Visibility perturbations on more than 100 decisions did not
change its actions when hidden opponent cards, deck order, or RNG changed.
See [FairValue results](Competitive/RESULTS.md). This was the main teacher and
control until TacticalVP passed promotion.

A two-action guided search improved uniform AlphaBeta1 play: pooled over 48
boards, **88/192** wins versus FairValue's **64/192**, a +12.5-point paired
difference with a positive interval. It did not improve the stronger mixed
table (**36/96** versus **38/96** on 24 boards), so it was retained as a
specialist. Giving that search a pilot Transformer branch prior produced no
clear gain. Later phase gating suggested guided opening choices were harmful
against the stronger mix. See [search experiment](Competitive/SEARCH_EXPERIMENT.md)
and [phase ablation](Competitive/PHASE_GUIDED_ABLATION.md).

We also screened bounded, visible-only structural changes: Road Building
continuations, robber victim tie-breaks, and a last-seat joint opening pair.
Each changed decisions in real games, passed its focused legality/visibility
checks, and then failed to clear the fresh strong-mix promotion gate. The
paired results are in [Road Building](Competitive/ROAD_BUILDING_SCREEN.md),
[robber victim](Competitive/ROBBER_VICTIM_SCREEN.md), and [opening pair](Competitive/OPENING_PAIR_SCREEN.md).
An attempted ±50% FairValue weight search was deferred after same-state probes
showed almost no action flips; no full parameter sweep or new configuration
selection occurred. See [policy search note](Competitive/FAIR_VALUE_POLICY_SEARCH.md).

### 3. Visible native dataset and first CPU architecture comparison

The visible adapter and v2 schema were built, including a fix for a former
ORE-port feature bit that encoded as zero. New schema versions reject old
checkpoints and datasets. A Gate 0 check completed 100 games with 29,676
decisions in about 55.3 seconds and passed 13 focused tests. Profiling showed
teacher scoring took 22.6 seconds, behavior selection 17.6 seconds, visible
encoding 10.3 seconds, and game stepping only 0.5 seconds. It also exposed a
validation reporting bug: padded action slots were included in the old
accuracy calculation; the loss itself had been masked correctly. See
[Gate 0](Transformer/GATE0_HANDOFF.md) and [profile](Transformer/runs/gate1_profile/DIAGNOSTICS.md).

The audited Gate 2 set then collected **29,899 exact-unique player-visible
decisions from 100 complete games**, retaining raw visible snapshots and
canonical legal actions. The 80/10/10 split is by whole board/game, and the
final partition remains sealed. The later v3 migration adds public opponent
resource and development-card **counts**, already in raw snapshots, without
changing labels or split membership. Its audit found zero duplicates or
hidden-field/metadata failures and 64/64 sampled tensor reconstructions
matched. See [v3 migration](Transformer/MIGRATION_V3.md).

On the same v2 data and validation games, the 13,729-parameter MLP reached
**46.8%** fair-teacher agreement on 1,723 nonforced choices after a 55.5-second
CPU run; the 13,857-parameter Transformer reached **60.2%** after the
590.9-second cap. Yet on fresh eight-board, all-seat games both were weak:
MLP **2/32 diverse, 0/32 strong**, Transformer **2/32 diverse, 0/32 strong**,
against FairValue's **12/32 and 15/32**. The Transformer was faster than the
teacher (about 0.8 ms warm p95), but a higher imitation score did not produce
competitive play. See [matched model comparison](Transformer/runs/gate3_matched/model_comparison.md)
and [game screen](Transformer/runs/gate4_screen/SCREEN.md).

The v3 Transformer trained on choices only and added public opponent counts;
its fresh game screen remained **1/32 in each mixed table**. A DAgger-style
collector then let the model play while FairValue labeled the visited states:
98 complete games yielded 60,000 supplemental rows. The audited merged set
had 89,899 unique rows and kept the original final boards sealed. A capped
CPU fine-tune gained about 0.3–0.4 mean VP but still won only **1/32** in
each mix against FairValue's **12/32 and 9/32**. See [merge audit](Transformer/MERGE_ON_POLICY.md)
and [fine-tune screen](Transformer/runs/gate5_dagger_screen/SCREEN.md).

Policy interventions helped locate the weakness. Replacing only the model's
opening decisions with FairValue gained about **1.0–1.3 VP** but did not close
the gap. Replacing only main-turn decisions gained about **1.3 VP** and
seven more wins out of 32 in each mix, yet the hybrid still trailed full
FairValue. Robber/trade-end/discard rescue was inconsistent across opponent
mixes. This supports a **multi-phase** failure diagnosis, not one exact cause.
See [selective rescue](Competitive/SELECTIVE_RESCUE.md) and
[main-turn rescue](Competitive/PHASE_RESCUE.md).

### 4. External CPU PPO/search reproduction and transfer

We built the published [Eli6th/catan-rl](https://github.com/Eli6th/catan-rl)
Rust engine and shipped 1.108M-parameter PPO MLP on the local Apple M5 CPU.
The shipped AlphaBot uses policy-ranked root actions plus full-game rollouts.
Our seat-rotated reproduction won **145/192 (75.5%)** against three upstream
Heuristic-v1 bots to 7 VP and **153/192 (79.7%)** to 10 VP. Those are
upstream-engine results; its search starts from the true hidden state and
cannot be submitted as a fair-information Catanatron bot.

We resumed its pretrained network for about **501 CPU optimizer seconds**
total and added **9.51 million decisions** in the corrected run. The raw
10-point policy improved by roughly **18.8–22.7 win-rate points** against two
upstream heuristic families on matched boards. Original trade-filtered
AlphaBot did not improve, but allowing search to consider trade offers with
the fine-tuned model won **164/192** versus **146/192** for shipped search
against evolved upstream bots across a screen and fresh confirmation. That
shows CPU training and search can work in a fast engine. It does **not** show
that these weights beat local Catanatron opponents; the rules, ports, action
codec, opponent trade behavior, and information available to search differ.
See [full reproduction](External/README.md).

A visible-only 1,350-feature/299-action adapter brought the fine-tuned raw
policy into Catanatron, including board graph mapping and legal-action
translation. In 16 fresh boards across four seats, it won **8/64** against
Value and **8/64** against AlphaBeta1; the then-incumbent FairValue won
**21/64** and **19/64**. The faster model did not transfer playing strength.
See [transfer report](External/TRANSFER_EXPERIMENT.md). The comparison was
against FairValue because TacticalVP had not yet been promoted.

### 5. TacticalVP promotion, family screens, and new distillation

An initial TacticalVP scanner activated only late and improved FairValue by
about **+1.5/+2.0 points** in two 100-board mixed confirmations, below the
predeclared +5-point strong-mix promotion gate. A direct FairValue-opponent
confirmation was inconclusive. See [late-only decision](Competitive/TACTICAL_VP_CONFIRM.md).

We next tested when to start a deterministic two-action VP scan. Shadow
play found more legal point lines at thresholds 5 and 0 without using hidden
draws. An eight-board selection screen chose threshold **0**; the frozen
candidate was then confirmed only on new seed ranges. Its main evidence:

| Fresh 100-board table, four seats per board | TacticalVP | FairValue | Paired difference, 95% board interval |
|---|---:|---:|---:|
| Weighted + Value + AlphaBeta1 | 207/400 | 168/400 | +9.75 points [+5.5, +14.0] |
| Value + two AlphaBeta1 | 183/400 | 142/400 | +10.25 points [+6.5, +14.0] |
| FairValue + Value + AlphaBeta1 | 128/400 | 116/400 | +3.0 points [+0.25, +5.75] |
| Three Value | 163/400 | 131/400 | +8.0 points [+4.0, +12.25] |
| Three AlphaBeta1 | 173/400 | 137/400 | +9.0 points [+5.25, +12.75] |

All **4,000 candidate/control games** completed without errors or caps.
TacticalVP warm p95 decision latency was **8.30 ms** diverse and **7.87 ms**
strong, versus FairValue's 3.90 and 3.62 ms. This passed the local promotion
rule; read the [exact confirmation](Competitive/TACTICAL_VP_MIN0_CONFIRM.md)
and raw linked JSON before changing it. TacticalVP still uses the FairValue
score; its exact scan is limited to known maritime trade/build continuations.

The first scalar family drill used eight fresh boards per variant against
Value + two AlphaBeta1 bots. TacticalVP and FairValue were legal in all
target-8, target-12, discard-at-5, friendly-robber, and random-number-layout
games. Tactical minus FairValue was **0.0, +3.1, −3.1, +15.6, and +6.3
points**, respectively; only friendly robber had a positive interval in this
small screen. Discard-at-5 may regress. These are compatibility screens,
not strength claims across Catan families. Three-player topology, different
board sizes, domestic trades, and new action APIs still need versioned
adapters. See [family readiness](FAMILY_READINESS.md).

Finally, 102 TacticalVP self-play games produced **30,929 unique visible
decisions**. A 77,569-parameter two-layer width-64 Transformer trained from
random initialization for **590 CPU seconds**. It reached **81.0% overall**
held-out teacher agreement, **68.0%** where there was a choice. On fresh
eight-board screens it won **5/32 versus TacticalVP's 12/32** diverse and
**5/32 versus 10/32** strong. It matched only **13.8%** of opening settlement,
**16.3%** of robber, and **28.7%** of road choices. A 6× rare-phase sampling
run improved opening agreement to 25.0% but dropped nonforced agreement to
43.7%; it won only **2/32 versus 19/32** diverse and **3/32 versus 12/32**
strong. Both models are rejected as replacements. See [distillation report](Transformer/runs/gate6_tactical_distill/DISTILLATION.md).

### 6. Rules deck, strategy research, and playable UI

The downloaded source deck is `/Users/anixd/Downloads/Hyde@IITB.pdf` (about
61 MB, 11 slides). It shows five resources, number-triggered production,
build costs, development cards, robber, bank/port exchanges, and a 10-VP
target. Slide 2 describes two players; it does not supply the event's final
rule and API contract. The [strategy document](HACKATHON_DECK_STRATEGY.md)
records concrete ideas: score initial placements jointly; compare expected
time to city, settlement, development card, and awards; search for immediate
VP lines; value robber denial; trade for specific builds; and switch from
growth to a finish race. Its public-source section links a six-page strategy
PDF, Colonist guides, and relevant Reddit discussions. A targeted X/Twitter
search produced no substantive verifiable strategy thread. Human advice is
being treated as a source of **candidate tests**, not ground truth.

A local [playable Catanatron UI](Competitive/PLAY.md) was wired up using the
pinned engine. A human can select built-in bots or `HYDE_FAIRVALUE` and play
at `http://127.0.0.1:3000` after setup. **Important naming gap:** the UI
still registers FairValue and labels it “measured incumbent”; the actual
benchmark incumbent is now TacticalVP. Update the UI registration/label
before telling anyone they are playing the strongest local bot.

## Evidence boundaries and risks

1. **Official event contract is still unknown.** The local manifest leaves
   player count, map/rules differences, observation fields, legal-action
   format, memory, CPU/memory budgets, scoring, and opponent mix empty. The
   introductory deck's two players do not validate our four-player scores.
2. **The tournament adapter still receives a full Catanatron `Game`.** Our
   policy code has visible-only perturbation tests; the interface itself is
   not a hard restricted-observation sandbox. Future event adapters must
   expose a typed visible snapshot plus legal actions only.
3. **Opponent and engine comparisons have different contracts.** Catanatron
   AlphaBeta1 receives full simulator state; the external AlphaBot's search
   uses true hidden state. Their scores are opponent/research baselines, not
   proof of fair-information superiority.
4. **No public leaderboard or human strength claim exists.** The confirmed
   promotion is relative to FairValue on one local ruleset and local bots.
   Matched board/seat seeds reduce variance but divergent policies can still
   produce different later dice paths.
5. **Repository state is not committed.** At handoff, `catanLabs/` is an
   untracked directory in Git, and the workspace has unrelated modified and
   untracked Connect Four files. Do not use a blanket reset or clean. Preserve
   generated checkpoints and JSON evidence before restructuring the repo.
6. **Some navigation docs are stale.** `Competitive/README.md` and
   `Competitive/PLAY.md` still call FairValue the incumbent. The execution
   plan's G6 ledger and this document carry the later distillation result.

## Next agent: ordered work

1. **On event-spec release:** fill
   [`competition_contract`](rules_manifest.json) and write a short diff
   against `local_rules`. Check player count first, then scoring/target VP,
   board topology, development and robber rules, player trades, visible
   fields, legal actions, timing, and packaging. Run one complete legal game
   through the official interface before any final strength claim.
2. **Preserve the local baseline:** keep
   [`TacticalVPAgent`](Competitive/tactical_vp_agent.py) plus
   [`tactical_vp_min0.json`](Competitive/configs/tactical_vp_min0.json)
   unchanged as the control. Give every candidate the same fresh boards,
   seats, opponent assignments, rules, and cap. Use the existing paired JSON
   comparator and report wins, VP, caps/illegal moves, and p95 full-decision
   CPU latency. Never select on the sealed dataset final boards or reuse a
   confirmation seed range for candidate tuning.
3. **First ML ticket under the current local rules:** try a **narrow
   model-assisted branch ranker** for the TacticalVP scan, or an action-type
   prior. TacticalVP must still validate every returned action and act as
   fallback on low-confidence/unsupported states. Audit where the model
   changes root ordering, search nodes/time saved, and whether it misses
   known winning lines. Compare against a same-data/same-budget MLP. Keep
   each new training run within **600 CPU seconds** and screen on fresh
   eight-to-sixteen-board all-seat matches before any 100-board confirmation.
4. **If the event is two-player or has trades:** build a separate rules and
   observation schema and two-seat evaluator. Do not pad a four-player
   checkpoint or import the external trade-enabled 85.4% as a two-player
   result. Use deck/public strategy ideas as feature and search hypotheses;
   verify each by matched games under the actual rules.
5. **Presentation/UI:** expose TacticalVP as a selectable bot alongside
   FairValue, correcting the latter's label. Prepare a short explanation of
   the chosen search rule: value legal actions using visible state, examine
   deterministic trade/build continuations for real VP gains, and fall back
   to FairValue otherwise. Show the matched promotion table rather than
   training accuracy.

The [goal ledger](NEXT_AGENT_PIPELINE.md#goal-ledger) is the project tracker:
G0–G5 are complete with the outcomes stated there; G6 is active and its two
completed distillation baselines are rejected. Update that ledger and link
new reports after each bounded experiment.
