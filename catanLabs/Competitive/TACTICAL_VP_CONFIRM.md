# Tactical VP confirmation and promotion decision

**Decision, 16 October 2026: keep FairValue as incumbent.** The scanner is a
separate research candidate. It passed legal-action, hidden-state, and
isolated-RNG checks, and its small mixed-table gains are real measurements,
but they do not pass the predeclared Gate 4 threshold of at least +5 win-rate
points in the main strong mixed table.

All games use provisional four-player Catanatron v5 rules. Candidate and
FairValue control share board seeds, four candidate seats, canonical actions,
and opponent assignments. The mixed confirmation used fresh seeds
20272100–20272199. All 1,600 games completed without reported errors or caps.
The paired differences and intervals use board-clustered bootstrapping.

| Opponents | Tactical wins | Fair wins | Paired difference | 95% interval | Mean VP difference |
|---|---:|---:|---:|---:|---:|
| Weighted + Value + AlphaBeta1 | 170/400 | 164/400 | +1.5 points | 0.0 to +3.25 | +0.055 |
| Value + two AlphaBeta1 | 142/400 | 134/400 | +2.0 points | +0.5 to +3.5 | +0.078 |

The [mixed candidate](runs/tactical_vp_confirm_100.json),
[control](runs/tactical_vp_confirm_fair_100.json), and
[paired report](runs/tactical_vp_vs_fair_confirm_100.json) contain per-game
outcomes. Tactical all-decision p95 latency was 5.25 ms and 5.51 ms in the
two tables, measured warm on this machine; the event CPU limit is unknown.

We also evaluated a table containing **one FairValue opponent** alongside
Value and AlphaBeta1. A 24-board screen on seeds 20272200–20272223 found
33/96 tactical wins versus 28/96 FairValue-control wins, a paired +5.2 points
with 95% interval +1.0 to +9.4. Its fresh 100-board confirmation on seeds
20272300–20272399 found **129/400 versus 127/400**, a paired +0.5 points with
95% interval **−1.0 to +2.0**. The FairValue control had one game capped at
20 seconds (seed 20272339, seat 1); tactical completed all 400. The direct
confirmation is therefore inconclusive and cannot rescue the failed main
mixed-table gate. See the [24-board comparison](runs/fair_head_to_head/comparison_24.json)
and [100-board comparison](runs/fair_head_to_head/comparison_100.json).

Next bounded strategy probe: FairValue scores `PLAY_ROAD_BUILDING` before its
free roads are placed. In the native 100-game dataset, the card was legal in
321 decisions across 30 games and FairValue chose it zero times. Test a
separate chance-safe continuation policy in shadow mode, then matched fresh
boards only if it finds useful lines. Do not tune on the sealed dataset final
split or reuse these confirmation seeds for candidate selection.
