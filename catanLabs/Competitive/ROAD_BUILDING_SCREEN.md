# Road Building continuation: matched game screen

**Decision, 16 October 2026: no promotion and no 100-board confirmation for
this version.** The corrected [Road Building candidate](road_building_agent.py)
reaches legal free-road continuations in a [shadow probe](ROAD_BUILDING_SHADOW.md),
but the first fresh game screens show negligible playing gain.

Both agents used the same provisional four-player v5 rules, board seeds,
candidate seats, opponent rotations, and FairValue settings. All candidate
and control games completed with zero reported errors or caps. The candidate
had three focused legality/visibility tests and a separate hidden-state/RNG
shadow check. Intervals are paired board-clustered bootstrap intervals.

| Fresh board seeds | Opponent mix | Road Building wins | FairValue wins | Paired difference | 95% interval |
|---|---|---:|---:|---:|---:|
| 20272500–20272507 | Weighted + Value + AlphaBeta1 | 14/32 | 14/32 | 0 | [0, 0] |
| 20272500–20272507 | Value + two AlphaBeta1 | 15/32 | 14/32 | +3.1 points | [0, +9.4] |
| 20272600–20272623 | Weighted + Value + AlphaBeta1 | 44/96 | 43/96 | +1.0 point | [0, +3.1] |
| 20272600–20272623 | Value + two AlphaBeta1 | 32/96 | 32/96 | 0 | [0, 0] |

The 24-board strong table also had −0.021 mean final VP for the candidate,
despite equal wins. Candidate p95 decision latency was 3.84 ms diverse and
3.73 ms strong, compared with 3.73 and 3.60 ms for FairValue. These are warm
local measurements; event latency limits are unknown. See the [eight-board
comparison](runs/road_building_vs_fair_screen_8.json) and [24-board
comparison](runs/road_building_vs_fair_screen_24.json) for per-game provenance.

This corrects a real one-ply valuation gap, but the correction did not clear
the predeclared +5-point strong-mix promotion threshold. Keep FairValue as
incumbent. The next candidate should target higher-impact decisions and must
again pass a fresh matched screen before confirmation.
