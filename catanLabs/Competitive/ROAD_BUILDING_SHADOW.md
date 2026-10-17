# Road Building continuation: small shadow diagnostic

[`road_building_agent.py`](road_building_agent.py) is a separate, unpromoted research candidate. It evaluates the Road Building card after one or two free road placements using copied states and validated legal actions. It compares the best leaf to the action selected by the existing FairValue policy, using FairValue's score for that action. Roll, development-card draw, theft, and Monopoly are handled by FairValue's existing chance-safe estimates, without executing their hidden outcomes. An unknown development-card purchase gets a conservative one-VP upper bound; a winning FairValue action is never displaced. When selected, the agent remembers the searched road actions and plays them only if they remain legal. The search allows at most 64 first roads and 256 second roads.

This targets a concrete one-ply issue: FairValue scores `PLAY_ROAD_BUILDING` as soon as the card is consumed, before either free road exists. In the saved 100-game v2 dataset, that card was legal in 321 decision rows across 30 games, and the Fair teacher selected it zero times. A focused synthetic position confirms the new search can legally find and then play a two-road line whose leaf exceeds ending the turn by about 99,000 FairValue score units. Three focused tests pass, including hidden-hand/deck/RNG invariance and the unknown-card guard.

## Shadow games

FairValue made every actual decision in these games. The new scanner only reported whether it would change that decision. All games used the strong mix (Value, AlphaBeta1, AlphaBeta1), all four seats, and Catanatron v5 rules.

| Seeds | Games completed | Fair decisions | Card legal | Proposed overrides | Caps/errors |
|---|---:|---:|---:|---:|---:|
| [20271200–20271201](runs/road_building_shadow_2boards.json) | 8/8 | 784 | 0 | 0 | 0 |
| [20271202–20271205](runs/road_building_shadow_4boards.json) | 16/16 | 1,494 | 2 | 0 | 0 |
| [20272400–20272423, pre-fix](runs/road_building_shadow_24boards.json) | 96/96 | 7,978 | 133 | 83 raw | 0 |
| [20272400–20272405, corrected](runs/road_building_shadow_postfix_6boards.json) | 24/24 | 1,879 | 30 | 11 | 0 |

The first six boards had only two legal-card decisions, both in one game; FairValue chose Knight both times. This prompted the larger 24-board shadow. That run found 133 legal-card decision states, but its 83 raw suggestions **must not be treated as valid overrides**: it preceded the conservative hidden VP-card guard and did not persist the planned roads. The corrected six-board rerun found 30 legal-card decision states and 11 proposed overrides. These are decision-level counts, with repeated states from the same game, and FairValue still made every actual move. The pre-fix scan had 18.2 ms median and 49.4 ms p95; the corrected six-board scan had 40.8 ms median and 46.3 ms p95. All games completed. No scan reached its branch cap; the corrected sample passed eight hidden-state perturbation checks and no shadow decision advanced live RNG.

These samples establish legality and show meaningful search disagreements, but they do not establish a win-rate gain. The full 24-board shadow used the earlier implementation, so corrected override frequency is only measured on six boards. Keep FairValue as incumbent. A future matched tournament must use the corrected candidate and fresh seeds, with latency and completed-game gates.
