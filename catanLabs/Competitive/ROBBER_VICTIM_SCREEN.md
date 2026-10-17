# Robber victim tie-break: matched game screen

**Decision, 16 October 2026: no promotion or 100-board confirmation.** The
[candidate](robber_victim_agent.py) changes FairValue only when the chosen
robber tile has multiple legal victims; it targets the opponent with the
highest public victory points. The [shadow probe](ROBBER_VICTIM_SHADOW.md)
established that the change occurs in real games and passed visibility/RNG
checks. Its fresh matched screens do not establish a general playing gain.

All games use the provisional four-player Catanatron v5 rules, all four
candidate seats per board, identical opponent rotations, and a 20-second game
cap. Candidate and FairValue control each completed all games without errors
or caps. Paired differences use board-clustered bootstrap intervals.

| Fresh seeds | Opponents | Candidate wins | Fair wins | Paired difference | 95% interval |
|---|---|---:|---:|---:|---:|
| 20272800–20272807 | Weighted + Value + AlphaBeta1 | 15/32 | 14/32 | +3.1 points | −6.2 to +12.5 |
| 20272800–20272807 | Value + two AlphaBeta1 | 14/32 | 13/32 | +3.1 points | −9.4 to +15.6 |
| 20273000–20273023 | Weighted + Value + AlphaBeta1 | 38/96 | 41/96 | −3.1 points | −10.4 to +3.1 |
| 20273000–20273023 | Value + two AlphaBeta1 | 31/96 | 28/96 | +3.1 points | −2.1 to +8.3 |

The 24-board strong-mix point estimate is below the predeclared +5-point
promotion threshold, its interval includes zero, and the diverse mix moves
the other way. FairValue remains incumbent. The [eight-board comparison](runs/robber_victim_vs_fair_screen_8.json)
and [24-board comparison](runs/robber_victim_vs_fair_screen_24.json) contain
per-game results and latency summaries. A larger tournament on this version
would not address the current promotion gate.
