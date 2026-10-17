# Last-seat opening pair: matched game screen

**Decision, 16 October 2026: no promotion or 100-board confirmation for this
version.** The [joint planner](opening_pair_agent.py) changes only the fourth
player's consecutive four-action opening. It falls back to FairValue in all
other seats and phases. The [shadow checks](OPENING_PAIR_SHADOW.md) found
legal, hidden-state-invariant changes on real boards; this report measures
whether those changes affected finished games.

On 16 fresh boards, seeds **20273200–20273215**, we rotated the candidate
through all four seats and both opponent compositions under provisional
Catanatron v5 rules. Candidate and FairValue control used identical boards,
seats, and opponent assignments. All 256 games completed without caps or
errors. Paired intervals bootstrap whole boards.

| Opponents | Opening pair wins | FairValue wins | Difference | Paired 95% interval |
|---|---:|---:|---:|---:|
| Weighted + Value + AlphaBeta1 | 26/64 | 25/64 | +1.6 points | −3.1 to +6.2 |
| Value + two AlphaBeta1 | 22/64 | 21/64 | +1.6 points | −4.7 to +7.8 |

The point estimates are below the predeclared +5-point strong-mix threshold
and both intervals contain zero. The change affects only one seat per board,
so this version has limited leverage even if its last-seat placements improve.
The [candidate](runs/opening_pair_screen_16.json), [control](runs/opening_pair_fair_screen_16.json),
and [paired comparison](runs/opening_pair_vs_fair_screen_16.json) preserve
per-game results. FairValue remains incumbent. Extending pair planning to
other seats requires explicit modeling of intervening opponent placements
and new visibility and matched-game checks; this result does not establish
that such an extension would help.
