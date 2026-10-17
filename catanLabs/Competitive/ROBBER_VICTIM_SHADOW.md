# Robber victim tie-break: research screen

**Status: unpromoted.** [`robber_victim_agent.py`](robber_victim_agent.py) keeps
FairValue's tile score and changes only which legal opponent to steal from
when that chosen tile has multiple victims. It picks the victim with the
highest public victory points. A tie keeps FairValue's original action.
The [research config](configs/robber_victim_research.json) is byte-identical
to the incumbent FairValue config.

This isolates a specific scoring omission: FairValue's robber scorer adds the
same `+0.45` for every nonempty victim, so two victim actions for one tile tie
regardless of which opponent is closer to 10 VP. In the saved 29,899-row v2
native raw corpus, Fair-behavior decisions included 321 robber choices. The
chosen tile had at least two legal victims in 174; in 40 decisions across 21
games, another legal victim on that tile had more public VP than the chosen
one. These are correlated decision counts from historical training games,
not a measured win-rate effect.

The [fresh two-board shadow report](runs/robber_victim_shadow_2boards.json)
used seeds 20272700–20272701, all four seats, Catanatron v5 first-to-10 rules,
and Value plus two AlphaBeta1 opponents. FairValue made every actual move.
All eight games completed without caps or errors. Of 730 candidate decisions,
31 were robber choices, 26 had multiple legal victims on the selected tile,
and the research agent proposed **six** victim changes. Eight sampled
hidden-hand, deck, and RNG perturbations preserved both choices; shadow
decisions did not advance the live RNG. The tie-break scanner took 0.0043 ms
median, 0.0053 ms p95, and 0.0077 ms maximum on the 26 eligible choices.

Two focused tests cover the higher-public-VP choice, same-tile and legal-action
constraints, validated execution, equal-VP fallback, and hidden-state/RNG
invariance. The shadow establishes incidence and implementation behavior only.
If tested for strength, use fresh matched boards, all seats, both mixed
opponent compositions, and the existing board-clustered paired comparison.
Keep FairValue as incumbent until such a screen and confirmation show a
meaningful gain.
