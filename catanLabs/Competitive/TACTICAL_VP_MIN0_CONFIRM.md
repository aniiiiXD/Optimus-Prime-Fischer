# Early tactical VP search: confirmation and promotion

**Decision, 16 October 2026: promote `TacticalVPAgent` with `tactical_min_vp=0` as the local incumbent.** It is the frozen FairValue policy plus a deterministic, player-visible two-action scan after rolling. The scan considers only our known maritime trades and builds, never samples dice, theft, hidden development cards, the deck, or opponent hands. It uses the frozen configuration in [`configs/tactical_vp_min0.json`](configs/tactical_vp_min0.json).

The lower threshold was selected in an eight-board exploration screen from three predeclared thresholds (0, 5, and 7 VP), then frozen. All promotion evidence below uses fresh, disjoint seed ranges that were never used in selection. Candidate and FairValue control use identical board seeds, all four candidate seats, opponent rotation, Catanatron v5 rules, and 20-second game caps. Every one of the 4,000 candidate/control games completed with zero errors or caps. Intervals are paired, board-clustered bootstrap 95% intervals.

| Fresh confirmation | Seed range | Tactical wins | FairValue wins | Paired win difference | 95% interval | Mean VP difference |
|---|---:|---:|---:|---:|---:|---:|
| Weighted + Value + AlphaBeta1 | 20274200–20274299 | 207/400 | 168/400 | +9.75 points | +5.5 to +14.0 | +0.565 |
| Value + two AlphaBeta1 | 20274200–20274299 | 183/400 | 142/400 | +10.25 points | +6.5 to +14.0 | +0.545 |
| FairValue + Value + AlphaBeta1 | 20274300–20274399 | 128/400 | 116/400 | +3.00 points | +0.25 to +5.75 | +0.300 |
| Three Value | 20274400–20274499 | 163/400 | 131/400 | +8.00 points | +4.0 to +12.25 | +0.552 |
| Three AlphaBeta1 | 20274400–20274499 | 173/400 | 137/400 | +9.00 points | +5.25 to +12.75 | +0.600 |

Tactical decision latency was 8.30 ms p95 in the diverse table and 7.87 ms p95 in the strong table, versus FairValue's 3.90 ms and 3.62 ms respectively, on this CPU. The event hardware limit remains unknown, but the candidate is well below the local 20-second per-game cap and has no illegal moves.

The main strong table exceeds the provisional +5-point promotion requirement with its lower paired interval above zero. It also has no regression in the two uniform strong tables and survives a direct table with the former incumbent. This promotes the candidate for local v5 work only; it does not claim performance under unreleased hackathon rules.

Raw, reproducible evidence:

- [mixed candidate](runs/tactical_vp_min0_confirm_100.json), [mixed control](runs/fair_tactical_vp_min0_confirm_100.json), and [paired mixed report](runs/tactical_vp_min0_vs_fair_confirm_100.json)
- [direct candidate](runs/tactical_vp_min0_head_to_head_confirm_100.json), [direct control](runs/fair_tactical_vp_min0_head_to_head_confirm_100.json), and [paired direct report](runs/tactical_vp_min0_vs_fair_head_to_head_confirm_100.json)
- [uniform candidate](runs/tactical_vp_min0_uniform_confirm_100.json), [uniform control](runs/fair_tactical_vp_min0_uniform_confirm_100.json), and [paired uniform report](runs/tactical_vp_min0_vs_fair_uniform_confirm_100.json)

Next gate: run the prepared Catan-family rule-variation drill against this new incumbent before claiming hackathon readiness. New candidates must be compared with this promoted policy, rather than the older FairValue-only policy.
