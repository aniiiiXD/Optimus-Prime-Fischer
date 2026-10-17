# Late-turn tactical VP scanner: matched screen

The scanner retains the fair-value policy and changes a move only when an
exact, deterministic one- or two-action line raises our actual victory points
more than FairValue's immediate move. The [shadow probe](TACTICAL_VP_SHADOW.md)
found a trade → road line that won while FairValue ended its turn. The agent
and its [configuration](configs/tactical_vp_research.json) are separate from
the unchanged incumbent.

All games use provisional Catanatron four-player v5 rules. The eight-board
screen used seeds **20271900–20271907** and found no win-rate change in either
mixed opponent table: 12/32 diverse and 9/32 strong for both agents. The
additional 24-board screen used fresh seeds **20272000–20272023**, all four
seats and the same opponent assignments. All 384 candidate/control games
completed without caps or errors.

| Mix | Tactical wins | Fair wins | Paired win difference | 95% board-bootstrap interval | Mean VP difference |
|---|---:|---:|---:|---:|---:|
| Weighted + Value + AlphaBeta1 | 41/96 | 39/96 | +2.1 points | 0.0 to +5.2 | +0.03 |
| Value + two AlphaBeta1 | 38/96 | 34/96 | +4.2 points | +1.0 to +8.3 | +0.06 |

These are small absolute gains and the diverse-mix interval reaches zero.
The scanner's all-decision p95 was 6.6 ms in both mixes; its eligible-scan
p95 was 25.9 ms in the two-board shadow probe. Event runtime limits remain
unknown. The [candidate](runs/tactical_vp_24.json), [fair control](runs/tactical_vp_fair_24.json),
and [paired comparison](runs/tactical_vp_vs_fair_24.json) preserve per-game
results. A fresh 100-board all-seat confirmation is required before promotion.
