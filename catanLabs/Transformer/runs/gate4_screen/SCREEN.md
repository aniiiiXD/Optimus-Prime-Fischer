# Gate 4 screen: trained policies versus incumbent

**Result:** neither learned policy advances to confirmation or replaces the
fair-value incumbent. This is an eight-board screening experiment under the
provisional four-player Catanatron v5 rules, not an event result.

All three agents played the same seeds **20271000–20271007** in all four seats
against each mixed opponent composition. Each report contains 32 completed
games per mix, with zero capped/error games and no illegal actions. The board
seeds are outside the 100-game training dataset and its sealed final split.
Opponent assignments also match exactly across reports. Once agents choose
different moves, later dice trajectories need not remain identical. AlphaBeta1
opponents receive the simulator's full game state, so their strength is an
opponent benchmark rather than a fair teacher label.

| Agent | Diverse mix wins | Strong mix wins | Diverse mean VP | Strong mean VP | Warm p95 decision latency |
|---|---:|---:|---:|---:|---:|
| [Fair-value incumbent](incumbent.json) | 12/32 | 15/32 | 7.19 | 8.25 | 3.05–3.42 ms |
| [13.7k MLP](mlp.json) | 2/32 | 0/32 | 3.97 | 3.56 | 0.54–0.56 ms |
| [13.9k Transformer](transformer.json) | 2/32 | 0/32 | 4.44 | 3.44 | 0.78–0.81 ms |

The Transformer minus incumbent paired win-rate difference is **−31.25
percentage points** in the diverse mix (board-bootstrap 95% interval
−46.9 to −15.6 points) and **−46.9 points** in the strong mix (−62.5 to
−31.25). The [paired comparison](transformer_vs_incumbent.json) clusters by
board seed. The [MLP comparison](mlp_vs_incumbent.json) is similarly negative.

The Transformer did achieve 60.2% teacher agreement on validation decisions
with a choice versus the MLP's 46.8% ([same-row diagnostic](../gate3_matched/model_comparison.md)).
That imitation gain did not make it competitive in these games. The exact
cause remains under investigation; weak opening placement, robber choice, and
on-policy distribution shift are hypotheses, not conclusions from this table.

**Decision:** preserve the incumbent and skip the 100-board confirmation.
Next run a bounded on-policy action trace and encoder audit, then choose one
specific correction to test on fresh screen seeds. Do not tune on the sealed
dataset final partition or these screen seeds.
