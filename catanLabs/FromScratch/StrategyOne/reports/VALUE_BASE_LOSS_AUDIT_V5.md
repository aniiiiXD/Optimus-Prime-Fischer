# Value Base loss audit — matched 26-game screen

## Scope

This is a descriptive audit only. Both models played the same 26 fresh Value Base games under the RNG-isolated Catanatron evaluation contract. The calibrated MLP won 3 games and lost 23; the original MLP won 1 game and lost 25.

## Matched win-rate change

| Model | Wins | Win rate |
| --- | ---: | ---: |
| Original MLP | 1 / 26 | 3.8% |
| Calibrated MLP | 3 / 26 | 11.5% |

## Average final state in losses: calibrated MLP minus Value Base

| Metric | Difference |
| --- | ---: |
| Actual VP | -7.00 |
| Resource cards | +3.74 |
| Production pips | -51.91 |
| Roads | -6.17 |
| Settlements | -1.57 |
| Cities | -1.65 |

## Average state difference by phase: calibrated MLP minus Value Base

| Phase | VP | Resources | Production pips | Roads | Settlements | Cities |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Opening | -0.08 | +0.59 | -2.37 | -0.54 | -0.13 | +0.02 |
| Midgame | -1.06 | +3.03 | -7.87 | -1.55 | -0.65 | -0.11 |
| Late game | -2.80 | +3.30 | -17.51 | -4.89 | -2.12 | +0.12 |

## Action-type disagreements inside the 23 calibrated-model losses

| Calibrated MLP action → Value Base action | Count |
| --- | ---: |
| End Turn → Buy Development Card | 152 |
| Build City → Buy Development Card | 12 |
| Build City → End Turn | 8 |
| Roll → Play Monopoly | 3 |
| Build Settlement → Build Road | 2 |
| Play Knight → Roll | 1 |

For comparison, the original MLP had 223 End Turn → Buy Development Card disagreements in its 25 losses on these seeds.
